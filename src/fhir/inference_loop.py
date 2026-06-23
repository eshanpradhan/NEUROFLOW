#!/usr/bin/env python3
"""
NEUROFLOW - FHIR Inference Loop
===============================
Continuously running SMART-on-FHIR inference: pulls encounter-scoped data from a
HAPI FHIR R4 server, builds the 40-channel / 63-hour feature tensor, runs the
trained NeuroflowTCN, applies calibration + conformal intervals, and writes a
RiskAssessment (plus an ABCDEF CarePlan when risk crosses threshold) back into
the patient's FHIR timeline.

Run:
    source /Users/eshanpradhan/Desktop/NEUROFLOW/.venv/bin/activate
    python src/fhir/inference_loop.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.tcn import build_model, CHANNELS, N_CHANNELS  # noqa: E402
from models.uncertainty import ConformalCalibrator  # noqa: E402
from fhir.fhir_client import (  # noqa: E402
    FHIRClient, FHIRError, VITAL_LOINC, build_fhir_reference,
    confidence_tier_for_width, fhir_now,
)

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    IsotonicRegression = None


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODELS_DIR = PROJECT_ROOT / "models"
TCN_PATH = MODELS_DIR / "neuroflow.pt"
SCALER_PATH = MODELS_DIR / "scaler.json"
CONFORMAL_PATH = MODELS_DIR / "conformal.json"
RESULTS_PATH = MODELS_DIR / "evaluation_results.json"

WINDOW_HOURS = 63
ROLLING_WINDOW = 6

VITAL_NAMES = ["heart_rate", "sbp", "dbp", "spo2", "resp_rate",
               "temperature", "rass", "gcs_total"]
# LOINC -> vital channel name (inverse of VITAL_LOINC from fhir_client).
LOINC_TO_VITAL = {code: name for name, code in VITAL_LOINC.items()}

# Clinically neutral fill for fully-missing vitals (channel order above).
VITAL_NEUTRAL = np.array([80, 120, 70, 97, 18, 98.6, -0.79, 14], dtype=np.float64)
VITAL_LOW = np.array([0, 0, 0, 0, 0, 70, -5, 3], dtype=np.float64)
VITAL_HIGH = np.array([350, 300, 250, 100, 80, 120, 4, 15], dtype=np.float64)

SEDATIVE_NAMES = ["propofol", "midazolam", "dexmedetomidine", "lorazepam", "fentanyl"]
SEDATIVE_HALFLIVES = np.array([5.0, 3.0, 2.0, 14.0, 4.0], dtype=np.float64)
# RxNorm ingredient code -> sedative index.
RXNORM_TO_SEDATIVE = {
    "3498": 0,    # propofol
    "41493": 1,   # midazolam
    "263913": 2,  # dexmedetomidine
    "6470": 3,    # lorazepam
    "4337": 4,    # fentanyl
}
SEDATIVE_NAME_HINTS = {
    "propofol": 0, "diprivan": 0,
    "midazolam": 1, "versed": 1,
    "dexmedetomidine": 2, "precedex": 2,
    "lorazepam": 3, "ativan": 3,
    "fentanyl": 4, "sublimaze": 4,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _parse_fhir_dt(s: str):
    if not s:
        return None
    s = s.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def ffill_bfill(a: np.ndarray) -> np.ndarray:
    """1-D forward-fill then backward-fill NaNs; all-NaN stays NaN."""
    out = a.copy()
    last = np.nan
    for i in range(out.size):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    nxt = np.nan
    for i in range(out.size - 1, -1, -1):
        if np.isnan(out[i]):
            out[i] = nxt
        else:
            nxt = out[i]
    return out


def hours_since_row(missing: np.ndarray) -> np.ndarray:
    """Hours since last real observation; WINDOW_HOURS before first obs."""
    out = np.empty(missing.size, dtype=np.float64)
    seen = False
    prev = float(WINDOW_HOURS)
    for i in range(missing.size):
        if not missing[i]:
            prev = 0.0
            seen = True
        else:
            prev = (prev + 1.0) if seen else float(WINDOW_HOURS)
        out[i] = prev
    return out


def rolling_std_row(a: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(a.size, dtype=np.float64)
    for j in range(a.size):
        lo = max(0, j - window + 1)
        seg = a[lo:j + 1]
        out[j] = float(np.std(seg)) if seg.size else 0.0
    return out


def zscore_row(a: np.ndarray) -> np.ndarray:
    m = a.mean()
    s = a.std()
    s = s if s > 1e-6 else 1.0
    return (a - m) / s


# --------------------------------------------------------------------------- #
# Inference engine
# --------------------------------------------------------------------------- #
class NeuroflowInference:
    """Loads the trained model + calibration and turns FHIR data into a risk."""

    def __init__(self):
        self.device = get_device()

        if not TCN_PATH.exists():
            raise FileNotFoundError(f"Model not found: {TCN_PATH}")
        ckpt = torch.load(TCN_PATH, map_location="cpu", weights_only=False)
        self.model = build_model(ckpt.get("model_name", "tcn"))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.channels = ckpt.get("channels", CHANNELS)
        if self.channels != CHANNELS:
            print("[WARN] checkpoint channel order differs from tcn.CHANNELS.")

        # Scaler (MIMIC-train statistics).
        with open(SCALER_PATH) as f:
            sc = json.load(f)
        self.scaler_mean = np.asarray(sc["mean"], dtype=np.float64)
        self.scaler_std = np.asarray(sc["std"], dtype=np.float64)
        self.standardize_idx = list(sc["standardize_idx"])

        # Conformal calibrator.
        self.conformal = (ConformalCalibrator.load(CONFORMAL_PATH)
                          if CONFORMAL_PATH.exists() else None)

        # Optional isotonic calibrator, refit from stored validation predictions.
        self.isotonic = self._load_isotonic()

        self.channel_index = {c: i for i, c in enumerate(CHANNELS)}

    def _load_isotonic(self):
        if IsotonicRegression is None or not RESULTS_PATH.exists():
            return None
        try:
            with open(RESULTS_PATH) as f:
                res = json.load(f)
            cc = res.get("test_calibrated", {}).get("calibration_curve", {})
            xs = cc.get("prob_pred", [])
            ys = cc.get("prob_true", [])
            if len(xs) >= 2 and len(xs) == len(ys):
                iso = IsotonicRegression(out_of_bounds="clip",
                                         y_min=0.0, y_max=1.0)
                iso.fit(np.asarray(xs, dtype=np.float64),
                        np.asarray(ys, dtype=np.float64))
                return iso
        except (ValueError, KeyError, OSError):
            return None
        return None

    # ----- feature construction ----- #
    def build_feature_vector(self, observations, med_administrations,
                             ref_time=None):
        """FHIR resources -> [40, 63] tensor (channel order == CHANNELS)."""
        if ref_time is None:
            ref_time = datetime.now(timezone.utc)
        window_start = ref_time - timedelta(hours=WINDOW_HOURS)

        # Hourly accumulation of vital means.
        vit_sum = np.zeros((len(VITAL_NAMES), WINDOW_HOURS), dtype=np.float64)
        vit_cnt = np.zeros((len(VITAL_NAMES), WINDOW_HOURS), dtype=np.float64)
        last_obs_hour = -1

        for obs in observations:
            name = self._observation_vital(obs)
            if name is None:
                continue
            t = _parse_fhir_dt(obs.get("effectiveDateTime", ""))
            val = self._observation_value(obs)
            if t is None or val is None:
                continue
            hour = int((t - window_start).total_seconds() // 3600)
            if hour < 0 or hour >= WINDOW_HOURS:
                continue
            vi = VITAL_NAMES.index(name)
            if not (VITAL_LOW[vi] <= val <= VITAL_HIGH[vi]):
                continue
            vit_sum[vi, hour] += val
            vit_cnt[vi, hour] += 1.0
            last_obs_hour = max(last_obs_hour, hour)

        with np.errstate(divide="ignore", invalid="ignore"):
            means = vit_sum / vit_cnt
        means[vit_cnt == 0] = np.nan
        missing = (vit_cnt == 0)

        # Sedative hourly amounts -> PK onboard.
        sed_amt = np.zeros((len(SEDATIVE_NAMES), WINDOW_HOURS), dtype=np.float64)
        for med in med_administrations:
            di = self._med_sedative_index(med)
            if di is None:
                continue
            t = self._med_time(med)
            amt = self._med_amount(med)
            if t is None:
                continue
            hour = int((t - window_start).total_seconds() // 3600)
            if hour < 0 or hour >= WINDOW_HOURS:
                continue
            sed_amt[di, hour] += amt
            last_obs_hour = max(last_obs_hour, hour)

        # Assemble channels in CHANNELS order.
        feat = np.zeros((N_CHANNELS, WINDOW_HOURS), dtype=np.float64)
        filled = np.empty_like(means)
        for vi, name in enumerate(VITAL_NAMES):
            f = ffill_bfill(means[vi])
            f = np.where(np.isnan(f), VITAL_NEUTRAL[vi], f)
            filled[vi] = f
            feat[self.channel_index[name]] = f
            feat[self.channel_index[f"{name}_mask"]] = missing[vi].astype(np.float64)
            feat[self.channel_index[f"{name}_hours_since"]] = \
                hours_since_row(missing[vi])

        hr = filled[VITAL_NAMES.index("heart_rate")]
        sbp = filled[VITAL_NAMES.index("sbp")]
        rr = filled[VITAL_NAMES.index("resp_rate")]
        feat[self.channel_index["hrv_proxy"]] = rolling_std_row(hr, ROLLING_WINDOW)
        composite = zscore_row(hr) + zscore_row(sbp) + zscore_row(rr)
        feat[self.channel_index["phys_instability"]] = \
            rolling_std_row(composite, ROLLING_WINDOW)
        feat[self.channel_index["charting_density"]] = vit_cnt.sum(axis=0)

        start_hour = window_start.hour
        clock = (start_hour + np.arange(WINDOW_HOURS)) % 24
        feat[self.channel_index["circadian_sin"]] = np.sin(2 * np.pi * clock / 24)
        feat[self.channel_index["circadian_cos"]] = np.cos(2 * np.pi * clock / 24)

        decay = np.power(0.5, 1.0 / SEDATIVE_HALFLIVES)
        for di, name in enumerate(SEDATIVE_NAMES):
            amt = sed_amt[di]
            feat[self.channel_index[f"amt_{name}"]] = amt
            onboard = np.zeros(WINDOW_HOURS, dtype=np.float64)
            prev = 0.0
            for t in range(WINDOW_HOURS):
                prev = prev * decay[di] + amt[t]
                onboard[t] = prev
            feat[self.channel_index[f"onboard_{name}"]] = onboard

        n_obs_hours = last_obs_hour + 1 if last_obs_hour >= 0 else 0
        pad = (np.arange(WINDOW_HOURS) >= n_obs_hours).astype(np.float64)
        feat[self.channel_index["pad"]] = pad

        # Standardize (MIMIC-train statistics) on the configured channels.
        for ci in self.standardize_idx:
            feat[ci] = (feat[ci] - self.scaler_mean[ci]) / self.scaler_std[ci]

        return feat, n_obs_hours

    @staticmethod
    def _observation_vital(obs):
        for coding in (obs.get("code", {}) or {}).get("coding", []):
            code = coding.get("code")
            if code in LOINC_TO_VITAL:
                return LOINC_TO_VITAL[code]
        return None

    @staticmethod
    def _observation_value(obs):
        vq = obs.get("valueQuantity")
        if vq and "value" in vq:
            try:
                return float(vq["value"])
            except (TypeError, ValueError):
                return None
        if "valueInteger" in obs:
            return float(obs["valueInteger"])
        return None

    @staticmethod
    def _med_sedative_index(med):
        mcc = (med.get("medicationCodeableConcept", {}) or {})
        for coding in mcc.get("coding", []):
            code = str(coding.get("code", ""))
            if code in RXNORM_TO_SEDATIVE:
                return RXNORM_TO_SEDATIVE[code]
            disp = (coding.get("display", "") or "").lower()
            for hint, di in SEDATIVE_NAME_HINTS.items():
                if hint in disp:
                    return di
        text = (mcc.get("text", "") or "").lower()
        for hint, di in SEDATIVE_NAME_HINTS.items():
            if hint in text:
                return di
        return None

    @staticmethod
    def _med_time(med):
        if "effectiveDateTime" in med:
            return _parse_fhir_dt(med["effectiveDateTime"])
        return _parse_fhir_dt((med.get("effectivePeriod", {}) or {}).get("start", ""))

    @staticmethod
    def _med_amount(med):
        dose = (med.get("dosage", {}) or {}).get("dose", {}) or {}
        try:
            return float(dose.get("value", 1.0))
        except (TypeError, ValueError):
            return 1.0

    # ----- prediction ----- #
    def predict(self, patient_id, encounter_id, client: FHIRClient) -> dict:
        observations = client.get_observations(patient_id, encounter_id,
                                                hours_back=WINDOW_HOURS)
        meds = client.get_medication_administrations(patient_id, encounter_id,
                                                     hours_back=WINDOW_HOURS)
        feat, n_obs_hours = self.build_feature_vector(observations, meds)

        x = torch.from_numpy(feat[None, :, :]).float().to(self.device)
        pad = torch.from_numpy(feat[self.channel_index["pad"]][None, :]).float() \
            .to(self.device)
        with torch.no_grad():
            idx = self.model.last_valid_index(pad)
            logit = self.model.predict_at(x, idx)
            prob_raw = float(torch.sigmoid(logit).cpu().numpy().ravel()[0])

        prob = prob_raw
        if self.isotonic is not None:
            prob = float(self.isotonic.predict([prob_raw])[0])
        prob = float(min(1.0, max(0.0, prob)))

        if self.conformal is not None:
            low, high, _ = self.conformal.interval(np.array([prob]))
            low, high = float(low[0]), float(high[0])
        else:
            low, high = max(0.0, prob - 0.15), min(1.0, prob + 0.15)
        width = max(0.0, high - low)

        return {
            "probability": round(prob, 4),
            "probability_raw": round(prob_raw, 4),
            "interval_low": round(low, 4),
            "interval_high": round(high, 4),
            "width": round(width, 4),
            "confidence_tier": confidence_tier_for_width(width),
            "n_observations": len(observations),
            "n_med_administrations": len(meds),
            "hours_of_data": int(n_obs_hours),
        }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _find_active_careplan(client: FHIRClient, patient_id, encounter_id):
    params = {
        "patient": build_fhir_reference("Patient", patient_id),
        "encounter": build_fhir_reference("Encounter", encounter_id),
        "status": "active",
        "_count": "1",
    }
    try:
        plans = client._search_all("CarePlan", params, max_pages=1)
        return plans[0]["id"] if plans else None
    except FHIRError:
        return None


def run_once(patient_id, encounter_id, client: FHIRClient,
             inference: NeuroflowInference, threshold: float = 0.3) -> dict:
    result = inference.predict(patient_id, encounter_id, client)

    ra_id = client.write_risk_assessment(
        patient_id, encounter_id,
        probability=result["probability"],
        interval_low=result["interval_low"],
        interval_high=result["interval_high"],
        confidence_tier=result["confidence_tier"],
    )
    result["risk_assessment_id"] = ra_id

    careplan_id = _find_active_careplan(client, patient_id, encounter_id)
    result["careplan_created"] = False
    if careplan_id is None:
        if result["probability"] >= threshold:
            careplan_id = client.create_abcdef_careplan(
                patient_id, encounter_id, result["probability"])
            result["careplan_created"] = True
    result["careplan_id"] = careplan_id

    if careplan_id is not None:
        result["abcdef_compliance"] = client.score_abcdef_compliance(
            patient_id, encounter_id, careplan_id)
    else:
        result["abcdef_compliance"] = None

    return result


def run_loop(interval_seconds: int = 3600,
             base_url: str = "http://localhost:8080/fhir",
             threshold: float = 0.5):
    client = FHIRClient(base_url)
    inference = NeuroflowInference()
    print(f"[{fhir_now()}] NEUROFLOW inference loop started "
          f"(interval {interval_seconds}s, threshold {threshold}).")
    while True:
        cycle_start = fhir_now()
        try:
            encounters = client.get_active_encounters()
        except FHIRError as e:
            print(f"[{cycle_start}] [ERROR] active-encounter discovery: {e}")
            time.sleep(interval_seconds)
            continue

        print(f"[{cycle_start}] cycle: {len(encounters)} active encounter(s).")
        for enc in encounters:
            enc_id = enc.get("id", "")
            subj = (enc.get("subject", {}) or {}).get("reference", "")
            pat_id = subj.split("/", 1)[1] if "/" in subj else ""
            if not enc_id or not pat_id:
                continue
            try:
                r = run_once(pat_id, enc_id, client, inference, threshold)
                tier = r["confidence_tier"]
                print(f"    Encounter/{enc_id}: risk {r['probability']:.3f} "
                      f"[{r['interval_low']:.2f}-{r['interval_high']:.2f}] "
                      f"({tier}); RA {r['risk_assessment_id']}; "
                      f"CarePlan {r['careplan_id']}"
                      f"{' (new)' if r['careplan_created'] else ''}")
            except (FHIRError, Exception) as e:  # noqa: BLE001
                print(f"    Encounter/{enc_id}: [ERROR] {type(e).__name__}: {e}")

        time.sleep(interval_seconds)


# --------------------------------------------------------------------------- #
# Synthetic seeding (DEMO/TEST ONLY - not real patient data)
# --------------------------------------------------------------------------- #
def seed_synthetic_patient(base_url: str = "http://localhost:8080/fhir"):
    """Create a SYNTHETIC Patient + Encounter + Observations + meds on HAPI.

    This data is entirely fabricated for demo/testing of the FHIR round-trip.
    It is NOT derived from MIMIC-IV, eICU, or any real patient record.
    """
    client = FHIRClient(base_url)
    now = datetime.now(timezone.utc)

    patient = {
        "resourceType": "Patient",
        "active": True,
        "name": [{"use": "official", "family": "SyntheticTest",
                  "given": ["Neuroflow"]}],
        "gender": "male",
        "birthDate": "1958-04-12",
    }
    pat_id = client._post("Patient", patient)["id"]

    encounter = {
        "resourceType": "Encounter",
        "status": "in-progress",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "ACUTE", "display": "inpatient acute",
        },
        "type": [{"text": "ICU admission (synthetic)"}],
        "subject": {"reference": build_fhir_reference("Patient", pat_id)},
        "period": {"start": (now - timedelta(hours=24)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")},
    }
    enc_id = client._post("Encounter", encounter)["id"]

    pat_ref = build_fhir_reference("Patient", pat_id)
    enc_ref = build_fhir_reference("Encounter", enc_id)

    # Realistic-ICU synthetic vital values per channel (within plausible ranges).
    vital_values = {
        "heart_rate": [88, 92, 101, 96], "sbp": [128, 118, 110, 122],
        "dbp": [70, 65, 60, 68], "spo2": [97, 96, 95, 98],
        "resp_rate": [18, 20, 22, 19], "temperature": [98.6, 99.1, 100.4, 99.5],
        "rass": [0, -1, -2, -1], "gcs_total": [15, 14, 13, 14],
    }
    hours_ago = [22, 16, 8, 2]
    n_obs = 0
    for name, series in vital_values.items():
        loinc = VITAL_LOINC[name]
        for val, ha in zip(series, hours_ago):
            t = (now - timedelta(hours=ha)).strftime("%Y-%m-%dT%H:%M:%SZ")
            obs = {
                "resourceType": "Observation",
                "status": "final",
                "category": [{"coding": [{
                    "system": ("http://terminology.hl7.org/CodeSystem/"
                               "observation-category"),
                    "code": "vital-signs"}]}],
                "code": {"coding": [{"system": "http://loinc.org",
                                     "code": loinc, "display": name}]},
                "subject": {"reference": pat_ref},
                "encounter": {"reference": enc_ref},
                "effectiveDateTime": t,
                "valueQuantity": {"value": val, "unit": name,
                                  "system": "http://unitsofmeasure.org",
                                  "code": "1"},
            }
            client._post("Observation", obs)
            n_obs += 1

    # Synthetic propofol administrations (RxNorm 3498).
    n_med = 0
    for ha, dose in [(20, 200.0), (12, 250.0), (4, 180.0)]:
        t = (now - timedelta(hours=ha)).strftime("%Y-%m-%dT%H:%M:%SZ")
        med = {
            "resourceType": "MedicationAdministration",
            "status": "completed",
            "medicationCodeableConcept": {
                "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "3498", "display": "propofol"}],
                "text": "propofol"},
            "subject": {"reference": pat_ref},
            "context": {"reference": enc_ref},
            "effectiveDateTime": t,
            "dosage": {"dose": {"value": dose, "unit": "mg",
                                "system": "http://unitsofmeasure.org",
                                "code": "mg"}},
        }
        client._post("MedicationAdministration", med)
        n_med += 1

    print(f"  Seeded SYNTHETIC Patient/{pat_id}, Encounter/{enc_id} "
          f"({n_obs} observations, {n_med} med administrations).")
    return pat_id, enc_id


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 70)
    print("  NEUROFLOW - FHIR Inference Loop (single-cycle demo)")
    print("=" * 70)
    base_url = "http://localhost:8080/fhir"
    client = FHIRClient(base_url)

    try:
        cap = client.capability()
        print(f"  FHIR server            : {cap.get('fhirVersion', '?')} "
              f"@ {base_url}")
    except FHIRError as e:
        print(f"  [ERROR] FHIR server not reachable: {e}")
        print("  Start it with: docker start neuroflow-fhir")
        sys.exit(1)

    print()
    print(">>> Seeding synthetic patient (demo data only)")
    pat_id, enc_id = seed_synthetic_patient(base_url)
    # Allow HAPI's search index to settle before searching by reference.
    time.sleep(2)

    print()
    print(">>> Loading model and running one inference cycle")
    inference = NeuroflowInference()
    print(f"  device                 : {inference.device}")
    print(f"  isotonic calibrator    : "
          f"{'loaded' if inference.isotonic is not None else 'none (raw sigmoid)'}")
    print(f"  conformal calibrator   : "
          f"{'loaded' if inference.conformal is not None else 'none'}")

    result = run_once(pat_id, enc_id, client, inference, threshold=0.5)

    print()
    print(">>> Result")
    print(json.dumps(result, indent=2))

    print()
    print(">>> FHIR write confirmation")
    ra_ok = bool(result.get("risk_assessment_id"))
    cp_ok = bool(result.get("careplan_id"))
    if ra_ok:
        ra = client._get(f"RiskAssessment/{result['risk_assessment_id']}")
        pred = (ra.get("prediction", [{}]) or [{}])[0]
        print(f"  RiskAssessment/{result['risk_assessment_id']} written: "
              f"probability {pred.get('probabilityDecimal')}")
    else:
        print("  [WARN] RiskAssessment not written.")
    if cp_ok:
        cp = client._get(f"CarePlan/{result['careplan_id']}")
        acts = cp.get("activity", [])
        print(f"  CarePlan/{result['careplan_id']} "
              f"{'created' if result['careplan_created'] else 'existing'}: "
              f"{len(acts)} ABCDEF activities")
    else:
        print("  No CarePlan (risk below threshold).")

    print()
    print("  Single-cycle demo complete.")
    print("  For continuous operation: run_loop(interval_seconds=3600)")
    print("  Next: src/fhir/dashboard.py")


if __name__ == "__main__":
    main()