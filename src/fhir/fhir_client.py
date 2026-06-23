#!/usr/bin/env python3
"""
NEUROFLOW - FHIR Read/Write Layer
=================================
Bidirectional FHIR R4 client connecting the trained NeuroflowTCN to a local
HAPI FHIR JPA Server (http://localhost:8080/fhir). Uses the `requests` library
only - no fhirclient SDK.

Reads (Observation, MedicationAdministration, Encounter, Patient) build the
40-channel feature vector; writes (RiskAssessment, CarePlan, Observation) push
calibrated predictions and ABCDEF-bundle compliance tracking back into the
patient's FHIR timeline, encounter-scoped and auditable.

Run (self-test against a running HAPI server):
    source /Users/eshanpradhan/Desktop/NEUROFLOW/.venv/bin/activate
    python src/fhir/fhir_client.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_BASE_URL = "http://localhost:8080/fhir"
FHIR_JSON = "application/fhir+json"
DEFAULT_TIMEOUT = 30

NEUROFLOW_EXT_BASE = "http://neuroflow.ai/fhir/ext"
EXT_CONFORMAL = f"{NEUROFLOW_EXT_BASE}/conformal-interval"
EXT_CONFIDENCE_TIER = f"{NEUROFLOW_EXT_BASE}/confidence-tier"

USCORE_RISKASSESSMENT = (
    "http://hl7.org/fhir/us/core/StructureDefinition/us-core-riskassessment"
)
USCORE_CAREPLAN = (
    "http://hl7.org/fhir/us/core/StructureDefinition/us-core-careplan"
)

# LOINC codes for the 8 vital channels (model channel order).
VITAL_LOINC = {
    "heart_rate":  "8867-4",
    "sbp":         "8480-6",
    "dbp":         "8462-4",
    "spo2":        "2708-6",
    "resp_rate":   "9279-1",
    "temperature": "8310-5",
    "rass":        "72651-2",
    "gcs_total":   "9269-2",
}
VITAL_LOINC_CODES = list(VITAL_LOINC.values())
LOINC_SYSTEM = "http://loinc.org"

# Delirium risk condition (SNOMED CT) used by CarePlan.addresses.
DELIRIUM_SNOMED = "2776000"  # Delirium (disorder)
DELIRIUM_DISPLAY = "Delirium (disorder)"

# ABCDEF bundle definition: (key, title, action, supporting LOINC code).
ABCDEF_ELEMENTS = [
    ("A", "Assess, prevent, and manage pain",
     "Document pain severity assessment (q4h while awake).", "72514-3"),
    ("B", "Both spontaneous awakening and breathing trials",
     "Coordinate daily SAT and SBT; document trial outcome.", "80330-2"),
    ("C", "Choice of analgesia and sedation",
     "Titrate sedation to RASS target -1 to 0; document RASS.", "72651-2"),
    ("D", "Delirium: assess, prevent, and manage",
     "Perform CAM-ICU assessment every nursing shift.", "72133-2"),
    ("E", "Early mobility and exercise",
     "Document highest activity/mobility level achieved.", "45970-1"),
    ("F", "Family engagement and empowerment",
     "Document family communication and involvement in care.", "85432-3"),
]
ABCDEF_LOINC = {k: loinc for (k, _t, _a, loinc) in ABCDEF_ELEMENTS}

CONFIDENCE_HIGH_MAX_WIDTH = 0.15
CONFIDENCE_MODERATE_MAX_WIDTH = 0.30


# --------------------------------------------------------------------------- #
# Errors & utilities
# --------------------------------------------------------------------------- #
class FHIRError(Exception):
    """Raised when the FHIR server returns a non-success status."""

    def __init__(self, status_code: int, body: str, context: str = ""):
        self.status_code = status_code
        self.body = body
        self.context = context
        msg = f"FHIR request failed [{status_code}]"
        if context:
            msg += f" ({context})"
        msg += f": {body[:500]}"
        super().__init__(msg)


def fhir_now() -> str:
    """Current UTC datetime in FHIR instant format (e.g. 2100-01-01T12:00:00Z)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fhir_time(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_fhir_reference(resource_type: str, resource_id: str) -> str:
    """Build a relative FHIR reference string, e.g. 'Patient/123'."""
    rid = str(resource_id)
    if "/" in rid and rid.split("/", 1)[0] == resource_type:
        return rid
    return f"{resource_type}/{rid}"


def confidence_tier_for_width(width: float) -> str:
    if width < CONFIDENCE_HIGH_MAX_WIDTH:
        return "high"
    if width < CONFIDENCE_MODERATE_MAX_WIDTH:
        return "moderate"
    return "low"


# --------------------------------------------------------------------------- #
# FHIR client
# --------------------------------------------------------------------------- #
class FHIRClient:
    """Thin requests-based wrapper over a HAPI FHIR R4 server."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": FHIR_JSON,
            "Content-Type": FHIR_JSON,
        })

    # ----- low-level HTTP ----- #
    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        try:
            r = self.session.get(self._url(path), params=params,
                                 timeout=self.timeout)
        except requests.RequestException as e:
            raise FHIRError(0, str(e), context=f"GET {path}")
        if r.status_code != 200:
            raise FHIRError(r.status_code, r.text, context=f"GET {path}")
        return r.json()

    def _post(self, resource_type: str, body: dict) -> dict:
        try:
            r = self.session.post(self._url(resource_type),
                                  data=json.dumps(body), timeout=self.timeout)
        except requests.RequestException as e:
            raise FHIRError(0, str(e), context=f"POST {resource_type}")
        if r.status_code not in (200, 201):
            raise FHIRError(r.status_code, r.text,
                            context=f"POST {resource_type}")
        return r.json()

    def capability(self) -> dict:
        return self._get("metadata")

    @staticmethod
    def _bundle_entries(bundle: dict) -> list:
        return [e["resource"] for e in bundle.get("entry", [])
                if "resource" in e]

    def _search_all(self, resource_type: str, params: dict,
                    max_pages: int = 20) -> list:
        """Search with automatic 'next' link pagination."""
        resources: list = []
        bundle = self._get(resource_type, params=params)
        pages = 0
        while bundle is not None and pages < max_pages:
            resources.extend(self._bundle_entries(bundle))
            next_url = None
            for link in bundle.get("link", []):
                if link.get("relation") == "next":
                    next_url = link.get("url")
                    break
            if not next_url:
                break
            try:
                r = self.session.get(next_url, timeout=self.timeout)
            except requests.RequestException as e:
                raise FHIRError(0, str(e), context="GET next page")
            if r.status_code != 200:
                raise FHIRError(r.status_code, r.text, context="GET next page")
            bundle = r.json()
            pages += 1
        return resources

    # ----- READ: resources ----- #
    def get_patient(self, patient_id: str) -> dict:
        return self._get(f"Patient/{patient_id}")

    def get_encounter(self, encounter_id: str) -> dict:
        return self._get(f"Encounter/{encounter_id}")

    def get_active_encounters(self) -> list:
        """All in-progress encounters (active ICU episodes)."""
        return self._search_all("Encounter",
                                {"status": "in-progress", "_count": "100"})

    def get_observations(self, patient_id: str, encounter_id: str,
                         loinc_codes: Optional[list] = None,
                         hours_back: int = 63) -> list:
        """Vital-sign Observations for the patient/encounter within hours_back,
        sorted ascending by effectiveDateTime."""
        codes = loinc_codes if loinc_codes is not None else VITAL_LOINC_CODES
        since = _fhir_time(datetime.now(timezone.utc)
                           - timedelta(hours=hours_back))
        params = {
            "patient": build_fhir_reference("Patient", patient_id),
            "encounter": build_fhir_reference("Encounter", encounter_id),
            "code": ",".join(f"{LOINC_SYSTEM}|{c}" for c in codes),
            "date": f"ge{since}",
            "_count": "1000",
            "_sort": "date",
        }
        obs = self._search_all("Observation", params)
        obs.sort(key=lambda o: o.get("effectiveDateTime", "") or "")
        return obs

    def get_medication_administrations(self, patient_id: str,
                                       encounter_id: str,
                                       hours_back: int = 63) -> list:
        """MedicationAdministration records for the patient/encounter within
        hours_back, sorted ascending by effective time."""
        since = _fhir_time(datetime.now(timezone.utc)
                           - timedelta(hours=hours_back))
        params = {
            "patient": build_fhir_reference("Patient", patient_id),
            "context": build_fhir_reference("Encounter", encounter_id),
            "effective-time": f"ge{since}",
            "_count": "1000",
            "_sort": "effective-time",
        }
        meds = self._search_all("MedicationAdministration", params)

        def _eff(m: dict) -> str:
            if "effectiveDateTime" in m:
                return m["effectiveDateTime"]
            return (m.get("effectivePeriod", {}) or {}).get("start", "") or ""

        meds.sort(key=_eff)
        return meds

    # ----- WRITE: RiskAssessment ----- #
    def write_risk_assessment(self, patient_id: str, encounter_id: str,
                              probability: float, interval_low: float,
                              interval_high: float,
                              confidence_tier: Optional[str] = None,
                              model_version: str = "neuroflow-tcn-v1") -> str:
        prob = float(max(0.0, min(1.0, probability)))
        low = float(max(0.0, min(1.0, interval_low)))
        high = float(max(0.0, min(1.0, interval_high)))
        width = max(0.0, high - low)
        tier = confidence_tier or confidence_tier_for_width(width)
        now = fhir_now()

        prediction = {
            "outcome": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": DELIRIUM_SNOMED,
                    "display": DELIRIUM_DISPLAY,
                }],
                "text": "ICU delirium within prediction horizon",
            },
            "probabilityDecimal": round(prob, 4),
            "extension": [
                {
                    "url": EXT_CONFORMAL,
                    "extension": [
                        {"url": "low", "valueDecimal": round(low, 4)},
                        {"url": "high", "valueDecimal": round(high, 4)},
                    ],
                },
                {
                    "url": EXT_CONFIDENCE_TIER,
                    "valueCode": tier,
                },
            ],
        }

        note_text = (
            f"NEUROFLOW delirium risk {prob:.2f} "
            f"(90% CI {low:.2f}-{high:.2f}, confidence: {tier}). "
            f"Model {model_version}. Generated {now}."
        )

        resource = {
            "resourceType": "RiskAssessment",
            "meta": {"profile": [USCORE_RISKASSESSMENT]},
            "status": "final",
            "code": {
                "coding": [{
                    "system": LOINC_SYSTEM,
                    "code": "72133-2",
                    "display": "Delirium risk assessment",
                }],
                "text": "ICU delirium risk assessment",
            },
            "subject": {"reference": build_fhir_reference("Patient", patient_id)},
            "encounter": {
                "reference": build_fhir_reference("Encounter", encounter_id)
            },
            "occurrenceDateTime": now,
            "performer": {
                "display": f"NEUROFLOW model {model_version}",
            },
            "prediction": [prediction],
            "note": [{"text": note_text}],
        }
        created = self._post("RiskAssessment", resource)
        return created.get("id", "")

    # ----- WRITE: continuous risk Observation (dashboard signal) ----- #
    def write_risk_observation(self, patient_id: str, encounter_id: str,
                               probability: float,
                               model_version: str = "neuroflow-tcn-v1") -> str:
        now = fhir_now()
        resource = {
            "resourceType": "Observation",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": ("http://terminology.hl7.org/CodeSystem/"
                               "observation-category"),
                    "code": "survey",
                    "display": "Survey",
                }],
            }],
            "code": {
                "coding": [{
                    "system": NEUROFLOW_EXT_BASE,
                    "code": "delirium-risk-score",
                    "display": "NEUROFLOW delirium risk score",
                }],
                "text": "NEUROFLOW delirium risk score",
            },
            "subject": {"reference": build_fhir_reference("Patient", patient_id)},
            "encounter": {
                "reference": build_fhir_reference("Encounter", encounter_id)
            },
            "effectiveDateTime": now,
            "valueQuantity": {
                "value": round(float(max(0.0, min(1.0, probability))), 4),
                "unit": "probability",
                "system": "http://unitsofmeasure.org",
                "code": "1",
            },
            "device": {
                "reference": build_fhir_reference("Device", model_version)
            },
        }
        created = self._post("Observation", resource)
        return created.get("id", "")

    # ----- WRITE: ABCDEF CarePlan ----- #
    def create_abcdef_careplan(self, patient_id: str, encounter_id: str,
                               risk_score: float) -> str:
        now = fhir_now()
        activities = []
        for key, title, action, _loinc in ABCDEF_ELEMENTS:
            activities.append({
                "detail": {
                    "status": "not-started",
                    "kind": "ServiceRequest",
                    "description": f"[{key}] {title}: {action}",
                    "code": {"text": f"ABCDEF-{key}: {title}"},
                },
            })

        resource = {
            "resourceType": "CarePlan",
            "meta": {"profile": [USCORE_CAREPLAN]},
            "status": "active",
            "intent": "plan",
            "category": [{
                "coding": [{
                    "system": ("http://hl7.org/fhir/us/core/CodeSystem/"
                               "careplan-category"),
                    "code": "assess-plan",
                }],
                "text": "ABCDEF bundle - ICU delirium prevention",
            }],
            "title": "ICU Liberation ABCDEF Bundle",
            "description": (
                f"ABCDEF delirium-prevention bundle initiated at delirium risk "
                f"{float(risk_score):.2f}."
            ),
            "subject": {"reference": build_fhir_reference("Patient", patient_id)},
            "encounter": {
                "reference": build_fhir_reference("Encounter", encounter_id)
            },
            "created": now,
            "addresses": [{
                "reference": "#delirium-risk",
                "display": DELIRIUM_DISPLAY,
            }],
            "contained": [{
                "resourceType": "Condition",
                "id": "delirium-risk",
                "clinicalStatus": {
                    "coding": [{
                        "system": ("http://terminology.hl7.org/CodeSystem/"
                                   "condition-clinical"),
                        "code": "active",
                    }],
                },
                "code": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": DELIRIUM_SNOMED,
                        "display": DELIRIUM_DISPLAY,
                    }],
                    "text": "Risk of ICU delirium",
                },
                "subject": {
                    "reference": build_fhir_reference("Patient", patient_id)
                },
            }],
            "activity": activities,
        }
        created = self._post("CarePlan", resource)
        return created.get("id", "")

    # ----- READ/SCORE: ABCDEF compliance ----- #
    def score_abcdef_compliance(self, patient_id: str, encounter_id: str,
                                careplan_id: Optional[str] = None) -> dict:
        """For each ABCDEF element, True if a supporting Observation exists in
        the last 12 hours for this patient/encounter."""
        since = _fhir_time(datetime.now(timezone.utc) - timedelta(hours=12))
        result: dict = {}
        for key, _title, _action, loinc in ABCDEF_ELEMENTS:
            params = {
                "patient": build_fhir_reference("Patient", patient_id),
                "encounter": build_fhir_reference("Encounter", encounter_id),
                "code": f"{LOINC_SYSTEM}|{loinc}",
                "date": f"ge{since}",
                "_count": "1",
            }
            try:
                obs = self._search_all("Observation", params, max_pages=1)
                result[key] = len(obs) > 0
            except FHIRError:
                result[key] = False
        if careplan_id is not None:
            result["careplan_id"] = careplan_id
        return result


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    print("=" * 70)
    print("  NEUROFLOW FHIR client - self-test")
    print("=" * 70)
    client = FHIRClient()

    try:
        cap = client.capability()
        print(f"  server reachable       : yes")
        print(f"  fhirVersion            : {cap.get('fhirVersion', 'unknown')}")
        sw = cap.get("software", {}) or {}
        print(f"  software               : {sw.get('name', '?')} "
              f"{sw.get('version', '?')}")
    except FHIRError as e:
        print(f"  [WARN] server not reachable: {e}")
        print("  Start it with: docker start neuroflow-fhir")
        print("  Skipping live read/write checks.")
        _offline_checks()
        return

    # Live checks: list active encounters; round-trip a write if one exists.
    try:
        encs = client.get_active_encounters()
        print(f"  active encounters      : {len(encs)}")
    except FHIRError as e:
        print(f"  [WARN] active encounter search failed: {e}")
        encs = []

    if encs:
        enc = encs[0]
        enc_id = enc.get("id", "")
        subj = (enc.get("subject", {}) or {}).get("reference", "")
        pat_id = subj.split("/", 1)[1] if "/" in subj else ""
        print(f"  sample encounter       : Encounter/{enc_id} "
              f"(patient {pat_id or 'unknown'})")
        if pat_id:
            try:
                obs = client.get_observations(pat_id, enc_id)
                meds = client.get_medication_administrations(pat_id, enc_id)
                print(f"    observations (63h)   : {len(obs)}")
                print(f"    med administrations  : {len(meds)}")
            except FHIRError as e:
                print(f"    [WARN] read failed: {e}")
            try:
                ra_id = client.write_risk_assessment(
                    pat_id, enc_id, probability=0.72,
                    interval_low=0.61, interval_high=0.83)
                print(f"    wrote RiskAssessment : {ra_id}")
                cp_id = client.create_abcdef_careplan(pat_id, enc_id, 0.72)
                print(f"    wrote CarePlan       : {cp_id}")
                comp = client.score_abcdef_compliance(pat_id, enc_id, cp_id)
                done = sum(1 for k in "ABCDEF" if comp.get(k))
                print(f"    ABCDEF compliance    : {done}/6 elements documented")
            except FHIRError as e:
                print(f"    [WARN] write failed: {e}")
    else:
        print("  No active encounters to exercise read/write; "
              "verifying resource construction offline.")
        _offline_checks()

    print()
    print("  Self-test complete.")
    print("  Next: src/fhir/smart_launch.py")


def _offline_checks() -> None:
    """Validate pure-logic helpers without a server."""
    assert build_fhir_reference("Patient", "123") == "Patient/123"
    assert build_fhir_reference("Patient", "Patient/123") == "Patient/123"
    assert confidence_tier_for_width(0.10) == "high"
    assert confidence_tier_for_width(0.20) == "moderate"
    assert confidence_tier_for_width(0.40) == "low"
    assert fhir_now().endswith("Z")
    print("  offline helper checks  : passed "
          "(references, confidence tiers, timestamp)")


if __name__ == "__main__":
    _self_test()