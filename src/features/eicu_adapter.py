#!/usr/bin/env python3
"""
NEUROFLOW - Stage 3: eICU Adapter (External Validation)
=======================================================
Converts eICU Collaborative Research Database v2.0 into the SAME 63-hour,
40-channel time-series format produced by build_timeseries.py, plus an eICU
cohort with delirium labels, so the MIMIC-trained model can be applied with NO
retraining.

Outputs
-------
  data/processed/eicu_cohort.csv        (one row per ICU stay)
  data/processed/eicu_timeseries.parquet (one row per stay x hour, 40 channels)

Mapping notes
-------------
  - eICU uses OFFSETS (minutes from unit admission); hour = floor(offset/60),
    window = first 63 hours.
  - Vitals: heartrate, BP, sao2, respiration, temperature. Blood pressure
    combines invasive readings from vitalPeriodic (systemicsystolic/diastolic)
    and non-invasive readings from vitalAperiodic (noninvasivesystolic/
    diastolic) into the SAME sbp/dbp channels. Temperature C -> F.
  - Each vital has a _mask flag and a _hours_since channel (hours since the last
    real observation; WINDOW_HOURS before the first-ever observation).
  - RASS and GCS Total come from nurseCharting.
  - Sedatives (infusionDrug): per-drug rate is capped, then integrated over time
    (rate x active-duration) into an hourly cumulative AMOUNT, dimensionally
    consistent with MIMIC's amount-based allocation. onboard_* uses MIMIC PK.
  - Labels: label_dx (diagnosisstring contains 'delirium'), label_score (a
    positive nurse-charting delirium scale/score/assessment, per Rocheteau
    et al. 2021, ACM CHIL), label_combined = dx OR score.

Imputation note: patients with zero real readings of a vital across their entire
stay are filled with that channel's clinically neutral default (not 0.0). The
_mask channel still flags every such hour as not observed.

Channel order is identical to build_timeseries.py / tcn.py.

DATA COMPLIANCE
---------------
All data is processed locally, chunked, filtered before loading. No individual
record or value is printed; only aggregate statistics. PhysioNet DUA 1.5.0 is
respected.

Run:
    source /Users/eshanpradhan/Desktop/NEUROFLOW/.venv/bin/activate
    python src/features/eicu_adapter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EICU_DIR = PROJECT_ROOT / "data" / "raw" / "eicu" / "2.0"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PATIENT_PATH = EICU_DIR / "patient.csv.gz"
VITALPERIODIC_PATH = EICU_DIR / "vitalPeriodic.csv.gz"
VITALAPERIODIC_PATH = EICU_DIR / "vitalAperiodic.csv.gz"
NURSECHARTING_PATH = EICU_DIR / "nurseCharting.csv.gz"
DIAGNOSIS_PATH = EICU_DIR / "diagnosis.csv.gz"
INFUSIONDRUG_PATH = EICU_DIR / "infusionDrug.csv.gz"

OUTPUT_COHORT = PROCESSED_DIR / "eicu_cohort.csv"
OUTPUT_PARQUET = PROCESSED_DIR / "eicu_timeseries.parquet"

WINDOW_HOURS = 63
ROLLING_WINDOW = 6
MIN_ICU_HOURS = 24.0

VITALPERIODIC_CHUNK = 3_000_000
VITALAPERIODIC_CHUNK = 2_000_000
NURSECHARTING_CHUNK = 3_000_000
INFUSIONDRUG_CHUNK = 1_000_000
DIAGNOSIS_CHUNK = 1_000_000

# 8 vital channels (name, low, high) - SAME order as MIMIC build_timeseries.
VITALS = [
    ("heart_rate",  0, 350),
    ("sbp",         0, 300),
    ("dbp",         0, 250),
    ("spo2",        0, 100),
    ("resp_rate",   0,  80),
    ("temperature", 70, 120),   # after Celsius -> Fahrenheit conversion
    ("rass",        -5,  4),
    ("gcs_total",    3, 15),
]
VITAL_NAMES = [v[0] for v in VITALS]
VLOW = np.array([v[1] for v in VITALS], dtype=np.float64)
VHIGH = np.array([v[2] for v in VITALS], dtype=np.float64)
N_VITALS = len(VITALS)

# Clinically neutral default per vital (VITAL_NAMES order) for fully-missing pts.
VITAL_NEUTRAL = np.array([80, 120, 70, 97, 18, 98.6, -0.79, 14], dtype=np.float64)
# Simple directly-measured vitalPeriodic columns -> (vital index, convert_C_to_F).
VP_SIMPLE = [
    ("heartrate",   0, False),
    ("sao2",        3, False),
    ("respiration", 4, False),
    ("temperature", 5, True),
]
RASS_IDX = 6
GCS_IDX = 7

# Sedatives: (name, name_regex incl. brand names, elimination_half_life_hours).
SEDATIVES = [
    ("propofol",        r"propofol|diprivan",        5.0),
    ("midazolam",       r"midazolam|versed",         3.0),
    ("dexmedetomidine", r"dexmedetomidine|precedex", 2.0),
    ("lorazepam",       r"lorazepam|ativan",        14.0),
    ("fentanyl",        r"fentanyl|sublimaze",       4.0),
]
SEDATIVE_NAMES = [s[0] for s in SEDATIVES]
SEDATIVE_PATTERNS = [s[1] for s in SEDATIVES]
SEDATIVE_HALFLIVES = np.array([s[2] for s in SEDATIVES], dtype=np.float64)
N_SEDATIVES = len(SEDATIVES)

# Physiologically plausible per-drug infusion-rate caps, applied to the rate
# BEFORE integration. Order: propofol, midazolam, dexmedetomidine, lorazepam,
# fentanyl.
SEDATIVE_RATE_CAPS = np.array([500.0, 50.0, 10.0, 10.0, 500.0], dtype=np.float64)

# Final channel order (identical to build_timeseries.py / tcn.py). 40 channels.
CHANNELS = (
    VITAL_NAMES
    + [c for n in VITAL_NAMES for c in (f"{n}_mask", f"{n}_hours_since")]
    + ["hrv_proxy", "phys_instability", "charting_density"]
    + ["circadian_sin", "circadian_cos"]
    + [f"amt_{n}" for n in SEDATIVE_NAMES]
    + [f"onboard_{n}" for n in SEDATIVE_NAMES]
    + ["pad"]
)


# --------------------------------------------------------------------------- #
# Print helpers
# --------------------------------------------------------------------------- #
def header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step(title: str) -> None:
    print()
    print(f">>> {title}")


def pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.2f}%" if d else "n/a"


# --------------------------------------------------------------------------- #
# Vectorized helpers (numpy, along the time axis)
# --------------------------------------------------------------------------- #
def ffill_rows(a: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(a)
    idx = np.where(mask, np.arange(a.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return a[np.arange(a.shape[0])[:, None], idx]


def bfill_rows(a: np.ndarray) -> np.ndarray:
    return ffill_rows(a[:, ::-1])[:, ::-1]


def impute_series(a: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill. Fully-missing rows remain all-NaN so the
    caller can fill them with a clinically neutral default (NOT 0.0)."""
    return bfill_rows(ffill_rows(a))


def hours_since_observed(missing: np.ndarray) -> np.ndarray:
    """Hours since the vital was last actually observed, per hour.

    missing: [N, T] (1/True = no observation that hour), from the _mask info and
    BEFORE imputation. Reset to 0 at each real observation; +1 each subsequent
    unobserved hour; WINDOW_HOURS before the first-ever observation.
    """
    missing = np.asarray(missing, dtype=bool)
    n, t = missing.shape
    observed = ~missing
    out = np.empty((n, t), dtype=np.float64)
    seen = np.zeros(n, dtype=bool)
    prev = np.full(n, float(WINDOW_HOURS), dtype=np.float64)
    for j in range(t):
        obs_j = observed[:, j]
        val = np.where(obs_j, 0.0,
                       np.where(seen, prev + 1.0, float(WINDOW_HOURS)))
        out[:, j] = val
        prev = val
        seen = seen | obs_j
    return out


def rolling_std(a: np.ndarray, window: int) -> np.ndarray:
    n, t = a.shape
    c1 = np.concatenate([np.zeros((n, 1)), np.cumsum(a, axis=1)], axis=1)
    c2 = np.concatenate([np.zeros((n, 1)), np.cumsum(a * a, axis=1)], axis=1)
    out = np.zeros((n, t), dtype=np.float64)
    for j in range(t):
        lo = max(0, j - window + 1)
        k = j - lo + 1
        s1 = c1[:, j + 1] - c1[:, lo]
        s2 = c2[:, j + 1] - c2[:, lo]
        var = s2 / k - (s1 / k) ** 2
        np.clip(var, 0.0, None, out=var)
        out[:, j] = np.sqrt(var)
    return out


def zscore_rows(a: np.ndarray) -> np.ndarray:
    m = a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, keepdims=True)
    s = np.where(s > 1e-6, s, 1.0)
    return (a - m) / s


def add_one(sum_acc, cnt_acc, vidx, s_idx, hour, val, low, high):
    """Scatter-add in-window, in-range readings for a single vital index."""
    ok = (hour >= 0) & (hour < WINDOW_HOURS) & (val >= low) & (val <= high)
    ok &= ~np.isnan(val)
    if not ok.any():
        return 0
    si = s_idx[ok].astype(np.intp)
    hb = hour[ok].astype(np.intp)
    vv = val[ok]
    vi = np.full(si.shape, vidx, dtype=np.intp)
    np.add.at(sum_acc, (si, vi, hb), vv)
    np.add.at(cnt_acc, (si, vi, hb), 1.0)
    return int(ok.sum())


# --------------------------------------------------------------------------- #
# Stage steps
# --------------------------------------------------------------------------- #
def check_inputs() -> None:
    needed = [PATIENT_PATH, VITALPERIODIC_PATH, VITALAPERIODIC_PATH,
              NURSECHARTING_PATH, DIAGNOSIS_PATH, INFUSIONDRUG_PATH]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print("[ERROR] Required eICU file(s) not found:")
        for p in missing:
            print(f"        {p}")
        print("Confirm the eICU v2.0 DUA is signed and files are downloaded.")
        sys.exit(1)


def _parse_age(x) -> float:
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x.startswith(">"):
        return 91.0
    try:
        return float(x)
    except ValueError:
        return np.nan


def load_cohort():
    """Load patient, keep ICU stays >= 24h, build stay index + demographics."""
    step("Loading patient table and applying >= 24h inclusion")
    pat = pd.read_csv(
        PATIENT_PATH,
        compression="gzip",
        usecols=["patientunitstayid", "age", "gender",
                 "unitadmittime24", "unitdischargeoffset"],
    )
    n_total = len(pat)

    pat["unitdischargeoffset"] = pd.to_numeric(pat["unitdischargeoffset"],
                                               errors="coerce")
    pat = pat.dropna(subset=["patientunitstayid", "unitdischargeoffset"])
    pat["los_hours"] = pat["unitdischargeoffset"] / 60.0
    pat = pat[pat["los_hours"] >= MIN_ICU_HOURS].copy()

    pat["patientunitstayid"] = pat["patientunitstayid"].astype("int64")
    pat = pat.sort_values("patientunitstayid").reset_index(drop=True)

    s = len(pat)
    if s == 0:
        print("[ERROR] No eICU stays met the inclusion criterion. Aborting.")
        sys.exit(1)

    stay_ids = pat["patientunitstayid"].to_numpy()
    s_map = pd.Series(np.arange(s, dtype=np.int64), index=pat["patientunitstayid"])

    n_obs = np.minimum(WINDOW_HOURS,
                       np.ceil(pat["los_hours"].to_numpy()).astype(np.int64))
    adm = pd.to_datetime(pat["unitadmittime24"], format="%H:%M:%S", errors="coerce")
    intime_hour = adm.dt.hour.fillna(0).astype(int).to_numpy()

    age = pat["age"].map(_parse_age).to_numpy()
    gender = pat["gender"].astype("string").fillna("").to_numpy()

    print(f"    eICU stays total          : {n_total:,}")
    print(f"    >= {MIN_ICU_HOURS:.0f}h (included cohort) : {s:,} "
          f"({pct(s, n_total)})")
    return (pat, stay_ids, s_map, n_obs, intime_hour, age, gender, s)


def accumulate_vitalperiodic(s_map, n_stays):
    """Scan vitalPeriodic (chunked) for the directly-measured vitals."""
    step("Scanning vitalPeriodic (heartrate, invasive BP, sao2, resp, temp)")
    vit_sum = np.zeros((n_stays, N_VITALS, WINDOW_HOURS), dtype=np.float64)
    vit_cnt = np.zeros((n_stays, N_VITALS, WINDOW_HOURS), dtype=np.float64)
    n_scanned = n_kept = 0

    reader = pd.read_csv(
        VITALPERIODIC_PATH, compression="gzip",
        usecols=["patientunitstayid", "observationoffset", "temperature", "sao2",
                 "heartrate", "respiration", "systemicsystolic",
                 "systemicdiastolic"],
        chunksize=VITALPERIODIC_CHUNK,
    )
    for chunk in tqdm(reader, desc="    vitalPeriodic", unit="chunk"):
        n_scanned += len(chunk)
        s_idx = chunk["patientunitstayid"].map(s_map)
        valid = s_idx.notna().to_numpy()
        if not valid.any():
            continue
        chunk = chunk.loc[valid]
        s_idx = s_idx[valid].to_numpy().astype(np.intp)
        off = pd.to_numeric(chunk["observationoffset"], errors="coerce").to_numpy()
        hour = np.floor(off / 60.0)

        for col, vidx, conv in VP_SIMPLE:
            val = pd.to_numeric(chunk[col], errors="coerce").to_numpy()
            if conv:
                val = val * 1.8 + 32.0
            n_kept += add_one(vit_sum, vit_cnt, vidx, s_idx, hour, val,
                              VLOW[vidx], VHIGH[vidx])

        sbp = pd.to_numeric(chunk["systemicsystolic"], errors="coerce").to_numpy()
        n_kept += add_one(vit_sum, vit_cnt, 1, s_idx, hour, sbp, VLOW[1], VHIGH[1])
        dbp = pd.to_numeric(chunk["systemicdiastolic"], errors="coerce").to_numpy()
        n_kept += add_one(vit_sum, vit_cnt, 2, s_idx, hour, dbp, VLOW[2], VHIGH[2])

    print(f"    vitalPeriodic rows scanned: {n_scanned:,}")
    print(f"    vital readings kept       : {n_kept:,}")
    return vit_sum, vit_cnt


def accumulate_vitalaperiodic(s_map, vit_sum, vit_cnt, n_stays):
    """Scan vitalAperiodic (chunked) for non-invasive BP into sbp/dbp channels."""
    step("Scanning vitalAperiodic (non-invasive BP -> sbp/dbp)")
    n_scanned = n_kept = 0

    reader = pd.read_csv(
        VITALAPERIODIC_PATH, compression="gzip",
        usecols=["patientunitstayid", "observationoffset",
                 "noninvasivesystolic", "noninvasivediastolic"],
        chunksize=VITALAPERIODIC_CHUNK,
    )
    for chunk in tqdm(reader, desc="    vitalAperiodic", unit="chunk"):
        n_scanned += len(chunk)
        s_idx = chunk["patientunitstayid"].map(s_map)
        valid = s_idx.notna().to_numpy()
        if not valid.any():
            continue
        chunk = chunk.loc[valid]
        s_idx = s_idx[valid].to_numpy().astype(np.intp)
        off = pd.to_numeric(chunk["observationoffset"], errors="coerce").to_numpy()
        hour = np.floor(off / 60.0)

        sbp = pd.to_numeric(chunk["noninvasivesystolic"], errors="coerce").to_numpy()
        n_kept += add_one(vit_sum, vit_cnt, 1, s_idx, hour, sbp, VLOW[1], VHIGH[1])
        dbp = pd.to_numeric(chunk["noninvasivediastolic"], errors="coerce").to_numpy()
        n_kept += add_one(vit_sum, vit_cnt, 2, s_idx, hour, dbp, VLOW[2], VHIGH[2])

    print(f"    vitalAperiodic rows scanned: {n_scanned:,}")
    print(f"    NIBP readings kept         : {n_kept:,}")
    return vit_sum, vit_cnt


def accumulate_nursecharting(s_map, vit_sum, vit_cnt, n_stays):
    """Scan nurseCharting (chunked) for RASS, GCS Total, and delirium scores."""
    step("Scanning nurseCharting (RASS, GCS Total, delirium scale/score)")
    delirium_nc: set = set()
    n_scanned = n_kept = 0

    reader = pd.read_csv(
        NURSECHARTING_PATH, compression="gzip",
        usecols=["patientunitstayid", "nursingchartoffset",
                 "nursingchartcelltypecat", "nursingchartcelltypevallabel",
                 "nursingchartcelltypevalname", "nursingchartvalue"],
        dtype={"nursingchartcelltypecat": "string",
               "nursingchartcelltypevallabel": "string",
               "nursingchartcelltypevalname": "string",
               "nursingchartvalue": "string"},
        chunksize=NURSECHARTING_CHUNK,
    )
    for chunk in tqdm(reader, desc="    nurseCharting", unit="chunk"):
        n_scanned += len(chunk)
        s_idx = chunk["patientunitstayid"].map(s_map)
        valid = s_idx.notna().to_numpy()
        if not valid.any():
            continue
        chunk = chunk.loc[valid]
        s_idx = s_idx[valid].to_numpy().astype(np.intp)
        off = pd.to_numeric(chunk["nursingchartoffset"], errors="coerce").to_numpy()
        hour = np.floor(off / 60.0)

        valname = chunk["nursingchartcelltypevalname"].str.lower()
        vallabel = chunk["nursingchartcelltypevallabel"].str.lower()
        value = chunk["nursingchartvalue"]

        rass_m = (valname.str.contains("rass", na=False)
                  | vallabel.str.contains("rass", na=False)).to_numpy(dtype=bool)
        if rass_m.any():
            v = pd.to_numeric(value[rass_m], errors="coerce").to_numpy()
            n_kept += add_one(vit_sum, vit_cnt, RASS_IDX,
                              s_idx[rass_m], hour[rass_m], v,
                              VLOW[RASS_IDX], VHIGH[RASS_IDX])

        gcs_m = (valname.str.contains("gcs", na=False)
                 & valname.str.contains("total", na=False)).to_numpy(dtype=bool)
        if gcs_m.any():
            v = pd.to_numeric(value[gcs_m], errors="coerce").to_numpy()
            n_kept += add_one(vit_sum, vit_cnt, GCS_IDX,
                              s_idx[gcs_m], hour[gcs_m], v,
                              VLOW[GCS_IDX], VHIGH[GCS_IDX])

        cat = chunk["nursingchartcelltypecat"].str.strip().str.lower()
        is_scores = (cat == "scores").fillna(False).to_numpy(dtype=bool)
        delir_name = (
            valname.str.contains("delirium scale", na=False)
            | valname.str.contains("delirium score", na=False)
            | valname.str.contains("delirium assessment", na=False)
            | vallabel.str.contains("delirium scale/score", na=False)
        ).to_numpy(dtype=bool)
        delir_m = is_scores & delir_name
        if delir_m.any():
            val_l = value.str.lower()
            val_num = pd.to_numeric(value, errors="coerce").to_numpy()
            positive = (
                val_l.str.contains("yes", na=False).to_numpy(dtype=bool)
                | val_l.str.contains("positive", na=False).to_numpy(dtype=bool)
                | value.str.contains("+", regex=False, na=False).to_numpy(dtype=bool)
                | ((~np.isnan(val_num)) & (val_num >= 4))
            )
            delir_pos = delir_m & positive
            if delir_pos.any():
                ids = chunk["patientunitstayid"].to_numpy()[delir_pos].astype("int64")
                delirium_nc.update(ids.tolist())

    print(f"    nurseCharting rows scanned: {n_scanned:,}")
    print(f"    RASS/GCS readings kept    : {n_kept:,}")
    print(f"    delirium-score positive   : {len(delirium_nc):,}")
    return delirium_nc


def load_delirium_diagnosis():
    """Scan diagnosis (chunked) for stays with 'delirium' in diagnosisstring."""
    step("Scanning diagnosis for 'delirium'")
    delirium: set = set()
    n_rows = 0
    reader = pd.read_csv(
        DIAGNOSIS_PATH, compression="gzip",
        usecols=["patientunitstayid", "diagnosisstring"],
        dtype={"diagnosisstring": "string"},
        chunksize=DIAGNOSIS_CHUNK,
    )
    for chunk in tqdm(reader, desc="    diagnosis", unit="chunk"):
        n_rows += len(chunk)
        m = chunk["diagnosisstring"].str.lower().str.contains("delirium", na=False)
        ids = pd.to_numeric(chunk.loc[m.to_numpy(dtype=bool), "patientunitstayid"],
                            errors="coerce").dropna().astype("int64")
        delirium.update(ids.tolist())
    print(f"    diagnosis rows scanned    : {n_rows:,}")
    print(f"    delirium-coded stays      : {len(delirium):,}")
    return delirium


def accumulate_infusiondrug(s_map, n_stays):
    """Scan infusionDrug (chunked); integrate capped rate x active-duration into
    hourly cumulative sedative amounts (dimensionally like MIMIC's amounts)."""
    step("Scanning infusionDrug for sedatives (rate x duration -> hourly amount)")
    rec_s, rec_d, rec_off, rec_rate = [], [], [], []
    n_scanned = n_events = 0

    reader = pd.read_csv(
        INFUSIONDRUG_PATH, compression="gzip",
        usecols=["patientunitstayid", "infusionoffset", "drugname",
                 "drugrate", "infusionrate"],
        dtype={"drugname": "string"},
        chunksize=INFUSIONDRUG_CHUNK,
    )
    for chunk in tqdm(reader, desc="    infusionDrug", unit="chunk"):
        n_scanned += len(chunk)
        s_idx = chunk["patientunitstayid"].map(s_map)
        valid = s_idx.notna().to_numpy()
        if not valid.any():
            continue
        chunk = chunk.loc[valid]
        s_arr = s_idx[valid].to_numpy().astype(np.int64)

        name_l = chunk["drugname"].str.lower()
        d_idx = np.full(len(chunk), -1, dtype=np.int64)
        for i, pat in enumerate(SEDATIVE_PATTERNS):
            m = name_l.str.contains(pat, regex=True, na=False).to_numpy(dtype=bool)
            d_idx[(d_idx < 0) & m] = i
        keep = d_idx >= 0
        if not keep.any():
            continue

        s2 = s_arr[keep]
        d2 = d_idx[keep]
        off = pd.to_numeric(chunk["infusionoffset"], errors="coerce").to_numpy()[keep]
        rate = pd.to_numeric(chunk["drugrate"], errors="coerce").to_numpy()[keep]
        inf = pd.to_numeric(chunk["infusionrate"], errors="coerce").to_numpy()[keep]
        rate = np.where(np.isnan(rate), inf, rate)

        ok = ~np.isnan(off) & ~np.isnan(rate) & (rate > 0)
        if not ok.any():
            continue
        s2, d2, off, rate = s2[ok], d2[ok], off[ok], rate[ok]
        # Cap the rate BEFORE integration.
        rate = np.minimum(rate, SEDATIVE_RATE_CAPS[d2])
        rec_s.append(s2)
        rec_d.append(d2)
        rec_off.append(off)
        rec_rate.append(rate)
        n_events += int(ok.sum())

    sed_amt = np.zeros((n_stays, N_SEDATIVES, WINDOW_HOURS), dtype=np.float64)
    if rec_s:
        s_all = np.concatenate(rec_s)
        d_all = np.concatenate(rec_d)
        off_all = np.concatenate(rec_off).astype(np.float64)
        rate_all = np.concatenate(rec_rate).astype(np.float64)

        # Sort by (stay, drug, offset) to find consecutive records per stay+drug.
        order = np.lexsort((off_all, d_all, s_all))
        s_all, d_all = s_all[order], d_all[order]
        off_all, rate_all = off_all[order], rate_all[order]

        next_off = np.full(s_all.shape, np.nan, dtype=np.float64)
        same_group = (s_all[1:] == s_all[:-1]) & (d_all[1:] == d_all[:-1])
        next_off[:-1] = np.where(same_group, off_all[1:], np.nan)
        gap = next_off - off_all

        # Duration (minutes): the gap if 0 < gap <= 60, else the remainder of the
        # current hour bin (no next entry, or gap exceeds 60 min).
        remainder = 60.0 - np.mod(off_all, 60.0)
        valid_gap = (~np.isnan(gap)) & (gap > 0.0) & (gap <= 60.0)
        duration = np.where(valid_gap, gap, remainder)
        duration = np.clip(duration, 0.0, 60.0)

        amount = rate_all * (duration / 60.0)
        hour = np.floor(off_all / 60.0)
        inwin = (hour >= 0) & (hour < WINDOW_HOURS)
        if inwin.any():
            np.add.at(
                sed_amt,
                (s_all[inwin].astype(np.intp),
                 d_all[inwin].astype(np.intp),
                 hour[inwin].astype(np.intp)),
                amount[inwin],
            )

    print(f"    infusionDrug rows scanned : {n_scanned:,}")
    print(f"    sedative records used     : {n_events:,}")
    return sed_amt


def build_channels(vit_sum, vit_cnt, sed_amt, n_obs, intime_hour, n_stays):
    step("Building channels (impute, time-delta, derive, PK, circadian, pad)")
    channels: dict[str, np.ndarray] = {}

    with np.errstate(divide="ignore", invalid="ignore"):
        means = vit_sum / vit_cnt
    means[vit_cnt == 0] = np.nan
    missing = (vit_cnt == 0)

    filled = np.empty_like(means)
    for vi, name in enumerate(VITAL_NAMES):
        f = impute_series(means[:, vi, :])
        # Fully-missing patients are still all-NaN here; fill with the channel's
        # clinically neutral default (NOT 0.0, which clip would push to the lower
        # bound and fabricate an implausible extreme). The mask still flags them.
        f = np.where(np.isnan(f), VITAL_NEUTRAL[vi], f)
        filled[:, vi, :] = f
        channels[name] = f
        channels[f"{name}_mask"] = missing[:, vi, :].astype(np.float64)
        channels[f"{name}_hours_since"] = hours_since_observed(missing[:, vi, :])

    hr = filled[:, VITAL_NAMES.index("heart_rate"), :]
    sbp = filled[:, VITAL_NAMES.index("sbp"), :]
    rr = filled[:, VITAL_NAMES.index("resp_rate"), :]
    channels["hrv_proxy"] = rolling_std(hr, ROLLING_WINDOW)
    composite = zscore_rows(hr) + zscore_rows(sbp) + zscore_rows(rr)
    channels["phys_instability"] = rolling_std(composite, ROLLING_WINDOW)
    channels["charting_density"] = vit_cnt.sum(axis=1)

    clock = (intime_hour[:, None] + np.arange(WINDOW_HOURS)[None, :]) % 24
    channels["circadian_sin"] = np.sin(2.0 * np.pi * clock / 24.0)
    channels["circadian_cos"] = np.cos(2.0 * np.pi * clock / 24.0)

    decay = np.power(0.5, 1.0 / SEDATIVE_HALFLIVES)
    for di, name in enumerate(SEDATIVE_NAMES):
        amt = sed_amt[:, di, :]
        channels[f"amt_{name}"] = amt
        onboard = np.zeros_like(amt)
        prev = np.zeros(n_stays, dtype=np.float64)
        d = decay[di]
        for t in range(WINDOW_HOURS):
            prev = prev * d + amt[:, t]
            onboard[:, t] = prev
        channels[f"onboard_{name}"] = onboard

    hour_grid = np.arange(WINDOW_HOURS)[None, :]
    channels["pad"] = (hour_grid >= n_obs[:, None]).astype(np.float64)
    return channels


def clip_vital_channels(channels):
    """Clip the 8 vital value channels to their plausible ranges (safety net,
    matching the bounds used in build_timeseries.py)."""
    for vi, name in enumerate(VITAL_NAMES):
        channels[name] = np.clip(channels[name], VLOW[vi], VHIGH[vi])
    return channels


def write_cohort(stay_ids, los_hours, age, gender, delirium, score_positive):
    step("Writing eicu_cohort.csv")
    label_dx = np.isin(stay_ids, np.fromiter(delirium, dtype=np.int64)
                       if delirium else np.empty(0, dtype=np.int64)).astype(int)
    label_score = np.isin(stay_ids, np.fromiter(score_positive, dtype=np.int64)
                          if score_positive else np.empty(0, dtype=np.int64)).astype(int)
    label_combined = ((label_dx == 1) | (label_score == 1)).astype(int)

    cohort = pd.DataFrame({
        "stay_id": stay_ids.astype(np.int64),
        "los_hours": los_hours,
        "age": age,
        "gender": gender,
        "label_dx": label_dx,
        "label_score": label_score,
        "label_combined": label_combined,
    })
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(OUTPUT_COHORT, index=False)
    print(f"    wrote {len(cohort):,} rows -> {OUTPUT_COHORT}")
    return cohort


def write_parquet(channels, stay_ids, n_stays):
    step("Writing eicu_timeseries.parquet")
    n_rows = n_stays * WINDOW_HOURS
    data = {
        "stay_id": np.repeat(stay_ids, WINDOW_HOURS).astype(np.int64),
        "hour": np.tile(np.arange(WINDOW_HOURS, dtype=np.int16), n_stays),
    }
    for name in CHANNELS:
        data[name] = channels[name].reshape(-1).astype(np.float32)

    df = pd.DataFrame(data, columns=["stay_id", "hour"] + CHANNELS)
    df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", index=False,
                  compression="snappy")
    size_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"    rows written              : {n_rows:,}")
    print(f"    columns                   : {df.shape[1]} "
          f"(stay_id, hour, + {len(CHANNELS)} channels)")
    print(f"    file size                 : {size_mb:,.1f} MB")


def print_summary(cohort, channels, n_stays):
    header("eICU ADAPTER SUMMARY (aggregate statistics only)")
    n = len(cohort)
    total_cells = n_stays * WINDOW_HOURS
    print(f"  Stays                       : {n:,}")
    print(f"  Window / channels           : {WINDOW_HOURS} h x 1 h, "
          f"{len(CHANNELS)} channels")
    print()
    print(f"  label_dx       positives    : {int(cohort['label_dx'].sum()):,}   "
          f"({pct(int(cohort['label_dx'].sum()), n)})")
    print(f"  label_score    positives    : {int(cohort['label_score'].sum()):,}   "
          f"({pct(int(cohort['label_score'].sum()), n)})")
    print(f"  label_combined positives    : "
          f"{int(cohort['label_combined'].sum()):,}   "
          f"({pct(int(cohort['label_combined'].sum()), n)})")
    print()
    print("  Vital coverage (% of stay-hours observed):")
    for name in VITAL_NAMES:
        observed = int((channels[f"{name}_mask"] == 0).sum())
        print(f"    {name:<13} {pct(observed, total_cells)}")
    print()
    print("  Sedative exposure (% of stay-hours with amount > 0):")
    for name in SEDATIVE_NAMES:
        exposed = int((channels[f"amt_{name}"] > 0).sum())
        print(f"    {name:<16} {pct(exposed, total_cells)}")
    print()
    print("  Notes:")
    print("    - 40 channels: each vital has _mask AND _hours_since. Same order")
    print("      as MIMIC; apply MIMIC-train standardization in evaluate.py.")
    print("    - BP combines invasive (vitalPeriodic) + non-invasive")
    print("      (vitalAperiodic) into the same sbp/dbp channels.")
    print("    - Sedatives: capped rate x active-duration -> hourly amount")
    print("      (dimensionally consistent with MIMIC inputevents amounts).")
    print("    - Fully-missing vitals filled with clinically neutral defaults,")
    print("      flagged via the mask channel (no fabricated extremes).")
    print("    - label_score uses nurse-charting delirium scale/score")
    print("      assessments (Rocheteau et al. 2021, ACM CHIL).")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    header("NEUROFLOW - Stage 3: eICU Adapter (External Validation)")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Outputs      : {OUTPUT_COHORT.name}, {OUTPUT_PARQUET.name}")

    check_inputs()

    (pat, stay_ids, s_map, n_obs, intime_hour,
     age, gender, n_stays) = load_cohort()
    los_hours = pat["los_hours"].to_numpy()

    vit_sum, vit_cnt = accumulate_vitalperiodic(s_map, n_stays)
    vit_sum, vit_cnt = accumulate_vitalaperiodic(s_map, vit_sum, vit_cnt, n_stays)
    delirium_nc = accumulate_nursecharting(s_map, vit_sum, vit_cnt, n_stays)
    delirium = load_delirium_diagnosis()
    sed_amt = accumulate_infusiondrug(s_map, n_stays)

    channels = build_channels(vit_sum, vit_cnt, sed_amt,
                              n_obs, intime_hour, n_stays)
    channels = clip_vital_channels(channels)
    del vit_sum, sed_amt

    cohort = write_cohort(stay_ids, los_hours, age, gender,
                          delirium, delirium_nc)
    write_parquet(channels, stay_ids, n_stays)
    print_summary(cohort, channels, n_stays)

    header("STAGE 3 COMPLETE")
    print("  Next: src/models/tcn.py")


if __name__ == "__main__":
    main()