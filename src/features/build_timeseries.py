#!/usr/bin/env python3
"""
NEUROFLOW - Stage 2: Time-Series Construction
=============================================
Transforms the cohort (data/processed/cohort.csv) plus MIMIC-IV chartevents and
inputevents into a per-stay 63-hour, 1-hour-resolution multivariate time-series
and writes data/processed/timeseries.parquet (long format: one row per
stay_id x hour).

Window
------
For each ICU stay, hour t = 0..62 covers [intime + t h, intime + (t+1) h).
Stays >= 63 h use their first 63 h fully; shorter stays are right-padded and the
`pad` channel marks beyond-discharge hours. (Cohort already requires >= 24 h.)

Channels (32 total, fixed order)
--------------------------------
  vitals (8)            : heart_rate, sbp, dbp, spo2, resp_rate, temperature,
                          rass, gcs_total            (hourly mean, imputed)
  vitals missingness(8) : <vital>_mask  (1 = no measurement that hour)
  derived (3)           : hrv_proxy            (rolling SD of HR - autonomic)
                          phys_instability     (rolling SD of HR/SBP/RR composite)
                          charting_density     (monitored-item readings / hour)
  circadian (2)         : circadian_sin, circadian_cos  (time-of-day; preserved
                          under MIMIC date shifting)
  sedatives amount (5)  : amt_<drug>   (hourly dose, infusion intervals allocated)
  sedatives onboard (5) : onboard_<drug>  (PK exponential-decay accumulation)
  pad (1)               : pad          (1 = hour beyond ICU discharge)

gcs_total is the true Glasgow Coma Scale: the sum of the three component
itemids (eye 220739 [1-4], verbal 223900 [1-5], motor 223901 [1-6]), range
[3,15]. An hour's gcs_total is observed only when all three components are
present; otherwise its mask is 1 and the value is ffill/bfill-imputed like the
other vitals.

Antipsychotics (haloperidol/quetiapine/olanzapine) are NOT features here - they
define the proxy label, so using them would leak the target.

Standardization is intentionally deferred to train.py (fit on the train split
only) to avoid leakage. This stage outputs raw (imputed) values plus flags.

DATA COMPLIANCE
---------------
All data is processed locally, chunked, and filtered by itemid before loading.
No individual record or value is printed; only aggregate statistics. PhysioNet
DUA 1.5.0 is respected.

Run:
    source /Users/eshanpradhan/Desktop/NEUROFLOW/.venv/bin/activate
    python src/features/build_timeseries.py
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
MIMIC_ICU = PROJECT_ROOT / "data" / "raw" / "mimiciv" / "3.1" / "icu"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COHORT_CSV = PROCESSED_DIR / "cohort.csv"
CHARTEVENTS_PATH = MIMIC_ICU / "chartevents.csv.gz"
INPUTEVENTS_PATH = MIMIC_ICU / "inputevents.csv.gz"
OUTPUT_PARQUET = PROCESSED_DIR / "timeseries.parquet"

WINDOW_HOURS = 63
ROLLING_WINDOW = 6  # hours, for variability/instability features

CHARTEVENTS_CHUNK = 2_000_000
INPUTEVENTS_CHUNK = 1_000_000

# Simple vitals: directly measured, one itemid each.
# (name, itemid, plausible_low, plausible_high). Order is fixed.
SIMPLE_VITALS = [
    ("heart_rate",  220045,   0, 350),
    ("sbp",         220179,   0, 300),
    ("dbp",         220180,   0, 250),
    ("spo2",        220277,   0, 100),
    ("resp_rate",   220210,   0,  80),
    ("temperature", 223761,  70, 120),   # Temperature Fahrenheit
    ("rass",        228096,  -5,   4),
]
SIMPLE_NAMES = [v[0] for v in SIMPLE_VITALS]
SIMPLE_ITEMID_TO_IDX = {v[1]: i for i, v in enumerate(SIMPLE_VITALS)}
SIMPLE_LOW = np.array([v[2] for v in SIMPLE_VITALS], dtype=np.float64)
SIMPLE_HIGH = np.array([v[3] for v in SIMPLE_VITALS], dtype=np.float64)
N_SIMPLE = len(SIMPLE_VITALS)

# GCS components: summed to form gcs_total (range [3,15]).
# (name, itemid, component_low, component_high). Order is fixed: eye, verbal, motor.
GCS_COMPONENTS = [
    ("gcs_eye",    220739, 1, 4),
    ("gcs_verbal", 223900, 1, 5),
    ("gcs_motor",  223901, 1, 6),
]
GCS_ITEMID_TO_IDX = {v[1]: i for i, v in enumerate(GCS_COMPONENTS)}
GCS_LOW = np.array([v[2] for v in GCS_COMPONENTS], dtype=np.float64)
GCS_HIGH = np.array([v[3] for v in GCS_COMPONENTS], dtype=np.float64)
N_GCS = len(GCS_COMPONENTS)

ALL_VITAL_ITEMIDS = set(SIMPLE_ITEMID_TO_IDX) | set(GCS_ITEMID_TO_IDX)

# The 8 output vital channels (7 simple + composite gcs_total).
VITAL_NAMES = SIMPLE_NAMES + ["gcs_total"]

# Sedatives: (name, itemid, elimination_half_life_hours). Order is fixed.
SEDATIVES = [
    ("propofol",        222168,  5.0),
    ("midazolam",       221668,  3.0),
    ("dexmedetomidine", 221749,  2.0),
    ("lorazepam",       221385, 14.0),
    ("fentanyl",        221744,  4.0),
]
SEDATIVE_NAMES = [s[0] for s in SEDATIVES]
SEDATIVE_ITEMIDS = {s[1] for s in SEDATIVES}
SEDATIVE_ITEMID_TO_IDX = {s[1]: i for i, s in enumerate(SEDATIVES)}
SEDATIVE_HALFLIVES = np.array([s[2] for s in SEDATIVES], dtype=np.float64)
N_SEDATIVES = len(SEDATIVES)

# Final channel order (must match the columns written to parquet).
CHANNELS = (
    VITAL_NAMES
    + [f"{n}_mask" for n in VITAL_NAMES]
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


# --------------------------------------------------------------------------- #
# Vectorized helpers (numpy, operate along the time axis)
# --------------------------------------------------------------------------- #
def ffill_rows(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along axis 1 (per row)."""
    mask = ~np.isnan(a)
    idx = np.where(mask, np.arange(a.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return a[np.arange(a.shape[0])[:, None], idx]


def bfill_rows(a: np.ndarray) -> np.ndarray:
    """Backward-fill NaNs along axis 1 (per row)."""
    return ffill_rows(a[:, ::-1])[:, ::-1]


def impute_series(a: np.ndarray) -> np.ndarray:
    """Forward-fill, then backward-fill, then zero any all-missing rows."""
    out = bfill_rows(ffill_rows(a))
    return np.nan_to_num(out, nan=0.0)


def rolling_std(a: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling standard deviation along axis 1 (min_periods=1)."""
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
    """Within-row z-score (robust to zero variance)."""
    m = a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, keepdims=True)
    s = np.where(s > 1e-6, s, 1.0)
    return (a - m) / s


# --------------------------------------------------------------------------- #
# Stage steps
# --------------------------------------------------------------------------- #
def check_inputs() -> None:
    missing = [p for p in (COHORT_CSV, CHARTEVENTS_PATH, INPUTEVENTS_PATH)
               if not p.exists()]
    if missing:
        print("[ERROR] Required input file(s) not found:")
        for p in missing:
            print(f"        {p}")
        if not COHORT_CSV.exists():
            print("Run src/features/extract_cohort.py first.")
        sys.exit(1)


def load_cohort():
    """Load cohort, return sorted stay frame plus lookup Series."""
    step("Loading cohort and building stay index")
    cohort = pd.read_csv(
        COHORT_CSV,
        usecols=["stay_id", "intime", "outtime", "los_hours"],
        parse_dates=["intime", "outtime"],
    )
    cohort = cohort.dropna(subset=["stay_id", "intime"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype("int64")
    cohort = cohort.sort_values("stay_id").reset_index(drop=True)

    s = len(cohort)
    if s == 0:
        print("[ERROR] Cohort is empty. Aborting.")
        sys.exit(1)

    stay_ids = cohort["stay_id"].to_numpy()
    s_map = pd.Series(np.arange(s, dtype=np.int64), index=cohort["stay_id"])
    intime_map = pd.Series(cohort["intime"].to_numpy(), index=cohort["stay_id"])

    # Observed-hour count and per-stay clock hour at intime (for pad / circadian).
    n_obs = np.minimum(
        WINDOW_HOURS,
        np.ceil(cohort["los_hours"].to_numpy()).astype(np.int64),
    )
    intime_hour = cohort["intime"].dt.hour.to_numpy().astype(np.int64)

    print(f"    cohort stays              : {s:,}")
    return cohort, stay_ids, s_map, intime_map, n_obs, intime_hour, s


def _accumulate(sum_acc, cnt_acc, s_idx, hour_bin, val, idx_float, low_arr, high_arr):
    """Scatter-add in-window, in-range readings into the (stay, idx, hour) grid."""
    sel = ~np.isnan(idx_float)
    if not sel.any():
        return 0
    vi = idx_float[sel].astype(np.intp)
    si = s_idx[sel]
    hb = hour_bin[sel]
    vv = val[sel]
    in_win = (hb >= 0) & (hb < WINDOW_HOURS)
    in_range = (vv >= low_arr[vi]) & (vv <= high_arr[vi])
    ok = in_win & in_range
    if not ok.any():
        return 0
    si, vi = si[ok], vi[ok]
    hb = hb[ok].astype(np.intp)
    vv = vv[ok]
    np.add.at(sum_acc, (si, vi, hb), vv)
    np.add.at(cnt_acc, (si, vi, hb), 1.0)
    return int(ok.sum())


def accumulate_chartevents(s_map, intime_map, n_stays):
    """Scan chartevents (chunked), accumulate hourly sums/counts for vitals."""
    step("Scanning chartevents for vitals (itemid-filtered, hourly-binned)")
    simp_sum = np.zeros((n_stays, N_SIMPLE, WINDOW_HOURS), dtype=np.float64)
    simp_cnt = np.zeros((n_stays, N_SIMPLE, WINDOW_HOURS), dtype=np.float64)
    gcs_sum = np.zeros((n_stays, N_GCS, WINDOW_HOURS), dtype=np.float64)
    gcs_cnt = np.zeros((n_stays, N_GCS, WINDOW_HOURS), dtype=np.float64)

    n_scanned = 0
    n_kept = 0

    reader = pd.read_csv(
        CHARTEVENTS_PATH,
        compression="gzip",
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        dtype={"itemid": "int64", "valuenum": "float64"},
        parse_dates=["charttime"],
        chunksize=CHARTEVENTS_CHUNK,
    )
    for chunk in tqdm(reader, desc="    chartevents", unit="chunk"):
        n_scanned += len(chunk)

        chunk = chunk[chunk["itemid"].isin(ALL_VITAL_ITEMIDS)]
        if chunk.empty:
            continue

        s_idx = chunk["stay_id"].map(s_map)
        keep = s_idx.notna() & chunk["charttime"].notna() & chunk["valuenum"].notna()
        chunk = chunk[keep]
        if chunk.empty:
            continue
        s_idx = s_idx[keep].to_numpy().astype(np.intp)

        intime = chunk["stay_id"].map(intime_map).to_numpy()
        hours = (chunk["charttime"].to_numpy() - intime) / np.timedelta64(1, "h")
        hour_bin = np.floor(hours).astype(np.int64)
        val = chunk["valuenum"].to_numpy()

        simple_idx = chunk["itemid"].map(SIMPLE_ITEMID_TO_IDX).to_numpy()
        gcs_idx = chunk["itemid"].map(GCS_ITEMID_TO_IDX).to_numpy()

        n_kept += _accumulate(simp_sum, simp_cnt, s_idx, hour_bin, val,
                              simple_idx, SIMPLE_LOW, SIMPLE_HIGH)
        n_kept += _accumulate(gcs_sum, gcs_cnt, s_idx, hour_bin, val,
                              gcs_idx, GCS_LOW, GCS_HIGH)

    print(f"    chartevents rows scanned  : {n_scanned:,}")
    print(f"    vital readings kept       : {n_kept:,}")
    return simp_sum, simp_cnt, gcs_sum, gcs_cnt


def _allocate_intervals(sed_amt, s_idx, d_idx, start_h, end_h, amount):
    """Allocate each infusion's amount across the hour bins it overlaps."""
    dur = end_h - start_h
    is_inf = dur > 0.0

    # Bolus / zero-duration events: assign full amount to the start bin.
    bol = ~is_inf
    if bol.any():
        bt = np.floor(start_h[bol]).astype(np.int64)
        valid = (bt >= 0) & (bt < WINDOW_HOURS)
        if valid.any():
            np.add.at(
                sed_amt,
                (s_idx[bol][valid].astype(np.intp),
                 d_idx[bol][valid].astype(np.intp),
                 bt[valid].astype(np.intp)),
                amount[bol][valid],
            )

    # Infusions: distribute amount proportional to per-bin time overlap.
    if is_inf.any():
        s2 = s_idx[is_inf].astype(np.intp)
        d2 = d_idx[is_inf].astype(np.intp)
        st, en, am, du = (start_h[is_inf], end_h[is_inf],
                          amount[is_inf], dur[is_inf])
        for t in range(WINDOW_HOURS):
            lo = np.maximum(st, t)
            hi = np.minimum(en, t + 1)
            ov = hi - lo
            np.clip(ov, 0.0, None, out=ov)
            alloc = am * (ov / du)
            nz = alloc > 0.0
            if nz.any():
                tt = np.full(int(nz.sum()), t, dtype=np.intp)
                np.add.at(sed_amt, (s2[nz], d2[nz], tt), alloc[nz])


def accumulate_inputevents(s_map, intime_map, n_stays):
    """Scan inputevents (chunked), accumulate hourly sedative amounts."""
    step("Scanning inputevents for sedatives (itemid-filtered, interval-allocated)")
    sed_amt = np.zeros((n_stays, N_SEDATIVES, WINDOW_HOURS), dtype=np.float64)

    n_scanned = 0
    n_events = 0

    reader = pd.read_csv(
        INPUTEVENTS_PATH,
        compression="gzip",
        usecols=["stay_id", "starttime", "endtime", "itemid", "amount"],
        dtype={"itemid": "int64", "amount": "float64"},
        parse_dates=["starttime", "endtime"],
        chunksize=INPUTEVENTS_CHUNK,
    )
    for chunk in tqdm(reader, desc="    inputevents", unit="chunk"):
        n_scanned += len(chunk)

        chunk = chunk[chunk["itemid"].isin(SEDATIVE_ITEMIDS)]
        if chunk.empty:
            continue

        s_idx = chunk["stay_id"].map(s_map)
        keep = (s_idx.notna() & chunk["starttime"].notna()
                & chunk["amount"].notna() & (chunk["amount"] > 0))
        chunk = chunk[keep]
        if chunk.empty:
            continue
        s_idx = s_idx[keep].to_numpy().astype(np.int64)

        intime = chunk["stay_id"].map(intime_map).to_numpy()
        start_h = (chunk["starttime"].to_numpy() - intime) / np.timedelta64(1, "h")
        end_arr = chunk["endtime"].to_numpy()
        end_h = (end_arr - intime) / np.timedelta64(1, "h")
        # Missing endtime -> treat as instantaneous bolus at starttime.
        end_h = np.where(np.isnan(end_h), start_h, end_h)

        d_idx = chunk["itemid"].map(SEDATIVE_ITEMID_TO_IDX).to_numpy().astype(np.int64)
        amount = chunk["amount"].to_numpy()

        # Keep only events overlapping the [0, WINDOW_HOURS) window.
        overlaps = (end_h > 0) & (start_h < WINDOW_HOURS)
        if not overlaps.any():
            continue
        s_idx, d_idx = s_idx[overlaps], d_idx[overlaps]
        start_h, end_h, amount = start_h[overlaps], end_h[overlaps], amount[overlaps]
        n_events += int(overlaps.sum())

        _allocate_intervals(sed_amt, s_idx, d_idx, start_h, end_h, amount)

    print(f"    inputevents rows scanned  : {n_scanned:,}")
    print(f"    sedative events used      : {n_events:,}")
    return sed_amt


def build_channels(simp_sum, simp_cnt, gcs_sum, gcs_cnt, sed_amt,
                   n_obs, intime_hour, n_stays):
    """Assemble the 32 channels as [n_stays, WINDOW_HOURS] float arrays."""
    step("Building channels (impute, derive, PK accumulation, circadian, pad)")
    channels: dict[str, np.ndarray] = {}

    # --- Simple vitals: hourly mean + missingness flag ---
    with np.errstate(divide="ignore", invalid="ignore"):
        simp_mean = simp_sum / simp_cnt
    simp_mean[simp_cnt == 0] = np.nan
    simp_missing = (simp_cnt == 0)

    filled_simple = np.empty_like(simp_mean)
    for vi, name in enumerate(SIMPLE_NAMES):
        f = impute_series(simp_mean[:, vi, :])
        filled_simple[:, vi, :] = f
        channels[name] = f
        channels[f"{name}_mask"] = simp_missing[:, vi, :].astype(np.float64)

    # --- gcs_total: sum of three component means (NaN if any missing) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        gcs_mean = gcs_sum / gcs_cnt
    gcs_mean[gcs_cnt == 0] = np.nan
    gcs_total_raw = gcs_mean[:, 0, :] + gcs_mean[:, 1, :] + gcs_mean[:, 2, :]
    gcs_total_mask = np.isnan(gcs_total_raw)
    channels["gcs_total"] = impute_series(gcs_total_raw)
    channels["gcs_total_mask"] = gcs_total_mask.astype(np.float64)

    # --- Derived physiological features (computed on imputed signals) ---
    hr = filled_simple[:, SIMPLE_NAMES.index("heart_rate"), :]
    sbp = filled_simple[:, SIMPLE_NAMES.index("sbp"), :]
    rr = filled_simple[:, SIMPLE_NAMES.index("resp_rate"), :]

    channels["hrv_proxy"] = rolling_std(hr, ROLLING_WINDOW)
    composite = zscore_rows(hr) + zscore_rows(sbp) + zscore_rows(rr)
    channels["phys_instability"] = rolling_std(composite, ROLLING_WINDOW)
    channels["charting_density"] = simp_cnt.sum(axis=1) + gcs_cnt.sum(axis=1)

    # --- Circadian phase (time-of-day, preserved under date shifting) ---
    clock = (intime_hour[:, None] + np.arange(WINDOW_HOURS)[None, :]) % 24
    channels["circadian_sin"] = np.sin(2.0 * np.pi * clock / 24.0)
    channels["circadian_cos"] = np.cos(2.0 * np.pi * clock / 24.0)

    # --- Sedatives: hourly amount + PK onboard (exponential decay) ---
    decay = np.power(0.5, 1.0 / SEDATIVE_HALFLIVES)  # per-hour retention factor
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

    # --- Pad flag (1 where hour is beyond ICU discharge) ---
    hour_grid = np.arange(WINDOW_HOURS)[None, :]
    channels["pad"] = (hour_grid >= n_obs[:, None]).astype(np.float64)

    return channels


def write_parquet(channels, stay_ids, n_stays):
    """Flatten channels to long format and write parquet."""
    step("Writing timeseries.parquet")
    n_rows = n_stays * WINDOW_HOURS
    data = {
        "stay_id": np.repeat(stay_ids, WINDOW_HOURS).astype(np.int64),
        "hour": np.tile(np.arange(WINDOW_HOURS, dtype=np.int16), n_stays),
    }
    for name in CHANNELS:
        data[name] = channels[name].reshape(-1).astype(np.float32)

    df = pd.DataFrame(data, columns=["stay_id", "hour"] + CHANNELS)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", index=False,
                  compression="snappy")

    size_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"    rows written              : {n_rows:,}")
    print(f"    columns                   : {df.shape[1]} "
          f"(stay_id, hour, + {len(CHANNELS)} channels)")
    print(f"    file size                 : {size_mb:,.1f} MB")
    return df


def print_summary(channels, n_stays):
    header("TIME-SERIES SUMMARY (aggregate statistics only)")
    total_cells = n_stays * WINDOW_HOURS
    print(f"  Stays                       : {n_stays:,}")
    print(f"  Window                      : {WINDOW_HOURS} h x 1 h")
    print(f"  Channels                    : {len(CHANNELS)}")
    print()
    print("  Vital coverage (% of stay-hours observed):")
    for name in VITAL_NAMES:
        observed = int((channels[f"{name}_mask"] == 0).sum())
        print(f"    {name:<13} {100.0 * observed / total_cells:6.2f}%")
    print()
    print("  Sedative exposure (% of stay-hours with dose > 0):")
    for name in SEDATIVE_NAMES:
        exposed = int((channels[f"amt_{name}"] > 0).sum())
        print(f"    {name:<16} {100.0 * exposed / total_cells:6.2f}%")
    print()
    print("  Notes:")
    print("    - gcs_total is the summed GCS (eye+verbal+motor); an hour is")
    print("      observed only when all three components are present.")
    print("    - Values are raw + imputed (ffill/bfill); standardize in train.py")
    print("      on the train split only. Use *_mask and pad as model inputs.")
    print("    - Join labels/split from cohort.csv on stay_id for training.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    header("NEUROFLOW - Stage 2: Time-Series Construction (MIMIC-IV v3.1)")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Output       : {OUTPUT_PARQUET}")

    check_inputs()

    _, stay_ids, s_map, intime_map, n_obs, intime_hour, n_stays = load_cohort()
    simp_sum, simp_cnt, gcs_sum, gcs_cnt = accumulate_chartevents(
        s_map, intime_map, n_stays)
    sed_amt = accumulate_inputevents(s_map, intime_map, n_stays)
    channels = build_channels(simp_sum, simp_cnt, gcs_sum, gcs_cnt, sed_amt,
                              n_obs, intime_hour, n_stays)

    # Free large sum buffers before writing.
    del simp_sum, gcs_sum

    write_parquet(channels, stay_ids, n_stays)
    print_summary(channels, n_stays)

    header("STAGE 2 COMPLETE")
    print("  Next: src/features/eicu_adapter.py")


if __name__ == "__main__":
    main()