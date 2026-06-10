#!/usr/bin/env python3
"""
NEUROFLOW - Stage 1: Cohort Extraction
======================================
Builds the analysis cohort and delirium labels from MIMIC-IV v3.1 and writes
data/processed/cohort.csv. Each row is one ICU stay (stay_id).

What this script does
---------------------
  1. Load icustays, compute stay duration, keep ICU stays >= 24 hours.
  2. Join patients to attach demographics and assign the temporal train/test
     split using anchor_year_group (NO random splits - this is required for
     valid time-series evaluation).
  3. Scan diagnoses_icd (chunked) and flag the primary ICD delirium label:
        ICD-10 F05*   (icd_version == 10)
        ICD-9  293*   (icd_version ==  9)
  4. Scan emar (chunked) for antipsychotic ADMINISTRATIONS (haloperidol,
     quetiapine, olanzapine), attribute each to the ICU stay whose
     [intime, outtime] window contains it, and flag the proxy label.
  5. Combine into label_combined = label_icd OR label_proxy.
  6. Write the cohort and print aggregate diagnostics.

Output columns
--------------
  subject_id, hadm_id, stay_id, first_careunit, last_careunit,
  intime, outtime, los_hours, gender, anchor_age, anchor_year_group, dod,
  label_icd, label_proxy, label_combined, split

DATA COMPLIANCE
---------------
All data files are processed locally. This script reads only the columns it
needs, never prints any individual record or value, and reports only aggregate
statistics (counts and prevalence). PhysioNet DUA 1.5.0 is respected.

Run:
    source /Users/eshanpradhan/Desktop/NEUROFLOW/.venv/bin/activate
    python src/features/extract_cohort.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIMIC_HOSP = PROJECT_ROOT / "data" / "raw" / "mimiciv" / "3.1" / "hosp"
MIMIC_ICU = PROJECT_ROOT / "data" / "raw" / "mimiciv" / "3.1" / "icu"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "cohort.csv"

ICUSTAYS_PATH = MIMIC_ICU / "icustays.csv.gz"
PATIENTS_PATH = MIMIC_HOSP / "patients.csv.gz"
DIAGNOSES_PATH = MIMIC_HOSP / "diagnoses_icd.csv.gz"
EMAR_PATH = MIMIC_HOSP / "emar.csv.gz"

# Inclusion criterion
MIN_ICU_HOURS = 24.0

# Chunk size for the two large hosp tables
CHUNK_SIZE = 1_000_000

# Delirium ICD code families (prefix match, version-scoped)
ICD10_DELIRIUM_PREFIX = "F05"   # Delirium due to known physiological condition
ICD9_DELIRIUM_PREFIX = "293"    # Transient mental disorders (incl. delirium 293.0/293.1)

# Antipsychotic proxy: case-insensitive substring match on emar.medication
ANTIPSYCHOTICS = ["haloperidol", "quetiapine", "olanzapine"]
ANTIPSYCHOTIC_REGEX = "|".join(ANTIPSYCHOTICS)

# emar.event_txt that denotes the drug was actually given contains "Administered"
# (e.g. "Administered", "Partial Administered"); "Not Given" / "Held" / "Stopped"
# do not. We keep only administration events so held/not-given orders are excluded.
ADMINISTERED_KEYWORD = "administered"

# Temporal validation split (exact MIMIC-IV anchor_year_group strings, " - " sep)
TRAIN_YEAR_GROUPS = {"2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"}
TEST_YEAR_GROUPS = {"2020 - 2022"}

OUTPUT_COLUMNS = [
    "subject_id", "hadm_id", "stay_id",
    "first_careunit", "last_careunit",
    "intime", "outtime", "los_hours",
    "gender", "anchor_age", "anchor_year_group", "dod",
    "label_icd", "label_proxy", "label_combined", "split",
]


# --------------------------------------------------------------------------- #
# Small print helpers
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


def to_bool_mask(series_mask: pd.Series) -> pd.Series:
    """Convert a possibly-nullable boolean Series to a plain bool Series."""
    return series_mask.fillna(False).astype(bool)


# --------------------------------------------------------------------------- #
# Stage steps
# --------------------------------------------------------------------------- #
def check_inputs() -> None:
    missing = [p for p in (ICUSTAYS_PATH, PATIENTS_PATH, DIAGNOSES_PATH, EMAR_PATH)
               if not p.exists()]
    if missing:
        print("[ERROR] Required MIMIC-IV file(s) not found:")
        for p in missing:
            print(f"        {p}")
        print("Run scripts/verify_setup.py to confirm data placement.")
        sys.exit(1)


def load_base_cohort() -> pd.DataFrame:
    """Load icustays, compute los_hours, keep stays >= MIN_ICU_HOURS."""
    step("Loading icustays and applying >= 24h inclusion")
    icu = pd.read_csv(
        ICUSTAYS_PATH,
        compression="gzip",
        usecols=["subject_id", "hadm_id", "stay_id",
                 "first_careunit", "last_careunit", "intime", "outtime"],
    )
    n_total = len(icu)

    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu = icu.dropna(subset=["intime", "outtime", "hadm_id"])
    n_valid = len(icu)

    icu["los_hours"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600.0
    cohort = icu[icu["los_hours"] >= MIN_ICU_HOURS].copy()
    n_kept = len(cohort)

    # Normalize identifier dtypes (needed for clean downstream merges).
    for col in ("subject_id", "hadm_id", "stay_id"):
        cohort[col] = pd.to_numeric(cohort[col], errors="coerce").astype("int64")

    print(f"    ICU stays total           : {n_total:,}")
    print(f"    with valid in/out times   : {n_valid:,}")
    print(f"    >= {MIN_ICU_HOURS:.0f}h (included cohort) : {n_kept:,} "
          f"({pct(n_kept, n_total)})")
    if n_kept == 0:
        print("[ERROR] No ICU stays met the inclusion criterion. Aborting.")
        sys.exit(1)
    return cohort


def attach_demographics_and_split(cohort: pd.DataFrame) -> pd.DataFrame:
    """Join patients, attach demographics, assign temporal split."""
    step("Attaching demographics and temporal split")
    pat = pd.read_csv(
        PATIENTS_PATH,
        compression="gzip",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year_group", "dod"],
    )
    pat["subject_id"] = pd.to_numeric(pat["subject_id"], errors="coerce").astype("int64")

    cohort = cohort.merge(pat, on="subject_id", how="left")

    def map_split(group: object) -> str:
        if group in TRAIN_YEAR_GROUPS:
            return "train"
        if group in TEST_YEAR_GROUPS:
            return "test"
        return "unknown"

    cohort["split"] = cohort["anchor_year_group"].map(map_split)

    n_unknown = int((cohort["split"] == "unknown").sum())
    print(f"    train stays               : {int((cohort['split'] == 'train').sum()):,}")
    print(f"    test  stays               : {int((cohort['split'] == 'test').sum()):,}")
    if n_unknown:
        print(f"    [WARN] {n_unknown:,} stays have an unrecognized anchor_year_group "
              f"and are marked 'unknown' (excluded from train/test).")
    return cohort


def load_delirium_hadm_ids() -> set:
    """Scan diagnoses_icd (chunked) for delirium-coded hospital admissions."""
    step("Scanning diagnoses_icd for ICD delirium codes (F05* / 293*)")
    delirium_hadms: set = set()
    n_rows = 0

    reader = pd.read_csv(
        DIAGNOSES_PATH,
        compression="gzip",
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": "string", "icd_version": "string"},
        chunksize=CHUNK_SIZE,
    )
    for chunk in tqdm(reader, desc="    diagnoses_icd", unit="chunk"):
        n_rows += len(chunk)
        code = chunk["icd_code"].astype("string").str.strip().str.upper()
        ver = chunk["icd_version"].astype("string").str.strip()

        is_icd10 = (ver == "10") & code.str.startswith(ICD10_DELIRIUM_PREFIX, na=False)
        is_icd9 = (ver == "9") & code.str.startswith(ICD9_DELIRIUM_PREFIX, na=False)
        mask = to_bool_mask(is_icd10 | is_icd9)

        hits = pd.to_numeric(chunk.loc[mask, "hadm_id"], errors="coerce").dropna()
        delirium_hadms.update(hits.astype("int64").tolist())

    print(f"    diagnoses rows scanned    : {n_rows:,}")
    print(f"    delirium-coded admissions : {len(delirium_hadms):,}")
    if not delirium_hadms:
        print("    [WARN] No delirium ICD codes matched. label_icd will be all 0.")
    return delirium_hadms


def load_antipsychotic_admins() -> pd.DataFrame:
    """Scan emar (chunked) for antipsychotic administration events."""
    step("Scanning emar for antipsychotic administrations")
    pieces = []
    n_rows = 0
    n_drug_match = 0
    n_admin_match = 0

    reader = pd.read_csv(
        EMAR_PATH,
        compression="gzip",
        usecols=["subject_id", "hadm_id", "charttime", "medication", "event_txt"],
        dtype={"medication": "string", "event_txt": "string"},
        chunksize=CHUNK_SIZE,
    )
    for chunk in tqdm(reader, desc="    emar", unit="chunk"):
        n_rows += len(chunk)

        med = chunk["medication"].astype("string")
        drug_mask = to_bool_mask(
            med.str.contains(ANTIPSYCHOTIC_REGEX, case=False, na=False, regex=True)
        )
        sub = chunk.loc[drug_mask]
        n_drug_match += len(sub)
        if sub.empty:
            continue

        evt = sub["event_txt"].astype("string").str.lower()
        admin_mask = to_bool_mask(evt.str.contains(ADMINISTERED_KEYWORD, na=False))
        sub = sub.loc[admin_mask, ["subject_id", "hadm_id", "charttime"]]
        n_admin_match += len(sub)
        if not sub.empty:
            pieces.append(sub)

    if pieces:
        admins = pd.concat(pieces, ignore_index=True)
    else:
        admins = pd.DataFrame(columns=["subject_id", "hadm_id", "charttime"])

    print(f"    emar rows scanned         : {n_rows:,}")
    print(f"    antipsychotic drug matches: {n_drug_match:,}")
    print(f"    of those, administrations : {n_admin_match:,}")
    if admins.empty:
        print("    [WARN] No antipsychotic administrations matched. "
              "label_proxy will be all 0.")
    return admins


def build_proxy_label(cohort: pd.DataFrame, admins: pd.DataFrame) -> set:
    """Attribute antipsychotic administrations to ICU stays by time window."""
    step("Attributing antipsychotic administrations to ICU stay windows")
    if admins.empty:
        return set()

    admins = admins.copy()
    admins["charttime"] = pd.to_datetime(admins["charttime"], errors="coerce")
    admins["subject_id"] = pd.to_numeric(admins["subject_id"], errors="coerce")
    admins["hadm_id"] = pd.to_numeric(admins["hadm_id"], errors="coerce")
    admins = admins.dropna(subset=["subject_id", "hadm_id", "charttime"])
    admins["subject_id"] = admins["subject_id"].astype("int64")
    admins["hadm_id"] = admins["hadm_id"].astype("int64")

    merged = admins.merge(
        cohort[["stay_id", "subject_id", "hadm_id", "intime", "outtime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    within = merged[
        (merged["charttime"] >= merged["intime"])
        & (merged["charttime"] <= merged["outtime"])
    ]
    proxy_stay_ids = set(within["stay_id"].unique().tolist())

    print(f"    admins mapped to a stay   : {len(merged):,}")
    print(f"    admins within ICU window  : {len(within):,}")
    print(f"    proxy-positive ICU stays  : {len(proxy_stay_ids):,}")
    return proxy_stay_ids


def assemble_and_save(cohort: pd.DataFrame,
                      delirium_hadms: set,
                      proxy_stay_ids: set) -> pd.DataFrame:
    """Build label columns, finalize, and write cohort.csv."""
    step("Assembling labels and writing cohort.csv")

    cohort["label_icd"] = cohort["hadm_id"].isin(delirium_hadms).astype(int)
    cohort["label_proxy"] = cohort["stay_id"].isin(proxy_stay_ids).astype(int)
    cohort["label_combined"] = (
        (cohort["label_icd"] == 1) | (cohort["label_proxy"] == 1)
    ).astype(int)

    cohort = cohort[OUTPUT_COLUMNS].sort_values("stay_id").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(OUTPUT_CSV, index=False)
    print(f"    wrote {len(cohort):,} rows -> {OUTPUT_CSV}")
    return cohort


def print_summary(cohort: pd.DataFrame) -> None:
    header("COHORT SUMMARY (aggregate statistics only)")
    n = len(cohort)
    n_icd = int(cohort["label_icd"].sum())
    n_proxy = int(cohort["label_proxy"].sum())
    n_comb = int(cohort["label_combined"].sum())
    n_both = int(((cohort["label_icd"] == 1) & (cohort["label_proxy"] == 1)).sum())
    n_icd_only = int(((cohort["label_icd"] == 1) & (cohort["label_proxy"] == 0)).sum())
    n_proxy_only = int(((cohort["label_icd"] == 0) & (cohort["label_proxy"] == 1)).sum())

    print(f"  Included ICU stays          : {n:,}")
    print()
    print(f"  label_icd      positives    : {n_icd:,}   ({pct(n_icd, n)})")
    print(f"  label_proxy    positives    : {n_proxy:,}   ({pct(n_proxy, n)})")
    print(f"  label_combined positives    : {n_comb:,}   ({pct(n_comb, n)})")
    print()
    print(f"  ICD & proxy (overlap)       : {n_both:,}")
    print(f"  ICD only                    : {n_icd_only:,}")
    print(f"  proxy only                  : {n_proxy_only:,}")

    print()
    print("  Per-split (label_combined prevalence):")
    for split_name in ("train", "test", "unknown"):
        sub = cohort[cohort["split"] == split_name]
        if len(sub) == 0:
            continue
        pos = int(sub["label_combined"].sum())
        print(f"    {split_name:<8} n = {len(sub):>8,}   positives = {pos:>7,} "
              f"({pct(pos, len(sub))})")

    print()
    print("  Notes:")
    print("    - ICD labels are admission-level (diagnoses_icd has hadm_id, not")
    print("      stay_id); all ICU stays in a delirium-coded admission inherit the")
    print("      label. Proxy labels are stay-level via charttime window.")
    print("    - Use label_icd for the ICD-only sensitivity arm and label_combined")
    print("      as the proxy-augmented arm in train/evaluate.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    header("NEUROFLOW - Stage 1: Cohort Extraction (MIMIC-IV v3.1)")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Output       : {OUTPUT_CSV}")

    check_inputs()

    cohort = load_base_cohort()
    cohort = attach_demographics_and_split(cohort)
    delirium_hadms = load_delirium_hadm_ids()
    admins = load_antipsychotic_admins()
    proxy_stay_ids = build_proxy_label(cohort, admins)
    cohort = assemble_and_save(cohort, delirium_hadms, proxy_stay_ids)
    print_summary(cohort)

    header("STAGE 1 COMPLETE")
    print("  Next: src/features/build_timeseries.py")


if __name__ == "__main__":
    main()