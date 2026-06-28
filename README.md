# NEUROFLOW

**FHIR-native ICU delirium prediction + ABCDEF bundle orchestration**

---

[![Python](https://img.shields.io/badge/python-3.9.6-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red)](https://pytorch.org)
[![FHIR](https://img.shields.io/badge/FHIR-R4-green)](https://hl7.org/fhir)
[![SMART](https://img.shields.io/badge/SMART--on--FHIR-EHR_Launch-purple)](https://smarthealthit.org)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Competition](https://img.shields.io/badge/AMIA_2026-FHIR_App_Competition-orange)](https://amia.org)

---

> **DATA COMPLIANCE NOTICE**
>
> This repository contains code only. No patient data is present anywhere in this codebase. MIMIC-IV v3.1 and eICU v2.0 are governed by the PhysioNet Credentialed Health Data Use Agreement 1.5.0 and must never be committed to any repository, shared through APIs, or transmitted to online platforms. To reproduce this work, obtain independent credentialed access at [physionet.org](https://physionet.org).

---

## The Problem

ICU delirium affects 7 million patients annually in the US. The standard detection method — CAM-ICU — happens only at nursing shift changes, leaving up to 12-hour blind spots. The warning signs are already there: heart rate variability collapsing, sedatives accumulating, circadian rhythms fragmenting. Recorded continuously. Never integrated into anything predictive.

The proven intervention is the ABCDEF bundle. Worldwide full compliance: 0–1% (54-country, 135-ICU study). The bottleneck is not prediction accuracy. It is intervention execution. NEUROFLOW addresses both.

---

## What It Does

A nurse opens her hospital dashboard inside the existing EHR. The NEUROFLOW panel shows a risk trajectory trending upward over the past 4 hours, currently reading 0.74 [CI: 0.61–0.83], flagged amber. Below it: an ABCDEF bundle compliance panel showing which of the six evidence-based intervention elements have been documented in the last 12 hours and which are still missing. No separate application. No login. No workflow disruption.

Under the hood: SMART-on-FHIR EHR launch receives patient and encounter context via real OAuth2 token exchange. The app queries FHIR R4 for Observation, MedicationAdministration, Encounter, and Patient resources. A 56,065-parameter Temporal Convolutional Network runs inference on the 40-channel, 63-hour synchronized time-series. A RiskAssessment resource with calibrated conformal prediction intervals is written back to FHIR via deterministic UUIDv5 idempotent PUT. If risk exceeds threshold, a US Core CarePlan with six discrete ABCDEF activities is generated and compliance is scored from existing Observations. Dashboard refreshes every hour. Every prediction is encounter-scoped, patient-bound, timestamped, and auditable.

---

## Results

| Metric | Value |
|--------|-------|
| MIMIC-IV test AUROC (temporal holdout, 2020–2022) | **0.8125** |
| MIMIC-IV test AUPRC | **0.5638** |
| Brier score (isotonic calibrated) | **0.1330** |
| Conformal coverage (target 90%) | **95.84%** |
| Median early warning lead time | **20.0 hours** |
| Patients warned ≥1h before first antipsychotic | **77.4%** (642/829) |
| eICU external AUROC (208 hospitals, no retraining) | **0.6568** |
| Real ICU stays processed | **207,729** |
| Raw data rows scanned | **~800 million** |

**Ablation (MIMIC-IV test set):**

| Model | AUROC | AUPRC |
|-------|-------|-------|
| NeuroflowTCN (40 channels, pharmacodynamic embedding) | 0.8125 | 0.5638 |
| XGBoost (128 engineered features) | 0.8044 | 0.5504 |
| LSTM baseline | 0.7911 | 0.5416 |
| Logistic regression (raw vitals) | 0.6899 | 0.3485 |

The eICU gap (0.8125 → 0.6568) reflects genuine cross-institutional domain shift across 208 community hospitals vs one academic medical center. Fifteen distinct diagnostic checks confirmed this is not a fixable pipeline defect — two real bugs were found and fixed during the investigation (blood pressure source error, imputation fabrication error).

---

## Two Genuine Novelty Claims

**1. Conformal prediction intervals as structured FHIR extensions.**
Every other clinical ML system that writes to FHIR RiskAssessment writes a bare probability. NEUROFLOW writes calibrated 90% conformal prediction intervals as nested sub-extensions on the prediction element — making uncertainty a first-class FHIR resource rather than a suppressed internal value. Zero published papers were found doing this.

**2. FHIR-native ABCDEF bundle CarePlan orchestration.**
NEUROFLOW connects AI risk prediction to the proven ABCDEF intervention protocol through standard FHIR R4 resources. When risk exceeds threshold, a US Core CarePlan with six discrete nursing activities is auto-generated. Compliance is scored hourly by querying existing FHIR Observations by LOINC code. This closes the loop between prediction and intervention. No published system was found doing this.

---

## Architecture

**Model:** NeuroflowTCN — dilated causal convolutional network with learned pharmacodynamic embedding

- Input: [B, 40, 63] — 40 channels, 63-hour window at 1-hour resolution
- Pharmacodynamic embedding: 10 sedative channels (drug identity + PK onboard) projected through learned 2-layer pointwise convolution before concatenation with 30 physiological channels
- Dilation schedule: 1, 2, 4, 8, 16, 32 — receptive field 64h ≥ 63h
- Causality formally verified: max delta from future perturbation = 0.00e+00
- 56,065 parameters

**40 Channels:**

| Group | Channels |
|-------|----------|
| Vitals (8) | heart_rate, sbp, dbp, spo2, resp_rate, temperature, rass, gcs_total |
| Missingness (8) | `{vital}_mask` — 1 if no observation that hour |
| Time-delta (8) | `{vital}_hours_since` — hours since last real observation; WINDOW_HOURS before first |
| Derived (3) | hrv_proxy, phys_instability, charting_density |
| Circadian (2) | circadian_sin, circadian_cos |
| Sedative amounts (5) | amt_propofol, amt_midazolam, amt_dexmedetomidine, amt_lorazepam, amt_fentanyl |
| Sedative onboard (5) | onboard_* (PK exponential-decay accumulation per drug half-life) |
| Pad (1) | 1 for hours beyond ICU discharge |

**Labels:** ICD-10 F05 / ICD-9 293 (primary) + antipsychotic administration proxy (haloperidol, quetiapine, olanzapine via emar). Combined label: 13.19% prevalence.

**Validation:** Strict temporal holdout — train on anchor_year_group 2008–2019, test on 2020–2022. No random splits.

**Uncertainty:** Split conformal prediction, locally risk-binned. 90% target coverage, 95.84% empirical on test set. Fit on validation set predictions, strictly separated from test set.

**Calibration:** Isotonic regression fit on chronological last 10% of training split (6,600 stays). Brier 0.1768 → 0.1330.

---

## FHIR Implementation

**Server:** HAPI FHIR JPA Server v8.8.0, FHIR R4 (4.0.1)

**SMART-on-FHIR:** Real EHR launch flow confirmed working against SMART Health IT public sandbox. `/launch` receives `iss` and `launch` token, fetches SMART configuration, performs OAuth2 authorization redirect. `/callback` exchanges code at token endpoint, redirects to dashboard with real patient/encounter context.

**RiskAssessment writeback (novel):**
```json
{
  "resourceType": "RiskAssessment",
  "id": "<UUIDv5 deterministic — one per encounter per hour>",
  "prediction": [{
    "probabilityDecimal": 0.74,
    "extension": [
      {
        "url": "http://neuroflow.ai/fhir/ext/conformal-interval",
        "extension": [
          {"url": "low", "valueDecimal": 0.61},
          {"url": "high", "valueDecimal": 0.83}
        ]
      },
      {
        "url": "http://neuroflow.ai/fhir/ext/confidence-tier",
        "valueCode": "high"
      }
    ]
  }]
}
```

Idempotent PUT via deterministic UUIDv5 keyed on (encounter_id, hour_bucket) — one clean RiskAssessment per encounter per hour, no database pollution.

**CarePlan (novel):** US Core CarePlan with 6 discrete ABCDEF activities auto-generated when risk ≥ 0.30. Compliance scored hourly from FHIR Observations by LOINC code (last 12 hours).

**ABCDEF LOINC mapping:**

| Element | Action | LOINC |
|---------|--------|-------|
| A — Pain | Pain severity assessment q4h | 72514-3 |
| B — Breathing | SAT/SBT documentation | 80330-2 |
| C — Sedation | RASS target −1 to 0 | 72651-2 |
| D — Delirium | CAM-ICU every shift | 72133-2 |
| E — Mobility | Activity level documented | 45970-1 |
| F — Family | Family communication | 85432-3 |

---

## Datasets

| Dataset | Role | Scale | Access |
|---------|------|-------|--------|
| MIMIC-IV v3.1 | Training + internal validation | 94,458 ICU stays, BIDMC 2008–2022 | [physionet.org](https://doi.org/10.13026/kpb9-mt58) |
| eICU v2.0 | External validation | 200,859 stays, 208 hospitals, 2014–2015 | [physionet.org](https://doi.org/10.13026/C2WM1R) |

---

## Stack

| Component | Tool |
|-----------|------|
| Model training | PyTorch 2.8.0, Apple Silicon MPS |
| Data processing | Pandas 2.3.3, NumPy 2.0.2 |
| Baselines + calibration | Scikit-learn 1.6.1, XGBoost 2.1.4 |
| FHIR server | HAPI FHIR JPA Server v8.8.0, Docker |
| FHIR client | Requests 2.32.5 (no SDK) |
| SMART-on-FHIR launch | Flask 3.1.3 |
| Dashboard | Plotly 6.8.0, Dash 4.3.0 |
| Hardware | MacBook Pro M5 Pro 24GB — no cloud compute |

---

## Project Structure

```
NEUROFLOW/
├── src/
│   ├── features/
│   │   ├── extract_cohort.py       Stage 1: cohort + labels from MIMIC-IV
│   │   ├── build_timeseries.py     Stage 2: 40-channel time-series
│   │   └── eicu_adapter.py         Stage 3: eICU external validation adapter
│   ├── models/
│   │   ├── tcn.py                  TCN architecture + pharmacodynamic embedding
│   │   ├── uncertainty.py          Conformal prediction calibrator
│   │   ├── train.py                Training loop (temporal holdout)
│   │   └── evaluate.py             Evaluation, ablation, subgroups, eICU
│   └── fhir/
│       ├── fhir_client.py          FHIR R4 read/write (requests-based, no SDK)
│       ├── inference_loop.py       Hourly inference + FHIR writeback
│       ├── dashboard.py            Plotly Dash clinical dashboard
│       └── smart_launch.py         SMART-on-FHIR OAuth2 EHR launch
├── scripts/
│   └── verify_setup.py             Environment verification (51 checks)
├── data/                           never committed (PhysioNet DUA)
├── models/                         never committed
├── .gitignore
└── README.md
```

---

## Reproducing This Work

**Prerequisites:**
1. Create account at [physionet.org](https://physionet.org)
2. Complete CITI Data or Specimens Only Research training
3. Sign MIMIC-IV DUA: [doi.org/10.13026/kpb9-mt58](https://doi.org/10.13026/kpb9-mt58)
4. Sign eICU DUA: [doi.org/10.13026/C2WM1R](https://doi.org/10.13026/C2WM1R)
5. Download MIMIC-IV files into `data/raw/mimiciv/3.1/` and eICU files into `data/raw/eicu/2.0/`

**Environment:**
```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 numpy pandas scikit-learn plotly dash flask \
            requests pyarrow tqdm xgboost
docker run -d --name neuroflow-fhir -p 8080:8080 hapiproject/hapi:latest
python scripts/verify_setup.py   # 51 checks, 0 failures expected
```

**Pipeline (run in order):**
```bash
python src/features/extract_cohort.py      # ~5 min
python src/features/build_timeseries.py    # ~15 min (scans 433M rows)
python src/features/eicu_adapter.py        # ~25 min (scans 324M rows)
python src/models/train.py                 # ~30 epochs, ~2h on M5 Pro MPS
python src/models/evaluate.py             # full ablation + eICU validation
```

**FHIR layer:**
```bash
python src/fhir/inference_loop.py          # seeds synthetic patient, runs demo
python src/fhir/dashboard.py               # http://localhost:8050

# For SMART EHR launch testing against public sandbox:
export NEUROFLOW_REDIRECT_URI=https://<your-ngrok-url>/callback
python src/fhir/smart_launch.py            # http://localhost:8051
# Then test at https://launch.smarthealthit.org
```

---

## Known Limitations

**eICU external AUROC (0.6568).** Fifteen diagnostic checks confirmed this reflects genuine cross-institutional domain shift (208 community hospitals vs one academic center), not a fixable pipeline defect. Temperature variance is 15× lower in eICU (88.97% of hours filled with neutral default); RASS is 98.73% missing. The remaining gap is not addressable without domain adaptation or prospective data from the target institution.

**FHIR layer tested with synthetic patients only.** Production deployment would require EHR vendor registration, OAuth2 certification, and hospital IT security review.

**Synchronous polling (MVP constraint).** The hourly inference loop makes sequential HTTP requests per encounter. A production deployment would adopt FHIR Subscription (R4B/R5 REST-hook or Websocket channels) or FHIR Bulk Data Access ($export) for parallel, event-driven data streaming.

**Conformal calibrator fit on raw probabilities.** Nonconformity scores are computed on raw TCN sigmoid outputs; intervals are applied to isotonic-calibrated probabilities at inference time. Fitting on calibrated probabilities would yield tighter bands and is noted as a future improvement.

---

## Citations

```
Johnson, A., et al. (2024). MIMIC-IV (version 3.1). PhysioNet. https://doi.org/10.13026/kpb9-mt58
Johnson, A.E.W., et al. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x
Pollard, T., et al. (2019). eICU Collaborative Research Database (version 2.0). PhysioNet. https://doi.org/10.13026/C2WM1R
Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation, 101(23), e215-e220.
Devlin, J.W., et al. (2018). Clinical Practice Guidelines for the Prevention and Management of Pain,
    Agitation/Sedation, Delirium, Immobility, and Sleep Disruption in Adult Patients in the ICU.
    Critical Care Medicine, 46(9), e825-e873. (ABCDEF bundle)
Rocheteau, E., et al. (2021). Temporal Pointwise Convolutional Networks for Length of Stay
    Prediction in the Intensive Care Unit. ACM CHIL 2021. (eICU delirium label methodology)
```

---

## License

Code: MIT. Data: Not included. Subject to [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/). Per the data use agreement, code used to produce openly disseminated results must be contributed to an open repository.