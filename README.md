# NEUROFLOW

**why predict delirium after when you can predict before**

---

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red)](https://pytorch.org)
[![FHIR](https://img.shields.io/badge/FHIR-R4-green)](https://hl7.org/fhir)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Competition](https://img.shields.io/badge/AMIA_2026-FHIR_App_Competition-orange)](https://amia.org)

---

> **DATA COMPLIANCE NOTICE**
>
> This repository contains code only. No patient data is present anywhere in this codebase. MIMIC-IV v3.1 and eICU v2.0 are governed by the PhysioNet Credentialed Health Data Use Agreement and must never be committed to any repository, shared through APIs, or transmitted to online platforms. To reproduce this work, obtain independent credentialed access at [physionet.org](https://physionet.org).

---

## The Problem

ICU delirium affects 7 million patients annually in the US. The standard detection method — CAM-ICU — happens only at nursing shift changes, leaving up to 12-hour blind spots. The warning signs are already there: heart rate variability collapsing, sedatives accumulating, sleep fragmenting. Recorded continuously. Never integrated into anything predictive. The intervention window is real and narrow. Current systems never reach it. NEUROFLOW does.

---

## What It Does

> A nurse opens her hospital dashboard at 3am. Inside the existing EHR interface she sees the NEUROFLOW panel: a risk trajectory trending upward over the past 4 hours, currently reading 0.74 ± 0.06, flagged amber. Below it: "Predicted delirium onset window: 6–14 hours." A recommended action bundle appears — reorientation protocol, sedation review, lighting adjustment. She taps acknowledge. The RiskAssessment resource is timestamped in the patient's FHIR timeline. No separate application. No login. No workflow disruption.
>
> Under the hood: SMART-on-FHIR launch receives patient and encounter context automatically. The app queries the FHIR server for Observation, MedicationAdministration, Condition, and Encounter resources scoped to the current encounter. A Temporal Convolutional Network runs inference on the retrieved 63-hour synchronized time-series. A RiskAssessment resource is written back to the FHIR server tied to that encounter ID and patient timeline. Dashboard refreshes every hour. Every prediction is encounter-scoped, patient-bound, timestamped, and auditable.

---

## Three Core Innovations

**Pharmacodynamic embedding.** Drug identity, dose, timing, and pharmacokinetic decay encoded as a learned temporal embedding. Continuous propofol for 48 hours is neurologically different from a single bolus 6 hours ago. No existing delirium prediction system encodes this properly.

**Uncertainty quantification.** Every prediction includes a calibrated confidence interval via conformal prediction. The RiskAssessment stores a probability range, not a point estimate. Point-only predictions are not clinically deployable.

**Bidirectional FHIR loop.** Reads structured clinical resources, runs inference, writes back structured clinical resources with full provenance. Every prediction is encounter-scoped, timestamped, and auditable.

---

## Clinical Action Ladder

| Risk | Duration | Response |
|------|----------|----------|
| < 0.40 | — | Background monitoring, sidebar only |
| 0.40 – 0.64 | — | Passive amber flag on next chart review |
| 0.65 – 0.79 | >= 2h sustained | Amber alert + non-pharmacological care bundle |
| >= 0.80 | >= 1h sustained | Red alert, attending notification, CAM-ICU order |

Alerts fire on sustained threshold crossings only. False alarm rate per ICU day is a primary reported metric.

---

## Datasets

| Dataset | Role | Scale | Access |
|---------|------|-------|--------|
| MIMIC-IV v3.1 | Training + internal validation | 94,458 ICU stays, BIDMC 2008–2022 | [physionet.org](https://doi.org/10.13026/kpb9-mt58) |
| eICU v2.0 | External validation | 200,000+ admissions, 335 hospitals, 2014–2015 | [physionet.org](https://doi.org/10.13026/C2WM1R) |

Delirium labels: ICD-10 F05 (primary) + antipsychotic proxy via `emar`/`emar_detail`. Sensitivity analysis on both label strategies reported.

---

## Model

**Architecture:** Temporal Convolutional Network with dilated causal convolutions
**Dilation schedule:** 1 → 2 → 4 → 8 → 16 → 32 — receptive field of 63 hours at 1-hour resolution

**Features:** heart rate variability proxy, sedation intensity index, sleep disruption proxy, physiological instability score, circadian phase alignment, forward-fill missing data with explicit missingness flags

**Validation:** Temporal holdout only — train on pre-cutoff admissions, test on post-cutoff. No random splits.

**Ablation baselines:** LSTM, logistic regression, SAPS-II, rule-based sedation threshold, TCN without medication embedding

---

## Stack

| Component | Tool |
|-----------|------|
| Model training | PyTorch 2.8.0, Apple Silicon MPS |
| Data processing | Pandas, NumPy |
| Baselines + calibration | Scikit-learn |
| FHIR server | HAPI FHIR JPA Server v8.8.0, Docker |
| SMART-on-FHIR interface | FastAPI + Uvicorn |
| Dashboard | Plotly |
| Exploration | Jupyter |

Runs entirely on Apple MacBook Pro M5 Pro 24GB. No cloud compute required. Full training cycle ~2–4 hours.

---

## Project Structure

    NEUROFLOW/
    ├── data/
    │   ├── raw/            never committed
    │   └── processed/      never committed
    ├── models/             never committed
    ├── notebooks/
    ├── src/
    │   ├── features/       feature engineering pipeline
    │   ├── models/         TCN architecture and training
    │   └── fhir/           FHIR read/write interface
    ├── scripts/
    ├── fhir-server/
    ├── .gitignore
    └── README.md

---

## Reproducing This Work

1. Create account at physionet.org
2. Complete CITI Data or Specimens Only Research at citiprogram.org
3. Sign MIMIC-IV DUA: https://doi.org/10.13026/kpb9-mt58
4. Sign eICU DUA: https://doi.org/10.13026/C2WM1R
5. Download MIMIC-IV files into `data/raw/mimiciv/3.1/` and eICU files into `data/raw/eicu/2.0/`
6. `pip install torch numpy pandas scikit-learn plotly fastapi uvicorn requests jupyter pyarrow tqdm`
7. `docker run -d --name neuroflow-fhir -p 8080:8080 hapiproject/hapi:latest`
8. Run in order: `extract_cohort.py` → `build_timeseries.py` → `train.py` → `inference_loop.py`

> Per PhysioNet policy, MIMIC-IV and eICU data must not be sent through external APIs or online platforms. Use a locally deployed model such as [Ollama](https://ollama.com) for any LLM-assisted data work.

---

## Citations

    Johnson, A., et al. (2024). MIMIC-IV (version 3.1). PhysioNet. https://doi.org/10.13026/kpb9-mt58
    Johnson, A.E.W., et al. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x
    Pollard, T., et al. (2019). eICU Collaborative Research Database (version 2.0). PhysioNet. https://doi.org/10.13026/C2WM1R
    Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation, 101(23), e215-e220.

---

## License

Code: MIT. Data: Not included. Subject to [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/). Derived datasets must be treated as sensitive and shared only under the same PhysioNet agreement. Per the data use agreement, code used to produce openly disseminated results must also be contributed to an open repository.
