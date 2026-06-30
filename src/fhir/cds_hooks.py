#!/usr/bin/env python3
"""
NEUROFLOW - CDS Hooks Service
=============================
A CDS Hooks (https://cds-hooks.org) service implementing the 'patient-view'
hook. At the point of care, it surfaces the most recent pre-computed NEUROFLOW
delirium RiskAssessment plus ABCDEF bundle compliance as a CDS card.

This service performs NO live TCN inference - it reads the already-written
RiskAssessment from FHIR so it can respond within the CDS Hooks sub-500ms
budget. Continuous inference is handled separately by inference_loop.py.

Runs on port 8052 (dashboard 8050, smart_launch 8051).

Run:
    source /Users/eshanpradhan/Desktop/NEUROFLOW/.venv/bin/activate
    python src/fhir/cds_hooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhir.fhir_client import (  # noqa: E402
    FHIRClient, FHIRError, build_fhir_reference, ABCDEF_ELEMENTS,
    EXT_CONFORMAL, EXT_CONFIDENCE_TIER,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
FHIR_BASE_URL = "http://localhost:8080/fhir"
CDS_HOST = "localhost"
CDS_PORT = 8052
CDS_BASE = f"http://{CDS_HOST}:{CDS_PORT}"
SERVICE_ID = "delirium-risk"
GITHUB_URL = "https://github.com/eshanpradhan/NEUROFLOW"

ABCDEF_TITLES = {k: t for (k, t, _a, _l) in ABCDEF_ELEMENTS}

_client = FHIRClient(FHIR_BASE_URL)
app = Flask(__name__)


# --------------------------------------------------------------------------- #
# FHIR reads (pre-computed only)
# --------------------------------------------------------------------------- #
def latest_risk_assessment(patient_id: str, encounter_id: str):
    params = {
        "patient": build_fhir_reference("Patient", patient_id),
        "encounter": build_fhir_reference("Encounter", encounter_id),
        "_sort": "-date",
        "_count": "1",
    }
    ras = _client._search_all("RiskAssessment", params, max_pages=1)
    return ras[0] if ras else None


def parse_risk_assessment(ra: dict):
    pred = (ra.get("prediction", [{}]) or [{}])[0]
    prob = pred.get("probabilityDecimal")
    if prob is None:
        return None
    low = high = None
    tier = None
    for ext in pred.get("extension", []):
        if ext.get("url") == EXT_CONFORMAL:
            for sub in ext.get("extension", []):
                if sub.get("url") == "low":
                    low = sub.get("valueDecimal")
                elif sub.get("url") == "high":
                    high = sub.get("valueDecimal")
        elif ext.get("url") == EXT_CONFIDENCE_TIER:
            tier = ext.get("valueCode")
    return {
        "id": ra.get("id", ""),
        "probability": float(prob),
        "low": float(low) if low is not None else float(prob),
        "high": float(high) if high is not None else float(prob),
        "tier": tier or "n/a",
        "time": ra.get("occurrenceDateTime", ""),
    }


def find_active_careplan(patient_id: str, encounter_id: str):
    params = {
        "patient": build_fhir_reference("Patient", patient_id),
        "encounter": build_fhir_reference("Encounter", encounter_id),
        "status": "active",
        "_count": "1",
    }
    try:
        plans = _client._search_all("CarePlan", params, max_pages=1)
        return plans[0]["id"] if plans else None
    except FHIRError:
        return None


def abcdef_status(patient_id: str, encounter_id: str):
    careplan_id = find_active_careplan(patient_id, encounter_id)
    if careplan_id is None:
        return None, None
    try:
        comp = _client.score_abcdef_compliance(patient_id, encounter_id,
                                               careplan_id)
    except FHIRError:
        return careplan_id, None
    return careplan_id, comp


# --------------------------------------------------------------------------- #
# Card construction
# --------------------------------------------------------------------------- #
def indicator_for(prob: float) -> str:
    if prob >= 0.8:
        return "critical"
    if prob >= 0.3:
        return "warning"
    return "info"


def build_detail(risk: dict, compliance) -> str:
    p, low, high, tier = (risk["probability"], risk["low"],
                          risk["high"], risk["tier"])
    lines = [
        f"## NEUROFLOW ICU Delirium Risk",
        "",
        f"**Risk probability:** {p:.0%}  ",
        f"**90% conformal interval:** [{low:.0%} – {high:.0%}]  ",
        f"**Confidence tier:** {tier}  ",
    ]
    if risk.get("time"):
        lines.append(f"**Last assessed:** {risk['time']}  ")
    lines.append("")

    if compliance is None:
        lines.append("_ABCDEF bundle: no active CarePlan for this encounter._")
    else:
        keys = [k for (k, _t, _a, _l) in ABCDEF_ELEMENTS]
        documented = [k for k in keys if compliance.get(k)]
        missing = [k for k in keys if not compliance.get(k)]
        lines.append(
            f"**ABCDEF bundle:** {len(documented)}/6 elements documented "
            f"in last 12h."
        )
        if missing:
            named = ", ".join(f"{k} ({ABCDEF_TITLES[k]})" for k in missing)
            lines.append(f"**Missing:** {named}")

    if p >= 0.65:
        lines.append("")
        lines.append("**Suggested actions:** non-pharmacologic bundle "
                     "(reorientation, sleep hygiene, sedation review); "
                     "formal CAM-ICU assessment.")
    return "\n".join(lines)


def build_card(risk: dict, compliance) -> dict:
    p = risk["probability"]
    missing_str = ""
    if compliance is not None:
        keys = [k for (k, _t, _a, _l) in ABCDEF_ELEMENTS]
        documented = sum(1 for k in keys if compliance.get(k))
        missing = [k for k in keys if not compliance.get(k)]
        missing_str = (f" | ABCDEF {documented}/6"
                       + (f", missing {', '.join(missing)}" if missing else ""))
    summary = (f"Delirium risk: {p:.0%} "
               f"([{risk['low']:.0%}-{risk['high']:.0%}], "
               f"{risk['tier']} confidence){missing_str}")
    return {
        "summary": summary[:140],
        "indicator": indicator_for(p),
        "detail": build_detail(risk, compliance),
        "source": {"label": "NEUROFLOW", "url": GITHUB_URL},
        "suggestions": [],
        "selectionBehavior": "any",
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/cds-services", methods=["GET"])
def discovery():
    return jsonify({
        "services": [{
            "hook": "patient-view",
            "title": "NEUROFLOW Delirium Risk",
            "description": ("Surfaces ICU delirium risk and ABCDEF bundle "
                            "compliance at the point of care."),
            "id": SERVICE_ID,
            "prefetch": {
                "patient": "Patient/{{context.patientId}}",
            },
        }]
    })


@app.route(f"/cds-services/{SERVICE_ID}", methods=["POST"])
def delirium_risk():
    body = request.get_json(silent=True) or {}
    context = body.get("context", {}) or {}
    patient_id = context.get("patientId")
    encounter_id = context.get("encounterId")

    if not patient_id:
        return jsonify({"cards": []}), 200

    try:
        if encounter_id:
            ra = latest_risk_assessment(patient_id, encounter_id)
        else:
            ra = None
    except FHIRError:
        return jsonify({"cards": []}), 200

    if ra is None:
        # No pre-computed assessment yet -> no advice to surface.
        return jsonify({"cards": []}), 200

    risk = parse_risk_assessment(ra)
    if risk is None:
        return jsonify({"cards": []}), 200

    _careplan_id, compliance = abcdef_status(patient_id, encounter_id)
    card = build_card(risk, compliance)
    return jsonify({"cards": [card]}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "neuroflow-cds-hooks"})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _print_banner() -> None:
    print("=" * 70)
    print("  NEUROFLOW - CDS Hooks Service (patient-view)")
    print("=" * 70)
    client = FHIRClient(FHIR_BASE_URL)
    try:
        cap = client.capability()
        print(f"  FHIR server   : {cap.get('fhirVersion', '?')} @ {FHIR_BASE_URL}")
    except FHIRError as e:
        print(f"  [WARN] FHIR server not reachable: {e}")
        print("  Start it with: docker start neuroflow-fhir")

    print()
    print(f"  Discovery     : {CDS_BASE}/cds-services")
    print(f"  Service       : POST {CDS_BASE}/cds-services/{SERVICE_ID}")
    print(f"  Health        : {CDS_BASE}/health")
    print(f"  Note          : reads pre-computed RiskAssessments (no live "
          f"inference, sub-500ms).")
    print()
    print("  Example test (replace IDs with a seeded patient/encounter):")
    print(f"    curl -s {CDS_BASE}/cds-services | python -m json.tool")
    print(f"    curl -s -X POST {CDS_BASE}/cds-services/{SERVICE_ID} \\")
    print("      -H 'Content-Type: application/json' \\")
    print("      -d '{\"hook\":\"patient-view\",\"hookInstance\":\"test-1\","
          "\"context\":{\"patientId\":\"<id>\",\"encounterId\":\"<id>\"}}'")
    print()
    print(f"  CDS Hooks service starting on {CDS_BASE}")
    print("=" * 70)


def main() -> None:
    _print_banner()
    app.run(host="0.0.0.0", port=CDS_PORT, debug=False)


if __name__ == "__main__":
    main()