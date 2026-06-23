#!/usr/bin/env python3
"""
NEUROFLOW - SMART-on-FHIR Launch
================================
Handles the real SMART-on-FHIR EHR launch flow from an external authorization
server (e.g. https://launch.smarthealthit.org): the EHR calls /launch with
?iss=<fhir_base>&launch=<token>, NEUROFLOW discovers the auth server, performs
the authorization-code redirect, exchanges the code for a token at /callback,
and redirects into the NEUROFLOW Dash dashboard with patient/encounter context.

A public client (no secret) is used, suitable for a student MVP. Set
NEUROFLOW_REDIRECT_URI to an externally reachable (e.g. ngrok) callback URL when
testing against a hosted SMART sandbox:

    export NEUROFLOW_REDIRECT_URI=https://sculpture-undocked-cornfield.ngrok-free.dev/callback

Run:
    source /Users/eshanpradhan/Desktop/NEUROFLOW/.venv/bin/activate
    python src/fhir/smart_launch.py
"""

from __future__ import annotations

import os
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests
from flask import Flask, jsonify, redirect, request, session

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhir.fhir_client import FHIRClient, FHIRError  # noqa: E402


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
FHIR_BASE_URL = "http://localhost:8080/fhir"
LAUNCH_HOST = "localhost"
LAUNCH_PORT = 8051
LAUNCH_BASE = f"http://{LAUNCH_HOST}:{LAUNCH_PORT}"
DASHBOARD_URL = "http://localhost:8050"

CLIENT_ID = "neuroflow_public"
REDIRECT_URI = os.environ.get("NEUROFLOW_REDIRECT_URI",
                              f"{LAUNCH_BASE}/callback")
LAUNCH_SCOPE = "launch launch/patient launch/encounter openid fhirUser"
HTTP_TIMEOUT = 30

SMART_CAPABILITIES = [
    "launch-ehr",
    "client-public",
    "context-ehr-patient",
    "context-ehr-encounter",
    "permission-patient",
    "permission-user",
]

# In-memory cache of SMART configuration keyed by issuer (avoids re-fetching).
_SMART_CONFIG_CACHE: dict = {}


def get_smart_config(iss: str) -> dict:
    """Fetch (and cache) the SMART configuration for an issuer."""
    if iss in _SMART_CONFIG_CACHE:
        return _SMART_CONFIG_CACHE[iss]
    url = f"{iss.rstrip('/')}/.well-known/smart-configuration"
    r = requests.get(url, headers={"Accept": "application/json"},
                     timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise FHIRError(r.status_code, r.text, context=f"GET {url}")
    cfg = r.json()
    _SMART_CONFIG_CACHE[iss] = cfg
    return cfg


# --------------------------------------------------------------------------- #
# Launch context from FHIR
# --------------------------------------------------------------------------- #
def _patient_name(patient: dict) -> str:
    names = patient.get("name", []) or []
    if not names:
        return "Unknown"
    n = names[0]
    if n.get("text"):
        return n["text"]
    given = " ".join(n.get("given", []) or [])
    family = n.get("family", "") or ""
    full = f"{given} {family}".strip()
    return full or "Unknown"


def launch_context_from_fhir(patient_id: str, encounter_id: str,
                             client: FHIRClient) -> dict:
    """Pull patient + encounter details from FHIR into a launch context dict."""
    context = {
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "patient_name": None,
        "gender": None,
        "birth_date": None,
        "encounter_type": None,
        "encounter_status": None,
        "admission_time": None,
        "fhir_base": client.base_url,
        "resolved": False,
        "error": None,
    }
    try:
        patient = client.get_patient(patient_id)
        encounter = client.get_encounter(encounter_id)
    except FHIRError as e:
        context["error"] = str(e)
        return context

    context["patient_name"] = _patient_name(patient)
    context["gender"] = patient.get("gender")
    context["birth_date"] = patient.get("birthDate")

    etype = encounter.get("type", []) or []
    if etype:
        t0 = etype[0]
        context["encounter_type"] = t0.get("text") or (
            (t0.get("coding", [{}]) or [{}])[0].get("display"))
    context["encounter_status"] = encounter.get("status")
    context["admission_time"] = (encounter.get("period", {}) or {}).get("start")
    context["resolved"] = True
    return context


# --------------------------------------------------------------------------- #
# Launch handler
# --------------------------------------------------------------------------- #
class SmartLaunchHandler:
    """Coordinates the SMART-on-FHIR launch sequence."""

    def __init__(self, fhir_base_url: str = FHIR_BASE_URL,
                 dashboard_url: str = DASHBOARD_URL):
        self.fhir_base_url = fhir_base_url
        self.dashboard_url = dashboard_url
        self.client = FHIRClient(fhir_base_url)
        # In-memory map of authorization codes -> launch context (demo only).
        self._codes: dict = {}
        self._tokens: dict = {}

    def launch(self, patient_id: str, encounter_id: str) -> dict:
        """Resolve context and mint a demo token (used by simulated paths)."""
        context = launch_context_from_fhir(patient_id, encounter_id, self.client)
        access_token = "demo." + secrets.token_urlsafe(24)
        self._tokens[access_token] = {
            "patient": patient_id, "encounter": encounter_id,
            "issued": datetime.now(timezone.utc).isoformat(),
            "scope": "launch patient/*.read user/*.read",
        }
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "launch patient/*.read user/*.read",
            "patient": patient_id,
            "encounter": encounter_id,
            "launch_context": context,
        }

    def issue_code(self, patient_id: str, encounter_id: str) -> str:
        code = secrets.token_urlsafe(16)
        self._codes[code] = {"patient": patient_id, "encounter": encounter_id,
                             "issued": time.time()}
        return code

    def exchange_code(self, code: str) -> dict:
        ctx = self._codes.pop(code, None)
        if ctx is None:
            return {"error": "invalid_grant",
                    "error_description": "Unknown or expired authorization code"}
        return self.launch(ctx["patient"], ctx["encounter"])

    def validate_context(self, context: dict) -> bool:
        if not isinstance(context, dict):
            return False
        if not context.get("patient_id") or not context.get("encounter_id"):
            return False
        if context.get("error"):
            return False
        return bool(context.get("resolved"))

    def get_launch_url(self, patient_id: str, encounter_id: str) -> str:
        """EHR-facing launch URL that kicks off the simulated sequence."""
        params = urlencode({"patient": patient_id, "encounter": encounter_id})
        return f"{LAUNCH_BASE}/launch?{params}"

    def dashboard_redirect_url(self, patient_id: str, encounter_id: str,
                               token: str) -> str:
        params = urlencode({
            "patient": patient_id,
            "encounter": encounter_id,
            "token": token,
        })
        return f"{self.dashboard_url}/?{params}"

    def smart_configuration(self) -> dict:
        return {
            "issuer": LAUNCH_BASE,
            "authorization_endpoint": f"{LAUNCH_BASE}/authorize",
            "token_endpoint": f"{LAUNCH_BASE}/token",
            "introspection_endpoint": f"{LAUNCH_BASE}/introspect",
            "capabilities": SMART_CAPABILITIES,
            "grant_types_supported": ["authorization_code"],
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
            "scopes_supported": [
                "launch", "launch/patient", "launch/encounter",
                "patient/*.read", "user/*.read", "openid", "fhirUser",
            ],
            "token_endpoint_auth_methods_supported": ["none"],
            "fhir_base_url": FHIR_BASE_URL,
        }


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
handler = SmartLaunchHandler()
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
_PENDING_STATES: dict = {}


@app.route("/.well-known/smart-configuration", methods=["GET"])
def smart_configuration():
    return jsonify(handler.smart_configuration())


@app.route("/launch", methods=["GET"])
def launch():
    """Real SMART EHR launch: ?iss=<fhir_base>&launch=<token>.

    Discovers the issuer's authorization endpoint and redirects the browser
    there with an authorization-code request (public client, PKCE optional).
    """
    iss = request.args.get("iss")
    launch_token = request.args.get("launch")
    if not iss or not launch_token:
        return jsonify({"error": "invalid_request",
                        "error_description":
                        "iss and launch parameters are required"}), 400

    try:
        cfg = get_smart_config(iss)
    except (FHIRError, requests.RequestException) as e:
        return jsonify({"error": "discovery_failed",
                        "error_description":
                        f"Could not fetch SMART configuration from {iss}: {e}"}), 502

    authorization_endpoint = cfg.get("authorization_endpoint")
    if not authorization_endpoint:
        return jsonify({"error": "discovery_failed",
                        "error_description":
                        "SMART configuration missing authorization_endpoint"}), 502

    state = secrets.token_urlsafe(24)
    _PENDING_STATES[state] = {"iss": iss, "issued": time.time()}

    auth_params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": LAUNCH_SCOPE,
        "launch": launch_token,
        "state": state,
        "aud": iss,
    }
    return redirect(f"{authorization_endpoint}?{urlencode(auth_params)}", code=302)


@app.route("/callback", methods=["GET"])
def callback():
    """Real OAuth2 callback: validate state, exchange code at the token endpoint,
    then redirect into the dashboard with patient/encounter context."""
    if "error" in request.args:
        return jsonify({
            "error": request.args.get("error"),
            "error_description": request.args.get("error_description", ""),
        }), 400

    code = request.args.get("code")
    state = request.args.get("state")
    if not code or not state:
        return jsonify({"error": "invalid_request",
                        "error_description": "Missing code or state"}), 400

    pending = _PENDING_STATES.pop(state, None)
    if not pending:
        return jsonify({"error": "invalid_state",
                        "error_description":
                        "State mismatch - possible CSRF"}), 400

    iss = pending["iss"]

    try:
        cfg = get_smart_config(iss)
        token_endpoint = cfg.get("token_endpoint")
        if not token_endpoint:
            raise FHIRError(0, "missing token_endpoint", context="smart-config")
        token_resp = requests.post(
            token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": CLIENT_ID,
            },
            headers={"Accept": "application/json",
                     "Content-Type": "application/x-www-form-urlencoded"},
            timeout=HTTP_TIMEOUT,
        )
    except (FHIRError, requests.RequestException) as e:
        return jsonify({"error": "token_request_failed",
                        "error_description": str(e)}), 502

    if token_resp.status_code != 200:
        return jsonify({"error": "token_error",
                        "status_code": token_resp.status_code,
                        "body": token_resp.text[:500]}), 400

    token_data = token_resp.json()
    patient_id = token_data.get("patient")
    encounter_id = token_data.get("encounter")

    # Clear single-use launch state.
    session.pop("oauth_state", None)

    if not patient_id:
        return jsonify({"error": "context_error",
                        "error_description":
                        "Token response did not include patient context",
                        "token_keys": list(token_data.keys())}), 400

    params = {"patient": patient_id}
    if encounter_id:
        params["encounter"] = encounter_id
    return redirect(f"{DASHBOARD_URL}/?{urlencode(params)}", code=302)


@app.route("/token", methods=["POST", "GET"])
def token():
    code = request.values.get("code")
    if not code:
        return jsonify({"error": "invalid_request",
                        "error_description": "Missing code"}), 400
    resp = handler.exchange_code(code)
    status = 400 if "error" in resp else 200
    return jsonify(resp), status


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "neuroflow-smart-launch"})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _print_banner() -> None:
    print("=" * 70)
    print("  NEUROFLOW - SMART-on-FHIR Launch")
    print("=" * 70)
    client = FHIRClient(FHIR_BASE_URL)
    try:
        cap = client.capability()
        print(f"  FHIR server   : {cap.get('fhirVersion', '?')} @ {FHIR_BASE_URL}")
    except FHIRError as e:
        print(f"  [WARN] FHIR server not reachable: {e}")
        print("  Start it with: docker start neuroflow-fhir")

    print()
    print(f"  SMART config  : {LAUNCH_BASE}/.well-known/smart-configuration")
    print(f"  Capabilities  : {', '.join(SMART_CAPABILITIES)}")
    print(f"  Dashboard     : {DASHBOARD_URL}")
    print(f"  client_id     : {CLIENT_ID} (public)")
    print(f"  redirect_uri  : {REDIRECT_URI}")
    print()
    print("  EHR launch endpoint (real SMART flow):")
    print(f"    {LAUNCH_BASE}/launch?iss=<fhir_base>&launch=<token>")
    print()
    print("  Test with the SMART App Launcher:")
    print("    https://launch.smarthealthit.org")
    print(f"    App Launch URL: {REDIRECT_URI.replace('/callback', '')}/launch")
    print(f"    Redirect URL  : {REDIRECT_URI}")
    print()
    if REDIRECT_URI.startswith("http://localhost"):
        print("  [NOTE] For an external sandbox, set NEUROFLOW_REDIRECT_URI to an")
        print("         ngrok URL ending in /callback before launching.")
    print()
    print(f"  Launch service starting on {LAUNCH_BASE}")
    print("=" * 70)


def main() -> None:
    _print_banner()
    app.run(host="0.0.0.0", port=LAUNCH_PORT, debug=False)


if __name__ == "__main__":
    main()