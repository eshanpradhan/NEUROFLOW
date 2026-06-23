#!/usr/bin/env python3
"""
NEUROFLOW - Clinical Dashboard
==============================
Plotly Dash dashboard (http://localhost:8050) rendering encounter-scoped ICU
delirium risk from a HAPI FHIR R4 server: risk trajectory with conformal band,
ABCDEF bundle compliance, and a patient summary sidebar. Reads historical
RiskAssessments from FHIR and can trigger a fresh inference cycle.

Run:
    source /Users/eshanpradhan/Desktop/NEUROFLOW/.venv/bin/activate
    pip install dash plotly --break-system-packages
    python src/fhir/dashboard.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhir.fhir_client import (  # noqa: E402
    FHIRClient, FHIRError, build_fhir_reference, ABCDEF_ELEMENTS,
    EXT_CONFORMAL, EXT_CONFIDENCE_TIER,
)
from fhir.inference_loop import NeuroflowInference, run_once  # noqa: E402


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
BASE_URL = "http://localhost:8080/fhir"
THRESHOLD = 0.3
WINDOW_HOURS = 63

BG = "#0a1628"
PANEL = "#0f2138"
PANEL_LIGHT = "#16314f"
TEXT = "#e6edf5"
MUTED = "#8aa0b8"
ACCENT = "#00d4ff"
GREEN = "#2ecc71"
YELLOW = "#f1c40f"
RED = "#e74c3c"
GRID = "rgba(138,160,184,0.15)"

FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"


def risk_color(p: float) -> str:
    if p < 0.3:
        return GREEN
    if p < 0.6:
        return YELLOW
    return RED


# --------------------------------------------------------------------------- #
# Shared engine (loaded once)
# --------------------------------------------------------------------------- #
_CLIENT = FHIRClient(BASE_URL)
try:
    _INFERENCE = NeuroflowInference()
    _INFERENCE_ERR = None
except Exception as e:  # noqa: BLE001
    _INFERENCE = None
    _INFERENCE_ERR = str(e)


# --------------------------------------------------------------------------- #
# FHIR helpers
# --------------------------------------------------------------------------- #
def _parse_dt(s: str):
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


def list_active_encounters():
    options = []
    try:
        encs = _CLIENT.get_active_encounters()
    except FHIRError:
        return options
    for enc in encs:
        enc_id = enc.get("id", "")
        subj = (enc.get("subject", {}) or {}).get("reference", "")
        pat_id = subj.split("/", 1)[1] if "/" in subj else ""
        if not enc_id or not pat_id:
            continue
        label = f"Patient {pat_id} - Encounter {enc_id}"
        options.append({"label": label, "value": f"{pat_id}|{enc_id}"})
    return options


def _extract_ra_point(ra: dict):
    """Return (datetime, prob, low, high, tier, ra_id) from a RiskAssessment."""
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
    t = _parse_dt(ra.get("occurrenceDateTime", ""))
    return {
        "time": t, "prob": float(prob),
        "low": float(low) if low is not None else float(prob),
        "high": float(high) if high is not None else float(prob),
        "tier": tier or "n/a", "id": ra.get("id", ""),
    }


def historical_risk(patient_id, encounter_id):
    params = {
        "patient": build_fhir_reference("Patient", patient_id),
        "encounter": build_fhir_reference("Encounter", encounter_id),
        "_sort": "date", "_count": "100",
    }
    try:
        ras = _CLIENT._search_all("RiskAssessment", params)
    except FHIRError:
        return []
    points = [p for p in (_extract_ra_point(r) for r in ras) if p]
    points = [p for p in points if p["time"] is not None]
    points.sort(key=lambda p: p["time"])
    return points


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(color=MUTED, size=16),
                       x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      height=420, margin=dict(l=40, r=20, t=60, b=40),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def trajectory_figure(points, current) -> go.Figure:
    fig = go.Figure()

    # Risk zone shading.
    zones = [(0.0, 0.3, "rgba(46,204,113,0.10)"),
             (0.3, 0.6, "rgba(241,196,15,0.10)"),
             (0.6, 1.0, "rgba(231,76,60,0.12)")]
    for lo, hi, color in zones:
        fig.add_hrect(y0=lo, y1=hi, line_width=0, fillcolor=color, layer="below")

    # Map historical points onto the 0-63h axis (oldest -> hour 0).
    n = len(points)
    if n >= 1:
        if n == 1:
            xs = [WINDOW_HOURS]
        else:
            span = WINDOW_HOURS
            xs = [span * i / (n - 1) for i in range(n)]
        probs = [p["prob"] for p in points]
        lows = [p["low"] for p in points]
        highs = [p["high"] for p in points]

        # Conformal band (filled).
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1], y=highs + lows[::-1],
            fill="toself", fillcolor="rgba(0,212,255,0.18)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            name="90% conformal interval"))

        fig.add_trace(go.Scatter(
            x=xs, y=probs, mode="lines+markers",
            line=dict(color=ACCENT, width=3),
            marker=dict(size=7, color=[risk_color(p) for p in probs],
                        line=dict(color="#04101f", width=1)),
            name="Delirium risk",
            hovertemplate="Hour %{x:.0f}<br>Risk %{y:.3f}<extra></extra>"))

        # Emphasize current point.
        if current is not None:
            fig.add_trace(go.Scatter(
                x=[xs[-1]], y=[current["prob"]], mode="markers",
                marker=dict(size=15, color=risk_color(current["prob"]),
                            line=dict(color="white", width=2)),
                name="Current",
                hovertemplate="Current %{y:.3f}<extra></extra>"))

    # Clinical action threshold.
    fig.add_hline(y=THRESHOLD, line_dash="dash", line_color=MUTED,
                  annotation_text="Action threshold 0.50",
                  annotation_font_color=MUTED,
                  annotation_position="top left")

    title = "Delirium Risk Trajectory"
    if current is not None:
        title = (f"Delirium Risk Trajectory  -  current "
                 f"{current['prob']:.2f} ({current['tier']} confidence)")

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=18)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT, family=FONT),
        height=440, margin=dict(l=55, r=25, t=60, b=50),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11),
                    orientation="h", y=-0.18),
        xaxis=dict(title="Hours (0-63h window)", range=[0, WINDOW_HOURS],
                   gridcolor=GRID, zeroline=False, color=MUTED),
        yaxis=dict(title="Risk probability", range=[0, 1],
                   gridcolor=GRID, zeroline=False, color=MUTED),
    )
    return fig


# --------------------------------------------------------------------------- #
# Component builders
# --------------------------------------------------------------------------- #
ABCDEF_TITLES = {k: t for (k, t, _a, _l) in ABCDEF_ELEMENTS}
ABCDEF_ACTIONS = {k: a for (k, _t, a, _l) in ABCDEF_ELEMENTS}


def abcdef_panel(compliance: dict):
    boxes = []
    done = 0
    for key, _title, _a, _l in ABCDEF_ELEMENTS:
        ok = bool(compliance.get(key)) if compliance else False
        done += int(ok)
        color = GREEN if ok else RED
        status = "DOCUMENTED" if ok else "NOT DOCUMENTED"
        boxes.append(html.Div([
            html.Div(key, style={"fontSize": "26px", "fontWeight": "700",
                                 "color": color}),
            html.Div(ABCDEF_TITLES[key], style={"fontSize": "11px",
                                                 "color": TEXT,
                                                 "fontWeight": "600",
                                                 "marginTop": "4px"}),
            html.Div(ABCDEF_ACTIONS[key], style={"fontSize": "10px",
                                                 "color": MUTED,
                                                 "marginTop": "4px",
                                                 "lineHeight": "1.3"}),
            html.Div(status, style={"fontSize": "10px", "color": color,
                                    "fontWeight": "700", "marginTop": "6px"}),
        ], style={
            "flex": "1", "minWidth": "150px", "backgroundColor": PANEL_LIGHT,
            "border": f"1px solid {color}", "borderRadius": "8px",
            "padding": "12px", "margin": "5px",
        }))

    overall = html.Div(
        f"ABCDEF Bundle Compliance: {done}/6 elements documented (last 12h)",
        style={"color": ACCENT, "fontSize": "15px", "fontWeight": "600",
               "marginBottom": "8px"})
    return html.Div([overall, html.Div(boxes, style={
        "display": "flex", "flexWrap": "wrap"})])


def stat_row(label, value, value_color=TEXT, big=False):
    return html.Div([
        html.Div(label, style={"color": MUTED, "fontSize": "11px",
                               "textTransform": "uppercase",
                               "letterSpacing": "0.5px"}),
        html.Div(value, style={"color": value_color,
                               "fontSize": "30px" if big else "15px",
                               "fontWeight": "700" if big else "500",
                               "marginTop": "2px"}),
    ], style={"marginBottom": "14px"})


def sidebar_panel(current):
    if current is None:
        return html.Div("No prediction available.",
                        style={"color": MUTED})
    p = current["prob"]
    tier = current["tier"]
    tier_color = {"high": GREEN, "moderate": YELLOW,
                  "low": RED}.get(tier, MUTED)
    updated = current["time"].strftime("%Y-%m-%d %H:%M UTC") \
        if current.get("time") else "n/a"
    return html.Div([
        stat_row("Current Delirium Risk", f"{p:.2f}",
                 value_color=risk_color(p), big=True),
        html.Div(tier.upper() + " CONFIDENCE", style={
            "display": "inline-block", "backgroundColor": tier_color,
            "color": "#04101f", "fontWeight": "700", "fontSize": "11px",
            "padding": "3px 10px", "borderRadius": "12px",
            "marginBottom": "16px"}),
        stat_row("90% Conformal Interval",
                 f"[{current['low']:.2f}, {current['high']:.2f}]"),
        stat_row("Interval Width", f"{current['high'] - current['low']:.2f}"),
        stat_row("Hours of Data", str(current.get("hours_of_data", "n/a"))),
        stat_row("Observations Used", str(current.get("n_observations", "n/a"))),
        stat_row("RiskAssessment ID", current.get("id", "n/a"),
                 value_color=ACCENT),
        stat_row("Last Updated", updated),
    ])


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = Dash(__name__, title="NEUROFLOW ICU Delirium Monitor")
server = app.server

CARD = {"backgroundColor": PANEL, "borderRadius": "10px", "padding": "18px",
        "border": f"1px solid {PANEL_LIGHT}"}

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Span("NEURO", style={"color": "white", "fontWeight": "800"}),
            html.Span("FLOW", style={"color": ACCENT, "fontWeight": "800"}),
        ], style={"fontSize": "26px", "letterSpacing": "1px"}),
        html.Div("FHIR-native ICU Delirium Prediction",
                 style={"color": MUTED, "fontSize": "13px"}),
    ], style={"marginBottom": "16px"}),

    html.Div([
        html.Div([
            html.Label("Active Encounter", style={"color": MUTED,
                                                  "fontSize": "12px"}),
            dcc.Dropdown(id="encounter-dropdown",
                         options=list_active_encounters(),
                         placeholder="Select an active ICU encounter...",
                         style={"color": "#04101f"}),
        ], style={"flex": "1", "marginRight": "12px"}),
        html.Button("REFRESH / RUN INFERENCE", id="refresh-btn", n_clicks=0,
                    style={"backgroundColor": ACCENT, "color": "#04101f",
                           "border": "none", "borderRadius": "8px",
                           "padding": "10px 18px", "fontWeight": "700",
                           "cursor": "pointer", "alignSelf": "flex-end"}),
    ], style={"display": "flex", "alignItems": "flex-end",
              "marginBottom": "8px"}),

    html.Div(id="status-line",
             style={"color": MUTED, "fontSize": "12px", "marginBottom": "14px",
                    "minHeight": "16px"}),

    html.Div([
        html.Div([
            html.Div(dcc.Graph(id="trajectory-graph",
                               figure=empty_fig("Select an encounter to begin.")),
                     style=CARD),
            html.Div(id="abcdef-panel", style={**CARD, "marginTop": "16px"}),
        ], style={"flex": "3", "marginRight": "16px"}),

        html.Div(id="sidebar", style={**CARD, "flex": "1", "minWidth": "240px"}),
    ], style={"display": "flex", "alignItems": "flex-start"}),

    dcc.Store(id="current-store"),
], style={"backgroundColor": BG, "minHeight": "100vh", "padding": "22px 28px",
          "fontFamily": FONT})


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@app.callback(
    Output("trajectory-graph", "figure"),
    Output("abcdef-panel", "children"),
    Output("sidebar", "children"),
    Output("status-line", "children"),
    Input("encounter-dropdown", "value"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_dashboard(selection, n_clicks):
    if not selection:
        return (empty_fig("Select an encounter to begin."),
                html.Div("No encounter selected.", style={"color": MUTED}),
                html.Div("No encounter selected.", style={"color": MUTED}),
                "")

    pat_id, enc_id = selection.split("|", 1)

    if _INFERENCE is None:
        status = f"[ERROR] Model not loaded: {_INFERENCE_ERR}"
        return (empty_fig("Model unavailable."),
                html.Div(status, style={"color": RED}),
                html.Div(status, style={"color": RED}), status)

    # Run a fresh inference cycle (writes a new RiskAssessment).
    current = None
    status = ""
    try:
        result = run_once(pat_id, enc_id, _CLIENT, _INFERENCE, threshold=THRESHOLD)
        current = {
            "prob": result["probability"],
            "low": result["interval_low"],
            "high": result["interval_high"],
            "tier": result["confidence_tier"],
            "id": result.get("risk_assessment_id", ""),
            "time": datetime.now(timezone.utc),
            "hours_of_data": result.get("hours_of_data"),
            "n_observations": result.get("n_observations"),
        }
        compliance = result.get("abcdef_compliance") or {}
        status = (f"Inference complete at "
                  f"{current['time'].strftime('%H:%M:%S UTC')}  -  "
                  f"RiskAssessment {current['id']}  -  "
                  f"{result.get('n_observations', 0)} observations, "
                  f"{result.get('hours_of_data', 0)}h of data.")
    except (FHIRError, Exception) as e:  # noqa: BLE001
        compliance = {}
        status = f"[ERROR] Inference failed: {type(e).__name__}: {e}"

    # Historical trajectory from stored RiskAssessments.
    points = historical_risk(pat_id, enc_id)
    if current is not None:
        points.append({"time": current["time"], "prob": current["prob"],
                       "low": current["low"], "high": current["high"],
                       "tier": current["tier"], "id": current["id"]})
        # carry summary fields to the sidebar via `current`
    elif points:
        current = points[-1]

    fig = trajectory_figure(points, current) if points \
        else empty_fig("No RiskAssessments available yet.")
    return fig, abcdef_panel(compliance), sidebar_panel(current), status


@app.callback(
    Output("encounter-dropdown", "options"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_encounters(_n):
    return list_active_encounters()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 70)
    print("  NEUROFLOW Clinical Dashboard")
    print("=" * 70)
    try:
        cap = _CLIENT.capability()
        print(f"  FHIR server   : {cap.get('fhirVersion', '?')} @ {BASE_URL}")
    except FHIRError as e:
        print(f"  [WARN] FHIR server not reachable: {e}")
        print("  Start it with: docker start neuroflow-fhir")
    if _INFERENCE is None:
        print(f"  [WARN] model not loaded: {_INFERENCE_ERR}")
    else:
        print(f"  model         : loaded on {_INFERENCE.device}")
    print(f"  active encounters discovered: {len(list_active_encounters())}")
    print()
    print("  Dashboard -> http://localhost:8050")
    print("=" * 70)
    app.run(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    main()