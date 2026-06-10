#!/usr/bin/env python3
"""
NEUROFLOW - Environment & Setup Verification
============================================
Confirms that the local development environment is correctly configured before
running any stage of the NEUROFLOW pipeline.

This script verifies:
  1. Python interpreter and virtual environment
  2. Required Python packages and versions
  3. PyTorch compute backend (Apple Silicon MPS)
  4. Project directory structure
  5. Presence of MIMIC-IV v3.1 data files  (existence only - contents NEVER read)
  6. Presence of eICU v2.0 data files       (existence only - contents NEVER read)
  7. HAPI FHIR JPA Server reachability       (localhost:8080)
  8. Ollama local LLM reachability           (localhost:11434)

DATA COMPLIANCE
---------------
This script inspects only filesystem METADATA (whether a path exists and its byte
size). It NEVER opens, reads, decompresses, or parses any data file. PhysioNet
Credentialed Health Data Use Agreement 1.5.0 compliance is preserved at all times.

Run:
    source /Users/eshanpradhan/Desktop/NEUROFLOW/.venv/bin/activate
    python scripts/verify_setup.py
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path

# requests is required for the FHIR / Ollama checks. Import defensively so the
# script can still produce a useful report if it is somehow missing.
try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class C:
    """ANSI color codes (no-ops when output is not a TTY)."""
    RESET = "\033[0m" if _USE_COLOR else ""
    BOLD = "\033[1m" if _USE_COLOR else ""
    DIM = "\033[2m" if _USE_COLOR else ""
    GREEN = "\033[32m" if _USE_COLOR else ""
    YELLOW = "\033[33m" if _USE_COLOR else ""
    RED = "\033[31m" if _USE_COLOR else ""
    CYAN = "\033[36m" if _USE_COLOR else ""


# Running tally of result severities (INFO is intentionally not tracked).
_tally = {"OK": 0, "WARN": 0, "FAIL": 0}


def line(level: str, label: str, detail: str = "") -> None:
    """Print one aligned status line and update the tally."""
    color = {
        "OK": C.GREEN,
        "WARN": C.YELLOW,
        "FAIL": C.RED,
        "INFO": C.CYAN,
    }.get(level, "")
    tag = f"{color}[{level:^4}]{C.RESET}"
    msg = f"  {tag}  {label}"
    if detail:
        msg += f"  {C.DIM}{detail}{C.RESET}"
    print(msg)
    if level in _tally:
        _tally[level] += 1


def section(title: str) -> None:
    print()
    print(f"{C.BOLD}{C.CYAN}{title}{C.RESET}")
    print(f"{C.DIM}{'-' * len(title)}{C.RESET}")


def banner() -> None:
    bar = "=" * 64
    print(f"{C.BOLD}{C.CYAN}{bar}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  NEUROFLOW - Environment & Setup Verification{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{bar}{C.RESET}")


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} PB"


# --------------------------------------------------------------------------- #
# 1. Python interpreter & virtual environment
# --------------------------------------------------------------------------- #
def check_python() -> None:
    section("1. Python interpreter & virtual environment")

    py_version = platform.python_version()
    expected = "3.9.6"
    if py_version == expected:
        line("OK", f"Python {py_version}")
    elif py_version.startswith("3.9"):
        line("WARN", f"Python {py_version}",
             f"expected {expected}; same 3.9 line, likely fine")
    else:
        line("WARN", f"Python {py_version}",
             f"expected {expected}; behavior may differ")

    line("INFO", f"Platform: {platform.platform()}")

    machine = platform.machine()
    if machine == "arm64":
        line("OK", f"Architecture: {machine} (Apple Silicon)")
    else:
        line("WARN", f"Architecture: {machine}", "expected arm64 (Apple Silicon)")

    venv = os.environ.get("VIRTUAL_ENV")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if venv:
        line("OK", "Virtual environment active", venv)
        if "NEUROFLOW" not in venv:
            line("WARN", "Active venv is not the NEUROFLOW .venv", venv)
    elif in_venv:
        line("WARN", "Running inside a venv but VIRTUAL_ENV is unset", sys.prefix)
    else:
        line("WARN", "No virtual environment detected",
             "activate: source .venv/bin/activate")

    line("INFO", f"Interpreter: {sys.executable}")


# --------------------------------------------------------------------------- #
# 2. Required Python packages
# --------------------------------------------------------------------------- #
# (distribution_name, import_name, expected_version_or_None, critical)
PACKAGES = [
    ("torch",        "torch",       "2.8.0",   True),
    ("torchvision",  "torchvision", "0.23.0",  False),
    ("torchaudio",   "torchaudio",  "2.8.0",   False),
    ("numpy",        "numpy",       "2.0.2",   True),
    ("pandas",       "pandas",      "2.3.3",   True),
    ("scikit-learn", "sklearn",     None,      True),
    ("plotly",       "plotly",      "6.8.0",   True),
    ("fastapi",      "fastapi",     "0.128.8", True),
    ("uvicorn",      "uvicorn",     None,      True),
    ("requests",     "requests",    None,      True),
    ("pyarrow",      "pyarrow",     "21.0.0",  True),
    ("tqdm",         "tqdm",        None,      True),
    ("jupyter",      "jupyter",     None,      False),
    ("ipykernel",    "ipykernel",   None,      False),
]


def check_packages() -> None:
    section("2. Required Python packages")
    for dist_name, import_name, expected, critical in PACKAGES:
        # Resolve installed version from distribution metadata first.
        try:
            installed = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            installed = None

        # Confirm the package actually imports (catches broken installs).
        import_err = ""
        try:
            module = importlib.import_module(import_name)
            if installed is None:
                installed = getattr(module, "__version__", "unknown")
        except Exception as e:  # ImportError or any import-time failure
            first = str(e).splitlines()[0] if str(e) else ""
            import_err = first or e.__class__.__name__

        if import_err or installed is None:
            level = "FAIL" if critical else "WARN"
            detail = f"cannot import: {import_err}" if import_err else "not installed"
            line(level, dist_name, detail)
            continue

        if expected is None or installed == expected:
            line("OK", f"{dist_name} {installed}")
        else:
            line("WARN", f"{dist_name} {installed}", f"expected {expected}")


# --------------------------------------------------------------------------- #
# 3. PyTorch compute backend (MPS)
# --------------------------------------------------------------------------- #
def check_torch_backend() -> None:
    section("3. PyTorch compute backend")
    try:
        import torch
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else e.__class__.__name__
        line("FAIL", "PyTorch not importable - cannot check backend", first)
        return

    line("INFO", f"torch.__version__ = {torch.__version__}")

    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend) and torch.backends.mps.is_built()
    mps_avail = bool(mps_backend) and torch.backends.mps.is_available()

    if mps_avail:
        line("OK", "MPS backend available", "device = mps")
        try:
            x = torch.ones(8, device="mps")
            result = (x * 2).sum().item()
            if result == 16.0:
                line("OK", "MPS tensor op smoke test passed")
            else:
                line("WARN", "MPS smoke test returned unexpected value", str(result))
        except Exception as e:
            first = str(e).splitlines()[0] if str(e) else e.__class__.__name__
            line("WARN", "MPS available but tensor op failed", first)
    else:
        detail = "built but not available" if mps_built else "not built into this torch"
        line("WARN", "MPS unavailable - pipeline will fall back to CPU", detail)

    if torch.cuda.is_available():
        line("INFO", "CUDA also available on this machine")


# --------------------------------------------------------------------------- #
# 4. Project directory structure
# --------------------------------------------------------------------------- #
EXPECTED_DIRS = [
    "data/raw/mimiciv/3.1/hosp",
    "data/raw/mimiciv/3.1/icu",
    "data/raw/eicu/2.0",
    "fhir-server",
    "notebooks",
    "scripts",
    "src/features",
    "src/models",
    "src/fhir",
]

# Written to by the pipeline - create if missing.
OUTPUT_DIRS = [
    "data/processed",
    "models",
]


def check_directories(root: Path) -> None:
    section("4. Project directory structure")
    for rel in EXPECTED_DIRS:
        p = root / rel
        if p.is_dir():
            line("OK", rel + "/")
        else:
            line("WARN", rel + "/", "missing")

    for rel in OUTPUT_DIRS:
        p = root / rel
        if p.is_dir():
            line("OK", rel + "/")
        else:
            try:
                p.mkdir(parents=True, exist_ok=True)
                line("OK", rel + "/", "created (output directory)")
            except Exception as e:
                line("FAIL", rel + "/", f"could not create: {e}")


# --------------------------------------------------------------------------- #
# 5 & 6. Data file presence (METADATA ONLY - files are never opened)
# --------------------------------------------------------------------------- #
MIMIC_FILES = [
    "data/raw/mimiciv/3.1/hosp/admissions.csv.gz",
    "data/raw/mimiciv/3.1/hosp/diagnoses_icd.csv.gz",
    "data/raw/mimiciv/3.1/hosp/emar.csv.gz",
    "data/raw/mimiciv/3.1/hosp/emar_detail.csv.gz",
    "data/raw/mimiciv/3.1/hosp/patients.csv.gz",
    "data/raw/mimiciv/3.1/icu/chartevents.csv.gz",
    "data/raw/mimiciv/3.1/icu/d_items.csv.gz",
    "data/raw/mimiciv/3.1/icu/icustays.csv.gz",
    "data/raw/mimiciv/3.1/icu/inputevents.csv.gz",
]

EICU_FILES = [
    "data/raw/eicu/2.0/diagnosis.csv.gz",
    "data/raw/eicu/2.0/infusionDrug.csv.gz",
    "data/raw/eicu/2.0/medication.csv.gz",
    "data/raw/eicu/2.0/nurseCharting.csv.gz",
    "data/raw/eicu/2.0/patient.csv.gz",
    "data/raw/eicu/2.0/vitalPeriodic.csv.gz",
]


def _check_file_list(root: Path, files: list[str], dataset_label: str):
    present, missing, empty = 0, 0, 0
    for rel in files:
        p = root / rel
        name = Path(rel).name
        if not p.exists():
            line("WARN", name, f"missing - required for {dataset_label}")
            missing += 1
            continue
        try:
            # stat() reads filesystem metadata only; the file is NOT opened.
            size = p.stat().st_size
        except OSError as e:
            line("WARN", name, f"cannot stat: {e}")
            missing += 1
            continue
        if size == 0:
            line("WARN", name, "0 bytes - download likely incomplete")
            empty += 1
        else:
            line("OK", name, human_size(size))
            present += 1
    return present, missing, empty


def check_data_files(root: Path) -> None:
    section("5. MIMIC-IV v3.1 data files  (primary training)")
    line("INFO", "Checking existence and byte size only - no file is opened or parsed")
    p, m, e = _check_file_list(root, MIMIC_FILES, "MIMIC-IV training")
    if m == 0 and e == 0:
        line("OK", f"All {len(MIMIC_FILES)} MIMIC-IV files present and non-empty")
    else:
        line("WARN", f"MIMIC-IV: {p} present, {m} missing, {e} empty",
             "training stages require these")

    section("6. eICU v2.0 data files  (external validation)")
    line("INFO", "Checking existence and byte size only - no file is opened or parsed")
    p2, m2, e2 = _check_file_list(root, EICU_FILES, "eICU external validation")
    if m2 == 0 and e2 == 0:
        line("OK", f"All {len(EICU_FILES)} eICU files present and non-empty")
    else:
        line("WARN", f"eICU: {p2} present, {m2} missing, {e2} empty",
             "external validation only; OK to defer until DUA signed")


# --------------------------------------------------------------------------- #
# 7. HAPI FHIR JPA Server
# --------------------------------------------------------------------------- #
FHIR_BASE = "http://localhost:8080/fhir"


def check_fhir_server() -> None:
    section("7. HAPI FHIR JPA Server  (localhost:8080)")
    if requests is None:
        line("FAIL", "requests not installed - cannot reach FHIR server")
        return

    url = f"{FHIR_BASE}/metadata"
    try:
        r = requests.get(url, headers={"Accept": "application/fhir+json"}, timeout=10)
    except requests.exceptions.ConnectionError:
        line("WARN", "FHIR server not reachable at localhost:8080",
             "start it: docker start neuroflow-fhir")
        return
    except requests.exceptions.Timeout:
        line("WARN", "FHIR server timed out", "container may still be starting up")
        return
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else e.__class__.__name__
        line("WARN", "FHIR server check failed", first)
        return

    if r.status_code != 200:
        line("WARN", f"FHIR /metadata returned HTTP {r.status_code}", "expected 200")
        return

    try:
        cap = r.json()
    except ValueError:
        line("WARN", "FHIR /metadata did not return JSON")
        return

    rtype = cap.get("resourceType")
    fhir_version = cap.get("fhirVersion", "unknown")
    software = cap.get("software", {}) or {}
    sw_name = software.get("name", "unknown")
    sw_version = software.get("version", "unknown")

    if rtype == "CapabilityStatement":
        line("OK", "FHIR server reachable - CapabilityStatement received")
    else:
        line("WARN", f"Unexpected resourceType at /metadata: {rtype}")

    if fhir_version == "4.0.1":
        line("OK", f"FHIR version {fhir_version} (R4)")
    else:
        line("WARN", f"FHIR version {fhir_version}", "expected 4.0.1 (R4)")

    line("INFO", f"Software: {sw_name} {sw_version}")


# --------------------------------------------------------------------------- #
# 8. Ollama local LLM
# --------------------------------------------------------------------------- #
OLLAMA_BASE = "http://localhost:11434"


def check_ollama() -> None:
    section("8. Ollama local LLM  (localhost:11434)")
    line("INFO", "Used only for data-specific debugging - keeps restricted data local")
    if requests is None:
        line("WARN", "requests not installed - cannot reach Ollama")
        return

    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    except requests.exceptions.ConnectionError:
        line("WARN", "Ollama not reachable at localhost:11434", "start it: ollama serve")
        return
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else e.__class__.__name__
        line("WARN", "Ollama check failed", first)
        return

    if r.status_code != 200:
        line("WARN", f"Ollama returned HTTP {r.status_code}")
        return

    try:
        models = [m.get("name", "") for m in r.json().get("models", [])]
    except ValueError:
        line("WARN", "Ollama response was not JSON")
        return

    line("OK", "Ollama reachable")
    match = next((name for name in models if name.startswith("llama3.1")), None)
    if match:
        line("OK", "llama3.1 model present", match)
    else:
        listed = ", ".join(m for m in models if m) if models else "none"
        line("WARN", "llama3.1 model not found",
             f"available: {listed}; pull: ollama pull llama3.1")


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summary() -> int:
    print()
    bar = "=" * 64
    print(f"{C.BOLD}{C.CYAN}{bar}{C.RESET}")
    print(f"{C.BOLD}  Summary{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{bar}{C.RESET}")

    ok = _tally["OK"]
    warn = _tally["WARN"]
    fail = _tally["FAIL"]
    print(f"  {C.GREEN}OK:   {ok:>3}{C.RESET}     "
          f"{C.YELLOW}WARN: {warn:>3}{C.RESET}     "
          f"{C.RED}FAIL: {fail:>3}{C.RESET}")
    print()

    if fail > 0:
        print(f"  {C.RED}{C.BOLD}Setup incomplete.{C.RESET} "
              f"Resolve the {fail} FAIL item(s) above before proceeding.")
        return 1
    if warn > 0:
        print(f"  {C.YELLOW}{C.BOLD}Core environment OK with {warn} warning(s).{C.RESET}")
        print(f"  {C.DIM}Warnings are non-blocking. Missing data files or a stopped FHIR{C.RESET}")
        print(f"  {C.DIM}container are expected early on and can be resolved as needed.{C.RESET}")
        return 0
    print(f"  {C.GREEN}{C.BOLD}All checks passed. Environment fully verified.{C.RESET}")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    banner()
    root = Path(__file__).resolve().parent.parent
    print()
    line("INFO", f"Project root: {root}")

    check_python()
    check_packages()
    check_torch_backend()
    check_directories(root)
    check_data_files(root)
    check_fhir_server()
    check_ollama()

    sys.exit(summary())


if __name__ == "__main__":
    main()