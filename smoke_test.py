"""Quick pre-demo sanity check for ROJ Guard.

Run from the repository root:
    python smoke_test.py
"""
from __future__ import annotations

import importlib.util
import os
import py_compile
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = [
    "dashboard_layer3.py", "main_layer1.py", "real_data_lab.py", "api_experience.py",
    "api_layer3.py", "api_layer4.py", "agents_layer4.py", "execution_layer4.py",
]
ARTIFACTS = [
    "roj_guard.db", "lead_time_model_layer3.joblib", "risk_classifier_layer3.joblib",
    "categorical_encoders_layer3.joblib", "requirements.txt", "start.sh", "render.yaml",
]
DEPS = ["fastapi", "uvicorn", "sqlalchemy", "pydantic", "pandas", "xgboost", "sklearn", "streamlit", "requests", "openpyxl"]

failures = []
print(f"Python: {sys.version.split()[0]}")
if sys.version_info[:2] != (3, 11):
    print("WARN: Python 3.11 is the tested runtime.")

for name in CORE:
    path = ROOT / name
    if not path.exists():
        failures.append(f"missing core file: {name}")
        continue
    try:
        py_compile.compile(str(path), doraise=True)
        print(f"PASS compile  {name}")
    except Exception as exc:
        failures.append(f"compile failed {name}: {exc}")

for name in ARTIFACTS:
    if (ROOT / name).exists():
        print(f"PASS artifact {name}")
    else:
        failures.append(f"missing artifact: {name}")

for dep in DEPS:
    if importlib.util.find_spec(dep) is None:
        failures.append(f"missing dependency: {dep}")
    else:
        print(f"PASS dep      {dep}")

try:
    from real_data_lab import template_csv_bytes, procurement_history_template_csv_bytes
    assert len(template_csv_bytes()) > 50
    assert len(procurement_history_template_csv_bytes()) > 50
    print("PASS Real Data Lab templates")
except Exception as exc:
    failures.append(f"Real Data Lab import/template check failed: {exc}")

backend = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
try:
    with urllib.request.urlopen(backend + "/", timeout=1.2) as response:
        if 200 <= response.status < 400:
            print(f"PASS backend   {backend}")
        else:
            print(f"WARN backend returned {response.status}")
except Exception:
    print("INFO backend is not running; start it before the live demo.")

if failures:
    print("\nFAILED CHECKS")
    for failure in failures:
        print(" -", failure)
    raise SystemExit(1)

print("\nROJ Guard smoke check passed.")
