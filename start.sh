#!/usr/bin/env bash
set -euo pipefail

if [[ "${STARTUP_SEED:-false}" == "true" ]]; then
  echo "Resetting deterministic demo dataset..."
  python seed_data_layer1.py
fi

uvicorn main_layer1:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
trap 'kill ${BACKEND_PID} 2>/dev/null || true' EXIT

# Wait briefly for the internal API before starting the public UI.
python - <<'PY'
import time, urllib.request
for _ in range(40):
    try:
        urllib.request.urlopen('http://127.0.0.1:8000/', timeout=1)
        break
    except Exception:
        time.sleep(.25)
PY

exec streamlit run dashboard_layer3.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-8501}" \
  --server.headless true
