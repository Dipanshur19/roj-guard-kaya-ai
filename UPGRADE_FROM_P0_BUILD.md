# Upgrade from the P0 build you are currently running

## Recommended method
Use this new folder as the working baseline rather than mixing old/new files.

1. Stop Uvicorn and Streamlit (`Ctrl+C` in both terminals).
2. Extract the new ZIP to a new folder.
3. Copy only your old `.env` into the new folder.
4. Reuse no old `venv`; create Python 3.11 environment or copy nothing except `.env`.
5. Install `requirements.txt`.
6. Start Uvicorn and Streamlit as before, or run `run_local.bat`.

The ZIP already includes a clean `roj_guard.db` and trained `.joblib` artifacts.

## If you must patch the existing folder
Replace/add these files:
- `dashboard_layer3.py`
- `main_layer1.py`
- `api_experience.py` (new)
- `.streamlit/config.toml` (new)
- `Dockerfile` (new)
- `.dockerignore` (new)
- `start.sh` (new)
- `render.yaml` (new)
- `DEPLOY_RENDER.md` (new)
- `run_local.bat` (new)

Your existing DB/models remain compatible, but the packaged DB is the recommended clean baseline.
