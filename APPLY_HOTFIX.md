# ROJ Guard AI Extraction 500 Hotfix

1. Stop FastAPI and Streamlit (`Ctrl+C` in both terminals).
2. Copy `gemini_extractor_layer1.py` into the project root and replace the existing file.
3. Copy `requirements.txt` into the project root and replace the existing file.
4. Activate the Python 3.11 venv.
5. Run `pip install -r requirements.txt` (adds `pypdf` for local PDF fallback).
6. Restart FastAPI: `uvicorn main_layer1:app --host 127.0.0.1 --port 8000 --reload`
7. Restart Streamlit: `streamlit run dashboard_layer3.py`
8. Re-run Incoming Intelligence.

The extraction layer now:
- Uses Gemini when available.
- Catches invalid key/model/quota/network/API failures instead of returning HTTP 500.
- Falls back to a deterministic local vendor-email parser.
- Extracts vendor, material, delay days, communication type, message date and summary for the hackathon-critical flow.
- Uses optional pypdf text extraction if Gemini PDF analysis fails.
