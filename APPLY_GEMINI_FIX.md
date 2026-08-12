# ROJ Guard — Gemini API migration fix

## Why the warning appeared
The project was still configured for `gemini-2.0-flash`, which Google now lists as shut down. The previous code also used the legacy `google-generativeai` SDK.

## What this patch changes
- Migrates Python calls to the current `google-genai` SDK (`from google import genai`).
- Defaults to stable `gemini-3.6-flash`.
- Ignores stale `.env` values that still specify shut-down Gemini 2.0 models.
- Tries a short supported-model fallback chain only for model-not-found errors.
- Keeps the deterministic local parser as an emergency fallback, but no longer exposes raw provider stack/error text in the judge UI.
- Uses the same current Gemini client for Layer 4 agent drafting.

## Apply to the project already open in VS Code
1. Stop FastAPI and Streamlit with `Ctrl+C`.
2. Copy/replace these files in the project root:
   - `gemini_extractor_layer1.py`
   - `agents_layer4.py`
   - `requirements.txt`
   - `.env.example` (optional but recommended)
3. Activate your Python 3.11 venv.
4. Remove the old SDK and install the current one:

```powershell
pip uninstall -y google-generativeai
pip install -U "google-genai>=2.0.0,<3.0.0"
```

Or simply run:

```powershell
pip install -r requirements.txt
```

5. Open your existing `.env` and set:

```env
GEMINI_MODEL=gemini-3.6-flash
```

The code also auto-ignores an old `GEMINI_MODEL=gemini-2.0-flash`, so this edit is recommended but not strictly required.

6. Restart FastAPI:

```powershell
uvicorn main_layer1:app --host 127.0.0.1 --port 8000 --reload
```

7. Restart Streamlit in the second terminal:

```powershell
streamlit run dashboard_layer3.py
```

8. Test **Incoming Intelligence → Analyze with AI** again. The note should say something like:

`Extracted with gemini-3.6-flash.`

instead of showing the old 404/decommissioned-model warning.
