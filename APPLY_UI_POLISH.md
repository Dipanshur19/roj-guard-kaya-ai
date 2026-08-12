# Apply the Overview UI Polish

Use this patch if the redesigned Round-2 project is already running on your computer.

1. Stop FastAPI and Streamlit with `Ctrl+C` in both terminals.
2. Copy `dashboard_layer3.py` and `api_layer3.py` from this patch into the project root.
3. Choose **Replace** when Windows asks whether to overwrite the files.
4. Restart FastAPI:

```powershell
uvicorn main_layer1:app --host 127.0.0.1 --port 8000 --reload
```

5. Restart Streamlit in the second terminal:

```powershell
streamlit run dashboard_layer3.py
```

No database reseed, model retraining, or dependency reinstall is required.
