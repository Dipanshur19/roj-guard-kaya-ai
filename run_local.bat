@echo off
setlocal
if not exist venv\Scripts\python.exe (
  echo Create the environment first: py -3.11 -m venv venv
  exit /b 1
)
start "ROJ Guard API" cmd /k "call venv\Scripts\activate.bat && uvicorn main_layer1:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 2 /nobreak >nul
start "ROJ Guard UI" cmd /k "call venv\Scripts\activate.bat && set BACKEND_URL=http://127.0.0.1:8000 && streamlit run dashboard_layer3.py"
endlocal
