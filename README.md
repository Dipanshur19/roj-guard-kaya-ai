# ROJ Guard — Round 2 Prototype

**Material ROJ Risk & Escalation Agent** for the Kaya AI IIT India Hackathon 2026, Track 2: Supply Chain.

ROJ Guard predicts whether active material lines will miss their Required-On-Job (ROJ) dates, explains the live risk drivers, and prepares mitigation actions with a mandatory human approval gate.

## Judge-facing workflow

**Upload/paste vendor update → AI extraction review → entity linking/project graph → feature refresh → ROJ risk change → mitigation draft → human approval → execution receipt.**

The redesigned Streamlit UI contains:

1. **Overview** — portfolio KPIs and ranked material risk.
2. **Incoming Intelligence** — paste vendor emails or upload PDFs; analyze, review and apply without Swagger.
3. **Material Intelligence** — forecast, risk drivers, vendor reliability, schedule context, activity timeline, mitigation agents.
4. **Project Graph** — visual Project → Material → Vendor → PO → Shipment → ROJ context.
5. **Live Demo** — one-button deterministic Low → High risk scenario with automatic mitigation drafting.
6. **Approval Center** — edit/approve/reject actions and view execution receipts.

## Architecture

- **Layer 1 — Data Ingestion:** FastAPI + Gemini extraction, manual/API inputs.
- **Layer 2 — Project Graph & Feature Store:** entity linking, SQLite/Postgres metadata, optional Neo4j sync, vendor/lead-time features.
- **Layer 3 — Prediction & Risk:** XGBoost lead-time regression + risk classifier + live evidence/rules.
- **Layer 4 — Agentic Action:** escalation, resequencing and alternate-supplier agents with human approval and safe execution.

## Local setup (Windows / VS Code)

Use **Python 3.11**.

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your Gemini key in `.env`:

```env
GEMINI_API_KEY=your_key
BACKEND_URL=http://127.0.0.1:8000
APP_TIMEZONE=Asia/Kolkata
DEMO_EMAIL_MODE=outbox
```

The ZIP already contains a clean deterministic database and trained model artifacts. To regenerate them:

```powershell
python seed_data_layer1.py
```

### Start manually
Terminal 1:

```powershell
uvicorn main_layer1:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2:

```powershell
streamlit run dashboard_layer3.py
```

Open `http://localhost:8501`.

### Or start both on Windows

```bat
run_local.bat
```

## Live Demo

Open **Live Demo** in the sidebar and click **Run Live Risk Scenario**. It:

1. selects a real Low-risk active material,
2. injects a 14-day vendor production/dispatch delay,
3. recomputes the operational feature row and ROJ risk,
4. moves the material to High risk (when supported by the signal),
5. drafts a vendor escalation automatically,
6. waits for human approval before executing,
7. stores an audit receipt.

Use **Reset Demo** to restore the baseline.

## Public deployment

The repository includes `Dockerfile`, `start.sh`, and `render.yaml`. See `DEPLOY_RENDER.md`.

The deployed container runs FastAPI internally and exposes only the Streamlit product UI to judges.

## Safety / secrets

- `.env` is intentionally excluded.
- `venv/` is intentionally excluded.
- Approved demo emails default to local `demo_outbox/`; no real vendor is contacted.
- Neo4j is optional; the prototype remains functional when it is unavailable.
