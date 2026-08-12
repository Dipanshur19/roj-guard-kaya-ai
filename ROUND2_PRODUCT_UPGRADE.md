# Round 2 Product Upgrade — What changed

## 1. Streamlit-native ingestion
- New **Incoming Intelligence** screen.
- Paste vendor emails / logistics text or upload PDFs.
- AI extraction is preview-only first.
- Confidence and extracted fields are visible.
- Human can edit the extraction before applying it.
- Applying a signal links entities, refreshes features and recomputes risk.
- Medium/High results proactively draft an escalation for the Approval Center.

Backend additions:
- `POST /api/extract/{doc_type}` — extraction preview only.
- `POST /api/apply-extraction/{doc_type}` — human-confirmed application with before/after risk snapshots.

## 2. Project Graph page
- Added material-centric Project → Material → Vendor → PO → Shipment → ROJ visualization.
- Added relationship table and Kaya/Amber integration narrative.
- Works from the relational graph projection even when Neo4j is not running; Neo4j sync remains optional.

## 3. Professional UI redesign
- New navy/orange visual system aligned to the Round-1 deck.
- Portfolio KPI cards.
- Cleaner material names: internal synthetic LOW/HIGH/HIST labels are hidden from judges.
- Explainable risk-driver cards.
- Material state panel and activity timeline.
- Separate Approval Center with execution audit trail.

## 4. Before → Signal → After panel
- Both real ingestion and Demo Mode show risk transition visually.

## 5. One-button Live Demo
- Deterministic Low-risk construction material selected automatically.
- Injects a 14-day vendor delay.
- Updates risk and creates a pending vendor escalation automatically.
- Human approves/edits and executes from the same screen.
- Reset endpoint restores the baseline.

Backend additions in `api_experience.py`:
- project/material catalog
- material context + risk drivers + activity
- graph projection
- demo scenario/run/reset

## 6. Deployment-ready
- `Dockerfile`
- `start.sh`
- `render.yaml`
- `.streamlit/config.toml`
- `DEPLOY_RENDER.md`

A Render Docker service can host both FastAPI (internal) and Streamlit (public) in one container.
