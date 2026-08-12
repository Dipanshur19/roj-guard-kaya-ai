# P0 implementation summary

1. Dynamic signals: vendor communications and shipping updates are consumed by Layer-3 inference and trigger immediate risk recomputation after ingestion.
2. ROJ/current-state features: days until ROJ, current transit, ETA-vs-ROJ, vendor delay days, submittal approval state, promised-ship slippage, critical path and float are model features.
3. Active-only dashboard: delivered materials are removed from live risk scoring; they remain historical training outcomes.
4. Realistic synthetic data: 360 historical completed deliveries + 30 active current materials with balanced operational scenarios.
5. Neo4j: fixed PO Cypher scope (`WITH po, m`) and vendor-delay metadata is synced into communication nodes.
6. Alternate suppliers: same-material-class capability history is mandatory; candidates are ranked by ROJ feasibility, reliability and historical lead time.
7. Execution: approved actions create actual sandboxed execution artifacts and ActionExecution audit receipts; SMTP test execution is optional.
8. Submission security: distribution excludes `.env`, `venv/`, caches and generated execution artifacts.
