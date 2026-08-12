# ROJ Guard — Overview UI Polish

This update adds the final overview-page polish requested after the Round-2 redesign.

## Changes

1. **PO / SKU identity in the risk table**
   - The risk API now includes the latest PO number for every active material.
   - The Overview table adds a compact `PO / SKU` column so similarly named procurement lines are visibly distinct.
   - Synthetic LOW/MEDIUM/HIGH markers continue to be hidden in the displayed SKU.

2. **Risk Distribution panel**
   - Adds a single stacked portfolio bar for High / Medium / Low risk.
   - Shows count and percentage for each risk tier.

3. **Top Risk Drivers panel**
   - Forecast arrivals beyond ROJ.
   - High/medium risks on the critical path.
   - Severe forecast slippage (7+ days).
   - Supplier concentration among high/medium-risk lines.

4. **Less table-heavy Overview**
   - The visual summary now appears before the detailed procurement table.
   - Table height is reduced so the highest-risk material and recent activity remain visible sooner.

## Files changed

- `dashboard_layer3.py`
- `api_layer3.py`

Restart FastAPI and Streamlit after copying these files.
