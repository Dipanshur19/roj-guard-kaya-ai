"""ROJ Guard Round-2 product UI.

Judge-facing Streamlit experience for the same four-layer prototype:
Ingest -> Graph/Features -> Predict Risk -> Agentic Action + Human Approval.

Run locally:
    streamlit run dashboard_layer3.py
"""

import json
import os
import re
from collections import Counter
from datetime import date

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="ROJ Guard | Predictive Procurement Intelligence",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Visual system - intentionally close to the navy/orange Round-1 deck language.
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
  --rg-navy: #081a33;
  --rg-navy-2: #0d2748;
  --rg-panel: #101a2b;
  --rg-panel-2: #142238;
  --rg-border: rgba(148,163,184,.18);
  --rg-text: #f8fafc;
  --rg-muted: #94a3b8;
  --rg-orange: #f59e0b;
  --rg-blue: #38bdf8;
  --rg-red: #ef4444;
  --rg-green: #22c55e;
}
.stApp { background: #070d18; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#081a33 0%,#091426 100%); border-right:1px solid var(--rg-border); }
[data-testid="stSidebar"] .stRadio label { font-weight: 600; }
.block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1480px; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.rg-brand { display:flex; align-items:center; gap:11px; margin: 4px 0 2px 0; }
.rg-logo { width:38px; height:38px; border-radius:11px; display:grid; place-items:center; font-weight:900; color:#071426; background:linear-gradient(135deg,#f59e0b,#38bdf8); box-shadow:0 8px 25px rgba(56,189,248,.16); }
.rg-brand-title {font-size:1.28rem;font-weight:800;letter-spacing:-.02em;color:white;}
.rg-brand-sub {font-size:.72rem;color:#94a3b8;margin-top:-2px;}
.rg-eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; color:#38bdf8; font-weight:700; }
.rg-title {font-size:2.15rem;font-weight:850;line-height:1.08;letter-spacing:-.035em;margin:.22rem 0 .2rem;color:#f8fafc;}
.rg-subtitle {font-size:.96rem;color:#94a3b8;max-width:900px;margin-bottom:1.2rem;}
.rg-card { background:linear-gradient(180deg,rgba(20,34,56,.94),rgba(13,26,45,.94)); border:1px solid var(--rg-border); border-radius:16px; padding:18px 19px; box-shadow:0 14px 40px rgba(0,0,0,.12); }
.rg-kpi { background:linear-gradient(180deg,rgba(20,34,56,.94),rgba(12,23,39,.94)); border:1px solid var(--rg-border); border-radius:15px; padding:16px 17px; min-height:112px; }
.rg-kpi-label { color:#94a3b8; font-size:.77rem; font-weight:650; text-transform:uppercase; letter-spacing:.055em; }
.rg-kpi-value { color:#f8fafc; font-size:2rem; font-weight:820; letter-spacing:-.04em; margin-top:7px; }
.rg-kpi-note { color:#64748b;font-size:.74rem;margin-top:2px; }
.rg-panel-title{font-size:.95rem;font-weight:790;color:#f8fafc;margin-bottom:4px}.rg-panel-sub{font-size:.75rem;color:#64748b;margin-bottom:14px}
.rg-dist-bar{height:15px;border-radius:999px;overflow:hidden;background:#1e293b;display:flex;margin:14px 0 16px;border:1px solid rgba(148,163,184,.12)}
.rg-dist-high{background:linear-gradient(90deg,#dc2626,#ef4444)}.rg-dist-medium{background:linear-gradient(90deg,#d97706,#f59e0b)}.rg-dist-low{background:linear-gradient(90deg,#16a34a,#22c55e)}
.rg-dist-row{display:grid;grid-template-columns:14px 1fr auto;align-items:center;gap:9px;margin:9px 0;color:#cbd5e1;font-size:.8rem}.rg-dot{width:9px;height:9px;border-radius:50%}.rg-dist-count{font-weight:800;color:#f8fafc}.rg-dist-meta{color:#64748b;font-size:.7rem}
.rg-driver-grid{display:grid;grid-template-columns:1fr;gap:8px}.rg-driver-item{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;padding:10px 11px;border:1px solid rgba(148,163,184,.13);border-radius:10px;background:rgba(15,23,42,.38)}.rg-driver-item b{color:#e2e8f0;font-size:.8rem}.rg-driver-item span{display:block;color:#64748b;font-size:.7rem;margin-top:2px}.rg-driver-num{font-weight:850;color:#f8fafc;font-size:1.05rem;white-space:nowrap}
.rg-section {font-size:1.22rem;font-weight:780;color:#f8fafc;margin:1.2rem 0 .6rem;letter-spacing:-.02em;}
.rg-chip {display:inline-flex;align-items:center;padding:4px 9px;border-radius:999px;font-size:.72rem;font-weight:750;border:1px solid transparent;}
.rg-high{background:rgba(239,68,68,.12);color:#fca5a5;border-color:rgba(239,68,68,.25)}
.rg-medium{background:rgba(245,158,11,.12);color:#fcd34d;border-color:rgba(245,158,11,.25)}
.rg-low{background:rgba(34,197,94,.12);color:#86efac;border-color:rgba(34,197,94,.25)}
.rg-neutral{background:rgba(56,189,248,.10);color:#7dd3fc;border-color:rgba(56,189,248,.20)}
.rg-driver {border-left:3px solid #64748b;background:rgba(15,23,42,.55);padding:10px 12px;border-radius:8px;margin:7px 0;}
.rg-driver.high{border-left-color:#ef4444}.rg-driver.medium{border-left-color:#f59e0b}.rg-driver.positive{border-left-color:#22c55e}
.rg-driver b{color:#e2e8f0}.rg-driver span{color:#94a3b8;font-size:.82rem;display:block;margin-top:2px}
.rg-timeline {position:relative;margin-left:9px;padding-left:22px;border-left:1px solid rgba(148,163,184,.24)}
.rg-event {position:relative;margin:0 0 15px 0}.rg-event:before{content:'';position:absolute;left:-27px;top:5px;width:9px;height:9px;border-radius:50%;background:#38bdf8;box-shadow:0 0 0 4px #0b1628}
.rg-event-date{color:#64748b;font-size:.71rem}.rg-event-title{color:#e2e8f0;font-weight:700;font-size:.86rem}.rg-event-detail{color:#94a3b8;font-size:.78rem;margin-top:2px}
.rg-flow {display:grid;grid-template-columns:1fr 54px 1fr 54px 1fr;align-items:stretch;gap:7px;margin:12px 0 18px}.rg-flow-arrow{display:grid;place-items:center;color:#64748b;font-size:1.7rem}
.rg-state {border:1px solid var(--rg-border);border-radius:15px;padding:18px;background:rgba(15,27,47,.88)}
.rg-state h4{margin:0 0 8px;color:#94a3b8;font-size:.76rem;text-transform:uppercase;letter-spacing:.08em}.rg-state .big{font-size:1.65rem;font-weight:820;color:#f8fafc}.rg-state p{color:#94a3b8;font-size:.8rem;margin:5px 0 0}
.rg-signal {border:1px solid rgba(245,158,11,.25);background:rgba(245,158,11,.07);border-radius:15px;padding:18px}.rg-signal h4{color:#f59e0b;margin:0 0 8px;font-size:.8rem;text-transform:uppercase;letter-spacing:.08em}.rg-signal p{color:#e2e8f0;font-size:.9rem}
.rg-graph {display:flex;align-items:stretch;gap:8px;overflow-x:auto;padding:16px 3px 20px}.rg-node{min-width:170px;max-width:210px;border:1px solid var(--rg-border);background:linear-gradient(180deg,#142238,#0d1b2f);border-radius:14px;padding:14px}.rg-node-type{font-size:.66rem;color:#38bdf8;text-transform:uppercase;letter-spacing:.09em;font-weight:750}.rg-node-label{font-size:.9rem;color:#f8fafc;font-weight:760;margin-top:8px}.rg-node-sub{font-size:.72rem;color:#94a3b8;margin-top:4px}.rg-edge{min-width:62px;display:flex;flex-direction:column;justify-content:center;align-items:center;color:#64748b}.rg-edge .arrow{font-size:1.6rem;line-height:1}.rg-edge .label{font-size:.6rem;text-transform:uppercase;letter-spacing:.05em;text-align:center;margin-top:5px}
.rg-confidence {height:7px;border-radius:999px;background:#1e293b;overflow:hidden;margin-top:8px}.rg-confidence > div{height:100%;background:linear-gradient(90deg,#38bdf8,#22c55e)}
.rg-note {font-size:.78rem;color:#94a3b8;line-height:1.5}
[data-testid="stDataFrame"] {border:1px solid var(--rg-border);border-radius:12px;overflow:hidden}
.stButton > button {border-radius:10px;font-weight:700;border:1px solid rgba(148,163,184,.20);}
.stButton > button[kind="primary"] {background:linear-gradient(90deg,#f59e0b,#fb923c);color:#111827;border:0;}
hr {border-color:rgba(148,163,184,.14)!important}
</style>
""",
    unsafe_allow_html=True,
)

RISK_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
ACTION_LABELS = {
    "ESCALATION": "Vendor Escalation",
    "RESEQUENCE": "Resequence Suggestion",
    "ALTERNATE_SUPPLIER": "Alternate Supplier",
}


def clean_name(value):
    if not value:
        return "—"
    return re.sub(r"\s*\[(?:LOW|MEDIUM|HIGH|HIST)-\d+\]\s*$", "", str(value), flags=re.I).strip()


def clean_sku(value):
    if not value:
        return "—"
    return re.sub(r"-(?:LOW|MEDIUM|HIGH|HIST)-", "-", str(value), flags=re.I)


def api_get(path, quiet=False):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        if not quiet:
            st.error(f"Backend request failed: {exc}")
        return None


def api_post(path, json_body=None, params=None, quiet=False):
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=json_body, params=params, timeout=90)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        if not quiet:
            detail = ""
            try:
                detail = r.json().get("detail", "")
            except Exception:
                pass
            st.error(f"Backend request failed: {detail or exc}")
        return None


def api_patch(path, json_body=None):
    try:
        r = requests.patch(f"{BACKEND_URL}{path}", json=json_body, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Backend request failed: {exc}")
        return None


def api_extract(doc_type, project_id, raw_text=None, uploaded_file=None):
    data = {"project_id": project_id}
    files = None
    if raw_text:
        data["raw_text"] = raw_text
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    try:
        r = requests.post(f"{BACKEND_URL}/api/extract/{doc_type}", data=data, files=files, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        detail = ""
        try:
            detail = r.json().get("detail", "")
        except Exception:
            pass
        st.error(f"Document analysis failed: {detail or exc}")
        return None


def header(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="rg-eyebrow">{eyebrow}</div><div class="rg-title">{title}</div><div class="rg-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def risk_chip(level):
    level = level or "Unknown"
    cls = level.lower() if level in ("High", "Medium", "Low") else "neutral"
    return f'<span class="rg-chip rg-{cls}">{level} risk</span>'


def kpi_card(label, value, note=""):
    return f'<div class="rg-kpi"><div class="rg-kpi-label">{label}</div><div class="rg-kpi-value">{value}</div><div class="rg-kpi-note">{note}</div></div>'


def render_before_after(before, signal, after):
    if not before or not after:
        return
    bp = float(before.get("miss_roj_probability") or 0) * 100
    ap = float(after.get("miss_roj_probability") or 0) * 100
    signal_text = signal.get("summary") if isinstance(signal, dict) else str(signal)
    delay = signal.get("delay_days") if isinstance(signal, dict) else None
    st.markdown(
        f"""
<div class="rg-flow">
  <div class="rg-state"><h4>Before</h4><div class="big">{bp:.0f}% • {before.get('risk_level','—')}</div><p>Predicted delay {before.get('predicted_delay_days','—')} day(s) • ROJ {before.get('roj_date','—')}</p></div>
  <div class="rg-flow-arrow">→</div>
  <div class="rg-signal"><h4>New supply-chain signal</h4><p>{signal_text or 'New operational update received.'}</p><span class="rg-chip rg-medium">{str(delay)+' day delay' if delay else 'AI extracted'}</span></div>
  <div class="rg-flow-arrow">→</div>
  <div class="rg-state"><h4>After</h4><div class="big">{ap:.0f}% • {after.get('risk_level','—')}</div><p>Predicted delay {after.get('predicted_delay_days','—')} day(s) • ROJ {after.get('roj_date','—')}</p></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_drivers(drivers):
    if not drivers:
        st.caption("No risk drivers available.")
        return
    html = ""
    for d in drivers:
        sev = d.get("severity", "medium")
        html += f'<div class="rg-driver {sev}"><b>{d.get("label","Signal")}</b><span>{d.get("detail","")}</span></div>'
    st.markdown(html, unsafe_allow_html=True)


def render_timeline(events):
    if not events:
        st.caption("No recent activity.")
        return
    html = '<div class="rg-timeline">'
    for e in events:
        html += (
            f'<div class="rg-event"><div class="rg-event-date">{e.get("date","—")} • {e.get("type","Event")}</div>'
            f'<div class="rg-event-title">{e.get("title","")}</div><div class="rg-event-detail">{e.get("detail","")}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def material_options():
    items = api_get("/api/experience/materials", quiet=True) or []
    return items, {x["material_id"]: f'{x["material_name"]}  •  {x.get("vendor_name") or "No vendor"}' for x in items}


# ---------------------------------------------------------------------------
# Sidebar / global status
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="rg-brand"><div class="rg-logo">R</div><div><div class="rg-brand-title">ROJ Guard</div><div class="rg-brand-sub">Predictive procurement intelligence</div></div></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

PAGES = ["Overview", "Incoming Intelligence", "Material Intelligence", "Project Graph", "Live Demo", "Approval Center"]
page = st.sidebar.radio("Workspace", PAGES, label_visibility="collapsed")

health = api_get("/", quiet=True)
if health:
    st.sidebar.success("Backend connected")
else:
    st.sidebar.error("Backend unavailable")
st.sidebar.caption(BACKEND_URL)

with st.sidebar.expander("Model & data controls"):
    if st.button("Recompute active risk", use_container_width=True):
        result = api_post("/api/risk/compute-all")
        if result:
            st.success(f"Scored {result.get('materials_processed',0)} active materials")
            st.rerun()
    if st.button("Retrain models", use_container_width=True):
        with st.spinner("Training XGBoost models..."):
            result = api_post("/api/risk/train")
        if result and result.get("status") == "trained":
            st.success(f"MAE {result['lead_time_model_mae_days']}d • AUC {result['risk_classifier_auc']}")

st.sidebar.markdown("---")
st.sidebar.caption("Human approval is required before any outbound action executes.")

# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------
if page == "Overview":
    header("Portfolio command center", "Project risk overview", "A predictive view of active procurement lines, schedule exposure, and the interventions that need attention now.")
    scores = api_get("/api/risk/scores") or []
    if not scores:
        st.info("No active risk scores are available. Recompute risk from the sidebar.")
    else:
        high = sum(s.get("risk_level") == "High" for s in scores)
        med = sum(s.get("risk_level") == "Medium" for s in scores)
        low = sum(s.get("risk_level") == "Low" for s in scores)
        critical = sum(bool(s.get("is_critical_path")) and s.get("risk_level") in ("High", "Medium") for s in scores)
        late = sum((s.get("predicted_delay_days") or 0) > 0 for s in scores)

        cols = st.columns(5)
        cards = [
            ("Active materials", len(scores), "Current procurement portfolio"),
            ("High risk", high, "Immediate mitigation candidates"),
            ("Medium risk", med, "Emerging schedule friction"),
            ("Critical-path exposure", critical, "High/medium on critical path"),
            ("Forecast late", late, "Arrival forecast beyond ROJ"),
        ]
        for col, card in zip(cols, cards):
            col.markdown(kpi_card(*card), unsafe_allow_html=True)

        # Fast visual read: distribution + aggregate drivers before the detailed table.
        panel_left, panel_right = st.columns([.9, 1.1])
        total = max(len(scores), 1)
        high_pct, med_pct, low_pct = high / total * 100, med / total * 100, low / total * 100
        with panel_left:
            st.markdown(
                f"""<div class="rg-card"><div class="rg-panel-title">Risk distribution</div><div class="rg-panel-sub">Current active procurement portfolio</div>
                <div class="rg-dist-bar"><div class="rg-dist-high" style="width:{high_pct:.1f}%"></div><div class="rg-dist-medium" style="width:{med_pct:.1f}%"></div><div class="rg-dist-low" style="width:{low_pct:.1f}%"></div></div>
                <div class="rg-dist-row"><div class="rg-dot" style="background:#ef4444"></div><div>High risk<div class="rg-dist-meta">Immediate mitigation</div></div><div class="rg-dist-count">{high} <span class="rg-dist-meta">• {high_pct:.0f}%</span></div></div>
                <div class="rg-dist-row"><div class="rg-dot" style="background:#f59e0b"></div><div>Medium risk<div class="rg-dist-meta">Emerging friction</div></div><div class="rg-dist-count">{med} <span class="rg-dist-meta">• {med_pct:.0f}%</span></div></div>
                <div class="rg-dist-row"><div class="rg-dot" style="background:#22c55e"></div><div>Low risk<div class="rg-dist-meta">Currently protected</div></div><div class="rg-dist-count">{low} <span class="rg-dist-meta">• {low_pct:.0f}%</span></div></div></div>""",
                unsafe_allow_html=True,
            )
        with panel_right:
            severe = sum((s.get("predicted_delay_days") or 0) >= 7 for s in scores)
            at_risk = [s for s in scores if s.get("risk_level") in ("High", "Medium")]
            vendor_counts = Counter((s.get("vendor_name") or "Unknown vendor") for s in at_risk)
            top_vendor, top_vendor_count = vendor_counts.most_common(1)[0] if vendor_counts else ("—", 0)
            st.markdown(
                f"""<div class="rg-card"><div class="rg-panel-title">Top risk drivers</div><div class="rg-panel-sub">What is creating schedule exposure right now</div><div class="rg-driver-grid">
                <div class="rg-driver-item"><div><b>Forecast arrival beyond ROJ</b><span>Active lines whose forecast crosses the required-on-job date</span></div><div class="rg-driver-num">{late}</div></div>
                <div class="rg-driver-item"><div><b>Critical-path exposure</b><span>High/medium-risk materials tied to critical activities</span></div><div class="rg-driver-num">{critical}</div></div>
                <div class="rg-driver-item"><div><b>Severe schedule slippage</b><span>Forecast delay of seven days or more</span></div><div class="rg-driver-num">{severe}</div></div>
                <div class="rg-driver-item"><div><b>Supplier concentration</b><span>{top_vendor} has the most high/medium-risk lines</span></div><div class="rg-driver-num">{top_vendor_count}</div></div>
                </div></div>""",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="rg-section">Materials requiring attention</div>', unsafe_allow_html=True)
        df = pd.DataFrame(scores)
        display = df[["material_name", "sku", "po_number", "vendor_name", "risk_level", "miss_roj_probability", "predicted_arrival_date", "roj_date", "predicted_delay_days", "is_critical_path"]].copy()
        display["material_name"] = display["material_name"].map(clean_name)
        display["sku"] = display["sku"].map(clean_sku)
        display["po_sku"] = display.apply(lambda row: f'{row.get("po_number") or "—"}  ·  {row.get("sku") or "—"}', axis=1)
        display["miss_roj_probability"] = (display["miss_roj_probability"] * 100).round(0).astype(int).astype(str) + "%"
        display = display[["material_name", "po_sku", "vendor_name", "risk_level", "miss_roj_probability", "predicted_arrival_date", "roj_date", "predicted_delay_days", "is_critical_path"]]
        display.columns = ["Material", "PO / SKU", "Vendor", "Risk", "Miss ROJ", "Forecast arrival", "ROJ", "Delay (days)", "Critical path"]
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=360,
            column_config={
                "Material": st.column_config.TextColumn(width="medium"),
                "PO / SKU": st.column_config.TextColumn(width="medium", help="Purchase-order number and unique material SKU"),
                "Vendor": st.column_config.TextColumn(width="medium"),
                "Risk": st.column_config.TextColumn(width="small"),
                "Miss ROJ": st.column_config.TextColumn(width="small"),
                "Critical path": st.column_config.CheckboxColumn(width="small"),
            },
        )

        c1, c2 = st.columns([1.2, .8])
        with c1:
            st.markdown('<div class="rg-section">Highest-risk material</div>', unsafe_allow_html=True)
            top = scores[0]
            ctx = api_get(f"/api/experience/material/{top['material_id']}")
            if ctx:
                risk = ctx["risk"]
                st.markdown(
                    f'<div class="rg-card"><div style="display:flex;justify-content:space-between;gap:12px"><div><div style="font-size:1.05rem;font-weight:800;color:white">{ctx["material"]["name"]}</div><div class="rg-note">{(ctx.get("vendor") or {}).get("name","—")} • {(ctx.get("po") or {}).get("po_number","—")}</div></div>{risk_chip(risk.get("risk_level"))}</div><div style="font-size:2rem;font-weight:850;margin-top:15px">{float(risk.get("miss_roj_probability") or 0)*100:.0f}%</div><div class="rg-note">Predicted miss-ROJ probability • forecast delay {risk.get("predicted_delay_days")} day(s)</div></div>',
                    unsafe_allow_html=True,
                )
                render_drivers(ctx.get("risk_drivers"))
        with c2:
            st.markdown('<div class="rg-section">Recent activity</div>', unsafe_allow_html=True)
            if ctx:
                render_timeline(ctx.get("activity", [])[:5])

# ---------------------------------------------------------------------------
# INCOMING INTELLIGENCE
# ---------------------------------------------------------------------------
elif page == "Incoming Intelligence":
    header("Layer 1 • Document intelligence", "Incoming intelligence", "Analyze a vendor email or project document, review the extracted entities, then apply the signal to the project. No Swagger or JSON API work is required.")
    projects = api_get("/api/experience/projects") or []
    if not projects:
        st.error("No project exists in the database.")
    else:
        project_map = {p["project_id"]: f'{p["project_name"]} • {p.get("location") or ""}' for p in projects}
        top1, top2 = st.columns([1.25, 1])
        with top1:
            project_id = st.selectbox("Project", list(project_map), format_func=lambda x: project_map[x])
        with top2:
            type_label = st.selectbox("Signal type", ["Vendor Email", "Shipping Update", "Purchase Order", "Submittal", "Schedule / ROJ"])
        doc_type = {"Vendor Email":"VENDOR_EMAIL", "Shipping Update":"SHIPPING", "Purchase Order":"PO", "Submittal":"SUBMITTAL", "Schedule / ROJ":"SCHEDULE"}[type_label]

        input_tab, file_tab = st.tabs(["Paste text", "Upload PDF"])
        raw_text = None
        uploaded = None
        with input_tab:
            placeholder = "Metro Civil Products confirms production is delayed by 14 days..." if doc_type == "VENDOR_EMAIL" else "Paste the document/update text here..."
            raw_text = st.text_area("Message / document text", height=190, placeholder=placeholder, key="incoming_raw")
        with file_tab:
            uploaded = st.file_uploader("Upload a PDF", type=["pdf"], help="Gemini reads the PDF directly; no separate OCR step is required.")

        analyze_col, note_col = st.columns([.28, .72])
        with analyze_col:
            analyze = st.button("Analyze with AI", type="primary", use_container_width=True)
        with note_col:
            st.markdown('<div class="rg-note" style="padding-top:9px">Analysis is preview-only. Nothing enters the project graph until you explicitly click <b>Apply to Project</b>.</div>', unsafe_allow_html=True)

        if analyze:
            if not raw_text and uploaded is None:
                st.warning("Paste text or upload a PDF first.")
            else:
                with st.spinner("Extracting entities and schedule signals..."):
                    result = api_extract(doc_type, project_id, raw_text=raw_text, uploaded_file=uploaded)
                if result:
                    st.session_state["extraction_preview"] = result
                    st.session_state.pop("extraction_apply", None)
                    st.session_state.pop("auto_agent_draft", None)

        preview = st.session_state.get("extraction_preview")
        if preview and preview.get("doc_type") == doc_type:
            st.markdown('<div class="rg-section">AI extraction review</div>', unsafe_allow_html=True)
            conf = float(preview.get("confidence") or 0)
            c1, c2 = st.columns([.72, .28])
            with c1:
                fields = preview.get("extracted_fields") or {}
                rows = [{"Field": k.replace("_", " ").title(), "Extracted value": v} for k, v in fields.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with c2:
                st.markdown(f'<div class="rg-card"><div class="rg-kpi-label">Extraction confidence</div><div class="rg-kpi-value">{conf*100:.0f}%</div><div class="rg-confidence"><div style="width:{conf*100:.0f}%"></div></div><div class="rg-note" style="margin-top:10px">{"Human confirmation recommended" if preview.get("requires_human_confirmation") else "Ready for confirmation"}</div></div>', unsafe_allow_html=True)
                if preview.get("notes"):
                    st.caption(preview["notes"])

            with st.expander("Review / edit extracted JSON"):
                json_text = st.text_area("Confirmed fields", value=json.dumps(preview.get("extracted_fields") or {}, indent=2, default=str), height=230, key="confirmed_json")

            if st.button("Apply to Project", type="primary"):
                try:
                    confirmed = json.loads(json_text)
                except json.JSONDecodeError as exc:
                    st.error(f"The edited extraction is not valid JSON: {exc}")
                else:
                    payload = {
                        "project_id": preview["project_id"],
                        "extracted_fields": confirmed,
                        "confidence": preview.get("confidence", 0),
                        "notes": preview.get("notes"),
                        "raw_text": preview.get("raw_text"),
                        "file_path": preview.get("file_path"),
                    }
                    with st.spinner("Linking entities, updating features and recomputing ROJ risk..."):
                        applied = api_post(f"/api/apply-extraction/{doc_type}", json_body=payload)
                    if applied:
                        st.session_state["extraction_apply"] = applied
                        after = applied.get("after_risk")
                        # When the new signal creates a material risk, the agent proactively prepares a draft.
                        if after and after.get("risk_level") in ("High", "Medium"):
                            draft = api_post(f"/api/actions/draft/{after['material_id']}", params={"action_type":"ESCALATION"}, quiet=True)
                            if draft:
                                st.session_state["auto_agent_draft"] = draft
                        st.success("Signal applied. Project intelligence and risk have been refreshed.")

        applied = st.session_state.get("extraction_apply")
        if applied:
            before = applied.get("before_risk")
            after = applied.get("after_risk")
            if after:
                signal = {"summary": (preview.get("extracted_fields") or {}).get("extracted_summary") or f"{type_label} applied to the project", "delay_days": (preview.get("extracted_fields") or {}).get("delay_days_mentioned")}
                st.markdown('<div class="rg-section">Risk reaction</div>', unsafe_allow_html=True)
                render_before_after(before or after, signal, after)
                draft = st.session_state.get("auto_agent_draft")
                if draft:
                    st.markdown(f'<div class="rg-card"><div class="rg-eyebrow">Agentic response prepared</div><div style="font-weight:800;color:white;margin:6px 0">Vendor escalation is waiting for human approval</div><div class="rg-note">{draft.get("reasoning","")}</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MATERIAL INTELLIGENCE
# ---------------------------------------------------------------------------
elif page == "Material Intelligence":
    header("Layer 3 + 4 • Explainable risk", "Material intelligence", "Inspect the forecast, live risk drivers, vendor context, schedule exposure, and mitigation actions for a single material line.")
    items, labels = material_options()
    if not items:
        st.info("No active materials available.")
    else:
        selected = st.selectbox("Material", [x["material_id"] for x in items], format_func=lambda x: labels[x])
        ctx = api_get(f"/api/experience/material/{selected}")
        if ctx:
            risk = ctx.get("risk") or {}
            material = ctx["material"]
            vendor = ctx.get("vendor") or {}
            shipment = ctx.get("shipment") or {}
            sched = ctx.get("schedule") or {}

            st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;gap:20px"><div><div style="font-size:1.45rem;font-weight:850;color:white">{material["name"]}</div><div class="rg-note">{material.get("sku","—")} • {vendor.get("name","—")}</div></div>{risk_chip(risk.get("risk_level"))}</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            metrics = [
                ("Miss ROJ", f'{float(risk.get("miss_roj_probability") or 0)*100:.0f}%', "Live probability"),
                ("Forecast delay", f'{risk.get("predicted_delay_days","—")}d', "Against ROJ"),
                ("Forecast arrival", risk.get("predicted_arrival_date","—"), "Operational forecast"),
                ("ROJ", risk.get("roj_date","—"), "Required on job"),
                ("Vendor reliability", f'{float(vendor.get("reliability") or 0)*100:.0f}%' if vendor.get("reliability") is not None else "—", "Historical delivery"),
            ]
            for col, metric in zip(cols, metrics): col.markdown(kpi_card(*metric), unsafe_allow_html=True)

            left, right = st.columns([1.15, .85])
            with left:
                st.markdown('<div class="rg-section">Why this risk?</div>', unsafe_allow_html=True)
                render_drivers(ctx.get("risk_drivers"))
                st.markdown('<div class="rg-section">Risk explanation</div>', unsafe_allow_html=True)
                st.info(risk.get("explanation") or "No explanation available.")
                if risk.get("risk_level") in ("High", "Medium"):
                    st.markdown('<div class="rg-section">Mitigation agents</div>', unsafe_allow_html=True)
                    a1, a2, a3 = st.columns(3)
                    actions = [("Vendor escalation","ESCALATION"),("Resequence work","RESEQUENCE"),("Find alternate supplier","ALTERNATE_SUPPLIER")]
                    for col, (label, action_type) in zip((a1,a2,a3), actions):
                        with col:
                            if st.button(label, key=f"act_{action_type}", use_container_width=True):
                                with st.spinner("Preparing action..."):
                                    result = api_post(f"/api/actions/draft/{selected}", params={"action_type": action_type})
                                if result:
                                    st.success("Draft moved to Approval Center")
            with right:
                st.markdown('<div class="rg-section">Current state</div>', unsafe_allow_html=True)
                state_rows = {
                    "Shipment": (shipment.get("status") or "unknown").replace("_"," ").title(),
                    "Current location": shipment.get("current_location") or "—",
                    "Tracking": shipment.get("tracking_number") or "—",
                    "Schedule task": sched.get("task_name") or "—",
                    "Float": f'{sched.get("float_days","—")} day(s)',
                    "Critical path": "Yes" if sched.get("is_critical_path") else "No",
                    "Latest vendor signal": (ctx.get("latest_vendor_signal") or {}).get("summary") or "—",
                }
                st.dataframe(pd.DataFrame([{"Field":k,"Value":v} for k,v in state_rows.items()]), use_container_width=True, hide_index=True)
                st.markdown('<div class="rg-section">Activity timeline</div>', unsafe_allow_html=True)
                render_timeline(ctx.get("activity", []))

# ---------------------------------------------------------------------------
# PROJECT GRAPH
# ---------------------------------------------------------------------------
elif page == "Project Graph":
    header("Layer 2 • Project graph", "Project graph", "A material-centric projection of the entities ROJ Guard links together. The same contract maps naturally onto Kaya/Amber's existing project graph.")
    items, labels = material_options()
    if items:
        selected = st.selectbox("Material graph", [x["material_id"] for x in items], format_func=lambda x: labels[x])
        graph = api_get(f"/api/experience/graph/{selected}")
        if graph:
            order = ["Project","Material","Vendor","PO","Shipment","ROJ"]
            nodes_by_type = {n["type"]: n for n in graph.get("nodes",[])}
            chain = [nodes_by_type[t] for t in order if t in nodes_by_type]
            html = '<div class="rg-graph">'
            for i, n in enumerate(chain):
                html += f'<div class="rg-node"><div class="rg-node-type">{n["type"]}</div><div class="rg-node-label">{clean_name(n["label"])}</div><div class="rg-node-sub">{n.get("subtitle","")}</div></div>'
                if i < len(chain)-1:
                    html += '<div class="rg-edge"><div class="arrow">→</div><div class="label">linked context</div></div>'
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)
            st.markdown('<div class="rg-section">Graph relationships</div>', unsafe_allow_html=True)
            node_lookup = {n["id"]: n for n in graph.get("nodes",[])}
            edge_rows=[]
            for e in graph.get("edges",[]):
                edge_rows.append({"From": node_lookup.get(e["source"],{}).get("type","Node"), "Relationship": e["label"], "To": node_lookup.get(e["target"],{}).get("type","Node")})
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)
            st.markdown("""<div class="rg-card"><div class="rg-eyebrow">Kaya / Amber integration</div><div style="color:white;font-weight:780;margin:6px 0">ROJ Guard is a specialist intelligence module, not a replacement graph.</div><div class="rg-note">In a Kaya deployment, these material/vendor/PO/shipment/ROJ entities can be consumed from Amber's project graph while ROJ Guard contributes forecast risk, explanations and mitigation actions back into the workflow.</div></div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LIVE DEMO
# ---------------------------------------------------------------------------
elif page == "Live Demo":
    header("60-second judge scenario", "Live risk reaction demo", "One deterministic scenario demonstrates the entire Round-1 promise: a healthy material receives a vendor delay signal, risk changes immediately, and an agent prepares mitigation for human approval.")

    top_a, top_b = st.columns([.72,.28])
    with top_b:
        if st.button("Reset Demo", use_container_width=True):
            with st.spinner("Restoring baseline..."):
                api_post("/api/experience/demo/reset")
            st.session_state.pop("demo_result", None)
            st.session_state.pop("demo_execution", None)
            st.rerun()
    existing_demo = st.session_state.get("demo_result")
    scenario = None if existing_demo else api_get("/api/experience/demo/scenario")
    with top_a:
        if existing_demo:
            st.markdown(f'<div class="rg-card"><div class="rg-eyebrow">Scenario completed</div><div style="font-size:1.1rem;color:white;font-weight:800;margin-top:5px">{existing_demo.get("material_name","Material")}</div><div class="rg-note">Use Reset Demo to restore the Low-risk baseline and run it again.</div></div>', unsafe_allow_html=True)
        elif scenario and scenario.get("ready"):
            st.markdown(f'<div class="rg-card"><div class="rg-eyebrow">Scenario ready</div><div style="font-size:1.1rem;color:white;font-weight:800;margin-top:5px">{scenario["material_name"]}</div><div class="rg-note">{scenario.get("vendor_name","—")} • currently {float(scenario["before"].get("miss_roj_probability") or 0)*100:.0f}% {scenario["before"].get("risk_level")} risk</div></div>', unsafe_allow_html=True)
        elif scenario:
            st.warning(scenario.get("message","Reset Demo Mode to restore a Low-risk baseline."))

    if scenario and scenario.get("ready") and not existing_demo:
        st.markdown('<div class="rg-section">What will happen</div>', unsafe_allow_html=True)
        render_before_after(
            scenario["before"],
            {"summary": scenario["signal"], "delay_days":14},
            {**scenario["before"], "risk_level":"High", "miss_roj_probability":.90, "predicted_delay_days":14},
        )
        if st.button("Run Live Risk Scenario", type="primary", use_container_width=True):
            with st.spinner("Receiving vendor signal → updating graph → recomputing features → scoring risk → preparing mitigation..."):
                result = api_post("/api/experience/demo/run", params={"material_id": scenario["material_id"]})
            if result:
                st.session_state["demo_result"] = result
                st.rerun()

    result = st.session_state.get("demo_result")
    if result:
        st.success("Live scenario completed: the new supply-chain signal changed project risk and triggered an agentic response.")
        render_before_after(result.get("before"), result.get("signal"), result.get("after"))
        after = result.get("after") or {}
        action = result.get("agent_action") or {}
        c1, c2 = st.columns([.56,.44])
        with c1:
            st.markdown('<div class="rg-section">Agent response</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="rg-card"><div class="rg-eyebrow">Human approval required</div><div style="font-size:1.05rem;color:white;font-weight:800;margin:6px 0">Vendor escalation prepared automatically</div><div class="rg-note">{action.get("reasoning","")}</div></div>', unsafe_allow_html=True)
            edited = st.text_area("Draft escalation", value=action.get("draft_content", ""), height=235, key="demo_draft")
            if not st.session_state.get("demo_execution"):
                if st.button("Approve & Execute", type="primary", use_container_width=True):
                    if edited != action.get("draft_content"):
                        api_patch(f"/api/actions/{action['action_id']}", {"draft_content": edited})
                    execution = api_post(f"/api/actions/{action['action_id']}/approve", params={"reviewed_by":"hackathon_judge"})
                    if execution:
                        st.session_state["demo_execution"] = execution
                        st.success(f"Executed • {execution.get('execution_status','executed')}")
            else:
                st.success("Human-approved action executed and audit receipt created.")
        with c2:
            st.markdown('<div class="rg-section">System reaction</div>', unsafe_allow_html=True)
            steps = [
                ("1", "Vendor signal received", "14-day delay notice"),
                ("2", "Entities linked", "Material + vendor + project context"),
                ("3", "Risk recomputed", f'{float(after.get("miss_roj_probability") or 0)*100:.0f}% • {after.get("risk_level")}'),
                ("4", "Mitigation drafted", "Vendor escalation pending approval"),
            ]
            for num, title, detail in steps:
                st.markdown(f'<div class="rg-driver positive"><b>{num}. {title}</b><span>{detail}</span></div>', unsafe_allow_html=True)
            execution = st.session_state.get("demo_execution")
            if execution:
                st.markdown(f'<div class="rg-driver positive"><b>5. Human-approved execution</b><span>{execution.get("execution_status")} • audit receipt created</span></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# APPROVAL CENTER
# ---------------------------------------------------------------------------
elif page == "Approval Center":
    header("Layer 4 • Human-in-the-loop", "Approval center", "Review, edit, approve or reject AI-drafted mitigation actions. Execution receipts remain visible for auditability.")
    pending = api_get("/api/actions/pending") or []
    materials, label_map = material_options()
    name_map = {m["material_id"]: m["material_name"] for m in materials}

    cols = st.columns(3)
    cols[0].markdown(kpi_card("Pending approvals", len(pending), "Human decisions required"), unsafe_allow_html=True)
    history = api_get("/api/actions/all") or []
    executed = sum((h.get("execution") or {}).get("status") == "executed" for h in history)
    approved = sum(h.get("status") == "approved" for h in history)
    cols[1].markdown(kpi_card("Approved", approved, "Human-approved actions"), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Executed", executed, "Audit receipts created"), unsafe_allow_html=True)

    st.markdown('<div class="rg-section">Pending decisions</div>', unsafe_allow_html=True)
    if not pending:
        st.info("No actions are waiting for approval.")
    for action in pending:
        material_name = name_map.get(action["material_id"], clean_name(action["material_id"]))
        label = ACTION_LABELS.get(action["action_type"], action["action_type"])
        with st.expander(f'{label} • {material_name} • {action.get("risk_level_at_creation","—")} risk'):
            st.caption(action.get("reasoning") or "AI mitigation recommendation")
            edited = st.text_area("Draft content", action.get("draft_content", ""), height=190, key=f'approval_{action["action_id"]}')
            a, b = st.columns(2)
            with a:
                if st.button("Approve & Execute", type="primary", key=f'ok_{action["action_id"]}', use_container_width=True):
                    if edited != action.get("draft_content"):
                        api_patch(f'/api/actions/{action["action_id"]}', {"draft_content": edited})
                    result = api_post(f'/api/actions/{action["action_id"]}/approve', params={"reviewed_by":"procurement_manager"})
                    if result:
                        st.success(f"Executed: {result.get('execution_status','approved')}")
                        st.rerun()
            with b:
                if st.button("Reject", key=f'no_{action["action_id"]}', use_container_width=True):
                    if api_post(f'/api/actions/{action["action_id"]}/reject', params={"reviewed_by":"procurement_manager"}):
                        st.rerun()

    st.markdown('<div class="rg-section">Action audit trail</div>', unsafe_allow_html=True)
    if history:
        rows=[]
        for h in history:
            ex=h.get("execution") or {}
            rows.append({
                "Material": name_map.get(h["material_id"], clean_name(h["material_id"])),
                "Action": ACTION_LABELS.get(h["action_type"], h["action_type"]),
                "Approval": h.get("status"),
                "Execution": ex.get("status","—"),
                "Created": str(h.get("created_at",""))[:19],
                "Receipt": ex.get("output_path","—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=360)
