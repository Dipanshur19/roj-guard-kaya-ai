"""ROJ Guard Round-2 product UI.

Judge-facing Streamlit experience for the same four-layer prototype:
Ingest -> Graph/Features -> Predict Risk -> Agentic Action + Human Approval.

Run locally:
    streamlit run dashboard_layer3.py
"""

import json
import os
import re
import hashlib
from collections import Counter
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from real_data_lab import (
    CANONICAL_FIELDS, suggest_column_mapping, excel_sheet_names, read_tabular_bytes,
    normalize_dataset, validate_dataset, build_real_feature_frames,
    train_real_data_models, score_with_real_model, score_with_prototype_model,
    build_supplier_intelligence, build_category_intelligence,
    train_procurement_lead_time_model, predict_procurement_scenario, compare_supplier_scenarios,
    template_csv_bytes, procurement_history_template_csv_bytes,
)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="ROJ Guard | Predictive Procurement Intelligence",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Premium product visual system: top navigation + glass surfaces + responsive UX.
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
  --rg-bg: #050a13; --rg-bg-2:#07101d; --rg-surface:rgba(13,25,43,.76);
  --rg-border:rgba(148,163,184,.16); --rg-text:#f8fafc; --rg-muted:#94a3b8;
  --rg-orange:#f59e0b; --rg-cyan:#38bdf8; --rg-red:#ef4444; --rg-green:#22c55e;
}
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp {
  color:var(--rg-text);
  background:
    radial-gradient(900px 520px at 12% -8%,rgba(56,189,248,.11),transparent 62%),
    radial-gradient(720px 460px at 92% 8%,rgba(245,158,11,.09),transparent 64%),
    radial-gradient(900px 560px at 50% 110%,rgba(59,130,246,.08),transparent 62%),
    linear-gradient(180deg,#050a13 0%,#070d18 48%,#050912 100%);
}
.stApp:before { content:"";position:fixed;inset:0;pointer-events:none;opacity:.13;z-index:0;background-image:linear-gradient(rgba(148,163,184,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.12) 1px,transparent 1px);background-size:64px 64px;mask-image:linear-gradient(to bottom,rgba(0,0,0,.5),transparent 85%); }
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"]{display:none!important}
[data-testid="stHeader"]{background:transparent!important}
.block-container{position:relative;z-index:1;padding-top:1.05rem;padding-bottom:4rem;max-width:1540px}
#MainMenu,footer{visibility:hidden}
.rg-topbar{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:11px 14px;margin:0 0 7px;border:1px solid var(--rg-border);border-radius:18px;background:linear-gradient(180deg,rgba(15,29,50,.80),rgba(8,18,32,.72));backdrop-filter:blur(18px);box-shadow:0 16px 50px rgba(0,0,0,.20),inset 0 1px 0 rgba(255,255,255,.035)}
.rg-brand{display:flex;align-items:center;gap:11px;min-width:0}.rg-logo{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;font-weight:900;color:#071426;background:linear-gradient(135deg,#f59e0b 8%,#67e8f9 92%);box-shadow:0 10px 35px rgba(56,189,248,.18),0 0 0 1px rgba(255,255,255,.10) inset}.rg-brand-title{font-size:1.12rem;font-weight:850;letter-spacing:-.025em;color:#fff;white-space:nowrap}.rg-brand-sub{font-size:.68rem;color:#7f91a8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:430px}
.rg-status-wrap{display:flex;align-items:center;gap:9px;flex-wrap:wrap;justify-content:flex-end}.rg-status{display:inline-flex;align-items:center;gap:7px;padding:7px 10px;border:1px solid rgba(34,197,94,.22);background:rgba(34,197,94,.08);border-radius:999px;color:#bbf7d0;font-size:.72rem;font-weight:720;white-space:nowrap}.rg-status-dot{width:7px;height:7px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 4px rgba(34,197,94,.10),0 0 16px rgba(34,197,94,.65)}.rg-trust{font-size:.70rem;color:#77889e;white-space:nowrap}
div[data-testid="stRadio"]{margin:.2rem 0 .55rem}div[data-testid="stRadio"]>div{overflow-x:auto;scrollbar-width:none}div[data-testid="stRadio"]>div::-webkit-scrollbar{display:none}div[data-testid="stRadio"] [role="radiogroup"]{display:flex!important;flex-direction:row!important;gap:6px!important;width:max-content;min-width:100%;padding:6px;border:1px solid var(--rg-border);border-radius:15px;background:rgba(8,18,32,.72);backdrop-filter:blur(14px)}div[data-testid="stRadio"] [role="radiogroup"] label{background:transparent;border:1px solid transparent;border-radius:10px;padding:7px 12px;transition:all .18s ease;min-height:38px;white-space:nowrap}div[data-testid="stRadio"] [role="radiogroup"] label:hover{background:rgba(148,163,184,.07);border-color:rgba(148,163,184,.10)}div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(135deg,rgba(245,158,11,.18),rgba(56,189,248,.10));border-color:rgba(245,158,11,.28)}div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p{color:#fff!important;font-weight:750}div[data-testid="stRadio"] [role="radiogroup"] p{color:#aebed1;font-size:.80rem;font-weight:620}div[data-testid="stRadio"] input{display:none}
.rg-control-strip{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:2px 0 14px;color:#7f91a8;font-size:.72rem}.rg-control-pill{padding:5px 9px;border-radius:999px;border:1px solid rgba(148,163,184,.13);background:rgba(15,23,42,.40)}
.rg-eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.68rem;color:#67e8f9;font-weight:780}.rg-title{font-size:2.05rem;font-weight:880;line-height:1.08;letter-spacing:-.045em;margin:.22rem 0 .25rem;color:#f8fafc}.rg-subtitle{font-size:.94rem;color:#8fa1b7;max-width:920px;margin-bottom:1.15rem;line-height:1.55}.rg-section{font-size:1.15rem;font-weight:800;color:#f8fafc;margin:1.25rem 0 .65rem;letter-spacing:-.025em}
.rg-card,.rg-kpi,.rg-state,.rg-node{background:linear-gradient(180deg,rgba(18,34,57,.84),rgba(9,20,36,.80));border:1px solid var(--rg-border);box-shadow:0 18px 55px rgba(0,0,0,.16),inset 0 1px 0 rgba(255,255,255,.028);backdrop-filter:blur(14px)}.rg-card{border-radius:17px;padding:18px 19px}.rg-card:hover,.rg-kpi:hover{border-color:rgba(125,211,252,.22);transform:translateY(-1px);transition:.18s ease}.rg-kpi{border-radius:16px;padding:16px 17px;min-height:110px}.rg-kpi-label{color:#8394aa;font-size:.70rem;font-weight:720;text-transform:uppercase;letter-spacing:.075em}.rg-kpi-value{color:#f8fafc;font-size:1.9rem;font-weight:880;letter-spacing:-.045em;margin-top:8px}.rg-kpi-note{color:#62748b;font-size:.70rem;margin-top:3px}.rg-panel-title{font-size:.92rem;font-weight:800;color:#f8fafc;margin-bottom:4px}.rg-panel-sub{font-size:.73rem;color:#62748b;margin-bottom:14px}.rg-note{font-size:.77rem;color:#8fa1b7;line-height:1.55}
.rg-chip{display:inline-flex;align-items:center;padding:4px 9px;border-radius:999px;font-size:.69rem;font-weight:780;border:1px solid transparent}.rg-high{background:rgba(239,68,68,.10);color:#fecaca;border-color:rgba(239,68,68,.22)}.rg-medium{background:rgba(245,158,11,.10);color:#fde68a;border-color:rgba(245,158,11,.22)}.rg-low{background:rgba(34,197,94,.10);color:#bbf7d0;border-color:rgba(34,197,94,.22)}.rg-neutral{background:rgba(56,189,248,.09);color:#bae6fd;border-color:rgba(56,189,248,.18)}
.rg-dist-bar{height:12px;border-radius:999px;overflow:hidden;background:#172235;display:flex;margin:14px 0 16px;border:1px solid rgba(148,163,184,.10)}.rg-dist-high{background:linear-gradient(90deg,#dc2626,#fb7185)}.rg-dist-medium{background:linear-gradient(90deg,#d97706,#fbbf24)}.rg-dist-low{background:linear-gradient(90deg,#16a34a,#4ade80)}.rg-dist-row{display:grid;grid-template-columns:14px 1fr auto;align-items:center;gap:9px;margin:9px 0;color:#cbd5e1;font-size:.78rem}.rg-dot{width:8px;height:8px;border-radius:50%}.rg-dist-count{font-weight:820;color:#f8fafc}.rg-dist-meta{color:#62748b;font-size:.68rem}
.rg-driver-grid{display:grid;grid-template-columns:1fr;gap:8px}.rg-driver-item{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;padding:10px 11px;border:1px solid rgba(148,163,184,.11);border-radius:11px;background:rgba(15,23,42,.30)}.rg-driver-item b{color:#e2e8f0;font-size:.78rem}.rg-driver-item span{display:block;color:#62748b;font-size:.68rem;margin-top:2px}.rg-driver-num{font-weight:880;color:#f8fafc;font-size:1rem;white-space:nowrap}.rg-driver{border-left:3px solid #64748b;background:rgba(15,23,42,.46);padding:10px 12px;border-radius:9px;margin:7px 0}.rg-driver.high{border-left-color:#ef4444}.rg-driver.medium{border-left-color:#f59e0b}.rg-driver.positive{border-left-color:#22c55e}.rg-driver b{color:#e2e8f0}.rg-driver span{color:#8fa1b7;font-size:.79rem;display:block;margin-top:2px}
.rg-timeline{position:relative;margin-left:9px;padding-left:22px;border-left:1px solid rgba(148,163,184,.22)}.rg-event{position:relative;margin:0 0 15px}.rg-event:before{content:'';position:absolute;left:-27px;top:5px;width:8px;height:8px;border-radius:50%;background:#38bdf8;box-shadow:0 0 0 4px #0b1628,0 0 14px rgba(56,189,248,.45)}.rg-event-date{color:#62748b;font-size:.69rem}.rg-event-title{color:#e2e8f0;font-weight:730;font-size:.83rem}.rg-event-detail{color:#8fa1b7;font-size:.75rem;margin-top:2px}
.rg-flow{display:grid;grid-template-columns:1fr 48px 1fr 48px 1fr;align-items:stretch;gap:8px;margin:12px 0 18px}.rg-flow-arrow{display:grid;place-items:center;color:#61728a;font-size:1.6rem}.rg-state{border-radius:16px;padding:18px}.rg-state h4{margin:0 0 8px;color:#8394aa;font-size:.71rem;text-transform:uppercase;letter-spacing:.09em}.rg-state .big{font-size:1.58rem;font-weight:870;color:#f8fafc}.rg-state p{color:#8394aa;font-size:.77rem;margin:5px 0 0}.rg-signal{border:1px solid rgba(245,158,11,.22);background:linear-gradient(180deg,rgba(245,158,11,.075),rgba(20,22,26,.72));border-radius:16px;padding:18px}.rg-signal h4{color:#fbbf24;margin:0 0 8px;font-size:.73rem;text-transform:uppercase;letter-spacing:.09em}.rg-signal p{color:#e2e8f0;font-size:.86rem}
.rg-graph{display:flex;align-items:stretch;gap:8px;overflow-x:auto;padding:16px 3px 20px}.rg-node{min-width:170px;max-width:210px;border-radius:14px;padding:14px}.rg-node-type{font-size:.63rem;color:#67e8f9;text-transform:uppercase;letter-spacing:.1em;font-weight:760}.rg-node-label{font-size:.86rem;color:#f8fafc;font-weight:770;margin-top:8px}.rg-node-sub{font-size:.70rem;color:#8394aa;margin-top:4px}.rg-edge{min-width:62px;display:flex;flex-direction:column;justify-content:center;align-items:center;color:#61728a}.rg-edge .arrow{font-size:1.5rem;line-height:1}.rg-edge .label{font-size:.58rem;text-transform:uppercase;letter-spacing:.05em;text-align:center;margin-top:5px}.rg-confidence{height:7px;border-radius:999px;background:#172235;overflow:hidden;margin-top:8px}.rg-confidence>div{height:100%;background:linear-gradient(90deg,#38bdf8,#22c55e)}
[data-testid="stDataFrame"]{border:1px solid var(--rg-border);border-radius:14px;overflow:hidden;box-shadow:0 18px 55px rgba(0,0,0,.12)}.stButton>button{border-radius:11px;font-weight:720;border:1px solid rgba(148,163,184,.18);background:rgba(15,29,50,.72);min-height:42px;transition:.16s ease}.stButton>button:hover{border-color:rgba(125,211,252,.28);transform:translateY(-1px)}.stButton>button[kind="primary"]{background:linear-gradient(100deg,#f59e0b,#fb923c);color:#111827;border:0;box-shadow:0 8px 28px rgba(245,158,11,.16)}[data-baseweb="input"]>div,[data-baseweb="select"]>div,textarea{border-radius:11px!important;background:rgba(12,24,41,.80)!important;border-color:rgba(148,163,184,.15)!important}[data-baseweb="tab-list"]{gap:8px}[data-baseweb="tab"]{border-radius:9px;padding:8px 12px}.streamlit-expanderHeader{border-radius:11px!important}hr{border-color:rgba(148,163,184,.12)!important}.rg-footer{margin-top:3rem;padding:18px 4px 0;border-top:1px solid rgba(148,163,184,.10);display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;color:#5f7188;font-size:.68rem}
@media(max-width:900px){.block-container{padding-left:.9rem;padding-right:.9rem;padding-top:.7rem}.rg-topbar{padding:9px 10px;border-radius:14px}.rg-brand-sub{max-width:230px}.rg-trust{display:none}.rg-title{font-size:1.75rem}.rg-subtitle{font-size:.86rem}.rg-flow{grid-template-columns:1fr;gap:8px}.rg-flow-arrow{height:28px;transform:rotate(90deg)}.rg-kpi{min-height:98px}.rg-kpi-value{font-size:1.65rem}}
@media(max-width:600px){.rg-brand-sub{display:none}.rg-logo{width:36px;height:36px}.rg-brand-title{font-size:1rem}.rg-status{font-size:.65rem;padding:6px 8px}.rg-control-strip{font-size:.65rem}div[data-testid="stRadio"] [role="radiogroup"] label{padding:6px 9px;min-height:34px}div[data-testid="stRadio"] [role="radiogroup"] p{font-size:.72rem}.rg-card{padding:15px}.rg-section{font-size:1.05rem}.rg-state .big{font-size:1.35rem}}

/* ---------------------------------------------------------------------
   ROJ Guard Glass UI V2 — modern enterprise SaaS / command-center layer
   --------------------------------------------------------------------- */
@keyframes rgAuroraA{0%,100%{transform:translate3d(-8%,-4%,0) scale(1)}50%{transform:translate3d(12%,8%,0) scale(1.18)}}
@keyframes rgAuroraB{0%,100%{transform:translate3d(8%,4%,0) scale(1.05)}50%{transform:translate3d(-10%,-7%,0) scale(1.22)}}
@keyframes rgPulse{0%,100%{opacity:.55;transform:scale(1)}50%{opacity:1;transform:scale(1.18)}}
@keyframes rgFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
.stApp:after{content:"";position:fixed;inset:-18%;z-index:0;pointer-events:none;background:radial-gradient(circle at 18% 20%,rgba(56,189,248,.14),transparent 23%),radial-gradient(circle at 82% 18%,rgba(245,158,11,.10),transparent 24%),radial-gradient(circle at 58% 76%,rgba(99,102,241,.11),transparent 27%);filter:blur(70px);animation:rgAuroraA 18s ease-in-out infinite alternate;opacity:.72}
.block-container{max-width:1600px!important;padding-left:1.35rem!important;padding-right:1.35rem!important}
.rg-topbar{position:sticky;top:8px;z-index:990;border-color:rgba(255,255,255,.10);background:linear-gradient(180deg,rgba(11,23,40,.84),rgba(6,14,27,.72));backdrop-filter:blur(28px) saturate(145%);-webkit-backdrop-filter:blur(28px) saturate(145%);box-shadow:0 18px 65px rgba(0,0,0,.34),inset 0 1px 0 rgba(255,255,255,.07),0 0 0 1px rgba(56,189,248,.02)}
.rg-topbar:after{content:"";position:absolute;left:18px;right:18px;bottom:-1px;height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,.35),rgba(245,158,11,.28),transparent)}
div[data-testid="stRadio"]{position:sticky;top:74px;z-index:985;margin:.35rem 0 .75rem;padding:0 1px;background:linear-gradient(180deg,rgba(5,10,19,.94),rgba(5,10,19,.78) 78%,transparent)}
div[data-testid="stRadio"] [role="radiogroup"]{border-color:rgba(255,255,255,.09)!important;background:linear-gradient(180deg,rgba(12,25,43,.82),rgba(7,16,29,.76))!important;backdrop-filter:blur(24px) saturate(145%)!important;-webkit-backdrop-filter:blur(24px) saturate(145%)!important;box-shadow:0 14px 44px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.04)}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(135deg,rgba(245,158,11,.22),rgba(56,189,248,.13))!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.06),0 8px 24px rgba(245,158,11,.08)}
.rg-logo{position:relative;overflow:hidden}.rg-logo:after{content:"";position:absolute;inset:-50%;background:linear-gradient(120deg,transparent 34%,rgba(255,255,255,.42) 50%,transparent 66%);transform:translateX(-80%) rotate(12deg);animation:rgAuroraB 5.5s ease-in-out infinite}
.rg-card,.rg-kpi,.rg-state,.rg-node{position:relative;overflow:hidden;background:linear-gradient(145deg,rgba(19,37,63,.74),rgba(8,18,33,.66));border:1px solid rgba(255,255,255,.085);backdrop-filter:blur(24px) saturate(135%);-webkit-backdrop-filter:blur(24px) saturate(135%);box-shadow:0 22px 70px rgba(0,0,0,.24),inset 0 1px 0 rgba(255,255,255,.055),inset 0 -1px 0 rgba(255,255,255,.015)}
.rg-card:before,.rg-kpi:before,.rg-state:before,.rg-node:before{content:"";position:absolute;inset:0;pointer-events:none;background:linear-gradient(120deg,rgba(255,255,255,.045),transparent 24%,transparent 76%,rgba(56,189,248,.025));opacity:.8}
.rg-card:hover,.rg-kpi:hover,.rg-node:hover{border-color:rgba(103,232,249,.22);box-shadow:0 28px 82px rgba(0,0,0,.28),0 0 34px rgba(56,189,248,.035),inset 0 1px 0 rgba(255,255,255,.07);transform:translateY(-2px)}
.rg-kpi{min-height:116px}.rg-kpi-value{background:linear-gradient(180deg,#ffffff,#d9e8f8);-webkit-background-clip:text;background-clip:text;color:transparent}
.rg-control-pill{backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);border-color:rgba(255,255,255,.085);background:rgba(12,24,41,.55)}

/* premium data surfaces */
[data-testid="stToolbar"],[data-testid="stDeployButton"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important}
[data-testid="stDataFrame"]{background:rgba(8,18,32,.58);backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);border-color:rgba(255,255,255,.09);box-shadow:0 22px 70px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.035)}
[data-testid="stDataFrame"] [role="columnheader"]{background:rgba(18,33,55,.82)!important;color:#a9bed3!important;font-weight:760!important}
.stButton>button{backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);box-shadow:inset 0 1px 0 rgba(255,255,255,.045),0 12px 34px rgba(0,0,0,.13)}
.stButton>button[kind="primary"]{background:linear-gradient(105deg,#f6b236,#fb8b4a 54%,#f59e0b);box-shadow:0 13px 38px rgba(245,158,11,.20),inset 0 1px 0 rgba(255,255,255,.35)}
.stButton>button[kind="primary"]:hover{filter:brightness(1.04);box-shadow:0 17px 48px rgba(245,158,11,.27),inset 0 1px 0 rgba(255,255,255,.40)}
[data-baseweb="input"]>div,[data-baseweb="select"]>div,textarea{background:linear-gradient(180deg,rgba(14,28,48,.78),rgba(8,18,32,.73))!important;border-color:rgba(255,255,255,.09)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.025)!important}
[data-baseweb="input"]>div:focus-within,[data-baseweb="select"]>div:focus-within,textarea:focus{border-color:rgba(56,189,248,.38)!important;box-shadow:0 0 0 3px rgba(56,189,248,.06)!important}

/* visual storytelling */
.rg-pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0 18px}.rg-pipe-step{position:relative;padding:15px 15px 14px;border:1px solid rgba(255,255,255,.08);border-radius:16px;background:linear-gradient(145deg,rgba(17,34,58,.72),rgba(8,18,32,.64));backdrop-filter:blur(20px);box-shadow:inset 0 1px 0 rgba(255,255,255,.045),0 16px 45px rgba(0,0,0,.13)}.rg-pipe-step:after{content:"";position:absolute;right:-11px;top:50%;width:11px;height:1px;background:linear-gradient(90deg,rgba(56,189,248,.52),rgba(56,189,248,.06))}.rg-pipe-step:last-child:after{display:none}.rg-pipe-num{font-size:.62rem;letter-spacing:.13em;text-transform:uppercase;color:#667d95;font-weight:800}.rg-pipe-title{font-size:.88rem;color:#ecf5ff;font-weight:800;margin-top:7px}.rg-pipe-desc{font-size:.68rem;color:#7288a0;line-height:1.45;margin-top:3px}.rg-pipe-step.active{border-color:rgba(56,189,248,.25);background:linear-gradient(145deg,rgba(25,53,80,.74),rgba(9,23,39,.72));box-shadow:0 18px 52px rgba(0,0,0,.17),0 0 40px rgba(56,189,248,.05),inset 0 1px 0 rgba(255,255,255,.06)}.rg-pipe-step.active .rg-pipe-num{color:#67e8f9}.rg-pipe-step.active:before{content:"";position:absolute;left:13px;right:13px;bottom:0;height:2px;border-radius:99px;background:linear-gradient(90deg,#38bdf8,#f59e0b)}
.rg-live-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#22c55e;margin-right:7px;box-shadow:0 0 0 5px rgba(34,197,94,.08),0 0 18px rgba(34,197,94,.75);animation:rgPulse 2s ease-in-out infinite}
.rg-hero-strip{display:grid;grid-template-columns:1.3fr .7fr;gap:12px;margin:14px 0 18px}.rg-hero-copy{padding:22px;border-radius:20px;border:1px solid rgba(255,255,255,.085);background:linear-gradient(135deg,rgba(18,38,65,.70),rgba(8,18,32,.64));backdrop-filter:blur(24px);box-shadow:0 22px 70px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.05)}.rg-hero-copy h3{margin:5px 0 8px;font-size:1.22rem;color:#f8fafc;letter-spacing:-.025em}.rg-hero-copy p{margin:0;color:#8298af;font-size:.78rem;line-height:1.55}.rg-hero-mini{display:grid;grid-template-columns:1fr 1fr;gap:10px}.rg-mini-stat{padding:16px;border-radius:17px;border:1px solid rgba(255,255,255,.08);background:linear-gradient(145deg,rgba(15,31,52,.72),rgba(7,16,29,.66));backdrop-filter:blur(20px);box-shadow:inset 0 1px 0 rgba(255,255,255,.04)}.rg-mini-stat span{display:block;color:#6d839a;font-size:.62rem;text-transform:uppercase;letter-spacing:.09em;font-weight:760}.rg-mini-stat b{display:block;color:#fff;font-size:1.15rem;margin-top:5px}.rg-mini-stat em{display:block;color:#6d839a;font-size:.64rem;font-style:normal;margin-top:2px}
.rg-donut-wrap{display:grid;grid-template-columns:160px 1fr;align-items:center;gap:24px}.rg-donut{width:156px;height:156px;border-radius:50%;position:relative;box-shadow:0 18px 55px rgba(0,0,0,.23),inset 0 0 22px rgba(255,255,255,.02)}.rg-donut:after{content:"";position:absolute;inset:18px;border-radius:50%;background:linear-gradient(160deg,rgba(9,21,37,.98),rgba(5,13,25,.98));border:1px solid rgba(255,255,255,.055);box-shadow:inset 0 1px 0 rgba(255,255,255,.035)}.rg-donut-center{position:absolute;z-index:2;inset:0;display:grid;place-items:center;text-align:center}.rg-donut-center b{display:block;font-size:1.65rem;color:#fff;line-height:1}.rg-donut-center span{display:block;color:#6e849a;font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;margin-top:5px}.rg-legend-row{display:flex;align-items:center;justify-content:space-between;padding:9px 0;border-bottom:1px solid rgba(255,255,255,.05);gap:14px}.rg-legend-row:last-child{border-bottom:0}.rg-legend-label{display:flex;align-items:center;gap:9px;color:#cbd8e6;font-size:.76rem}.rg-legend-value{font-size:.78rem;color:#fff;font-weight:820}
.rg-risk-cockpit{display:grid;grid-template-columns:200px 1fr;gap:18px;align-items:stretch;margin:13px 0 15px}.rg-risk-ring-card{display:grid;place-items:center;border-radius:20px;border:1px solid rgba(255,255,255,.085);background:linear-gradient(145deg,rgba(16,34,58,.74),rgba(8,18,32,.66));backdrop-filter:blur(24px);box-shadow:0 22px 70px rgba(0,0,0,.20),inset 0 1px 0 rgba(255,255,255,.05);padding:18px}.rg-risk-ring{width:142px;height:142px;border-radius:50%;position:relative;display:grid;place-items:center}.rg-risk-ring:after{content:"";position:absolute;inset:14px;border-radius:50%;background:#081423;border:1px solid rgba(255,255,255,.06);box-shadow:inset 0 10px 30px rgba(0,0,0,.26)}.rg-risk-ring-inner{position:relative;z-index:2;text-align:center}.rg-risk-ring-inner b{font-size:1.78rem;color:#fff}.rg-risk-ring-inner span{display:block;font-size:.64rem;text-transform:uppercase;letter-spacing:.08em;color:#71869d;margin-top:2px}.rg-risk-copy{padding:20px;border-radius:20px;border:1px solid rgba(255,255,255,.085);background:linear-gradient(145deg,rgba(17,35,59,.70),rgba(8,18,32,.65));backdrop-filter:blur(24px);box-shadow:0 22px 70px rgba(0,0,0,.17),inset 0 1px 0 rgba(255,255,255,.045)}.rg-risk-copy h3{font-size:1.23rem;margin:4px 0 6px;color:#f8fafc}.rg-risk-copy p{font-size:.78rem;color:#8095ac;line-height:1.55;margin:0}.rg-risk-tags{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}
.rg-glass-banner{position:relative;overflow:hidden;padding:18px 20px;border:1px solid rgba(255,255,255,.085);border-radius:19px;background:linear-gradient(130deg,rgba(18,38,65,.70),rgba(8,18,32,.62));backdrop-filter:blur(24px);box-shadow:0 20px 65px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.05);margin:9px 0 16px}.rg-glass-banner:after{content:"";position:absolute;width:240px;height:240px;border-radius:50%;right:-80px;top:-120px;background:radial-gradient(circle,rgba(56,189,248,.14),transparent 68%);filter:blur(3px)}.rg-glass-banner h3{margin:4px 0 7px;color:#f8fafc;font-size:1.05rem}.rg-glass-banner p{margin:0;color:#7f94ab;font-size:.76rem;line-height:1.5;max-width:900px}.rg-banner-badges{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.rg-banner-badge{padding:5px 9px;border-radius:999px;background:rgba(56,189,248,.07);border:1px solid rgba(56,189,248,.14);font-size:.66rem;color:#9bdff7}
.rg-flow .rg-state,.rg-flow .rg-signal{box-shadow:0 22px 70px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.045);backdrop-filter:blur(22px)}.rg-signal{position:relative;overflow:hidden}.rg-signal:after{content:"";position:absolute;inset:auto -40px -70px auto;width:180px;height:180px;border-radius:50%;background:radial-gradient(circle,rgba(245,158,11,.13),transparent 67%)}
.rg-section{margin-top:1.45rem}.rg-title{font-size:2.2rem}.rg-subtitle{max-width:980px}

@media(max-width:980px){.rg-pipeline{grid-template-columns:repeat(2,1fr)}.rg-pipe-step:after{display:none}.rg-hero-strip{grid-template-columns:1fr}.rg-donut-wrap{grid-template-columns:140px 1fr}.rg-donut{width:136px;height:136px}.rg-risk-cockpit{grid-template-columns:170px 1fr}.rg-risk-ring{width:126px;height:126px}}
@media(max-width:700px){.block-container{padding-left:.72rem!important;padding-right:.72rem!important}.rg-topbar{top:5px}.rg-status-wrap .rg-trust{display:none}div[data-testid="stRadio"]{top:66px}.rg-pipeline{grid-template-columns:1fr 1fr;gap:7px}.rg-pipe-step{padding:12px}.rg-hero-mini{grid-template-columns:1fr 1fr}.rg-donut-wrap{grid-template-columns:1fr;text-align:center}.rg-donut{margin:0 auto}.rg-risk-cockpit{grid-template-columns:1fr}.rg-risk-ring-card{padding:14px}.rg-glass-banner{padding:16px}.rg-title{font-size:1.82rem}}
@media(max-width:440px){.rg-pipeline{grid-template-columns:1fr}.rg-hero-mini{grid-template-columns:1fr 1fr}.rg-mini-stat{padding:12px}.rg-title{font-size:1.65rem}.rg-subtitle{font-size:.82rem}}



/* ========================================================================
   ROJ GUARD — FINAL CINEMATIC UI V3
   Floating command dock • glassmorphism • spatial depth • motion system
   ======================================================================== */
:root{
  --rg-ink:#eaf4ff;--rg-ice:#9bdcff;--rg-sky:#4cc9ff;--rg-amber:#ffb646;
  --rg-violet:#7c5cff;--rg-deep:#030711;--rg-panel:rgba(9,18,33,.58);
  --rg-glass:rgba(13,27,48,.55);--rg-line:rgba(180,220,255,.10);
  --rg-shadow:0 28px 90px rgba(0,0,0,.42);
}
@keyframes rgAuroraA{0%,100%{transform:translate3d(-4%,-2%,0) scale(1)}50%{transform:translate3d(6%,5%,0) scale(1.14)}}
@keyframes rgFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@keyframes rgRise{from{opacity:0;transform:translateY(18px) scale(.985)}to{opacity:1;transform:translateY(0) scale(1)}}
@keyframes rgSweep{0%{transform:translateX(-120%) skewX(-18deg)}100%{transform:translateX(260%) skewX(-18deg)}}
@keyframes rgGlowPulse{0%,100%{box-shadow:0 0 0 0 rgba(56,189,248,.0),0 16px 55px rgba(0,0,0,.20)}50%{box-shadow:0 0 0 5px rgba(56,189,248,.035),0 20px 70px rgba(0,0,0,.28)}}
@keyframes rgOrbit{to{transform:rotate(360deg)}}
@keyframes rgBlink{0%,100%{opacity:.35}50%{opacity:1}}
@keyframes rgGradientShift{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}

.stApp{overflow-x:hidden;background:
  radial-gradient(900px 520px at 8% -5%,rgba(25,143,255,.15),transparent 68%),
  radial-gradient(760px 520px at 94% 2%,rgba(124,92,255,.13),transparent 68%),
  radial-gradient(740px 520px at 78% 92%,rgba(255,182,70,.09),transparent 70%),
  linear-gradient(180deg,#030711 0%,#050a14 40%,#030711 100%)!important;}
.stApp:after{content:"";position:fixed;inset:-18vh -14vw;z-index:0;pointer-events:none;opacity:.82;background:
  radial-gradient(circle at 22% 28%,rgba(55,190,255,.085),transparent 22%),
  radial-gradient(circle at 68% 24%,rgba(124,92,255,.075),transparent 20%),
  radial-gradient(circle at 72% 72%,rgba(255,182,70,.055),transparent 24%);
  filter:blur(18px);animation:rgAuroraA 17s ease-in-out infinite alternate;}
.block-container{max-width:1600px!important;padding-top:1.1rem!important;padding-left:clamp(.8rem,2.4vw,2.5rem)!important;padding-right:clamp(.8rem,2.4vw,2.5rem)!important;}

/* Floating top command surface */
.rg-topbar{position:sticky!important;top:14px!important;z-index:999!important;margin:0 auto 12px!important;max-width:1480px!important;padding:13px 16px!important;border-radius:24px!important;
  border:1px solid rgba(196,226,255,.13)!important;background:linear-gradient(135deg,rgba(11,23,42,.76),rgba(7,16,30,.60))!important;
  backdrop-filter:blur(34px) saturate(175%)!important;-webkit-backdrop-filter:blur(34px) saturate(175%)!important;
  box-shadow:0 26px 90px rgba(0,0,0,.48),0 8px 28px rgba(9,120,255,.07),inset 0 1px 0 rgba(255,255,255,.085),inset 0 -1px 0 rgba(255,255,255,.025)!important;
  animation:rgRise .65s cubic-bezier(.2,.8,.2,1) both;isolation:isolate;}
.rg-topbar:before{content:"";position:absolute;inset:-1px;border-radius:24px;z-index:-1;padding:1px;background:linear-gradient(110deg,rgba(76,201,255,.28),transparent 24%,transparent 72%,rgba(255,182,70,.23));
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;}
.rg-topbar:after{content:""!important;position:absolute!important;left:12%!important;right:12%!important;bottom:-16px!important;height:18px!important;background:radial-gradient(ellipse at center,rgba(63,169,255,.14),transparent 70%)!important;filter:blur(8px);pointer-events:none;}
.rg-logo{position:relative;overflow:hidden;width:44px!important;height:44px!important;border-radius:15px!important;background:linear-gradient(135deg,#ffbe52 0%,#8ce9ff 58%,#6f77ff 100%)!important;box-shadow:0 12px 34px rgba(85,203,255,.20),inset 0 1px 0 rgba(255,255,255,.55)!important;}
.rg-logo:after{content:"";position:absolute;inset:-45% auto -45% -45%;width:35%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.72),transparent);animation:rgSweep 4.6s ease-in-out infinite;}
.rg-brand-title{font-size:1.16rem!important;letter-spacing:-.035em!important}.rg-brand-sub{color:#6f849c!important;letter-spacing:.01em}
.rg-status{background:rgba(34,197,94,.07)!important;border-color:rgba(92,255,153,.18)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.025)}
.rg-status-dot{animation:rgBlink 1.8s ease-in-out infinite}.rg-control-pill{background:rgba(255,255,255,.025)!important;border-color:rgba(255,255,255,.075)!important;color:#8096ad!important;}

/* Floating navigation dock */
div[data-testid="stRadio"]{position:sticky!important;top:88px!important;z-index:990!important;margin:0 auto 18px!important;max-width:1080px!important;filter:drop-shadow(0 18px 32px rgba(0,0,0,.30));animation:rgRise .72s .06s cubic-bezier(.2,.8,.2,1) both;}
div[data-testid="stRadio"] [role="radiogroup"]{justify-content:center!important;min-width:0!important;width:100%!important;padding:7px!important;border-radius:18px!important;border:1px solid rgba(185,220,255,.105)!important;background:linear-gradient(180deg,rgba(10,21,38,.76),rgba(6,15,28,.66))!important;backdrop-filter:blur(28px) saturate(160%)!important;-webkit-backdrop-filter:blur(28px) saturate(160%)!important;box-shadow:0 22px 55px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.055)!important;}
div[data-testid="stRadio"] [role="radiogroup"] label{position:relative;padding:8px 15px!important;border-radius:12px!important;overflow:hidden;}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(135deg,rgba(42,149,255,.19),rgba(124,92,255,.12),rgba(255,182,70,.10))!important;border-color:rgba(106,205,255,.20)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.07),0 8px 24px rgba(0,0,0,.13)!important;}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked):after{content:"";position:absolute;left:18%;right:18%;bottom:2px;height:2px;border-radius:8px;background:linear-gradient(90deg,#4cc9ff,#8a7cff,#ffb646);box-shadow:0 0 14px rgba(76,201,255,.55)}
.rg-control-strip{justify-content:center;opacity:.78;margin-top:-4px!important;margin-bottom:18px!important;}

/* Cinematic typography */
.rg-eyebrow{display:inline-flex;align-items:center;gap:7px;color:#69d8ff!important;letter-spacing:.17em!important;font-size:.64rem!important;}
.rg-title{font-size:clamp(2rem,3.2vw,3.35rem)!important;line-height:.98!important;letter-spacing:-.058em!important;margin-top:.45rem!important;background:linear-gradient(110deg,#ffffff 0%,#dceeff 44%,#9cdcff 72%,#ffffff 100%);background-size:220% 100%;-webkit-background-clip:text;background-clip:text;color:transparent!important;animation:rgGradientShift 8s ease-in-out infinite;}
.rg-subtitle{font-size:clamp(.86rem,1.05vw,1rem)!important;color:#8096ad!important;max-width:1020px!important;}
.rg-section{position:relative;padding-left:13px;margin-top:1.8rem!important}.rg-section:before{content:"";position:absolute;left:0;top:.2em;bottom:.2em;width:3px;border-radius:8px;background:linear-gradient(#4cc9ff,#7c5cff,#ffb646);box-shadow:0 0 18px rgba(76,201,255,.32)}

/* High fidelity glass system */
.rg-card,.rg-kpi,.rg-state,.rg-node,.rg-risk-ring-card,.rg-risk-copy,.rg-glass-banner,.rg-pipe-step{position:relative;overflow:hidden;border-color:rgba(185,220,255,.10)!important;background:linear-gradient(145deg,rgba(15,31,54,.60),rgba(6,16,30,.50))!important;backdrop-filter:blur(28px) saturate(150%)!important;-webkit-backdrop-filter:blur(28px) saturate(150%)!important;box-shadow:0 22px 75px rgba(0,0,0,.25),inset 0 1px 0 rgba(255,255,255,.055)!important;animation:rgRise .65s cubic-bezier(.2,.8,.2,1) both;}
.rg-card:before,.rg-kpi:before,.rg-glass-banner:before,.rg-risk-copy:before{content:"";position:absolute;left:8%;right:8%;top:0;height:1px;background:linear-gradient(90deg,transparent,rgba(128,220,255,.42),rgba(255,255,255,.22),transparent);opacity:.7}
.rg-card:hover,.rg-kpi:hover,.rg-pipe-step:hover,.rg-risk-copy:hover{transform:translateY(-4px) perspective(800px) rotateX(.6deg)!important;border-color:rgba(91,197,255,.22)!important;box-shadow:0 32px 100px rgba(0,0,0,.34),0 0 36px rgba(76,201,255,.045),inset 0 1px 0 rgba(255,255,255,.08)!important;transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease!important;}
.rg-kpi{min-height:122px!important}.rg-kpi-value{font-size:2.05rem!important}.rg-kpi-label{color:#91a7bd!important}.rg-kpi-note{color:#5f7690!important}
.rg-driver-item{transition:.2s ease;border-color:rgba(185,220,255,.08)!important;background:rgba(9,20,36,.42)!important}.rg-driver-item:hover{transform:translateX(4px);border-color:rgba(76,201,255,.17)!important;background:rgba(12,28,49,.56)!important}

/* Animated intelligence rail */
.rg-signal-rail{position:relative;display:flex;align-items:center;gap:9px;overflow:hidden;margin:10px 0 18px;padding:10px 12px;border-radius:15px;border:1px solid rgba(185,220,255,.08);background:rgba(6,14,27,.46);backdrop-filter:blur(18px)}
.rg-signal-rail:before{content:"";position:absolute;top:0;bottom:0;left:-20%;width:18%;background:linear-gradient(90deg,transparent,rgba(76,201,255,.08),transparent);animation:rgSweep 5.5s linear infinite}
.rg-signal-pulse{width:8px;height:8px;border-radius:50%;background:#4cc9ff;box-shadow:0 0 0 5px rgba(76,201,255,.07),0 0 22px rgba(76,201,255,.70);animation:rgBlink 1.7s infinite}.rg-signal-copy{font-size:.72rem;color:#7f96ad}.rg-signal-copy b{color:#cfeeff}.rg-signal-sep{color:#30445b}

/* Interactive panel accents */
[data-testid="stDataFrame"]{border-radius:18px!important;border:1px solid rgba(185,220,255,.09)!important;overflow:hidden;box-shadow:0 26px 80px rgba(0,0,0,.27)!important;animation:rgRise .7s ease both}
[data-testid="stExpander"]{border-radius:17px!important;border:1px solid rgba(185,220,255,.09)!important;background:rgba(8,19,35,.46)!important;backdrop-filter:blur(22px)!important;box-shadow:0 18px 55px rgba(0,0,0,.18)!important;overflow:hidden}
[data-testid="stFileUploader"] section{border-radius:18px!important;border:1px dashed rgba(83,197,255,.20)!important;background:linear-gradient(145deg,rgba(15,31,52,.54),rgba(7,17,31,.44))!important;}
.stButton>button{border-radius:14px!important;transition:transform .20s ease,box-shadow .20s ease,filter .20s ease!important;min-height:45px}.stButton>button:hover{transform:translateY(-2px)}
.stButton>button[kind="primary"]{position:relative;overflow:hidden!important;background:linear-gradient(110deg,#ffbc46 0%,#ff8d4f 45%,#e87c58 100%)!important;color:#0a111c!important;font-weight:800!important;box-shadow:0 14px 42px rgba(255,146,68,.20),inset 0 1px 0 rgba(255,255,255,.45)!important}
.stButton>button[kind="primary"]:before{content:"";position:absolute;inset:-60% auto -60% -40%;width:28%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.55),transparent);animation:rgSweep 4s ease-in-out infinite}

/* Live demo visual theatre */
.rg-flow{perspective:1200px}.rg-state,.rg-signal{transition:.28s ease;animation:rgRise .68s ease both}.rg-state:hover,.rg-signal:hover{transform:translateY(-5px) rotateX(1.2deg)}
.rg-signal{border-color:rgba(255,182,70,.22)!important;box-shadow:0 24px 80px rgba(0,0,0,.25),0 0 50px rgba(255,182,70,.045)!important}
.rg-flow-arrow{animation:rgFloat 2.4s ease-in-out infinite;color:#607d98!important}
.rg-event:before{animation:rgGlowPulse 2.2s ease-in-out infinite}.rg-timeline{border-left-color:rgba(76,201,255,.18)!important}

@media(max-width:980px){div[data-testid="stRadio"]{max-width:100%!important;top:86px!important}.rg-topbar{top:8px!important}.rg-status-wrap .rg-control-pill{display:none!important}.rg-title{font-size:2.25rem!important}}
@media(max-width:700px){.block-container{padding-top:.55rem!important}.rg-topbar{top:6px!important;padding:10px 11px!important;border-radius:18px!important}.rg-brand-sub{display:none!important}.rg-logo{width:38px!important;height:38px!important;border-radius:13px!important}.rg-status-wrap .rg-trust{display:none!important}div[data-testid="stRadio"]{top:67px!important;margin-bottom:12px!important}div[data-testid="stRadio"] [role="radiogroup"]{justify-content:flex-start!important;overflow-x:auto!important;width:max-content!important;min-width:100%!important}.rg-control-strip{display:none!important}.rg-title{font-size:1.95rem!important}.rg-card:hover,.rg-kpi:hover,.rg-pipe-step:hover,.rg-risk-copy:hover{transform:none!important}.rg-signal-rail{overflow-x:auto;white-space:nowrap}}
@media(max-width:430px){.rg-topbar .rg-status{padding:6px 8px!important}.rg-status-wrap{gap:5px!important}.rg-brand-title{font-size:1rem!important}.rg-title{font-size:1.72rem!important}.rg-subtitle{font-size:.80rem!important}}

/* ========================================================================== 
   FINAL JUDGE DASHBOARD OVERRIDES
   ========================================================================== */
.block-container{max-width:1660px!important;padding-left:clamp(.8rem,2.4vw,2.5rem)!important;padding-right:clamp(.8rem,2.4vw,2.5rem)!important;padding-top:.75rem!important}
.rg-title{font-size:clamp(1.55rem,2vw,2.05rem)!important;line-height:1.08!important;letter-spacing:-.038em!important;margin:.24rem 0 .26rem!important;background:none!important;color:#f4f8fc!important;animation:none!important}
.rg-subtitle{font-size:clamp(.82rem,.92vw,.94rem)!important;line-height:1.58!important;margin-bottom:.9rem!important;max-width:980px!important;color:#8194aa!important}
.rg-eyebrow{font-size:.60rem!important;letter-spacing:.15em!important}.rg-section{font-size:1.02rem!important;margin-top:1.25rem!important}
.rg-topbar{position:sticky!important;top:10px!important;z-index:1000!important;max-width:1580px!important;margin:0 auto 7px!important;border-radius:22px!important;padding:10px 14px!important;background:linear-gradient(180deg,rgba(10,22,40,.78),rgba(5,14,28,.68))!important;border:1px solid rgba(169,211,244,.12)!important;backdrop-filter:blur(30px) saturate(160%)!important;-webkit-backdrop-filter:blur(30px) saturate(160%)!important;box-shadow:0 26px 70px rgba(0,0,0,.34),0 8px 28px rgba(38,125,195,.06),inset 0 1px 0 rgba(255,255,255,.07)!important}
.rg-topbar:after{content:"";position:absolute;left:5%;right:5%;bottom:-1px;height:1px;background:linear-gradient(90deg,transparent,rgba(76,201,255,.28),rgba(151,116,255,.20),transparent);pointer-events:none}.rg-brand-sub{font-size:.64rem!important}.rg-trust{font-size:.65rem!important}
div[data-testid="stRadio"]{position:sticky!important;top:78px!important;z-index:995!important;max-width:1580px!important;width:100%!important;margin:0 auto 15px!important;filter:drop-shadow(0 20px 36px rgba(0,0,0,.28))!important}
div[data-testid="stRadio"]>div{overflow-x:auto!important;overflow-y:hidden!important;scrollbar-width:none!important}div[data-testid="stRadio"]>div::-webkit-scrollbar{display:none!important}
div[data-testid="stRadio"] [role="radiogroup"]{display:flex!important;flex-direction:row!important;flex-wrap:nowrap!important;align-items:center!important;justify-content:space-between!important;gap:4px!important;width:100%!important;min-width:1120px!important;padding:6px!important;border-radius:18px!important;background:linear-gradient(180deg,rgba(9,20,36,.82),rgba(5,14,27,.72))!important;border:1px solid rgba(169,211,244,.11)!important;backdrop-filter:blur(28px) saturate(160%)!important;-webkit-backdrop-filter:blur(28px) saturate(160%)!important;box-shadow:0 20px 55px rgba(0,0,0,.26),inset 0 1px 0 rgba(255,255,255,.055)!important}
div[data-testid="stRadio"] [role="radiogroup"] label{flex:1 1 auto!important;min-width:max-content!important;justify-content:center!important;padding:7px 10px!important;border-radius:11px!important;white-space:nowrap!important}div[data-testid="stRadio"] [role="radiogroup"] p{font-size:clamp(.64rem,.69vw,.74rem)!important;font-weight:660!important;white-space:nowrap!important}
.rg-pipeline,.rg-pipe-step,.rg-control-strip,.rg-signal-rail{display:none!important}
.rg-dashboard-label{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:2px 0 9px}.rg-dashboard-title{font-size:1.62rem;font-weight:820;letter-spacing:-.04em;color:#f4f8fc}.rg-dashboard-meta{font-size:.70rem;color:#6f839a}
.rg-dashboard-grid{display:grid;grid-template-columns:1.02fr .98fr 1.08fr;gap:12px;margin:12px 0 14px}.rg-dashboard-panel{min-height:252px;padding:17px 18px;border-radius:19px;border:1px solid rgba(176,216,248,.10);background:linear-gradient(145deg,rgba(13,29,50,.62),rgba(5,15,28,.54));backdrop-filter:blur(24px) saturate(145%);-webkit-backdrop-filter:blur(24px) saturate(145%);box-shadow:0 22px 65px rgba(0,0,0,.23),inset 0 1px 0 rgba(255,255,255,.045);position:relative;overflow:hidden}.rg-dashboard-panel:before{content:"";position:absolute;left:8%;right:8%;top:0;height:1px;background:linear-gradient(90deg,transparent,rgba(104,208,255,.38),transparent)}
.rg-dash-panel-title{font-size:.86rem;font-weight:780;color:#eef6ff}.rg-dash-panel-sub{font-size:.67rem;color:#657b93;margin-top:3px;margin-bottom:13px}
.rg-exposure-row{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid rgba(255,255,255,.05)}.rg-exposure-row:last-child{border-bottom:0}.rg-exposure-name{font-size:.73rem;color:#a7b8ca}.rg-exposure-name small{display:block;color:#5f748c;font-size:.62rem;margin-top:2px}.rg-exposure-value{font-size:1.04rem;font-weight:830;color:#f4f8fc}.rg-meter{height:5px;border-radius:99px;background:rgba(95,116,140,.16);overflow:hidden;margin-top:6px}.rg-meter span{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#42c6ff,#7d78ff,#ffb44a);box-shadow:0 0 14px rgba(66,198,255,.22)}
.rg-priority-list{display:grid;gap:7px}.rg-priority-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:center;padding:9px 10px;border:1px solid rgba(255,255,255,.055);border-radius:11px;background:rgba(6,16,30,.35)}.rg-priority-name{font-size:.72rem;font-weight:720;color:#dbe9f6;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.rg-priority-meta{font-size:.61rem;color:#60758d;margin-top:2px}.rg-priority-score{font-size:.88rem;font-weight:840;color:#fff}.rg-risk-mini{height:3px;border-radius:99px;background:rgba(255,255,255,.07);margin-top:5px;overflow:hidden}.rg-risk-mini span{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#ffb24a,#ff6a6a)}
.rg-activity-compact{display:grid;gap:8px}.rg-activity-row{display:grid;grid-template-columns:8px 1fr;gap:9px;align-items:start}.rg-activity-dot{width:7px;height:7px;margin-top:5px;border-radius:50%;background:#4cc9ff;box-shadow:0 0 0 4px rgba(76,201,255,.06),0 0 13px rgba(76,201,255,.36)}.rg-activity-title{font-size:.70rem;color:#d7e5f2;font-weight:680}.rg-activity-detail{font-size:.61rem;color:#60758d;line-height:1.4;margin-top:2px}
.rg-compact-stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}.rg-compact-stat{padding:10px 11px;border-radius:12px;border:1px solid rgba(255,255,255,.06);background:rgba(8,20,36,.42)}.rg-compact-stat span{display:block;font-size:.57rem;letter-spacing:.08em;text-transform:uppercase;color:#60758d}.rg-compact-stat b{display:block;font-size:1.02rem;color:#f4f8fc;margin-top:4px}
.rg-card{padding:15px 16px!important}.rg-kpi{min-height:96px!important;padding:13px 14px!important}.rg-kpi-value{font-size:1.62rem!important}.rg-kpi-note{font-size:.64rem!important}.rg-panel-sub{margin-bottom:10px!important}.rg-glass-banner{padding:15px 17px!important;margin:7px 0 12px!important}.rg-glass-banner h3{font-size:.94rem!important}.rg-glass-banner p{font-size:.72rem!important}
@media(max-width:1180px){.rg-dashboard-grid{grid-template-columns:1fr 1fr}.rg-dashboard-panel:last-child{grid-column:1/-1}.rg-status-wrap .rg-control-pill{display:none!important}}
@media(max-width:760px){.block-container{padding-left:.65rem!important;padding-right:.65rem!important}.rg-topbar{top:6px!important;border-radius:18px!important}.rg-topbar .rg-trust{display:none!important}div[data-testid="stRadio"]{top:65px!important;margin-bottom:10px!important}div[data-testid="stRadio"] [role="radiogroup"]{min-width:980px!important;justify-content:flex-start!important}.rg-dashboard-grid{grid-template-columns:1fr}.rg-dashboard-panel:last-child{grid-column:auto}.rg-dashboard-panel{min-height:0}.rg-compact-stat-row{grid-template-columns:1fr 1fr}.rg-dashboard-title{font-size:1.38rem}.rg-dashboard-meta{display:none}.rg-title{font-size:1.42rem!important}}

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



def render_risk_donut(high, medium, low):
    total = max(high + medium + low, 1)
    hp = high / total * 100
    mp = medium / total * 100
    h_end = hp
    m_end = hp + mp
    st.markdown(
        f'''<div class="rg-card"><div class="rg-panel-title">Portfolio risk mix</div><div class="rg-panel-sub">Live distribution across active procurement lines</div>
        <div class="rg-donut-wrap">
          <div class="rg-donut" style="background:conic-gradient(#ef4444 0 {h_end:.1f}%,#f59e0b {h_end:.1f}% {m_end:.1f}%,#22c55e {m_end:.1f}% 100%)"><div class="rg-donut-center"><div><b>{total}</b><span>active lines</span></div></div></div>
          <div>
            <div class="rg-legend-row"><div class="rg-legend-label"><span class="rg-dot" style="background:#ef4444"></span>High risk</div><div class="rg-legend-value">{high} · {hp:.0f}%</div></div>
            <div class="rg-legend-row"><div class="rg-legend-label"><span class="rg-dot" style="background:#f59e0b"></span>Medium risk</div><div class="rg-legend-value">{medium} · {mp:.0f}%</div></div>
            <div class="rg-legend-row"><div class="rg-legend-label"><span class="rg-dot" style="background:#22c55e"></span>Low risk</div><div class="rg-legend-value">{low} · {low/total*100:.0f}%</div></div>
          </div>
        </div></div>''', unsafe_allow_html=True)


def render_risk_cockpit(material, vendor, risk, shipment, schedule):
    prob = float(risk.get("miss_roj_probability") or 0) * 100
    level = risk.get("risk_level") or "Unknown"
    color = RISK_COLORS.get(level, "#38bdf8")
    delay = risk.get("predicted_delay_days")
    state = (shipment.get("status") or "unknown").replace("_", " ").title()
    crit = "Critical path" if schedule.get("is_critical_path") else "Non-critical path"
    vendor_name = vendor.get("name") or "Vendor unavailable"
    sku = material.get("sku") or "—"
    st.markdown(
        f'''<div class="rg-risk-cockpit">
        <div class="rg-risk-ring-card"><div class="rg-risk-ring" style="background:conic-gradient({color} 0 {prob:.1f}%,rgba(51,65,85,.42) {prob:.1f}% 100%)"><div class="rg-risk-ring-inner"><b>{prob:.0f}%</b><span>miss ROJ</span></div></div><div style="margin-top:10px">{risk_chip(level)}</div></div>
        <div class="rg-risk-copy"><div class="rg-eyebrow"><span class="rg-live-dot"></span>Live material intelligence</div><h3>{clean_name(material.get("name"))}</h3><p>{vendor_name} · {clean_sku(sku)} · {state}. Forecast delay is <b style="color:#e8f4ff">{delay if delay is not None else '—'} day(s)</b> against the Required-On-Job date.</p><div class="rg-risk-tags"><span class="rg-chip rg-neutral">{crit}</span><span class="rg-chip rg-neutral">ROJ {risk.get("roj_date") or '—'}</span><span class="rg-chip rg-neutral">Forecast {risk.get("predicted_arrival_date") or '—'}</span></div></div>
        </div>''', unsafe_allow_html=True)


def render_glass_banner(title, body, badges=None, eyebrow="Live intelligence"):
    badges = badges or []
    badge_html = ''.join(f'<span class="rg-banner-badge">{b}</span>' for b in badges)
    st.markdown(
        f'<div class="rg-glass-banner"><div class="rg-eyebrow"><span class="rg-live-dot"></span>{eyebrow}</div><h3>{title}</h3><p>{body}</p><div class="rg-banner-badges">{badge_html}</div></div>',
        unsafe_allow_html=True,
    )



def render_signal_rail(items=None):
    items = items or [
        "Vendor signals linked to active materials",
        "Lead-time forecast online",
        "ROJ risk continuously recomputed",
        "Human approval boundary active",
    ]
    content = '<span class="rg-signal-pulse"></span>'
    for i, item in enumerate(items[:5]):
        if i:
            content += '<span class="rg-signal-sep">•</span>'
        content += f'<span class="rg-signal-copy">{item}</span>'
    st.markdown(f'<div class="rg-signal-rail">{content}</div>', unsafe_allow_html=True)


def render_3d_hero(title, subtitle, metrics=None, accent="#4cc9ff", mode="overview", compact=False):
    metrics = metrics or []
    metric_html = "".join(
        f'<div class="metric"><span>{str(label)}</span><b>{str(value)}</b></div>'
        for label, value in metrics[:4]
    )
    cfg = json.dumps({"mode": mode, "accent": accent})
    height = 250 if compact else 382
    iframe_h = height + 12
    html = f'''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
*{{box-sizing:border-box}}html,body{{margin:0;height:100%;overflow:hidden;background:transparent;font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;color:#eef7ff}}
.scene{{height:{height}px;position:relative;overflow:hidden;border-radius:28px;border:1px solid rgba(190,224,255,.12);background:linear-gradient(135deg,rgba(8,19,36,.96),rgba(3,9,18,.98));box-shadow:0 36px 110px rgba(0,0,0,.36),inset 0 1px 0 rgba(255,255,255,.065)}}
.scene:before{{content:"";position:absolute;inset:0;z-index:1;pointer-events:none;background:radial-gradient(circle at 75% 25%,rgba(76,201,255,.10),transparent 30%),radial-gradient(circle at 88% 78%,rgba(124,92,255,.08),transparent 26%),linear-gradient(90deg,rgba(3,8,16,.97) 0%,rgba(3,8,16,.88) 38%,rgba(3,8,16,.26) 66%,rgba(3,8,16,.03) 100%)}}
.scene:after{{content:"";position:absolute;inset:0;z-index:1;pointer-events:none;opacity:.14;background-image:linear-gradient(rgba(126,184,226,.14) 1px,transparent 1px),linear-gradient(90deg,rgba(126,184,226,.14) 1px,transparent 1px);background-size:52px 52px;mask-image:linear-gradient(90deg,transparent 12%,#000 60%,#000)}}
#three{{position:absolute;inset:0}}canvas{{width:100%!important;height:100%!important}}
.copy{{position:absolute;z-index:3;left:clamp(22px,3.4vw,50px);top:clamp(24px,4vw,48px);max-width:min(690px,58%)}}
.kicker{{display:inline-flex;align-items:center;gap:8px;color:#78dcff;text-transform:uppercase;letter-spacing:.17em;font-size:10px;font-weight:800}}.kicker i{{width:7px;height:7px;border-radius:50%;background:#4cc9ff;box-shadow:0 0 0 5px rgba(76,201,255,.07),0 0 20px rgba(76,201,255,.75);animation:pulse 1.8s ease-in-out infinite}}
h1{{font-size:clamp(30px,3.7vw,54px);line-height:.98;letter-spacing:-.055em;margin:14px 0 12px;max-width:700px;background:linear-gradient(110deg,#fff,#ddecff 48%,#9edfff 76%,#fff);background-size:220% 100%;-webkit-background-clip:text;color:transparent;animation:shift 8s ease-in-out infinite}}p{{font-size:13px;line-height:1.62;color:#8ba0b7;margin:0;max-width:610px}}
.metrics{{display:flex;gap:9px;flex-wrap:wrap;margin-top:22px}}.metric{{min-width:112px;padding:10px 12px;border-radius:14px;border:1px solid rgba(185,220,255,.11);background:linear-gradient(145deg,rgba(17,34,58,.50),rgba(7,17,31,.40));backdrop-filter:blur(16px);box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 15px 35px rgba(0,0,0,.14);animation:float 4.5s ease-in-out infinite}}.metric:nth-child(2){{animation-delay:.5s}}.metric:nth-child(3){{animation-delay:1s}}.metric:nth-child(4){{animation-delay:1.5s}}.metric span{{display:block;color:#677d94;font-size:8px;text-transform:uppercase;letter-spacing:.10em;font-weight:750}}.metric b{{display:block;color:#f8fbff;font-size:17px;margin-top:4px}}
.hud{{position:absolute;z-index:3;right:22px;top:20px;display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end;max-width:340px}}.pill{{padding:6px 9px;border-radius:999px;border:1px solid rgba(185,220,255,.10);background:rgba(5,14,27,.52);backdrop-filter:blur(12px);font-size:9px;color:#7f97ae}}.legend{{position:absolute;z-index:3;right:22px;bottom:18px;display:flex;align-items:center;gap:8px;padding:8px 10px;border:1px solid rgba(76,201,255,.12);border-radius:999px;background:rgba(5,14,27,.52);backdrop-filter:blur(12px);font-size:9px;color:#91bad0}}.legend i{{width:18px;height:1px;background:linear-gradient(90deg,#4cc9ff,#7c5cff,#ffb646);box-shadow:0 0 12px rgba(76,201,255,.35)}}
.label{{position:absolute;z-index:3;padding:6px 9px;border-radius:10px;border:1px solid rgba(185,220,255,.10);background:rgba(5,14,27,.62);backdrop-filter:blur(12px);font-size:9px;color:#9fb5ca;box-shadow:0 14px 40px rgba(0,0,0,.22);animation:float 5s ease-in-out infinite}}.l1{{right:24%;top:22%}}.l2{{right:10%;top:47%;animation-delay:.7s}}.l3{{right:30%;bottom:16%;animation-delay:1.3s}}
@keyframes pulse{{0%,100%{{opacity:.45;transform:scale(.92)}}50%{{opacity:1;transform:scale(1.12)}}}}@keyframes float{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-6px)}}}}@keyframes shift{{0%,100%{{background-position:0% 50%}}50%{{background-position:100% 50%}}}}
@media(max-width:760px){{.scene{{height:{max(height,300)}px;border-radius:20px}}.copy{{left:20px;right:18px;top:22px;max-width:92%}}h1{{font-size:30px;max-width:92%}}p{{font-size:11px;max-width:82%}}.metrics{{margin-top:16px}}.metric{{min-width:92px;padding:8px 9px}}.hud,.label{{display:none}}.legend{{right:12px;bottom:11px}}}}
</style></head><body><div class="scene"><div id="three"></div><div class="copy"><div class="kicker"><i></i>ROJ Guard • spatial intelligence</div><h1>{title}</h1><p>{subtitle}</p><div class="metrics">{metric_html}</div></div><div class="hud"><span class="pill">LIVE GRAPH</span><span class="pill">PREDICTIVE</span><span class="pill">HUMAN GOVERNED</span></div><div class="label l1">VENDOR SIGNALS</div><div class="label l2">MATERIAL GRAPH</div><div class="label l3">ROJ CONSTRAINT</div><div class="legend"><i></i>live project digital twin</div></div>
<script>const cfg={cfg};try{{const root=document.getElementById('three'),W=root.clientWidth||1280,H={height};const scene=new THREE.Scene();const camera=new THREE.PerspectiveCamera(46,W/H,.1,100);camera.position.set(0,0,9.2);const renderer=new THREE.WebGLRenderer({{alpha:true,antialias:true,powerPreference:'high-performance'}});renderer.setPixelRatio(Math.min(devicePixelRatio,1.7));renderer.setSize(W,H);root.appendChild(renderer.domElement);const world=new THREE.Group();world.position.set(W<760?2.2:2.75,.05,0);scene.add(world);const cyan=0x4cc9ff,violet=0x7c5cff,amber=0xffb646,green=0x22c55e,red=0xef4444;function mat(c,o=.65){{return new THREE.MeshBasicMaterial({{color:c,transparent:true,opacity:o,wireframe:true}})}}const core=new THREE.Mesh(new THREE.IcosahedronGeometry(1.15,2),mat(cyan,.28));world.add(core);const halo1=new THREE.Mesh(new THREE.TorusGeometry(1.75,.012,8,120),new THREE.MeshBasicMaterial({{color:cyan,transparent:true,opacity:.24}}));halo1.rotation.x=1.03;world.add(halo1);const halo2=new THREE.Mesh(new THREE.TorusGeometry(2.2,.009,8,140),new THREE.MeshBasicMaterial({{color:violet,transparent:true,opacity:.17}}));halo2.rotation.x=.32;halo2.rotation.y=.72;world.add(halo2);const halo3=new THREE.Mesh(new THREE.TorusGeometry(2.65,.007,8,150),new THREE.MeshBasicMaterial({{color:amber,transparent:true,opacity:.13}}));halo3.rotation.x=1.42;halo3.rotation.z=.35;world.add(halo3);const pts=[],nodes=[],colors=[cyan,cyan,violet,amber,green,red];for(let i=0;i<34;i++){{const a=Math.random()*Math.PI*2,b=(Math.random()-.5)*Math.PI,r=1.55+Math.random()*1.75;const v=new THREE.Vector3(Math.cos(a)*Math.cos(b)*r,Math.sin(b)*r,Math.sin(a)*Math.cos(b)*r);const n=new THREE.Mesh(new THREE.SphereGeometry(i%8===0?.07:.032,10,10),new THREE.MeshBasicMaterial({{color:colors[i%colors.length],transparent:true,opacity:i%8===0?.95:.70}}));n.position.copy(v);world.add(n);nodes.push(n)}}for(let i=0;i<nodes.length;i++){{for(let j=i+1;j<nodes.length;j++){{if(nodes[i].position.distanceTo(nodes[j].position)<1.35)pts.push(nodes[i].position.clone(),nodes[j].position.clone())}}}}world.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({{color:0x8bdcff,transparent:true,opacity:.085}})));const dust=[];for(let i=0;i<240;i++)dust.push(new THREE.Vector3((Math.random()-.5)*17,(Math.random()-.5)*8,(Math.random()-.5)*10));const dustObj=new THREE.Points(new THREE.BufferGeometry().setFromPoints(dust),new THREE.PointsMaterial({{color:0x4b6885,size:.016,transparent:true,opacity:.48}}));scene.add(dustObj);let signal=null,signalT=0;if(cfg.mode==='demo'){{const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-2.6,-1.2,0),new THREE.Vector3(2.2,1.35,0)]),new THREE.LineDashedMaterial({{color:amber,dashSize:.16,gapSize:.10,transparent:true,opacity:.35}}));line.computeLineDistances();scene.add(line);signal=new THREE.Mesh(new THREE.SphereGeometry(.09,16,16),new THREE.MeshBasicMaterial({{color:amber}}));scene.add(signal)}}if(cfg.mode==='ingest'){{for(let i=0;i<4;i++){{const pl=new THREE.Mesh(new THREE.PlaneGeometry(.78,1.02),new THREE.MeshBasicMaterial({{color:i%2?cyan:violet,transparent:true,opacity:.07,side:THREE.DoubleSide}}));pl.position.set(-1.7+i*.28,-1+i*.38,-.3+i*.15);pl.rotation.y=.35+i*.08;world.add(pl)}}}}if(cfg.mode==='approval'){{const shield=new THREE.Mesh(new THREE.OctahedronGeometry(.82,1),mat(green,.22));shield.scale.y=1.15;world.add(shield)}}if(cfg.mode==='material'){{const rr=new THREE.Mesh(new THREE.TorusGeometry(.95,.045,12,120),new THREE.MeshBasicMaterial({{color:red,transparent:true,opacity:.42}}));rr.rotation.x=.62;world.add(rr)}}let mx=0,my=0;window.addEventListener('pointermove',e=>{{mx=(e.clientX/W-.5)*.34;my=(e.clientY/H-.5)*.20}});const clock=new THREE.Clock();function animate(){{requestAnimationFrame(animate);const t=clock.getElapsedTime();world.rotation.y=t*.075+mx;world.rotation.x=Math.sin(t*.3)*.05+my;core.rotation.z=-t*.12;halo1.rotation.z=t*.05;halo2.rotation.z=-t*.035;halo3.rotation.y=t*.025;dustObj.rotation.y=t*.004;nodes.forEach((n,i)=>n.scale.setScalar(1+Math.sin(t*2.1+i*.7)*.14));if(signal){{signalT=(signalT+.0045)%1;signal.position.lerpVectors(new THREE.Vector3(-2.6,-1.2,0),new THREE.Vector3(2.2,1.35,0),signalT);signal.scale.setScalar(1+Math.sin(t*8)*.28)}}renderer.render(scene,camera)}}animate();window.addEventListener('resize',()=>{{const nw=root.clientWidth||W;camera.aspect=nw/H;camera.updateProjectionMatrix();renderer.setSize(nw,H)}})}}catch(e){{console.log('Three.js visual fallback',e)}}</script></body></html>'''
    components.html(html, height=iframe_h, scrolling=False)



def render_dashboard_hero(metrics):
    """Large evaluator-facing spatial dashboard; Three.js is visual only."""
    metric_html = "".join(f'<div class="metric"><span>{label}</span><b>{value}</b></div>' for label, value in metrics[:4])
    html = f"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
*{{box-sizing:border-box}}html,body{{margin:0;background:transparent;overflow:hidden;font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;color:#f4f8fc}}
.hero{{height:474px;position:relative;overflow:hidden;border-radius:28px;border:1px solid rgba(182,222,252,.12);background:radial-gradient(650px 360px at 73% 44%,rgba(45,157,229,.09),transparent 62%),radial-gradient(460px 280px at 92% 12%,rgba(123,91,246,.07),transparent 70%),linear-gradient(135deg,#06101e 0%,#050b15 58%,#07101f 100%);box-shadow:0 38px 120px rgba(0,0,0,.38),inset 0 1px 0 rgba(255,255,255,.06)}}
.hero:before{{content:'';position:absolute;inset:0;z-index:1;pointer-events:none;background:linear-gradient(90deg,rgba(4,10,19,.98) 0%,rgba(4,10,19,.90) 35%,rgba(4,10,19,.35) 56%,rgba(4,10,19,.03) 78%),linear-gradient(180deg,transparent 55%,rgba(3,8,16,.30) 100%)}}
.hero:after{{content:'';position:absolute;inset:0;z-index:1;pointer-events:none;opacity:.10;background-image:linear-gradient(rgba(137,188,229,.13) 1px,transparent 1px),linear-gradient(90deg,rgba(137,188,229,.13) 1px,transparent 1px);background-size:54px 54px;mask-image:linear-gradient(90deg,transparent 18%,#000 58%,#000)}}
#scene{{position:absolute;inset:0}}canvas{{width:100%!important;height:100%!important}}.copy{{position:absolute;z-index:3;left:clamp(26px,3.2vw,50px);top:50%;transform:translateY(-50%);width:min(610px,48%)}}
.kicker{{display:flex;align-items:center;gap:8px;color:#73d8ff;text-transform:uppercase;letter-spacing:.16em;font-size:9px;font-weight:800}}.pulse{{width:7px;height:7px;border-radius:50%;background:#4cc9ff;box-shadow:0 0 0 5px rgba(76,201,255,.06),0 0 20px rgba(76,201,255,.65);animation:pulse 1.9s ease-in-out infinite}}
h1{{font-size:clamp(31px,3.15vw,47px);line-height:1.01;letter-spacing:-.052em;margin:14px 0 12px;max-width:600px;color:#f8fbff;font-weight:820}}p{{font-size:13px;line-height:1.62;color:#8498ae;margin:0;max-width:580px}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(96px,1fr));gap:8px;margin-top:23px;max-width:560px}}.metric{{padding:10px 11px;border-radius:13px;border:1px solid rgba(186,222,251,.10);background:linear-gradient(145deg,rgba(16,32,53,.52),rgba(6,16,30,.42));backdrop-filter:blur(16px);box-shadow:inset 0 1px 0 rgba(255,255,255,.035),0 14px 34px rgba(0,0,0,.15)}}.metric span{{display:block;font-size:7px;text-transform:uppercase;letter-spacing:.10em;color:#647a91;font-weight:760}}.metric b{{display:block;font-size:18px;color:#f6f9fc;margin-top:4px;letter-spacing:-.03em}}
.live{{position:absolute;z-index:3;right:20px;top:18px;display:flex;align-items:center;gap:8px;padding:7px 10px;border-radius:999px;border:1px solid rgba(78,201,255,.12);background:rgba(4,13,25,.50);backdrop-filter:blur(14px);font-size:8px;color:#7fa9bd;text-transform:uppercase;letter-spacing:.10em}}.live i{{width:6px;height:6px;border-radius:50%;background:#34d399;box-shadow:0 0 12px rgba(52,211,153,.65)}}
.legend{{position:absolute;z-index:3;right:20px;bottom:18px;padding:8px 10px;border:1px solid rgba(186,222,251,.09);border-radius:11px;background:rgba(4,13,25,.45);backdrop-filter:blur(14px);font-size:8px;color:#6d849b}}.legend b{{color:#a8d9ed;font-weight:700}}@keyframes pulse{{0%,100%{{opacity:.45;transform:scale(.90)}}50%{{opacity:1;transform:scale(1.1)}}}}
@media(max-width:800px){{.hero{{height:430px;border-radius:20px}}.copy{{left:20px;right:18px;top:30px;transform:none;width:auto;max-width:91%}}h1{{font-size:31px;max-width:88%}}p{{font-size:11px;max-width:84%}}.metrics{{grid-template-columns:1fr 1fr;max-width:280px;margin-top:16px}}.metric{{padding:8px 9px}}.metric b{{font-size:15px}}.live,.legend{{display:none}}}}
</style></head><body><div class="hero"><div id="scene"></div><div class="copy"><div class="kicker"><span class="pulse"></span>Live procurement intelligence</div><h1>Predict procurement risk before it impacts the schedule.</h1><p>ROJ Guard connects vendor updates, purchase orders, shipments and Required-On-Job constraints into one live risk view, then prepares mitigation while every outbound action remains under human control.</p><div class="metrics">{metric_html}</div></div><div class="live"><i></i>Spatial project graph online</div><div class="legend"><b>Digital twin:</b> vendor → material → PO → shipment → ROJ</div></div>
<script>try{{const root=document.getElementById('scene');let W=root.clientWidth||1400,H=474;const scene=new THREE.Scene();const camera=new THREE.PerspectiveCamera(43,W/H,.1,100);camera.position.set(0,.25,10.8);const renderer=new THREE.WebGLRenderer({{alpha:true,antialias:true,powerPreference:'high-performance'}});renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.7));renderer.setSize(W,H);root.appendChild(renderer.domElement);const world=new THREE.Group();world.position.set(W<800?2.6:3.15,.15,0);scene.add(world);const cyan=0x4cc9ff,violet=0x8075ff,amber=0xffb44a,green=0x34d399,red=0xff6b6b;function basic(c,o=.65,wire=false){{return new THREE.MeshBasicMaterial({{color:c,transparent:true,opacity:o,wireframe:wire}})}}const shell=new THREE.Mesh(new THREE.IcosahedronGeometry(1.32,2),basic(cyan,.24,true));world.add(shell);const inner=new THREE.Mesh(new THREE.IcosahedronGeometry(.82,1),basic(violet,.12,true));world.add(inner);const core=new THREE.Mesh(new THREE.SphereGeometry(.18,22,22),basic(amber,.95,false));world.add(core);const rings=[];[[1.85,1.10,.25,cyan,.22],[2.25,.42,.78,violet,.17],[2.72,1.42,.35,amber,.13],[3.15,.80,1.25,cyan,.08]].forEach((r,i)=>{{const m=new THREE.Mesh(new THREE.TorusGeometry(r[0],.009,6,160),basic(r[3],r[4],false));m.rotation.x=r[1];m.rotation.y=r[2];m.rotation.z=i*.26;world.add(m);rings.push(m)}});const nodes=[],positions=[],nodeColors=[cyan,cyan,cyan,violet,amber,green,red];for(let i=0;i<44;i++){{const a=(i/44)*Math.PI*2+(Math.random()-.5)*.25;const band=i%2;const rr=band?2.1+Math.random()*1.3:1.55+Math.random()*.8;const y=(Math.random()-.5)*(band?3.2:2.1);const v=new THREE.Vector3(Math.cos(a)*rr,y,Math.sin(a)*rr*.68);positions.push(v);const size=i%9===0?.075:.035;const n=new THREE.Mesh(new THREE.SphereGeometry(size,10,10),basic(nodeColors[i%nodeColors.length],i%9===0?.95:.65,false));n.position.copy(v);world.add(n);nodes.push(n)}}const linePts=[];for(let i=0;i<positions.length;i++){{let nearest=[];for(let j=0;j<positions.length;j++)if(i!==j)nearest.push([positions[i].distanceTo(positions[j]),j]);nearest.sort((a,b)=>a[0]-b[0]);for(const pair of nearest.slice(0,2))linePts.push(positions[i],positions[pair[1]])}}const geo=new THREE.BufferGeometry().setFromPoints(linePts);world.add(new THREE.LineSegments(geo,new THREE.LineBasicMaterial({{color:0x4d7898,transparent:true,opacity:.13}})));const spokePts=[];positions.filter((_,i)=>i%3===0).forEach(v=>spokePts.push(new THREE.Vector3(0,0,0),v));world.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(spokePts),new THREE.LineBasicMaterial({{color:cyan,transparent:true,opacity:.10}})));const grid=new THREE.GridHelper(11,28,0x1e5f83,0x173249);grid.material.transparent=true;grid.material.opacity=.10;grid.position.y=-2.75;world.add(grid);const pCount=180,pa=new Float32Array(pCount*3);for(let i=0;i<pCount;i++){{pa[i*3]=(Math.random()-.5)*14;pa[i*3+1]=(Math.random()-.5)*8;pa[i*3+2]=(Math.random()-.5)*8-1}}const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(pa,3));const particles=new THREE.Points(pg,new THREE.PointsMaterial({{color:0x5abde8,size:.022,transparent:true,opacity:.28}}));scene.add(particles);const movers=[];function addSignal(a,b,c,color,offset){{const curve=new THREE.QuadraticBezierCurve3(a,b,c);world.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(70)),new THREE.LineBasicMaterial({{color,transparent:true,opacity:.12}})));const dot=new THREE.Mesh(new THREE.SphereGeometry(.055,12,12),basic(color,.95,false));world.add(dot);movers.push({{curve,dot,offset}})}}addSignal(new THREE.Vector3(-3,1.2,-.3),new THREE.Vector3(-.9,2.2,.7),new THREE.Vector3(0,0,0),cyan,0);addSignal(new THREE.Vector3(3,-1.5,.2),new THREE.Vector3(1.2,-2,1),new THREE.Vector3(0,0,0),amber,.34);addSignal(new THREE.Vector3(2.6,1.9,-.4),new THREE.Vector3(.8,2.4,-1.2),new THREE.Vector3(0,0,0),violet,.67);let mx=0,my=0;window.addEventListener('pointermove',e=>{{mx=e.clientX/window.innerWidth-.5;my=e.clientY/window.innerHeight-.5}});const clock=new THREE.Clock();function frame(){{requestAnimationFrame(frame);const t=clock.getElapsedTime();shell.rotation.x=t*.075;shell.rotation.y=t*.11;inner.rotation.x=-t*.10;inner.rotation.z=t*.08;rings.forEach((r,i)=>r.rotation.z+=.00055*(i%2?1:-1));nodes.forEach((n,i)=>{{const s=1+Math.sin(t*1.45+i*.75)*.13;n.scale.setScalar(s)}});movers.forEach(m=>m.dot.position.copy(m.curve.getPoint((t*.095+m.offset)%1)));particles.rotation.y=t*.008;world.rotation.y+=(mx*.08-world.rotation.y)*.018;world.rotation.x+=(-my*.035-world.rotation.x)*.018;renderer.render(scene,camera)}}frame();window.addEventListener('resize',()=>{{W=root.clientWidth||1400;H=window.innerWidth<800?430:474;camera.aspect=W/H;camera.updateProjectionMatrix();renderer.setSize(W,H)}})}}catch(e){{console.warn('3D dashboard fallback',e)}}</script></body></html>"""
    components.html(html, height=486, scrolling=False)


def render_dashboard_summary(scores, high, med, low, critical, late, ctx=None):
    total = max(len(scores), 1)
    severe = sum((s.get("predicted_delay_days") or 0) >= 7 for s in scores)
    top5 = sorted(scores, key=lambda x: float(x.get("miss_roj_probability") or 0), reverse=True)[:5]
    top_html = '<div class="rg-priority-list">'
    for s in top5:
        p = float(s.get("miss_roj_probability") or 0) * 100
        top_html += f'<div class="rg-priority-row"><div><div class="rg-priority-name">{clean_name(s.get("material_name"))}</div><div class="rg-priority-meta">{s.get("vendor_name") or "—"} · {s.get("po_number") or "—"} · delay {s.get("predicted_delay_days") or 0}d</div><div class="rg-risk-mini"><span style="width:{min(max(p,2),100):.0f}%"></span></div></div><div class="rg-priority-score">{p:.0f}%</div></div>'
    top_html += '</div>'
    events = (ctx or {}).get("activity", [])[:4]
    if events:
        event_html = '<div class="rg-activity-compact">'
        for e in events:
            event_html += f'<div class="rg-activity-row"><span class="rg-activity-dot"></span><div><div class="rg-activity-title">{e.get("title") or e.get("type") or "Project update"}</div><div class="rg-activity-detail">{e.get("date") or "—"} · {e.get("detail") or ""}</div></div></div>'
        event_html += '</div>'
    else:
        event_html = '<div class="rg-note">No recent material events are available yet.</div>'
    risk_pct=(high+med)/total*100; late_pct=late/total*100; crit_pct=critical/total*100; severe_pct=severe/total*100
    html = f'''<div class="rg-dashboard-grid">
      <div class="rg-dashboard-panel"><div class="rg-dash-panel-title">Portfolio risk</div><div class="rg-dash-panel-sub">Current distribution across active procurement lines</div><div class="rg-donut-wrap" style="grid-template-columns:128px 1fr;gap:18px"><div class="rg-donut" style="width:124px;height:124px;background:conic-gradient(#ef4444 0 {high/total*100:.1f}%,#f59e0b {high/total*100:.1f}% {(high+med)/total*100:.1f}%,#22c55e {(high+med)/total*100:.1f}% 100%)"><div class="rg-donut-center"><div><b>{total}</b><span>active</span></div></div></div><div><div class="rg-legend-row"><div class="rg-legend-label"><span class="rg-dot" style="background:#ef4444"></span>High</div><div class="rg-legend-value">{high}</div></div><div class="rg-legend-row"><div class="rg-legend-label"><span class="rg-dot" style="background:#f59e0b"></span>Medium</div><div class="rg-legend-value">{med}</div></div><div class="rg-legend-row"><div class="rg-legend-label"><span class="rg-dot" style="background:#22c55e"></span>Low</div><div class="rg-legend-value">{low}</div></div></div></div></div>
      <div class="rg-dashboard-panel"><div class="rg-dash-panel-title">Schedule exposure</div><div class="rg-dash-panel-sub">Where procurement pressure is touching milestones</div><div class="rg-exposure-row"><div class="rg-exposure-name">High + medium risk<small>Lines requiring monitoring or intervention</small><div class="rg-meter"><span style="width:{risk_pct:.0f}%"></span></div></div><div class="rg-exposure-value">{high+med}</div></div><div class="rg-exposure-row"><div class="rg-exposure-name">Forecast beyond ROJ<small>Predicted arrival later than site requirement</small><div class="rg-meter"><span style="width:{late_pct:.0f}%"></span></div></div><div class="rg-exposure-value">{late}</div></div><div class="rg-exposure-row"><div class="rg-exposure-name">Critical-path exposure<small>At-risk critical activities</small><div class="rg-meter"><span style="width:{crit_pct:.0f}%"></span></div></div><div class="rg-exposure-value">{critical}</div></div><div class="rg-exposure-row"><div class="rg-exposure-name">Severe slippage<small>Forecast delay of seven days or more</small><div class="rg-meter"><span style="width:{severe_pct:.0f}%"></span></div></div><div class="rg-exposure-value">{severe}</div></div></div>
      <div class="rg-dashboard-panel"><div class="rg-dash-panel-title">Priority materials</div><div class="rg-dash-panel-sub">Highest miss-ROJ probability right now</div>{top_html}</div></div>'''
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(f'<div class="rg-dashboard-grid" style="grid-template-columns:1.35fr .65fr;margin-top:0"><div class="rg-dashboard-panel" style="min-height:180px"><div class="rg-dash-panel-title">Recent project signals</div><div class="rg-dash-panel-sub">Latest linked material and supplier events</div>{event_html}</div><div class="rg-dashboard-panel" style="min-height:180px"><div class="rg-dash-panel-title">Control state</div><div class="rg-dash-panel-sub">Decision support remains human governed</div><div class="rg-compact-stat-row" style="grid-template-columns:1fr 1fr"><div class="rg-compact-stat"><span>Forecast late</span><b>{late}</b></div><div class="rg-compact-stat"><span>Severe slips</span><b>{severe}</b></div><div class="rg-compact-stat"><span>At risk</span><b>{high+med}</b></div><div class="rg-compact-stat"><span>Approval</span><b>Human</b></div></div></div></div>',unsafe_allow_html=True)

def render_top_shell(health_ok):
    status = '<span class="rg-status"><span class="rg-status-dot"></span>System live</span>' if health_ok else '<span class="rg-status" style="border-color:rgba(239,68,68,.28);background:rgba(239,68,68,.08);color:#fecaca"><span class="rg-status-dot" style="background:#ef4444"></span>Backend unavailable</span>'
    st.markdown(f'<div class="rg-topbar"><div class="rg-brand"><div class="rg-logo">R</div><div><div class="rg-brand-title">ROJ Guard</div><div class="rg-brand-sub">Predictive procurement intelligence • live construction control tower</div></div></div><div class="rg-status-wrap"><span class="rg-control-pill">Bangalore Hyperscale DC • Phase 1</span><span class="rg-control-pill">AI + ML online</span>{status}<span class="rg-trust">Human-governed execution</span></div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top application shell / global status
# ---------------------------------------------------------------------------
health = api_get("/", quiet=True)
render_top_shell(bool(health))
PAGES = ["Dashboard", "Incoming Intelligence", "Real Data Lab", "Material Intelligence", "Project Graph", "Live Demo", "Approval Center"]
page = st.radio("Workspace", PAGES, horizontal=True, label_visibility="collapsed", key="top_workspace")
with st.expander("System controls", expanded=False):
    cc1, cc2, cc3 = st.columns([1,1,2])
    with cc1:
        if st.button("Recompute active risk", use_container_width=True):
            result = api_post("/api/risk/compute-all")
            if result:
                st.success(f"Scored {result.get('materials_processed',0)} active materials")
                st.rerun()
    with cc2:
        if st.button("Retrain models", use_container_width=True):
            with st.spinner("Training XGBoost models..."):
                result = api_post("/api/risk/train")
            if result and result.get("status") == "trained":
                st.success(f"MAE {result['lead_time_model_mae_days']}d • AUC {result['risk_classifier_auc']}")
    with cc3:
        st.caption("Controls are intentionally secondary. The main navigation is optimized for evaluator workflows on desktop and mobile.")

# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------
if page == "Dashboard":
    scores = api_get("/api/risk/scores") or []
    if not scores:
        header("Live project intelligence", "Dashboard", "No active risk scores are available yet. Use System controls to compute current material risk.")
        st.info("No active risk scores are available. Recompute active risk from System controls above.")
    else:
        high = sum(s.get("risk_level") == "High" for s in scores)
        med = sum(s.get("risk_level") == "Medium" for s in scores)
        low = sum(s.get("risk_level") == "Low" for s in scores)
        critical = sum(bool(s.get("is_critical_path")) and s.get("risk_level") in ("High", "Medium") for s in scores)
        late = sum((s.get("predicted_delay_days") or 0) > 0 for s in scores)
        st.markdown('<div class="rg-dashboard-label"><div class="rg-dashboard-title">Dashboard</div><div class="rg-dashboard-meta">Live material risk · schedule exposure · intervention priority</div></div>', unsafe_allow_html=True)
        render_dashboard_hero([("Active lines",len(scores)),("High risk",high),("Critical path",critical),("Forecast late",late)])
        top = sorted(scores, key=lambda x: float(x.get("miss_roj_probability") or 0), reverse=True)[0]
        ctx = api_get(f"/api/experience/material/{top['material_id']}", quiet=True)
        render_dashboard_summary(scores, high, med, low, critical, late, ctx)
        st.markdown('<div class="rg-section">Materials requiring attention</div>', unsafe_allow_html=True)
        df = pd.DataFrame(sorted(scores, key=lambda x: float(x.get("miss_roj_probability") or 0), reverse=True))
        display = df[["material_name","sku","po_number","vendor_name","risk_level","miss_roj_probability","predicted_arrival_date","roj_date","predicted_delay_days","is_critical_path"]].copy()
        display["material_name"] = display["material_name"].map(clean_name)
        display["sku"] = display["sku"].map(clean_sku)
        display["po_sku"] = display.apply(lambda row: f'{row.get("po_number") or "—"}  ·  {row.get("sku") or "—"}', axis=1)
        display["miss_roj_probability"] = (display["miss_roj_probability"]*100).round(0).astype(int).astype(str)+"%"
        display = display[["material_name","po_sku","vendor_name","risk_level","miss_roj_probability","predicted_arrival_date","roj_date","predicted_delay_days","is_critical_path"]]
        display.columns = ["Material","PO / SKU","Vendor","Risk","Miss ROJ","Forecast arrival","ROJ","Delay (days)","Critical path"]
        st.dataframe(display,use_container_width=True,hide_index=True,height=390,column_config={"Material":st.column_config.TextColumn(width="medium"),"PO / SKU":st.column_config.TextColumn(width="medium",help="Purchase-order number and unique material SKU"),"Vendor":st.column_config.TextColumn(width="medium"),"Risk":st.column_config.TextColumn(width="small"),"Miss ROJ":st.column_config.TextColumn(width="small"),"Critical path":st.column_config.CheckboxColumn(width="small")})
        if ctx:
            c1,c2=st.columns([1.15,.85])
            with c1:
                st.markdown('<div class="rg-section">Highest-risk material</div>',unsafe_allow_html=True)
                risk=ctx["risk"]
                st.markdown(f'<div class="rg-card"><div style="display:flex;justify-content:space-between;gap:12px"><div><div style="font-size:.96rem;font-weight:780;color:white">{clean_name(ctx["material"]["name"])}</div><div class="rg-note">{(ctx.get("vendor") or {}).get("name","—")} · {(ctx.get("po") or {}).get("po_number","—")}</div></div>{risk_chip(risk.get("risk_level"))}</div><div style="font-size:1.72rem;font-weight:840;margin-top:12px">{float(risk.get("miss_roj_probability") or 0)*100:.0f}%</div><div class="rg-note">Predicted miss-ROJ probability · forecast delay {risk.get("predicted_delay_days")} day(s)</div></div>',unsafe_allow_html=True)
                render_drivers(ctx.get("risk_drivers"))
            with c2:
                st.markdown('<div class="rg-section">Latest activity</div>',unsafe_allow_html=True)
                render_timeline(ctx.get("activity",[])[:5])

# ---------------------------------------------------------------------------
# INCOMING INTELLIGENCE
# ---------------------------------------------------------------------------
elif page == "Incoming Intelligence":
    header("AI document intelligence", "Incoming intelligence", "Analyze a vendor email or project document, review the extracted entities, then apply the signal to the project. No Swagger or JSON API work is required.")
    render_3d_hero("Unstructured supplier updates become structured project signals.", "Documents, emails and logistics messages flow into one entity-linked intelligence layer before any signal is allowed to affect project risk.", [("Input", "Email / PDF"), ("Extraction", "Gemini"), ("Review", "Human"), ("Graph", "Controlled")], mode="ingest", compact=True)
    render_glass_banner(
        "Turn an unstructured supplier update into project intelligence.",
        "Paste a vendor message or upload a PDF. ROJ Guard extracts the entities and schedule signal first, then asks a human to confirm the result before anything touches the project graph.",
        ["Gemini extraction", "Confidence score", "Human confirmation", "Safe fallback"],
        eyebrow="AI ingestion workspace",
    )
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
# REAL DATA LAB
# ---------------------------------------------------------------------------
elif page == "Real Data Lab":
    header(
        "Bring your own procurement data",
        "Real Data Lab",
        "Upload a CSV/XLSX export. ROJ Guard detects whether it contains project ROJ data or procurement history, validates the schema, learns from legitimate historical outcomes, and keeps model provenance explicit.",
    )
    render_3d_hero(
        "Turn unfamiliar procurement exports into decision-ready intelligence.",
        "The lab supports two honest workflows: direct ROJ project scoring when schedule fields exist, or Historical Procurement Mode when the file contains order/delivery history but no ROJ dates.",
        [("Input", "CSV / XLSX"), ("Mapping", "Automatic"), ("Quality", "Validated"), ("Models", "Isolated")],
        mode="ingest", compact=True,
    )
    render_glass_banner(
        "Real data stays isolated from the deterministic demo baseline.",
        "Nothing here overwrites roj_guard.db or the baseline hackathon models. Historical Procurement Mode trains a separate order-to-delivery XGBoost model and converts its forecast into ROJ risk only after you provide a real project ROJ date.",
        ["Schema mapping", "Supplier KPIs", "Leakage-reduced history", "Scenario prediction"],
        eyebrow="BYOD • safe evaluation workspace",
    )

    intro_a, intro_b = st.columns([0.76, 0.24])
    with intro_a:
        uploaded_real = st.file_uploader(
            "Upload procurement export",
            type=["csv", "xlsx", "xlsm"],
            key="real_data_upload",
            help="CSV/XLSX with one row per PO/material line. ROJ Guard can work with project schedule exports or pure procurement KPI history.",
        )
    with intro_b:
        st.download_button(
            "ROJ project template",
            data=template_csv_bytes(),
            file_name="roj_guard_real_data_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Procurement history template",
            data=procurement_history_template_csv_bytes(),
            file_name="roj_guard_procurement_history_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Two schemas; company column names can still differ.")

    if uploaded_real is None:
        st.markdown('<div class="rg-section">Supported real-data workflows</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.markdown("""<div class="rg-card"><div class="rg-eyebrow">Project ROJ mode</div><div style="font-weight:820;color:white;margin:7px 0">Score active project materials</div><div class="rg-note">Best when the export contains supplier, material, ROJ date, PO/shipment context and current status. Historical rows can retrain the full miss-ROJ classifier when actual outcomes are available.</div></div>""", unsafe_allow_html=True)
        with b:
            st.markdown("""<div class="rg-card"><div class="rg-eyebrow">Historical procurement mode</div><div style="font-weight:820;color:white;margin:7px 0">Learn real lead-time + supplier behavior</div><div class="rg-note">Works when there is no ROJ column. Uses clean Delivered orders to train a real order-to-delivery model, builds supplier quality/compliance intelligence, then lets you enter an ROJ for a new procurement scenario.</div></div>""", unsafe_allow_html=True)
    else:
        file_bytes = uploaded_real.getvalue()
        signature = hashlib.sha1(file_bytes).hexdigest()
        previous_sig = st.session_state.get("real_data_signature")
        if signature != previous_sig:
            for key in [
                "real_mapping", "real_normalized", "real_validation", "real_training_df", "real_active_df",
                "real_model_bundle", "real_model_metrics", "real_predictions", "real_train_status",
                "real_proc_bundle", "real_proc_metrics", "real_proc_prediction", "real_proc_scenario",
            ]:
                st.session_state.pop(key, None)
            st.session_state["real_data_signature"] = signature

        sheet_name = None
        if uploaded_real.name.lower().endswith((".xlsx", ".xlsm")):
            try:
                sheets = excel_sheet_names(file_bytes)
                sheet_name = st.selectbox("Workbook sheet", sheets, key="real_sheet")
            except Exception as exc:
                st.error(f"Could not inspect workbook sheets: {exc}")

        try:
            raw_real = read_tabular_bytes(file_bytes, uploaded_real.name, sheet_name=sheet_name)
        except Exception as exc:
            st.error(f"Could not read the uploaded dataset: {exc}")
            raw_real = pd.DataFrame()

        if not raw_real.empty:
            suggestions, map_conf = suggest_column_mapping(raw_real.columns)
            if "real_mapping" not in st.session_state:
                st.session_state["real_mapping"] = suggestions.copy()

            likely_procurement = bool(
                not suggestions.get("roj_date") and suggestions.get("order_date")
                and suggestions.get("actual_delivered_date") and suggestions.get("vendor_name")
            )
            workflow_guess = "Historical procurement" if likely_procurement else "ROJ / project data"

            st.markdown('<div class="rg-section">1 • File intelligence</div>', unsafe_allow_html=True)
            fa, fb, fc, fd = st.columns(4)
            fa.markdown(kpi_card("Rows", f"{len(raw_real):,}", uploaded_real.name), unsafe_allow_html=True)
            fb.markdown(kpi_card("Columns", len(raw_real.columns), "Raw source schema"), unsafe_allow_html=True)
            fc.markdown(kpi_card("Auto-mapped", sum(bool(v) for v in suggestions.values()), "Canonical fields"), unsafe_allow_html=True)
            fd.markdown(kpi_card("Likely workflow", workflow_guess, "Confirmed after validation"), unsafe_allow_html=True)

            with st.expander("Preview uploaded rows", expanded=False):
                st.dataframe(raw_real.head(25), use_container_width=True, hide_index=True)

            st.markdown('<div class="rg-section">2 • Schema mapping</div>', unsafe_allow_html=True)
            mapping_preview = []
            for field, meta in CANONICAL_FIELDS.items():
                src = suggestions.get(field)
                if src:
                    mapping_preview.append({
                        "ROJ Guard field": meta["label"], "Suggested source": src,
                        "Confidence": f"{map_conf.get(field, 0)*100:.0f}%", "Group": meta["group"],
                    })
            if mapping_preview:
                st.dataframe(pd.DataFrame(mapping_preview), use_container_width=True, hide_index=True, height=260)
            st.caption("Exact synonyms are matched first, then conservative fuzzy matching. Every source column is auto-assigned at most once.")

            choices = ["— Not mapped —"] + list(raw_real.columns)
            mapping_now = {}
            groups = []
            for meta in CANONICAL_FIELDS.values():
                if meta["group"] not in groups:
                    groups.append(meta["group"])
            with st.expander("Review / correct column mapping", expanded=True):
                for group in groups:
                    st.markdown(f"**{group}**")
                    fields = [f for f, meta in CANONICAL_FIELDS.items() if meta["group"] == group]
                    grid = st.columns(3)
                    for i, field in enumerate(fields):
                        suggested = st.session_state["real_mapping"].get(field)
                        current = suggested if suggested in raw_real.columns else "— Not mapped —"
                        idx = choices.index(current) if current in choices else 0
                        with grid[i % 3]:
                            selected_col = st.selectbox(
                                CANONICAL_FIELDS[field]["label"], choices, index=idx,
                                key=f"map_{signature[:8]}_{field}",
                            )
                            mapping_now[field] = None if selected_col == "— Not mapped —" else selected_col

            vcol, note_col = st.columns([.28, .72])
            with vcol:
                validate_click = st.button("Validate & Build Dataset", type="primary", use_container_width=True)
            with note_col:
                st.markdown('<div class="rg-note" style="padding-top:9px">Validation is in-memory only. The uploaded dataset never replaces the prepared hackathon database.</div>', unsafe_allow_html=True)

            if validate_click:
                try:
                    normalized = normalize_dataset(raw_real, mapping_now)
                    report_obj = validate_dataset(normalized)
                    train_df, active_df = build_real_feature_frames(normalized)
                    st.session_state["real_mapping"] = mapping_now
                    st.session_state["real_normalized"] = normalized
                    st.session_state["real_validation"] = report_obj.as_dict()
                    st.session_state["real_training_df"] = train_df
                    st.session_state["real_active_df"] = active_df
                    for key in ["real_model_bundle", "real_model_metrics", "real_predictions", "real_proc_bundle", "real_proc_metrics", "real_proc_prediction"]:
                        st.session_state.pop(key, None)
                except Exception as exc:
                    st.error(f"Validation failed: {exc}")

            report = st.session_state.get("real_validation")
            normalized = st.session_state.get("real_normalized")
            train_df = st.session_state.get("real_training_df")
            active_df = st.session_state.get("real_active_df")

            if report:
                mode = report.get("mode", "generic")
                if mode == "procurement_history":
                    st.markdown('<div class="rg-section">3 • Historical Procurement Mode</div>', unsafe_allow_html=True)
                    st.success("Procurement history detected. ROJ dates are not present, so ROJ Guard will learn real order-to-delivery behavior first and will request an ROJ only when you create a prediction scenario.")
                    qa, qb, qc, qd, qe = st.columns(5)
                    qa.markdown(kpi_card("Quality", f'{report["quality_score"]}%', "Core procurement fields"), unsafe_allow_html=True)
                    qb.markdown(kpi_card("Clean delivered", f'{report["clean_delivered_rows"]:,}', "Eligible real outcomes"), unsafe_allow_html=True)
                    qc.markdown(kpi_card("Suppliers", report["vendors"], "Supplier population"), unsafe_allow_html=True)
                    qd.markdown(kpi_card("Categories", report["categories"], "Procurement classes"), unsafe_allow_html=True)
                    comp = report.get("compliance_rate")
                    qe.markdown(kpi_card("Compliance", f'{comp*100:.1f}%' if comp is not None else "—", "Observed source field"), unsafe_allow_html=True)

                    for err in report.get("errors", []): st.error(err)
                    for warn in report.get("warnings", []): st.warning(warn)

                    # Data-quality evidence and order-state composition.
                    left_q, right_q = st.columns([.46, .54])
                    with left_q:
                        st.markdown('<div class="rg-card"><div class="rg-panel-title">Data cleaning trace</div><div class="rg-panel-sub">What is and is not allowed into supervised lead-time training</div>', unsafe_allow_html=True)
                        clean_rows = report.get("clean_delivered_rows", 0)
                        st.markdown(f"**{clean_rows:,}** rows pass: `Delivered` + valid Order Date + valid Delivery Date + non-negative lead time.")
                        st.markdown(f"- Delivered status rows: **{report.get('delivered_rows', 0):,}**")
                        st.markdown(f"- Missing Delivery Date: **{report.get('missing_delivery_dates', 0):,}**")
                        st.markdown(f"- Invalid order→delivery sequence: **{report.get('invalid_order_delivery_rows', 0):,}**")
                        st.markdown("- Cancelled / pending / partial rows remain available for supplier KPIs but are **not** used as completed lead-time targets.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with right_q:
                        status_chart = pd.DataFrame({"Orders": [
                            report.get("delivered_rows", 0), report.get("pending_rows", 0),
                            report.get("partial_rows", 0), report.get("cancelled_rows", 0),
                        ]}, index=["Delivered", "Pending/open", "Partially delivered", "Cancelled"])
                        st.markdown('<div class="rg-panel-title">Order-state composition</div><div class="rg-panel-sub">Source status distribution</div>', unsafe_allow_html=True)
                        st.bar_chart(status_chart, height=250)

                    st.markdown('<div class="rg-section">4 • Supplier intelligence</div>', unsafe_allow_html=True)
                    supplier_kpis = build_supplier_intelligence(normalized)
                    category_kpis = build_category_intelligence(normalized)
                    if not supplier_kpis.empty:
                        sleft, sright = st.columns([1.18, .82])
                        with sleft:
                            table = supplier_kpis.copy()
                            for col in ["Completion rate", "Compliance rate", "Defect rate", "Weighted savings rate"]:
                                table[col] = (table[col] * 100).round(1).astype(str) + "%"
                            for col in ["Avg lead time days", "Median lead time days", "P90 lead time days", "Supplier health score"]:
                                table[col] = pd.to_numeric(table[col], errors="coerce").round(1)
                            show_cols = ["Supplier", "Orders", "Clean delivered", "Completion rate", "Compliance rate", "Defect rate", "Weighted savings rate", "Avg lead time days", "P90 lead time days", "Supplier health score"]
                            st.dataframe(table[show_cols], use_container_width=True, hide_index=True, height=300)
                        with sright:
                            lead_chart = supplier_kpis.set_index("Supplier")[["Avg lead time days", "P90 lead time days"]].copy()
                            st.markdown('<div class="rg-panel-title">Supplier lead-time profile</div><div class="rg-panel-sub">Observed clean Delivered history</div>', unsafe_allow_html=True)
                            st.bar_chart(lead_chart, height=300)
                        dl1, dl2 = st.columns(2)
                        dl1.download_button("Download supplier KPI CSV", supplier_kpis.to_csv(index=False).encode("utf-8"), "roj_guard_supplier_intelligence.csv", "text/csv", use_container_width=True)
                        if not category_kpis.empty:
                            dl2.download_button("Download category lead-time CSV", category_kpis.to_csv(index=False).encode("utf-8"), "roj_guard_category_lead_times.csv", "text/csv", use_container_width=True)

                    st.markdown('<div class="rg-section">5 • Train real procurement lead-time model</div>', unsafe_allow_html=True)
                    st.markdown("""<div class="rg-card"><div class="rg-eyebrow">Real supervised target</div><div style="font-size:.98rem;color:white;font-weight:820;margin:6px 0">Order Date → Actual Delivery Date</div><div class="rg-note">Only clean Delivered rows create the target. The model uses supplier/category/commercial context plus prior supplier history. A temporal 80/20 holdout is used; no historical ROJ classifier is fabricated.</div></div>""", unsafe_allow_html=True)
                    if st.button("Train Lead-Time Model on Real Data", type="primary", use_container_width=True, disabled=report.get("clean_delivered_rows", 0) < 50):
                        with st.spinner("Building leakage-reduced supplier history → temporal holdout → training XGBoost lead-time model..."):
                            trained = train_procurement_lead_time_model(normalized)
                        if trained.get("status") == "trained":
                            st.session_state["real_proc_bundle"] = trained.pop("bundle")
                            st.session_state["real_proc_metrics"] = trained
                            st.session_state.pop("real_proc_prediction", None)
                        else:
                            st.session_state.pop("real_proc_bundle", None)
                            st.session_state["real_proc_metrics"] = trained

                    pm = st.session_state.get("real_proc_metrics")
                    if pm:
                        if pm.get("status") == "trained":
                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.markdown(kpi_card("Rows used", pm["rows_used"], pm.get("model_grade", "history")), unsafe_allow_html=True)
                            m2.markdown(kpi_card("Holdout", pm["holdout_rows"], "Temporal 20%"), unsafe_allow_html=True)
                            m3.markdown(kpi_card("Model MAE", f'{pm["lead_time_mae_days"]:.2f}d', "Real holdout"), unsafe_allow_html=True)
                            m4.markdown(kpi_card("Baseline MAE", f'{pm["baseline_mae_days"]:.2f}d', "Training-mean baseline"), unsafe_allow_html=True)
                            lift = pm.get("model_lift_vs_baseline_pct", 0)
                            m5.markdown(kpi_card("Model lift", f'{lift:+.1f}%', "vs naive baseline"), unsafe_allow_html=True)
                            st.caption(f'Median absolute error {pm["median_abs_error_days"]:.2f}d • RMSE {pm["rmse_days"]:.2f}d • R² {pm["r2"] if pm.get("r2") is not None else "N/A"}. These are holdout statistics from the uploaded dataset, not production claims.')
                            if lift < 5:
                                st.info("This dataset has limited predictive signal beyond the overall historical lead-time level. ROJ Guard reports that instead of overstating model quality; supplier KPIs and forecast uncertainty remain usable for scenario analysis.")
                        else:
                            st.warning(pm.get("note", "Could not train the procurement model."))

                    bundle = st.session_state.get("real_proc_bundle")
                    if bundle:
                        st.markdown('<div class="rg-section">6 • Predict a new procurement against a real ROJ</div>', unsafe_allow_html=True)
                        suppliers = list(bundle["supplier_kpis"]["Supplier"].astype(str)) if not bundle["supplier_kpis"].empty else sorted(normalized["vendor_name"].dropna().astype(str).unique())
                        categories = sorted(normalized["material_class"].dropna().astype(str).unique())
                        q_default = float(pd.to_numeric(normalized["quantity"], errors="coerce").replace(0, pd.NA).dropna().median() or 1)
                        up_default = float(pd.to_numeric(normalized["unit_price"], errors="coerce").replace(0, pd.NA).dropna().median() or 1)
                        np_default = float(pd.to_numeric(normalized["negotiated_price"], errors="coerce").replace(0, pd.NA).dropna().median() or up_default)

                        with st.form("procurement_scenario_form", clear_on_submit=False):
                            r1a, r1b, r1c = st.columns(3)
                            supplier = r1a.selectbox("Supplier", suppliers)
                            material_class = r1b.selectbox("Item / material category", categories)
                            quantity = r1c.number_input("Quantity", min_value=0.0, value=max(1.0, q_default), step=1.0)
                            r2a, r2b, r2c = st.columns(3)
                            unit_price = r2a.number_input("Unit price", min_value=0.0, value=max(0.01, up_default), step=1.0)
                            negotiated_price = r2b.number_input("Negotiated price", min_value=0.0, value=max(0.01, np_default), step=1.0)
                            float_days = r2c.number_input("Schedule float (days)", min_value=0.0, value=0.0, step=1.0)
                            r3a, r3b, r3c = st.columns(3)
                            order_dt = r3a.date_input("New PO / order date", value=date.today())
                            roj_dt = r3b.date_input("Required-On-Job (ROJ) date", value=date.today() + timedelta(days=30))
                            critical = r3c.checkbox("Critical-path material", value=False)
                            submit_scenario = st.form_submit_button("Predict ROJ Risk", type="primary", use_container_width=True)

                        if submit_scenario:
                            if roj_dt <= order_dt:
                                st.error("ROJ date must be after the order date.")
                            else:
                                scenario = dict(
                                    supplier=supplier, material_class=material_class, quantity=quantity,
                                    unit_price=unit_price, negotiated_price=negotiated_price,
                                    order_date=order_dt, roj_date=roj_dt, float_days=float_days,
                                    is_critical_path=critical,
                                )
                                pred = predict_procurement_scenario(bundle, **scenario)
                                st.session_state["real_proc_prediction"] = pred
                                st.session_state["real_proc_scenario"] = scenario

                        pred = st.session_state.get("real_proc_prediction")
                        if pred:
                            risk = pred["Risk"]
                            risk_class = "rg-high" if risk == "High" else ("rg-medium" if risk == "Medium" else "rg-low")
                            st.markdown(f"""<div class="rg-card"><div class="rg-eyebrow">Real-data scenario assessment</div><div style="display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-top:7px"><div><div style="font-size:1.34rem;font-weight:860;color:#fff">{pred["Supplier"]} • {pred["Category"]}</div><div class="rg-note">Real procurement lead-time model + supplied project ROJ</div></div><span class="rg-chip {risk_class}" style="font-size:.88rem;padding:7px 12px">{risk.upper()} • {pred["Miss ROJ probability"]*100:.1f}%</span></div></div>""", unsafe_allow_html=True)
                            c1, c2, c3, c4, c5 = st.columns(5)
                            c1.markdown(kpi_card("Predicted lead", f'{pred["Predicted lead time days"]:.1f}d', "Order → delivery"), unsafe_allow_html=True)
                            c2.markdown(kpi_card("Forecast arrival", pred["Forecast arrival"], "Model forecast"), unsafe_allow_html=True)
                            c3.markdown(kpi_card("ROJ", pred["ROJ date"], "Required on job"), unsafe_allow_html=True)
                            c4.markdown(kpi_card("Forecast margin", f'{pred["Forecast margin days"]:+d}d', "Positive = before ROJ"), unsafe_allow_html=True)
                            c5.markdown(kpi_card("Supplier health", f'{pred["Supplier health score"]:.0f}/100', "Composite observed KPI"), unsafe_allow_html=True)
                            st.caption(f'Statistical P(miss ROJ) {pred["Statistical miss probability"]*100:.1f}% • estimated schedule-impact probability {pred["Schedule impact probability"]*100:.1f}% • forecast uncertainty σ≈{pred["Uncertainty sigma days"]:.1f}d.')
                            st.markdown('<div class="rg-section">Why this scenario received that risk</div>', unsafe_allow_html=True)
                            for d in pred["Drivers"]:
                                st.markdown(f'<div class="rg-driver"><b>{d}</b></div>', unsafe_allow_html=True)

                            st.markdown('<div class="rg-section">Supplier what-if comparison</div>', unsafe_allow_html=True)
                            compare = compare_supplier_scenarios(bundle, **st.session_state["real_proc_scenario"])
                            st.dataframe(compare, use_container_width=True, hide_index=True)
                            st.caption("This is a model-based scenario comparison using identical quantity, commercial terms, order date and ROJ; it is not an RFQ or a claim that every supplier can supply every category. Capability qualification remains a human procurement gate.")
                            st.download_button("Download scenario comparison CSV", compare.to_csv(index=False).encode("utf-8"), "roj_guard_supplier_scenario_comparison.csv", "text/csv", use_container_width=True)

                elif mode == "roj_project":
                    st.markdown('<div class="rg-section">3 • Project ROJ data quality gate</div>', unsafe_allow_html=True)
                    qa, qb, qc, qd, qe = st.columns(5)
                    qa.markdown(kpi_card("Quality", f'{report["quality_score"]}%', "Key-field completeness"), unsafe_allow_html=True)
                    qb.markdown(kpi_card("Completed history", f'{report["historical_rows"]:,}', "Potential training outcomes"), unsafe_allow_html=True)
                    qc.markdown(kpi_card("Active rows", f'{report["active_rows"]:,}', "Materials to score"), unsafe_allow_html=True)
                    qd.markdown(kpi_card("Vendors", report["vendors"], "Unique suppliers"), unsafe_allow_html=True)
                    qe.markdown(kpi_card("Missed ROJ history", report["missed_roj_rows"], f'vs {report["on_time_rows"]} on-time'), unsafe_allow_html=True)
                    for err in report.get("errors", []): st.error(err)
                    for warn in report.get("warnings", []): st.warning(warn)

                    if not report.get("errors"):
                        st.markdown('<div class="rg-section">4 • Choose model provenance</div>', unsafe_allow_html=True)
                        roj_mode = st.radio("Inference mode", ["Prototype model + real project data", "Retrain on supplied historical data"], horizontal=True, key="real_inference_mode")
                        if roj_mode == "Retrain on supplied historical data":
                            if st.button("Train Real-Data ROJ Model", type="primary", disabled=(train_df is None or len(train_df) < 50), use_container_width=True):
                                with st.spinner("Temporal holdout → training lead-time + miss-ROJ models..."):
                                    trained = train_real_data_models(train_df)
                                if trained.get("status") == "trained":
                                    st.session_state["real_model_bundle"] = trained.pop("bundle")
                                    st.session_state["real_model_metrics"] = trained
                                else:
                                    st.session_state["real_model_metrics"] = trained
                            metrics = st.session_state.get("real_model_metrics")
                            if metrics:
                                if metrics.get("status") == "trained":
                                    ma, mb, mc, md, me = st.columns(5)
                                    ma.markdown(kpi_card("Rows used", metrics["rows_used"], metrics.get("model_grade", "real history")), unsafe_allow_html=True)
                                    mb.markdown(kpi_card("Holdout", metrics["holdout_rows"], "Temporal validation"), unsafe_allow_html=True)
                                    mc.markdown(kpi_card("Lead-time MAE", f'{metrics["lead_time_mae_days"]:.2f}d', "Lower is better"), unsafe_allow_html=True)
                                    md.markdown(kpi_card("Risk AUC", f'{metrics["risk_auc"]:.3f}' if metrics.get("risk_auc") is not None else "N/A", "Temporal holdout"), unsafe_allow_html=True)
                                    me.markdown(kpi_card("Recall", f'{metrics["recall"]*100:.0f}%', "Late-material detection"), unsafe_allow_html=True)
                                else:
                                    st.warning(metrics.get("note", "Real-data retraining could not be completed."))
                        else:
                            st.info("Uses the existing prototype statistical model with live features derived from the uploaded real project data. Model provenance remains explicitly labeled.")

                        st.markdown('<div class="rg-section">5 • Score active materials</div>', unsafe_allow_html=True)
                        can_score = active_df is not None and not active_df.empty
                        if roj_mode == "Retrain on supplied historical data":
                            can_score = can_score and bool(st.session_state.get("real_model_bundle"))
                        if st.button("Score Active Materials", type="primary", use_container_width=True, disabled=not can_score):
                            with st.spinner("Forecasting lead time → computing miss-ROJ risk..."):
                                preds = score_with_real_model(active_df, st.session_state["real_model_bundle"]) if roj_mode == "Retrain on supplied historical data" else score_with_prototype_model(active_df)
                            st.session_state["real_predictions"] = preds
                        predictions = st.session_state.get("real_predictions")
                        if isinstance(predictions, pd.DataFrame) and not predictions.empty:
                            counts = Counter(predictions["Risk"])
                            pa, pb, pc, pdx = st.columns(4)
                            pa.markdown(kpi_card("Scored", len(predictions), "Active procurement lines"), unsafe_allow_html=True)
                            pb.markdown(kpi_card("High", counts.get("High", 0), "Immediate attention"), unsafe_allow_html=True)
                            pc.markdown(kpi_card("Medium", counts.get("Medium", 0), "Watch / mitigate"), unsafe_allow_html=True)
                            pdx.markdown(kpi_card("Low", counts.get("Low", 0), "Currently protected"), unsafe_allow_html=True)
                            display = predictions.copy(); display["Risk %"] = (display["Miss ROJ probability"] * 100).round(1).astype(str) + "%"
                            cols_show = ["Material", "PO", "Vendor", "Risk", "Risk %", "Forecast arrival", "ROJ date", "Forecast delay days", "Critical path", "Model source", "Why"]
                            st.dataframe(display[cols_show], use_container_width=True, hide_index=True, height=430)
                            st.download_button("Download prediction results CSV", predictions.to_csv(index=False).encode("utf-8"), "roj_guard_real_data_predictions.csv", "text/csv", use_container_width=True)
                else:
                    for err in report.get("errors", []): st.error(err)
                    for warn in report.get("warnings", []): st.warning(warn)

# ---------------------------------------------------------------------------
# MATERIAL INTELLIGENCE
# ---------------------------------------------------------------------------
elif page == "Material Intelligence":
    header("Layer 3 + 4 • Explainable risk", "Material intelligence", "Inspect the forecast, live risk drivers, vendor context, schedule exposure, and mitigation actions for a single material line.")
    render_3d_hero("One material. Every signal. One risk state.", "ROJ Guard fuses vendor history, shipment state, schedule constraints and live communications into a continuously explainable material risk profile.", [("Risk", "Live"), ("Forecast", "Dynamic"), ("Drivers", "Explainable"), ("Actions", "3 agents")], mode="material", compact=True)
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

            render_risk_cockpit(material, vendor, risk, shipment, sched)
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
    render_3d_hero("A project graph built for decisions, not decoration.", "Trace every material from project and vendor context through purchase order, shipment and Required-On-Job constraints. Risk intelligence stays attached to the entities that created it.", [("Graph layer", "Live"), ("Entity linking", "Active"), ("Risk context", "Attached")], accent="#38bdf8", mode="graph")
    header("Layer 2 • Project graph", "Project graph", "A material-centric projection of the entities ROJ Guard links together. The same contract maps naturally onto Kaya/Amber's existing project graph.")
    render_glass_banner("Context travels with the material.", "Every forecast is anchored to the project entities that created it — material, vendor, PO, shipment and ROJ. That gives the risk engine traceable context and makes the output easier to audit.", ["Entity-linked", "Traceable", "Kaya / Amber compatible"], eyebrow="Graph intelligence")
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
    render_3d_hero("Watch a vendor signal travel through the project intelligence graph.", "A 14-day supplier delay propagates from communication to material context, forecast, ROJ risk and finally a human-governed mitigation action.", [("Before", "4% Low"), ("Signal", "+14 days"), ("After", "90% High"), ("Action", "Escalate")], mode="demo", compact=True)
    render_glass_banner("Watch one supplier signal propagate through the whole system.", "This scenario is deterministic by design: it starts with a healthy material, injects a 14-day vendor delay, recomputes ROJ risk and automatically prepares a mitigation draft for human approval.", ["4% → 90%", "14-day signal", "Agent draft", "Human approval"], eyebrow="Live scenario armed")

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
    render_3d_hero("AI can prepare the move. People still authorize it.", "Every vendor escalation, resequencing proposal and alternate-supplier intervention crosses an explicit approval boundary before execution.", [("Drafts", "Editable"), ("Approval", "Required"), ("Execution", "Controlled"), ("Audit", "Retained")], mode="approval", compact=True)
    render_glass_banner("AI prepares. People decide.", "ROJ Guard can draft escalation, resequencing and alternate-supplier actions, but the execution boundary remains explicit: review, edit, approve or reject — with a retained audit record.", ["Human approval", "Editable drafts", "Execution receipts", "Audit trail"], eyebrow="Governance layer")
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


st.markdown('<div class="rg-footer"><span>ROJ Guard • Spatial predictive procurement intelligence</span><span>Kaya AI IIT India Hackathon 2026 • Predict early. Act safely.</span></div>', unsafe_allow_html=True)
