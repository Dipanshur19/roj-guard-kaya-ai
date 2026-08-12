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
import streamlit.components.v1 as components

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



def render_intelligence_pipeline(active="Predict"):
    steps = [
        ("01", "Ingest", "POs, emails, logistics"),
        ("02", "Link", "Entity + project context"),
        ("03", "Predict", "Lead time + ROJ risk"),
        ("04", "Act", "Human-governed mitigation"),
    ]
    html = '<div class="rg-pipeline">'
    for num, title, desc in steps:
        cls = ' active' if title.lower() == str(active).lower() else ''
        html += f'<div class="rg-pipe-step{cls}"><div class="rg-pipe-num">{num} • Layer</div><div class="rg-pipe-title">{title}</div><div class="rg-pipe-desc">{desc}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


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


def render_top_shell(health_ok):
    status = '<span class="rg-status"><span class="rg-status-dot"></span>System live</span>' if health_ok else '<span class="rg-status" style="border-color:rgba(239,68,68,.28);background:rgba(239,68,68,.08);color:#fecaca"><span class="rg-status-dot" style="background:#ef4444"></span>Backend unavailable</span>'
    st.markdown(f'<div class="rg-topbar"><div class="rg-brand"><div class="rg-logo">R</div><div><div class="rg-brand-title">ROJ Guard</div><div class="rg-brand-sub">Predictive procurement intelligence • live construction control tower</div></div></div><div class="rg-status-wrap"><span class="rg-control-pill">Bangalore Hyperscale DC • Phase 1</span><span class="rg-control-pill">AI + ML online</span>{status}<span class="rg-trust">Human-governed execution</span></div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top application shell / global status
# ---------------------------------------------------------------------------
health = api_get("/", quiet=True)
render_top_shell(bool(health))
PAGES = ["Overview", "Incoming Intelligence", "Material Intelligence", "Project Graph", "Live Demo", "Approval Center"]
page = st.radio("Workspace", PAGES, horizontal=True, label_visibility="collapsed", key="top_workspace")
st.markdown(f'<div class="rg-control-strip"><span class="rg-control-pill">API: {BACKEND_URL}</span><span class="rg-control-pill">Human approval required before outbound execution</span><span class="rg-control-pill">XGBoost + Gemini + Project Graph</span></div>',unsafe_allow_html=True)
render_signal_rail()
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

        render_3d_hero(
            "See procurement risk before it reaches the site.",
            "ROJ Guard links supplier signals, shipment context and schedule constraints into a live predictive view — then prepares mitigation while keeping every outbound action under human control.",
            [("Active lines", len(scores)), ("High risk", high), ("Critical path", critical), ("Forecast late", late)],
            mode="overview",
        )
        render_intelligence_pipeline("Predict")
        render_glass_banner(
            "One operating picture for procurement risk.",
            "Live material context, vendor signals, shipment state and schedule exposure are connected before risk is scored — so the team sees not only what is late, but what is likely to become late next.",
            ["Real-time signals", "Explainable risk", "Human-governed actions"],
            eyebrow="Command center online",
        )

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
            render_risk_donut(high, med, low)
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
    render_3d_hero("Unstructured supplier updates become structured project signals.", "Documents, emails and logistics messages flow into one entity-linked intelligence layer before any signal is allowed to affect project risk.", [("Input", "Email / PDF"), ("Extraction", "Gemini"), ("Review", "Human"), ("Graph", "Controlled")], mode="ingest", compact=True)
    render_intelligence_pipeline("Ingest")
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
# MATERIAL INTELLIGENCE
# ---------------------------------------------------------------------------
elif page == "Material Intelligence":
    header("Layer 3 + 4 • Explainable risk", "Material intelligence", "Inspect the forecast, live risk drivers, vendor context, schedule exposure, and mitigation actions for a single material line.")
    render_3d_hero("One material. Every signal. One risk state.", "ROJ Guard fuses vendor history, shipment state, schedule constraints and live communications into a continuously explainable material risk profile.", [("Risk", "Live"), ("Forecast", "Dynamic"), ("Drivers", "Explainable"), ("Actions", "3 agents")], mode="material", compact=True)
    render_intelligence_pipeline("Predict")
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
    render_intelligence_pipeline("Link")
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
    render_intelligence_pipeline("Act")
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
    render_intelligence_pipeline("Act")
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
