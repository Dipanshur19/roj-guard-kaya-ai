"""Layer 4 specialist agents. Every output is a draft until human approval."""

import os
from datetime import date
from time_utils import current_date
from sqlalchemy.orm import Session

try:
    from google import genai
except Exception:
    genai = None

from models_layer1 import (
    Material, Vendor, PurchaseOrder, ScheduleROJ, ShippingLogistics,
)
from models_layer2 import VendorReliability
from models_layer3 import MaterialRiskScore

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
ENV_MODEL = os.environ.get("GEMINI_MODEL", "").strip()
DECOMMISSIONED_MODELS = {"gemini-2.0-flash", "gemini-2.0-flash-lite"}
MODEL_CANDIDATES = []
if ENV_MODEL and ENV_MODEL not in DECOMMISSIONED_MODELS:
    MODEL_CANDIDATES.append(ENV_MODEL)
for _m in ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"):
    if _m not in MODEL_CANDIDATES:
        MODEL_CANDIDATES.append(_m)


def _generate(prompt: str, fallback: str) -> str:
    if genai is None or not GEMINI_API_KEY:
        return fallback
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        for model_name in MODEL_CANDIDATES:
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                if response.text:
                    return response.text.strip()
            except Exception as exc:
                msg = str(exc).lower()
                if "404" in msg or "not_found" in msg or "not found" in msg or "unsupported" in msg:
                    continue
                break
    except Exception:
        pass
    return fallback


def _gather_context(db: Session, material_id: str) -> dict:
    material = db.query(Material).filter(Material.material_id == material_id).first()
    risk = db.query(MaterialRiskScore).filter(MaterialRiskScore.material_id == material_id).first()
    po = (
        db.query(PurchaseOrder).filter(PurchaseOrder.material_id == material_id)
        .order_by(PurchaseOrder.created_at.desc()).first()
    )
    vendor = db.query(Vendor).filter(Vendor.vendor_id == po.vendor_id).first() if po else None
    schedule = (
        db.query(ScheduleROJ).filter(ScheduleROJ.material_id == material_id)
        .order_by(ScheduleROJ.created_at.desc()).first()
    )
    return {"material": material, "risk": risk, "po": po, "vendor": vendor, "schedule": schedule}


def draft_vendor_escalation(db: Session, material_id: str) -> dict:
    ctx = _gather_context(db, material_id)
    material, risk, vendor, po = ctx["material"], ctx["risk"], ctx["vendor"], ctx["po"]
    if not material or not risk or not vendor:
        return {"error": "Missing material, risk score, or vendor data for this material."}

    fallback = (
        f"Subject: Urgent status confirmation - {po.po_number if po else material.sku}\n\n"
        f"Dear {vendor.vendor_name} team,\n\n"
        f"Our current ROJ risk assessment for {material.description} indicates a "
        f"{risk.predicted_delay_days:.0f}-day delay against the required-on-job date {risk.roj_date}. "
        f"Please confirm current fabrication/shipping status, the latest committed delivery date, "
        f"and whether expediting is available. This item is being actively coordinated against the project schedule.\n\n"
        f"Regards,\nROJ Guard Procurement Team"
    )
    prompt = f"""
Draft a professional vendor escalation email under 150 words.
PO: {po.po_number if po else 'N/A'}
Material: {material.description} ({material.sku})
Vendor: {vendor.vendor_name}
Predicted arrival: {risk.predicted_arrival_date}
ROJ: {risk.roj_date}
Predicted delay: {risk.predicted_delay_days} days
Risk: {risk.risk_level}
Ask for status, committed date, and expediting options. Do not fabricate facts.
"""
    draft_content = _generate(prompt, fallback)
    reasoning = (
        f"{vendor.vendor_name} is {risk.risk_level} risk with "
        f"{risk.miss_roj_probability*100:.0f}% estimated probability of missing ROJ."
    )
    return {
        "action_type": "ESCALATION", "material_id": material_id,
        "vendor_id": vendor.vendor_id, "risk_level_at_creation": risk.risk_level,
        "draft_content": draft_content, "reasoning": reasoning,
    }


def draft_resequence_suggestion(db: Session, material_id: str) -> dict:
    ctx = _gather_context(db, material_id)
    material, risk, schedule = ctx["material"], ctx["risk"], ctx["schedule"]
    if not material or not risk or not schedule:
        return {"error": "Missing material, risk score, or schedule data for this material."}

    delay = max(0, int(round(risk.predicted_delay_days or 0)))
    net_impact = max(0, delay - int(schedule.float_days or 0))
    fallback = (
        f"Proposed schedule scenario: keep '{schedule.task_name}' as the constrained activity, "
        f"consume the available {schedule.float_days or 0} day(s) of float, and resequence crews "
        f"to parallel prerequisite/adjacent work until the material arrives. The current predicted "
        f"{delay}-day material delay leaves approximately {net_impact} day(s) of residual schedule impact. "
        f"Validate this scenario with the project scheduler before publishing it to the master schedule."
    )
    prompt = f"""
Propose ONE specific construction schedule resequencing scenario in under 120 words.
Material: {material.description}
Dependent task: {schedule.task_name}
ROJ: {risk.roj_date}
Predicted delay: {delay} days
Float: {schedule.float_days or 0} days
Critical path: {bool(schedule.is_critical_path)}
Do not invent task IDs or dates not supplied. State that the scenario requires scheduler validation.
"""
    draft_content = _generate(prompt, fallback)
    reasoning = (
        f"'{schedule.task_name}' has {schedule.float_days or 0} day(s) float against "
        f"a {delay}-day forecast delay; residual impact is ~{net_impact} day(s)."
    )
    return {
        "action_type": "RESEQUENCE", "material_id": material_id, "vendor_id": None,
        "risk_level_at_creation": risk.risk_level, "draft_content": draft_content,
        "reasoning": reasoning,
    }


def _candidate_stats(db: Session, vendor: Vendor, material_class: str):
    rows = (
        db.query(ShippingLogistics, Material)
        .join(Material, Material.material_id == ShippingLogistics.material_id)
        .filter(
            ShippingLogistics.vendor_id == vendor.vendor_id,
            Material.material_class == material_class,
            ShippingLogistics.status == "delivered",
            ShippingLogistics.actual_delivered_date.isnot(None),
            ShippingLogistics.shipped_date.isnot(None),
        )
        .all()
    )
    if not rows:
        return None
    lead_times = [(sh.actual_delivered_date - sh.shipped_date).days for sh, _ in rows]
    vr = db.query(VendorReliability).filter(VendorReliability.vendor_id == vendor.vendor_id).first()
    return {
        "vendor": vendor,
        "same_class_orders": len(rows),
        "avg_transit_days": sum(lead_times) / len(lead_times),
        "reliability": vr.reliability_score if vr else 0.5,
    }


def find_alternate_supplier(db: Session, material_id: str) -> dict:
    """Only compare vendors with proven history for the *same material class*."""
    ctx = _gather_context(db, material_id)
    material, risk, po, current_vendor, schedule = (
        ctx["material"], ctx["risk"], ctx["po"], ctx["vendor"], ctx["schedule"]
    )
    if not material or not po:
        return {"error": "Missing material or PO data for this material."}

    days_available = (schedule.roj_date - current_date()).days if schedule and schedule.roj_date else 999
    candidates = []
    for vendor in db.query(Vendor).filter(Vendor.vendor_id != po.vendor_id).all():
        stats = _candidate_stats(db, vendor, material.material_class)
        if not stats:
            continue
        # Prototype replacement estimate = 7-day sourcing/mobilization allowance + historical transit.
        stats["estimated_replacement_lead_days"] = round(7 + stats["avg_transit_days"], 1)
        stats["can_meet_roj"] = stats["estimated_replacement_lead_days"] <= max(days_available, 0)
        candidates.append(stats)

    candidates.sort(key=lambda c: (
        not c["can_meet_roj"], -c["reliability"], c["estimated_replacement_lead_days"]
    ))
    candidates = candidates[:3]

    if not candidates:
        return {
            "action_type": "ALTERNATE_SUPPLIER", "material_id": material_id,
            "vendor_id": None, "risk_level_at_creation": risk.risk_level if risk else None,
            "draft_content": f"No alternate vendor with verified {material.material_class} delivery history exists in the current project data.",
            "reasoning": "Compatibility filter rejected vendors without same-material-class delivery history.",
        }

    best = candidates[0]
    candidate_summary = "\n".join(
        f"- {c['vendor'].vendor_name}: reliability {c['reliability']:.2f}, "
        f"{c['same_class_orders']} same-class deliveries, estimated replacement lead "
        f"{c['estimated_replacement_lead_days']:.1f} days, can meet current ROJ: {c['can_meet_roj']}"
        for c in candidates
    )
    fallback = (
        f"Recommended alternate: {best['vendor'].vendor_name}. It has "
        f"{best['same_class_orders']} recorded {material.material_class} deliveries, reliability "
        f"{best['reliability']:.2f}, and an estimated replacement lead of "
        f"{best['estimated_replacement_lead_days']:.1f} days. "
        f"ROJ feasibility from current date: {'Yes' if best['can_meet_roj'] else 'No - use as contingency/expedite option'}. "
        f"Commercial terms and live capacity still require procurement confirmation."
    )
    prompt = f"""
Recommend an alternate supplier in under 100 words using ONLY the supplied evidence.
Material: {material.description}
Material class: {material.material_class}
Current vendor: {current_vendor.vendor_name if current_vendor else 'Unknown'}
Days until ROJ: {days_available}
Candidates:\n{candidate_summary}
Mention that live capacity, price and commercial terms require confirmation.
"""
    draft_content = _generate(prompt, fallback)
    return {
        "action_type": "ALTERNATE_SUPPLIER", "material_id": material_id,
        "vendor_id": best["vendor"].vendor_id,
        "risk_level_at_creation": risk.risk_level if risk else None,
        "draft_content": draft_content,
        "reasoning": (
            f"Filtered to vendors with verified {material.material_class} history; ranked by ROJ feasibility, "
            f"reliability and same-class lead time."
        ),
    }
