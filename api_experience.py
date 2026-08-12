"""Product-experience endpoints for the Round-2 prototype.

These endpoints do not replace the four-layer architecture. They expose a
judge-friendly projection of the existing data: material context, project graph,
activity timeline, and a deterministic live demo scenario.
"""

import re
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database_layer1 import get_db
from models_layer1 import (
    Project, Vendor, Material, SourceDocument, PurchaseOrder, Submittal,
    VendorCommunication, ShippingLogistics, ScheduleROJ,
)
from models_layer2 import VendorReliability
from models_layer3 import MaterialRiskScore
from models_layer4 import AgentAction, ActionExecution
from feature_engineering_layer3 import build_inference_row
from inference_layer3 import compute_risk_for_material
from api_layer3 import _store_result, _serialize_score
from agents_layer4 import draft_vendor_escalation
from sync_service_layer2 import sync_record_to_graph
from time_utils import current_date

router = APIRouter(prefix="/api/experience", tags=["Round 2 - Product Experience"])

DEMO_MARKER = "ROJ_GUARD_DEMO_SIGNAL"


def clean_material_name(name: str | None) -> str:
    if not name:
        return "Unknown material"
    # Keep internal unique labels in the data layer, never expose risk-bucket labels to judges.
    return re.sub(r"\s*\[(?:LOW|MEDIUM|HIGH|HIST)-\d+\]\s*$", "", name, flags=re.I).strip()


def clean_sku(sku: str | None) -> str | None:
    if not sku:
        return sku
    return re.sub(r"-(?:LOW|MEDIUM|HIGH|HIST)-", "-", sku, flags=re.I)


def _latest(model, db: Session, material_id: str):
    return (
        db.query(model).filter(model.material_id == material_id)
        .order_by(model.created_at.desc()).first()
    )


def _risk_driver_list(row: dict, score: MaterialRiskScore | None) -> list[dict]:
    drivers = []
    vendor_delay = int(row.get("latest_vendor_delay_days") or 0)
    eta_delay = int(row.get("estimated_delay_vs_roj_days") or 0)
    float_days = int(row.get("float_days") or 0)
    days_until = int(row.get("days_until_roj") or 0)
    reliability = float(row.get("vendor_reliability_score") or 0.65)
    status = (row.get("shipment_status") or "unknown").lower()
    submittal = (row.get("submittal_status") or "unknown").lower()

    if vendor_delay > 0:
        drivers.append({"severity": "high" if vendor_delay >= 8 else "medium", "label": "Vendor delay signal", "detail": f"Vendor reported {vendor_delay} day(s) of delay."})
    if eta_delay > 0:
        drivers.append({"severity": "high" if eta_delay >= 8 else "medium", "label": "ETA beyond ROJ", "detail": f"Current ETA is {eta_delay} day(s) after the required-on-job date."})
    if status == "delayed":
        drivers.append({"severity": "high", "label": "Shipment exception", "detail": "Latest logistics status is delayed."})
    if row.get("is_critical_path"):
        drivers.append({"severity": "high" if float_days <= 2 else "medium", "label": "Critical-path exposure", "detail": f"Material supports a critical-path activity with {float_days} day(s) float."})
    if submittal not in ("approved", "approved_with_comments"):
        drivers.append({"severity": "medium", "label": "Submittal not approved", "detail": f"Latest submittal status is {submittal}."})
    if reliability < 0.60:
        drivers.append({"severity": "medium", "label": "Vendor reliability", "detail": f"Historical reliability score is {reliability:.0%}."})
    elif reliability >= 0.82:
        drivers.append({"severity": "positive", "label": "Vendor reliability", "detail": f"Historical reliability is strong at {reliability:.0%}."})
    if days_until <= 14:
        drivers.append({"severity": "medium", "label": "Limited response window", "detail": f"Only {days_until} day(s) remain until ROJ."})
    if not drivers:
        drivers.append({"severity": "positive", "label": "No dominant exception", "detail": "No explicit delivery exception is currently present."})

    if score and score.predicted_delay_days is not None and score.predicted_delay_days > 0:
        drivers.insert(0, {"severity": "high" if score.predicted_delay_days >= 7 else "medium", "label": "Forecast schedule impact", "detail": f"Operational arrival is forecast {score.predicted_delay_days:.0f} day(s) after ROJ."})
    return drivers[:6]


@router.get("/projects")
def projects(db: Session = Depends(get_db)):
    return [
        {"project_id": p.project_id, "project_name": p.project_name, "location": p.location}
        for p in db.query(Project).order_by(Project.created_at.desc()).all()
    ]


@router.get("/materials")
def active_materials(db: Session = Depends(get_db)):
    scores = db.query(MaterialRiskScore).all()
    result = []
    for r in scores:
        m = db.query(Material).filter(Material.material_id == r.material_id).first()
        v = db.query(Vendor).filter(Vendor.vendor_id == r.vendor_id).first() if r.vendor_id else None
        if not m:
            continue
        result.append({
            "material_id": m.material_id,
            "project_id": m.project_id,
            "material_name": clean_material_name(m.description),
            "sku": clean_sku(m.sku),
            "vendor_id": r.vendor_id,
            "vendor_name": v.vendor_name if v else None,
            "risk_level": r.risk_level,
            "miss_roj_probability": r.miss_roj_probability,
        })
    order = {"High": 0, "Medium": 1, "Low": 2}
    result.sort(key=lambda x: (order.get(x["risk_level"], 9), -(x["miss_roj_probability"] or 0)))
    return result


@router.get("/material/{material_id}")
def material_context(material_id: str, db: Session = Depends(get_db)):
    material = db.query(Material).filter(Material.material_id == material_id).first()
    if not material:
        raise HTTPException(404, "Material not found")
    project = db.query(Project).filter(Project.project_id == material.project_id).first()
    po = _latest(PurchaseOrder, db, material_id)
    shipment = _latest(ShippingLogistics, db, material_id)
    schedule = _latest(ScheduleROJ, db, material_id)
    submittal = _latest(Submittal, db, material_id)
    comm = _latest(VendorCommunication, db, material_id)
    risk = db.query(MaterialRiskScore).filter(MaterialRiskScore.material_id == material_id).first()
    vendor = db.query(Vendor).filter(Vendor.vendor_id == (po.vendor_id if po else None)).first() if po else None
    reliability = None
    if vendor:
        rel = (
            db.query(VendorReliability).filter(VendorReliability.vendor_id == vendor.vendor_id)
            .order_by(VendorReliability.computed_at.desc()).first()
        )
        reliability = rel.reliability_score if rel else vendor.reliability_score

    row = build_inference_row(db, material_id) or {}
    events = []
    if po:
        events.append({"date": str(po.order_date or po.created_at.date()), "type": "Purchase Order", "title": f"{po.po_number} placed", "detail": f"Committed with {vendor.vendor_name if vendor else 'vendor'}; promised ship {po.promised_ship_date}."})
    if submittal:
        events.append({"date": str(submittal.approved_date or submittal.submitted_date or submittal.created_at.date()), "type": "Submittal", "title": f"Submittal {submittal.approval_status or 'updated'}", "detail": f"Spec {submittal.spec_section or 'N/A'}."})
    if shipment:
        events.append({"date": str(shipment.shipped_date or shipment.created_at.date()), "type": "Logistics", "title": f"Shipment {shipment.status or 'updated'}", "detail": f"ETA {shipment.estimated_arrival or 'N/A'} • {shipment.current_location or 'Location unavailable'}."})
    comms = (
        db.query(VendorCommunication).filter(VendorCommunication.material_id == material_id)
        .order_by(VendorCommunication.message_date.desc()).limit(4).all()
    )
    for c in comms:
        events.append({"date": str(c.message_date or c.created_at.date()), "type": "Vendor Signal", "title": "Delay notice" if c.delay_days_mentioned else "Vendor update", "detail": c.extracted_summary or "Vendor communication received."})
    actions = (
        db.query(AgentAction).filter(AgentAction.material_id == material_id)
        .order_by(AgentAction.created_at.desc()).limit(4).all()
    )
    for a in actions:
        events.append({"date": str(a.created_at.date()), "type": "Agent Action", "title": f"{a.action_type.replace('_', ' ').title()} • {a.status}", "detail": a.reasoning or "AI mitigation action created."})
    events.sort(key=lambda x: x["date"], reverse=True)

    return {
        "project": {"project_id": project.project_id, "project_name": project.project_name, "location": project.location} if project else None,
        "material": {"material_id": material.material_id, "name": clean_material_name(material.description), "sku": clean_sku(material.sku), "material_class": material.material_class},
        "vendor": {"vendor_id": vendor.vendor_id, "name": vendor.vendor_name, "email": vendor.contact_email, "reliability": reliability} if vendor else None,
        "po": {"po_id": po.po_id, "po_number": po.po_number, "quantity": float(po.quantity or 0), "order_date": po.order_date, "promised_ship_date": po.promised_ship_date, "unit_price": float(po.unit_price or 0), "incoterms": po.incoterms} if po else None,
        "shipment": {"shipment_id": shipment.shipment_id, "status": shipment.status, "carrier": shipment.carrier, "tracking_number": shipment.tracking_number, "current_location": shipment.current_location, "shipped_date": shipment.shipped_date, "estimated_arrival": shipment.estimated_arrival} if shipment else None,
        "schedule": {"schedule_id": schedule.schedule_id, "task_name": schedule.task_name, "roj_date": schedule.roj_date, "float_days": schedule.float_days, "is_critical_path": bool(schedule.is_critical_path)} if schedule else None,
        "submittal": {"status": submittal.approval_status, "submitted_date": submittal.submitted_date, "approved_date": submittal.approved_date} if submittal else None,
        "latest_vendor_signal": {"comm_type": comm.comm_type, "message_date": comm.message_date, "summary": comm.extracted_summary, "delay_days": comm.delay_days_mentioned} if comm else None,
        "risk": _serialize_score(db, risk) if risk else None,
        "risk_drivers": _risk_driver_list(row, risk),
        "activity": events[:10],
        "live_features": {
            "days_until_roj": row.get("days_until_roj"),
            "vendor_reliability_score": row.get("vendor_reliability_score"),
            "shipment_status": row.get("shipment_status"),
            "estimated_delay_vs_roj_days": row.get("estimated_delay_vs_roj_days"),
            "latest_vendor_delay_days": row.get("latest_vendor_delay_days"),
            "float_days": row.get("float_days"),
            "submittal_status": row.get("submittal_status"),
        },
    }


@router.get("/graph/{material_id}")
def material_graph(material_id: str, db: Session = Depends(get_db)):
    ctx = material_context(material_id, db)
    nodes = []
    edges = []

    def add_node(node_id, node_type, label, subtitle="", status=None):
        if not node_id:
            return
        nodes.append({"id": str(node_id), "type": node_type, "label": str(label), "subtitle": str(subtitle or ""), "status": status})

    p, m, v, po, sh, sched = ctx["project"], ctx["material"], ctx["vendor"], ctx["po"], ctx["shipment"], ctx["schedule"]
    add_node(p["project_id"], "Project", p["project_name"], p.get("location", "") if p else "")
    add_node(m["material_id"], "Material", m["name"], m.get("sku", ""))
    if v: add_node(v["vendor_id"], "Vendor", v["name"], f"Reliability {v.get('reliability'):.0%}" if v.get("reliability") is not None else "")
    if po: add_node(po["po_id"], "PO", po["po_number"], f"Promised ship {po.get('promised_ship_date')}")
    if sh: add_node(sh["shipment_id"], "Shipment", sh.get("status", "Shipment").replace("_", " ").title(), f"ETA {sh.get('estimated_arrival')}", sh.get("status"))
    if sched: add_node(sched["schedule_id"], "ROJ", str(sched["roj_date"]), sched.get("task_name", ""), "critical" if sched.get("is_critical_path") else "planned")

    edges.append({"source": p["project_id"], "target": m["material_id"], "label": "contains"})
    if po: edges.append({"source": m["material_id"], "target": po["po_id"], "label": "ordered via"})
    if v and po: edges.append({"source": po["po_id"], "target": v["vendor_id"], "label": "placed with"})
    if sh: edges.append({"source": m["material_id"], "target": sh["shipment_id"], "label": "tracked by"})
    if sched: edges.append({"source": m["material_id"], "target": sched["schedule_id"], "label": "required by"})
    return {"nodes": nodes, "edges": edges, "risk_level": ctx.get("risk", {}).get("risk_level") if ctx.get("risk") else None}


def _choose_demo_material(db: Session):
    # Prefer a structural-steel line for a construction-native story.
    candidates = (
        db.query(MaterialRiskScore, Material)
        .join(Material, Material.material_id == MaterialRiskScore.material_id)
        .filter(MaterialRiskScore.risk_level == "Low")
        .all()
    )
    if not candidates:
        return None, None
    preferred = [x for x in candidates if (x[1].material_class or "") == "structural_steel"]
    return (preferred or candidates)[0]


@router.get("/demo/scenario")
def demo_scenario(db: Session = Depends(get_db)):
    risk, material = _choose_demo_material(db)
    if not material:
        return {"ready": False, "message": "No Low-risk material is available. Reset Demo Mode first."}
    ctx = material_context(material.material_id, db)
    return {
        "ready": True,
        "material_id": material.material_id,
        "material_name": clean_material_name(material.description),
        "vendor_name": (ctx.get("vendor") or {}).get("name"),
        "before": _serialize_score(db, risk),
        "signal": "Vendor reports a 14-day production and dispatch delay.",
    }


@router.post("/demo/run")
def run_demo(material_id: str | None = None, db: Session = Depends(get_db)):
    if material_id:
        risk = db.query(MaterialRiskScore).filter(MaterialRiskScore.material_id == material_id).first()
        material = db.query(Material).filter(Material.material_id == material_id).first()
    else:
        risk, material = _choose_demo_material(db)
    if not material or not risk:
        raise HTTPException(409, "No demo-ready Low-risk material. Reset Demo Mode first.")
    if risk.risk_level != "Low":
        raise HTTPException(409, "Selected material is no longer Low risk. Reset Demo Mode first.")

    po = _latest(PurchaseOrder, db, material.material_id)
    vendor = db.query(Vendor).filter(Vendor.vendor_id == (po.vendor_id if po else None)).first() if po else None
    if not vendor:
        raise HTTPException(400, "Demo material has no linked vendor")

    before = _serialize_score(db, risk)
    message = (
        f"{DEMO_MARKER}: {vendor.vendor_name} confirms a production and dispatch delay for "
        f"{clean_material_name(material.description)}. Delivery is expected to slip by 14 days due to "
        "manufacturing constraints. Please update the project team."
    )
    doc = SourceDocument(
        project_id=material.project_id, doc_type="VENDOR_EMAIL", input_method="DEMO",
        raw_text=message, extraction_confidence=1.0, needs_review=False, uploaded_by="demo_mode",
    )
    db.add(doc); db.flush()
    comm = VendorCommunication(
        doc_id=doc.doc_id, project_id=material.project_id, vendor_id=vendor.vendor_id,
        material_id=material.material_id, comm_type="delay_notice", message_date=current_date(),
        extracted_summary=f"{vendor.vendor_name} confirmed a 14-day production and dispatch delay.",
        delay_days_mentioned=14,
    )
    db.add(comm); db.flush()
    try:
        sync_record_to_graph(db, "VENDOR_EMAIL", comm)
    except Exception:
        pass
    db.commit()

    result = compute_risk_for_material(db, material.material_id)
    _store_result(db, material.material_id, result)
    updated = db.query(MaterialRiskScore).filter(MaterialRiskScore.material_id == material.material_id).first()
    after = _serialize_score(db, updated)

    existing = (
        db.query(AgentAction).filter(
            AgentAction.material_id == material.material_id,
            AgentAction.action_type == "ESCALATION",
            AgentAction.status == "pending_approval",
            AgentAction.reasoning.like("Demo Mode:%"),
        ).first()
    )
    if existing:
        action = existing
    else:
        draft = draft_vendor_escalation(db, material.material_id)
        if "error" in draft:
            raise HTTPException(400, draft["error"])
        action = AgentAction(
            material_id=draft["material_id"], vendor_id=draft.get("vendor_id"),
            action_type="ESCALATION", risk_level_at_creation=draft.get("risk_level_at_creation"),
            draft_content=draft["draft_content"], reasoning="Demo Mode: " + (draft.get("reasoning") or "Escalation proposed after new delay signal."),
            status="pending_approval",
        )
        db.add(action); db.commit(); db.refresh(action)

    return {
        "status": "demo_completed",
        "material_id": material.material_id,
        "material_name": clean_material_name(material.description),
        "vendor_name": vendor.vendor_name,
        "before": before,
        "signal": {"type": "Vendor delay notice", "delay_days": 14, "summary": comm.extracted_summary},
        "after": after,
        "agent_action": {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "status": action.status,
            "draft_content": action.draft_content,
            "reasoning": action.reasoning,
        },
    }


@router.post("/demo/reset")
def reset_demo(db: Session = Depends(get_db)):
    demo_docs = db.query(SourceDocument).filter(SourceDocument.input_method == "DEMO").all()
    doc_ids = [d.doc_id for d in demo_docs]
    material_ids = []
    if doc_ids:
        comms = db.query(VendorCommunication).filter(VendorCommunication.doc_id.in_(doc_ids)).all()
        material_ids = list({c.material_id for c in comms if c.material_id})
        db.query(VendorCommunication).filter(VendorCommunication.doc_id.in_(doc_ids)).delete(synchronize_session=False)
        db.query(SourceDocument).filter(SourceDocument.doc_id.in_(doc_ids)).delete(synchronize_session=False)

    demo_actions = db.query(AgentAction).filter(AgentAction.reasoning.like("Demo Mode:%")).all()
    action_ids = [a.action_id for a in demo_actions]
    if action_ids:
        db.query(ActionExecution).filter(ActionExecution.action_id.in_(action_ids)).delete(synchronize_session=False)
        db.query(AgentAction).filter(AgentAction.action_id.in_(action_ids)).delete(synchronize_session=False)
    db.commit()

    restored = []
    for mid in material_ids:
        result = compute_risk_for_material(db, mid)
        if "error" not in result and not result.get("inactive"):
            _store_result(db, mid, result)
            restored.append({"material_id": mid, "risk_level": result.get("risk_level"), "probability": result.get("miss_roj_probability")})
    return {"status": "reset", "demo_signals_removed": len(doc_ids), "demo_actions_removed": len(action_ids), "restored": restored}
