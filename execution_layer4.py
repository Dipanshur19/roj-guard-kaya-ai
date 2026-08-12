"""Safe execution adapters for approved Layer-4 actions.

Default mode is fully functional but sandboxed:
- escalation -> writes a real outbound-email artifact to demo_outbox/
- resequence -> writes a schedule-change proposal JSON to demo_executions/
- alternate supplier -> writes a sourcing-request JSON to demo_executions/

Set DEMO_EMAIL_MODE=smtp plus SMTP_* variables to send escalation messages to a
controlled test inbox instead of the local outbox.
"""

from datetime import datetime, timedelta
from email.message import EmailMessage
import json
import os
import smtplib
from sqlalchemy.orm import Session

from models_layer1 import Material, Vendor, PurchaseOrder, ScheduleROJ
from models_layer3 import MaterialRiskScore
from models_layer4 import AgentAction, ActionExecution

OUTBOX_DIR = os.environ.get("DEMO_OUTBOX_DIR", "demo_outbox")
EXECUTION_DIR = os.environ.get("DEMO_EXECUTION_DIR", "demo_executions")
os.makedirs(OUTBOX_DIR, exist_ok=True)
os.makedirs(EXECUTION_DIR, exist_ok=True)


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)[:80]


def _context(db: Session, action: AgentAction):
    material = db.query(Material).filter(Material.material_id == action.material_id).first()
    po = (
        db.query(PurchaseOrder).filter(PurchaseOrder.material_id == action.material_id)
        .order_by(PurchaseOrder.created_at.desc()).first()
    )
    schedule = (
        db.query(ScheduleROJ).filter(ScheduleROJ.material_id == action.material_id)
        .order_by(ScheduleROJ.created_at.desc()).first()
    )
    risk = db.query(MaterialRiskScore).filter(MaterialRiskScore.material_id == action.material_id).first()
    vendor = db.query(Vendor).filter(Vendor.vendor_id == action.vendor_id).first() if action.vendor_id else None
    return material, po, schedule, risk, vendor


def _execute_escalation(db: Session, action: AgentAction):
    material, po, _, _, vendor = _context(db, action)
    recipient = vendor.contact_email if vendor and vendor.contact_email else "demo-vendor@example.com"
    subject = f"ROJ Guard escalation - {po.po_number if po else (material.sku if material else action.material_id)}"
    mode = os.environ.get("DEMO_EMAIL_MODE", "outbox").lower()

    if mode == "smtp":
        host = os.environ.get("SMTP_HOST")
        port = int(os.environ.get("SMTP_PORT", "587"))
        username = os.environ.get("SMTP_USERNAME")
        password = os.environ.get("SMTP_PASSWORD")
        sender = os.environ.get("SMTP_FROM", username or "roj-guard@example.com")
        if not host:
            raise RuntimeError("DEMO_EMAIL_MODE=smtp requires SMTP_HOST")
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.set_content(action.draft_content)
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            if os.environ.get("SMTP_STARTTLS", "true").lower() == "true":
                smtp.starttls()
            if username:
                smtp.login(username, password or "")
            smtp.send_message(msg)
        return {"mode": "smtp", "recipient": recipient, "subject": subject}, None

    filename = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{_safe_name(action.action_id)}.txt"
    path = os.path.join(OUTBOX_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TO: {recipient}\nSUBJECT: {subject}\n\n{action.draft_content}\n")
    return {"mode": "local_outbox", "recipient": recipient, "subject": subject}, path


def _execute_resequence(db: Session, action: AgentAction):
    material, _, schedule, risk, _ = _context(db, action)
    if not schedule:
        raise RuntimeError("No schedule record available for resequence execution")
    delay = max(0, int(round(risk.predicted_delay_days or 0))) if risk else 0
    float_days = int(schedule.float_days or 0)
    residual = max(0, delay - float_days)
    proposed_impact_date = schedule.roj_date + timedelta(days=residual)
    payload = {
        "type": "schedule_change_proposal",
        "action_id": action.action_id,
        "material_id": action.material_id,
        "material": material.description if material else None,
        "task_name": schedule.task_name,
        "original_roj_date": str(schedule.roj_date),
        "available_float_days": float_days,
        "predicted_material_delay_days": delay,
        "residual_schedule_impact_days": residual,
        "proposed_impact_date_if_no_other_mitigation": str(proposed_impact_date),
        "proposal_text": action.draft_content,
        "status": "APPROVED_SCENARIO_PENDING_MASTER_SCHEDULE_PUBLISH",
        "generated_at": datetime.utcnow().isoformat(),
    }
    path = os.path.join(EXECUTION_DIR, f"resequence_{_safe_name(action.action_id)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload, path


def _execute_alternate_supplier(db: Session, action: AgentAction):
    material, po, schedule, risk, vendor = _context(db, action)
    payload = {
        "type": "alternate_supplier_sourcing_request",
        "action_id": action.action_id,
        "material_id": action.material_id,
        "material": material.description if material else None,
        "sku": material.sku if material else None,
        "source_po": po.po_number if po else None,
        "recommended_vendor_id": vendor.vendor_id if vendor else None,
        "recommended_vendor": vendor.vendor_name if vendor else None,
        "target_roj_date": str(schedule.roj_date) if schedule else None,
        "risk_level": risk.risk_level if risk else None,
        "request_text": action.draft_content,
        "status": "APPROVED_FOR_RFQ_CAPACITY_AND_COMMERCIAL_CONFIRMATION",
        "generated_at": datetime.utcnow().isoformat(),
    }
    path = os.path.join(EXECUTION_DIR, f"alternate_supplier_{_safe_name(action.action_id)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload, path


def execute_approved_action(db: Session, action: AgentAction) -> ActionExecution:
    try:
        if action.action_type == "ESCALATION":
            details, output_path = _execute_escalation(db, action)
        elif action.action_type == "RESEQUENCE":
            details, output_path = _execute_resequence(db, action)
        elif action.action_type == "ALTERNATE_SUPPLIER":
            details, output_path = _execute_alternate_supplier(db, action)
        else:
            raise RuntimeError(f"Unsupported action type: {action.action_type}")
        status = "executed"
    except Exception as exc:
        details, output_path, status = {"error": str(exc)}, None, "failed"

    execution = ActionExecution(
        action_id=action.action_id,
        execution_type=action.action_type,
        status=status,
        output_path=output_path,
        details_json=json.dumps(details, default=str),
        executed_at=datetime.utcnow(),
    )
    db.add(execution)
    db.commit()
    db.refresh(execution)
    return execution
