"""Layer 4 API: draft -> human approval -> safe execution -> audit trail."""

import json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database_layer1 import get_db
from models_layer4 import AgentAction, ActionExecution
from agents_layer4 import draft_vendor_escalation, draft_resequence_suggestion, find_alternate_supplier
from execution_layer4 import execute_approved_action

router = APIRouter(prefix="/api/actions", tags=["Layer 4 - Agentic Action Layer"])

AGENT_MAP = {
    "ESCALATION": draft_vendor_escalation,
    "RESEQUENCE": draft_resequence_suggestion,
    "ALTERNATE_SUPPLIER": find_alternate_supplier,
}


class ActionEdit(BaseModel):
    draft_content: str


@router.post("/draft/{material_id}")
def draft_action(material_id: str, action_type: str, db: Session = Depends(get_db)):
    action_type = action_type.upper()
    if action_type not in AGENT_MAP:
        raise HTTPException(400, f"action_type must be one of {list(AGENT_MAP.keys())}")
    result = AGENT_MAP[action_type](db, material_id)
    if "error" in result:
        raise HTTPException(400, result["error"])

    action = AgentAction(
        material_id=result["material_id"], vendor_id=result.get("vendor_id"),
        action_type=result["action_type"], risk_level_at_creation=result.get("risk_level_at_creation"),
        draft_content=result["draft_content"], reasoning=result.get("reasoning"), status="pending_approval",
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return _serialize_action(action)


def _serialize_action(a: AgentAction):
    return {
        "action_id": a.action_id, "material_id": a.material_id, "vendor_id": a.vendor_id,
        "status": a.status, "action_type": a.action_type,
        "risk_level_at_creation": a.risk_level_at_creation,
        "draft_content": a.draft_content, "reasoning": a.reasoning,
        "reviewed_by": a.reviewed_by, "reviewed_at": a.reviewed_at, "created_at": a.created_at,
    }


@router.patch("/{action_id}")
def edit_action(action_id: str, payload: ActionEdit, db: Session = Depends(get_db)):
    action = db.query(AgentAction).filter(AgentAction.action_id == action_id).first()
    if not action:
        raise HTTPException(404, "Action not found")
    if action.status != "pending_approval":
        raise HTTPException(409, "Only pending actions can be edited")
    action.draft_content = payload.draft_content
    db.commit()
    db.refresh(action)
    return _serialize_action(action)


@router.get("/pending")
def get_pending_actions(db: Session = Depends(get_db)):
    return [_serialize_action(a) for a in db.query(AgentAction).filter(AgentAction.status == "pending_approval").all()]


@router.get("/all")
def get_all_actions(db: Session = Depends(get_db)):
    actions = db.query(AgentAction).order_by(AgentAction.created_at.desc()).all()
    result = []
    for a in actions:
        item = _serialize_action(a)
        execution = (
            db.query(ActionExecution).filter(ActionExecution.action_id == a.action_id)
            .order_by(ActionExecution.executed_at.desc()).first()
        )
        if execution:
            item["execution"] = {
                "execution_id": execution.execution_id,
                "status": execution.status,
                "output_path": execution.output_path,
                "details": json.loads(execution.details_json or "{}"),
                "executed_at": execution.executed_at,
            }
        else:
            item["execution"] = None
        result.append(item)
    return result


@router.post("/{action_id}/approve")
def approve_action(action_id: str, reviewed_by: str = "user", db: Session = Depends(get_db)):
    action = db.query(AgentAction).filter(AgentAction.action_id == action_id).first()
    if not action:
        raise HTTPException(404, "Action not found")
    if action.status != "pending_approval":
        raise HTTPException(409, f"Action is already {action.status}")

    action.status = "approved"
    action.reviewed_by = reviewed_by
    action.reviewed_at = datetime.utcnow()
    db.commit()
    execution = execute_approved_action(db, action)
    return {
        "status": "approved",
        "action_id": action_id,
        "execution_status": execution.status,
        "execution_output": execution.output_path,
        "execution_details": json.loads(execution.details_json or "{}"),
    }


@router.post("/{action_id}/reject")
def reject_action(action_id: str, reviewed_by: str = "user", db: Session = Depends(get_db)):
    action = db.query(AgentAction).filter(AgentAction.action_id == action_id).first()
    if not action:
        raise HTTPException(404, "Action not found")
    if action.status != "pending_approval":
        raise HTTPException(409, f"Action is already {action.status}")
    action.status = "rejected"
    action.reviewed_by = reviewed_by
    action.reviewed_at = datetime.utcnow()
    db.commit()
    return {"status": "rejected", "action_id": action_id}
