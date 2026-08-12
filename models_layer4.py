"""
Layer 4 persistence models.

AgentAction stores the human-in-the-loop approval lifecycle.
ActionExecution stores what actually happened after approval so the prototype
has a verifiable execution/audit trail without contacting real vendors by
default.
"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from models_layer1 import Base, gen_uuid


class AgentAction(Base):
    __tablename__ = "agent_actions"
    action_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    material_id = Column(UUID(as_uuid=False), ForeignKey("materials.material_id"))
    vendor_id = Column(UUID(as_uuid=False), ForeignKey("vendors.vendor_id"), nullable=True)

    action_type = Column(String(30), nullable=False)  # ESCALATION, RESEQUENCE, ALTERNATE_SUPPLIER
    risk_level_at_creation = Column(String(20), nullable=True)

    draft_content = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=True)

    status = Column(String(20), default="pending_approval")  # pending_approval, approved, rejected
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ActionExecution(Base):
    """Immutable execution receipt created only after human approval."""

    __tablename__ = "action_executions"
    execution_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    action_id = Column(UUID(as_uuid=False), ForeignKey("agent_actions.action_id"), nullable=False)
    execution_type = Column(String(30), nullable=False)
    status = Column(String(30), nullable=False)  # executed, failed
    output_path = Column(Text, nullable=True)
    details_json = Column(Text, nullable=True)
    executed_at = Column(DateTime, default=datetime.utcnow)
