"""
Layer 3 schema: stores the output of the forecasting model and risk
classifier so the dashboard (and Layer 4 agentic actions) can read it
without re-running inference every time.
"""

from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from models_layer1 import Base, gen_uuid


class MaterialRiskScore(Base):
    __tablename__ = "material_risk_scores"
    score_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    material_id = Column(UUID(as_uuid=False), ForeignKey("materials.material_id"))
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.project_id"))
    vendor_id = Column(UUID(as_uuid=False), ForeignKey("vendors.vendor_id"), nullable=True)

    predicted_lead_time_days = Column(Float, nullable=True)
    predicted_arrival_date = Column(String(20), nullable=True)  # ISO date string
    roj_date = Column(String(20), nullable=True)
    predicted_delay_days = Column(Float, nullable=True)

    miss_roj_probability = Column(Float, nullable=True)   # 0-1, from classifier
    risk_level = Column(String(20), nullable=True)         # Low, Medium, High
    explanation = Column(Text, nullable=True)

    is_critical_path = Column(Boolean, default=False)
    computed_at = Column(DateTime, default=datetime.utcnow)
