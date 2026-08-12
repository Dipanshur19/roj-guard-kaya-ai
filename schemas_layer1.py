"""
Pydantic schemas for request/response validation.
Used by both manual-entry endpoints and OCR/Gemini extraction output validation.
"""

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class PurchaseOrderIn(BaseModel):
    project_id: str
    material_id: Optional[str] = None
    vendor_id: Optional[str] = None
    po_number: Optional[str] = None
    quantity: Optional[float] = None
    order_date: Optional[date] = None
    promised_ship_date: Optional[date] = None
    unit_price: Optional[float] = None
    incoterms: Optional[str] = None


class SubmittalIn(BaseModel):
    project_id: str
    material_id: Optional[str] = None
    spec_section: Optional[str] = None
    approval_status: Optional[str] = None
    submitted_date: Optional[date] = None
    approved_date: Optional[date] = None


class VendorCommunicationIn(BaseModel):
    project_id: str
    vendor_id: Optional[str] = None
    material_id: Optional[str] = None
    comm_type: Optional[str] = None
    message_date: Optional[date] = None
    extracted_summary: Optional[str] = None
    delay_days_mentioned: Optional[int] = None


class ShippingLogisticsIn(BaseModel):
    project_id: str
    material_id: Optional[str] = None
    vendor_id: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    current_location: Optional[str] = None
    shipped_date: Optional[date] = None
    estimated_arrival: Optional[date] = None
    actual_delivered_date: Optional[date] = None
    status: Optional[str] = None


class ScheduleROJIn(BaseModel):
    project_id: str
    material_id: Optional[str] = None
    task_name: Optional[str] = None
    roj_date: date
    float_days: Optional[int] = None
    is_critical_path: Optional[bool] = False


# ---- Gemini extraction output schema (the AI agent must return this shape) ----

class GeminiExtractionResult(BaseModel):
    doc_type: str = Field(..., description="One of PO, SUBMITTAL, VENDOR_EMAIL, SHIPPING, SCHEDULE")
    confidence: float = Field(..., ge=0.0, le=1.0)
    extracted_fields: dict
    notes: Optional[str] = None
