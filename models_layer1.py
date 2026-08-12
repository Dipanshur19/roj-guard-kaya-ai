"""Layer 1 database models only. Do not import API routers here."""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, Date, DateTime, Integer, ForeignKey, Text, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()

def gen_uuid():
    return str(uuid.uuid4())

class Project(Base):
    __tablename__ = "projects"
    project_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    project_name = Column(String(255), nullable=False)
    location = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

class Vendor(Base):
    __tablename__ = "vendors"
    vendor_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    vendor_name = Column(String(255), nullable=False)
    contact_email = Column(String(255))
    reliability_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Material(Base):
    __tablename__ = "materials"
    material_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.project_id"))
    sku = Column(String(100))
    description = Column(Text)
    material_class = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class SourceDocument(Base):
    __tablename__ = "source_documents"
    doc_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.project_id"))
    doc_type = Column(String(50), nullable=False)
    input_method = Column(String(20), nullable=False)
    file_path = Column(Text)
    raw_text = Column(Text)
    extraction_confidence = Column(Float)
    needs_review = Column(Boolean, default=False)
    uploaded_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    po_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    doc_id = Column(UUID(as_uuid=False), ForeignKey("source_documents.doc_id"))
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.project_id"))
    material_id = Column(UUID(as_uuid=False), ForeignKey("materials.material_id"))
    vendor_id = Column(UUID(as_uuid=False), ForeignKey("vendors.vendor_id"))
    po_number = Column(String(100))
    quantity = Column(Numeric)
    order_date = Column(Date)
    promised_ship_date = Column(Date)
    unit_price = Column(Numeric)
    incoterms = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

class Submittal(Base):
    __tablename__ = "submittals"
    submittal_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    doc_id = Column(UUID(as_uuid=False), ForeignKey("source_documents.doc_id"))
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.project_id"))
    material_id = Column(UUID(as_uuid=False), ForeignKey("materials.material_id"))
    spec_section = Column(String(100))
    approval_status = Column(String(50))
    submitted_date = Column(Date)
    approved_date = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)

class VendorCommunication(Base):
    __tablename__ = "vendor_communications"
    comm_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    doc_id = Column(UUID(as_uuid=False), ForeignKey("source_documents.doc_id"))
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.project_id"))
    vendor_id = Column(UUID(as_uuid=False), ForeignKey("vendors.vendor_id"))
    material_id = Column(UUID(as_uuid=False), ForeignKey("materials.material_id"))
    comm_type = Column(String(50))
    message_date = Column(Date)
    extracted_summary = Column(Text)
    delay_days_mentioned = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class ShippingLogistics(Base):
    __tablename__ = "shipping_logistics"
    shipment_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    doc_id = Column(UUID(as_uuid=False), ForeignKey("source_documents.doc_id"))
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.project_id"))
    material_id = Column(UUID(as_uuid=False), ForeignKey("materials.material_id"))
    vendor_id = Column(UUID(as_uuid=False), ForeignKey("vendors.vendor_id"))
    carrier = Column(String(100))
    tracking_number = Column(String(100))
    current_location = Column(String(255))
    shipped_date = Column(Date)
    estimated_arrival = Column(Date)
    actual_delivered_date = Column(Date)
    status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

class ScheduleROJ(Base):
    __tablename__ = "schedule_roj"
    schedule_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    doc_id = Column(UUID(as_uuid=False), ForeignKey("source_documents.doc_id"))
    project_id = Column(UUID(as_uuid=False), ForeignKey("projects.project_id"))
    material_id = Column(UUID(as_uuid=False), ForeignKey("materials.material_id"))
    task_name = Column(String(255))
    roj_date = Column(Date, nullable=False)
    float_days = Column(Integer)
    is_critical_path = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
