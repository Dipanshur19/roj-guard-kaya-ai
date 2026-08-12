"""
Layer 1 (Data Ingestion) FastAPI backend -- also mounts Layer 2, 3, and 4
routers so the entire pipeline is served from this single entry point.

Supports two input paths for all 5 data units (PO, Submittal, Vendor Comm,
Shipping, Schedule/ROJ):
  1. MANUAL entry  -> direct form/JSON submission
  2. OCR/AI entry  -> PDF upload (or raw pasted text) is sent to Gemini, which
                      acts as the extraction agent and returns structured JSON

Records with low Gemini confidence are flagged `needs_review` and held in a
review queue instead of being auto-inserted.

Graph sync (Neo4j) is wrapped in try/except everywhere it's called -- if no
Neo4j instance is running, ingestion still succeeds and only the graph sync
step is skipped with a printed warning, so the demo never breaks over
optional infrastructure.
"""

import os
import shutil
import uuid
from datetime import date
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # loads GEMINI_API_KEY, NEO4J_URI, etc. from a local .env file if present

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database_layer1 import init_db, get_db, engine
from models_layer1 import (
    SourceDocument, PurchaseOrder, Submittal, VendorCommunication,
    ShippingLogistics, ScheduleROJ
)
from schemas_layer1 import (
    PurchaseOrderIn, SubmittalIn, VendorCommunicationIn,
    ShippingLogisticsIn, ScheduleROJIn
)
from gemini_extractor_layer1 import extract_from_pdf, extract_from_text

from models_layer2 import patch_schema
from sync_service_layer2 import resolve_and_attach_ids, sync_record_to_graph
from feature_api_layer2 import router as feature_router

from api_layer3 import router as risk_router, compute_and_store_risk, _serialize_score
from api_layer4 import router as action_router
from api_experience import router as experience_router
from feature_engineering_layer2 import compute_lead_time_features, compute_vendor_reliability

app = FastAPI(title="ROJ Guard - Full Pipeline API")

app.include_router(feature_router)
app.include_router(risk_router)
app.include_router(action_router)
app.include_router(experience_router)

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.75

UNIT_MODEL_MAP = {
    "PO": PurchaseOrder,
    "SUBMITTAL": Submittal,
    "VENDOR_EMAIL": VendorCommunication,
    "SHIPPING": ShippingLogistics,
    "SCHEDULE": ScheduleROJ,
}




def _refresh_material_risk(db: Session, material_id: str, recompute_delivery_features: bool = False):
    """Recompute the live risk immediately after an operational input changes."""
    if not material_id:
        return None
    try:
        if recompute_delivery_features:
            compute_lead_time_features(db)
            compute_vendor_reliability(db)
        return compute_and_store_risk(material_id, db)
    except Exception as e:
        print(f"[WARNING] Risk refresh skipped for material {material_id}: {e}")
        return {"warning": str(e)}

def _safe_graph_sync(db: Session, doc_type: str, record):
    """Never let a missing/unreachable Neo4j instance break ingestion."""
    try:
        sync_record_to_graph(db, doc_type, record)
    except Exception as e:
        print(f"[WARNING] Graph sync skipped for {doc_type} ({record}): {e}")


@app.on_event("startup")
def on_startup():
    init_db()
    patch_schema(engine)


# ---------------------------------------------------------------------------
# MANUAL ENTRY ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/api/manual-entry/po")
def manual_entry_po(payload: PurchaseOrderIn, db: Session = Depends(get_db)):
    doc = SourceDocument(project_id=payload.project_id, doc_type="PO", input_method="MANUAL", uploaded_by="user")
    db.add(doc)
    db.flush()

    record = PurchaseOrder(doc_id=doc.doc_id, **payload.dict())
    db.add(record)
    db.flush()
    _safe_graph_sync(db, "PO", record)
    db.commit()
    db.refresh(record)
    risk = _refresh_material_risk(db, record.material_id)
    return {"status": "created", "po_id": record.po_id, "risk_refresh": risk}


@app.post("/api/manual-entry/submittal")
def manual_entry_submittal(payload: SubmittalIn, db: Session = Depends(get_db)):
    doc = SourceDocument(project_id=payload.project_id, doc_type="SUBMITTAL", input_method="MANUAL", uploaded_by="user")
    db.add(doc)
    db.flush()

    record = Submittal(doc_id=doc.doc_id, **payload.dict())
    db.add(record)
    db.flush()
    _safe_graph_sync(db, "SUBMITTAL", record)
    db.commit()
    db.refresh(record)
    risk = _refresh_material_risk(db, record.material_id)
    return {"status": "created", "submittal_id": record.submittal_id, "risk_refresh": risk}


@app.post("/api/manual-entry/vendor-comm")
def manual_entry_vendor_comm(payload: VendorCommunicationIn, db: Session = Depends(get_db)):
    doc = SourceDocument(project_id=payload.project_id, doc_type="VENDOR_EMAIL", input_method="MANUAL", uploaded_by="user")
    db.add(doc)
    db.flush()

    record = VendorCommunication(doc_id=doc.doc_id, **payload.dict())
    db.add(record)
    db.flush()
    _safe_graph_sync(db, "VENDOR_EMAIL", record)
    db.commit()
    db.refresh(record)
    risk = _refresh_material_risk(db, record.material_id)
    return {"status": "created", "comm_id": record.comm_id, "risk_refresh": risk}


@app.post("/api/manual-entry/shipping")
def manual_entry_shipping(payload: ShippingLogisticsIn, db: Session = Depends(get_db)):
    doc = SourceDocument(project_id=payload.project_id, doc_type="SHIPPING", input_method="MANUAL", uploaded_by="user")
    db.add(doc)
    db.flush()

    record = ShippingLogistics(doc_id=doc.doc_id, **payload.dict())
    db.add(record)
    db.flush()
    _safe_graph_sync(db, "SHIPPING", record)
    db.commit()
    db.refresh(record)
    risk = _refresh_material_risk(db, record.material_id, recompute_delivery_features=True)
    return {"status": "created", "shipment_id": record.shipment_id, "risk_refresh": risk}


@app.post("/api/manual-entry/schedule")
def manual_entry_schedule(payload: ScheduleROJIn, db: Session = Depends(get_db)):
    doc = SourceDocument(project_id=payload.project_id, doc_type="SCHEDULE", input_method="MANUAL", uploaded_by="user")
    db.add(doc)
    db.flush()

    record = ScheduleROJ(doc_id=doc.doc_id, **payload.dict())
    db.add(record)
    db.flush()
    _safe_graph_sync(db, "SCHEDULE", record)
    db.commit()
    db.refresh(record)
    risk = _refresh_material_risk(db, record.material_id)
    return {"status": "created", "schedule_id": record.schedule_id, "risk_refresh": risk}




class ExtractionApplyPayload(BaseModel):
    project_id: str
    extracted_fields: dict
    confidence: float = 1.0
    notes: Optional[str] = None
    raw_text: Optional[str] = None
    file_path: Optional[str] = None


def _score_snapshot(db: Session, material_id: str):
    if not material_id:
        return None
    from models_layer3 import MaterialRiskScore
    record = db.query(MaterialRiskScore).filter(MaterialRiskScore.material_id == material_id).first()
    return _serialize_score(db, record) if record else None


@app.post("/api/extract/{doc_type}")
async def preview_extraction(
    doc_type: str,
    project_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
):
    """Analyze a document/email without mutating the project.

    The Streamlit experience uses this for an explicit Analyze -> Review -> Apply
    workflow, so judges can see what the LLM extracted before anything changes.
    """
    doc_type = doc_type.upper()
    if doc_type not in UNIT_MODEL_MAP:
        raise HTTPException(400, f"Invalid doc_type. Must be one of {list(UNIT_MODEL_MAP.keys())}")

    file_path = None
    if file is not None:
        ext = os.path.splitext(file.filename)[1].lower() or ".pdf"
        if ext != ".pdf":
            raise HTTPException(400, "Document upload currently supports PDF. Paste text for email/text inputs.")
        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = extract_from_pdf(file_path, doc_type)
    elif raw_text and raw_text.strip():
        result = extract_from_text(raw_text, doc_type)
    else:
        raise HTTPException(400, "Provide a PDF or raw text")

    return {
        "status": "extracted",
        "project_id": project_id,
        "doc_type": doc_type,
        "confidence": result.confidence,
        "extracted_fields": result.extracted_fields,
        "notes": result.notes,
        "raw_text": raw_text,
        "file_path": file_path,
        "requires_human_confirmation": result.confidence < CONFIDENCE_THRESHOLD,
    }


@app.post("/api/apply-extraction/{doc_type}")
def apply_extraction(doc_type: str, payload: ExtractionApplyPayload, db: Session = Depends(get_db)):
    """Human-confirmed application of a previously previewed AI extraction."""
    doc_type = doc_type.upper()
    if doc_type not in UNIT_MODEL_MAP:
        raise HTTPException(400, f"Invalid doc_type. Must be one of {list(UNIT_MODEL_MAP.keys())}")

    resolved = resolve_and_attach_ids(db, payload.project_id, payload.extracted_fields)
    material_id = resolved.get("material_id")
    before = _score_snapshot(db, material_id)

    doc = SourceDocument(
        project_id=payload.project_id, doc_type=doc_type, input_method="AI_CONFIRMED",
        file_path=payload.file_path, raw_text=payload.raw_text,
        extraction_confidence=payload.confidence, needs_review=False, uploaded_by="streamlit_user",
    )
    db.add(doc); db.flush()
    record = _insert_extracted_record(db, doc.doc_id, payload.project_id, doc_type, resolved)
    db.commit()
    after = _refresh_material_risk(
        db, getattr(record, "material_id", None), recompute_delivery_features=(doc_type == "SHIPPING")
    )
    # Fetch the stored representation so before/after cards have identical keys.
    after_snapshot = _score_snapshot(db, getattr(record, "material_id", None))
    return {
        "status": "applied",
        "record_id": str(getattr(record, list(record.__table__.primary_key.columns)[0].name)),
        "resolved_fields": resolved,
        "before_risk": before,
        "after_risk": after_snapshot or after,
    }


# ---------------------------------------------------------------------------
# OCR / GEMINI AGENT ENTRY ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/api/upload/{doc_type}")
async def upload_and_extract(
    doc_type: str,
    project_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    doc_type = doc_type.upper()
    if doc_type not in UNIT_MODEL_MAP:
        raise HTTPException(400, f"Invalid doc_type. Must be one of {list(UNIT_MODEL_MAP.keys())}")

    file_path = None
    if file is not None:
        ext = os.path.splitext(file.filename)[1] or ".pdf"
        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = extract_from_pdf(file_path, doc_type)
    elif raw_text:
        result = extract_from_text(raw_text, doc_type)
    else:
        raise HTTPException(400, "Must provide either a file or raw_text")

    needs_review = result.confidence < CONFIDENCE_THRESHOLD

    doc = SourceDocument(
        project_id=project_id, doc_type=doc_type, input_method="OCR",
        file_path=file_path, raw_text=raw_text,
        extraction_confidence=result.confidence, needs_review=needs_review, uploaded_by="user",
    )
    db.add(doc)
    db.flush()

    response = {
        "doc_id": doc.doc_id, "confidence": result.confidence, "needs_review": needs_review,
        "extracted_fields": result.extracted_fields, "notes": result.notes,
    }

    if needs_review:
        db.commit()
        response["status"] = "pending_review"
        return response

    record = _insert_extracted_record(db, doc.doc_id, project_id, doc_type, result.extracted_fields)
    db.commit()
    response["status"] = "auto_inserted"
    response["record_id"] = str(getattr(record, list(record.__table__.primary_key.columns)[0].name))
    response["risk_refresh"] = _refresh_material_risk(
        db, getattr(record, "material_id", None), recompute_delivery_features=(doc_type == "SHIPPING")
    )
    return response


def _as_date(value):
    if value is None or isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return value


def _insert_extracted_record(db: Session, doc_id: str, project_id: str, doc_type: str, fields: dict):
    fields = resolve_and_attach_ids(db, project_id, fields)

    if doc_type == "PO":
        record = PurchaseOrder(
            doc_id=doc_id, project_id=project_id, material_id=fields.get("material_id"),
            vendor_id=fields.get("vendor_id"), po_number=fields.get("po_number"),
            quantity=fields.get("quantity"), order_date=_as_date(fields.get("order_date")),
            promised_ship_date=_as_date(fields.get("promised_ship_date")), unit_price=fields.get("unit_price"),
            incoterms=fields.get("incoterms"),
        )
    elif doc_type == "SUBMITTAL":
        record = Submittal(
            doc_id=doc_id, project_id=project_id, material_id=fields.get("material_id"),
            spec_section=fields.get("spec_section"), approval_status=fields.get("approval_status"),
            submitted_date=_as_date(fields.get("submitted_date")), approved_date=_as_date(fields.get("approved_date")),
        )
    elif doc_type == "VENDOR_EMAIL":
        record = VendorCommunication(
            doc_id=doc_id, project_id=project_id, material_id=fields.get("material_id"),
            vendor_id=fields.get("vendor_id"), comm_type=fields.get("comm_type"),
            message_date=_as_date(fields.get("message_date")), extracted_summary=fields.get("extracted_summary"),
            delay_days_mentioned=fields.get("delay_days_mentioned"),
        )
    elif doc_type == "SHIPPING":
        record = ShippingLogistics(
            doc_id=doc_id, project_id=project_id, material_id=fields.get("material_id"),
            vendor_id=fields.get("vendor_id"), carrier=fields.get("carrier"),
            tracking_number=fields.get("tracking_number"), current_location=fields.get("current_location"),
            shipped_date=_as_date(fields.get("shipped_date")), estimated_arrival=_as_date(fields.get("estimated_arrival")),
            actual_delivered_date=_as_date(fields.get("actual_delivered_date")), status=fields.get("status"),
        )
    elif doc_type == "SCHEDULE":
        record = ScheduleROJ(
            doc_id=doc_id, project_id=project_id, material_id=fields.get("material_id"),
            task_name=fields.get("task_name"), roj_date=_as_date(fields.get("roj_date")),
            float_days=fields.get("float_days"), is_critical_path=fields.get("is_critical_path", False),
        )
    else:
        raise HTTPException(400, "Unsupported doc_type")

    db.add(record)
    db.flush()
    _safe_graph_sync(db, doc_type, record)
    return record


# ---------------------------------------------------------------------------
# REVIEW QUEUE ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/api/review-queue")
def get_review_queue(db: Session = Depends(get_db)):
    docs = db.query(SourceDocument).filter(SourceDocument.needs_review == True).all()  # noqa: E712
    return [
        {"doc_id": d.doc_id, "doc_type": d.doc_type, "confidence": d.extraction_confidence,
         "raw_text": d.raw_text, "file_path": d.file_path, "created_at": d.created_at}
        for d in docs
    ]


@app.post("/api/review-queue/{doc_id}/confirm")
def confirm_review(doc_id: str, corrected_fields: dict, db: Session = Depends(get_db)):
    doc = db.query(SourceDocument).filter(SourceDocument.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    record = _insert_extracted_record(db, doc.doc_id, doc.project_id, doc.doc_type, corrected_fields)
    doc.needs_review = False
    db.commit()
    risk = _refresh_material_risk(
        db, getattr(record, "material_id", None), recompute_delivery_features=(doc.doc_type == "SHIPPING")
    )
    return {"status": "confirmed_and_inserted", "record_id": str(record), "risk_refresh": risk}


@app.get("/")
def health_check():
    return {"status": "ROJ Guard full pipeline API running"}
