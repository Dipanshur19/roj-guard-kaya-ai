"""
Layer 2: Sync Service
Glue code that connects Layer 1 ingestion (main_layer1.py) to Entity Linking
and the Project Graph DB. Call `sync_record_to_graph()` right after any record
is inserted (manual or Gemini-extracted) to keep the graph up to date in real time.
"""

from sqlalchemy.orm import Session
from entity_linking_layer2 import resolve_vendor, resolve_material
from graph_builder_layer2 import graph_client
from models_layer1 import Project, Vendor, Material


def ensure_project_in_graph(db: Session, project_id: str):
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if project:
        graph_client.upsert_project(project.project_id, project.project_name, project.location)


def sync_record_to_graph(db: Session, doc_type: str, record):
    """
    Pushes a newly inserted record (PurchaseOrder, Submittal, VendorCommunication,
    ShippingLogistics, or ScheduleROJ) into the Neo4j graph, resolving/creating
    its Material and Vendor nodes first if they aren't already there.
    """
    project_id = record.project_id
    ensure_project_in_graph(db, project_id)

    material = db.query(Material).filter(Material.material_id == record.material_id).first() \
        if getattr(record, "material_id", None) else None
    if material:
        graph_client.upsert_material(
            material.material_id, material.project_id,
            material.description, material.sku, material.material_class,
        )

    vendor = db.query(Vendor).filter(Vendor.vendor_id == getattr(record, "vendor_id", None)).first() \
        if getattr(record, "vendor_id", None) else None
    if vendor:
        graph_client.upsert_vendor(vendor.vendor_id, vendor.vendor_name, vendor.reliability_score)

    if doc_type == "PO" and material and vendor:
        graph_client.upsert_purchase_order(
            record.po_id, material.material_id, vendor.vendor_id,
            record.po_number, record.order_date, record.promised_ship_date,
        )

    elif doc_type == "SUBMITTAL" and material:
        graph_client.upsert_submittal(
            record.submittal_id, material.material_id,
            record.approval_status, record.submitted_date,
        )

    elif doc_type == "SHIPPING" and material:
        graph_client.upsert_shipment(
            record.shipment_id, material.material_id, record.status,
            record.shipped_date, record.estimated_arrival,
        )

    elif doc_type == "SCHEDULE" and material:
        graph_client.upsert_schedule_task(
            record.schedule_id, material.material_id,
            record.task_name, record.roj_date, record.is_critical_path,
        )

    elif doc_type == "VENDOR_EMAIL" and material and vendor:
        graph_client.upsert_vendor_comm(
            record.comm_id, vendor.vendor_id, material.material_id,
            record.comm_type, record.message_date, record.extracted_summary,
            getattr(record, "delay_days_mentioned", None),
        )


def resolve_and_attach_ids(db: Session, project_id: str, fields: dict) -> dict:
    """
    Call this BEFORE inserting a record, to turn raw vendor_name /
    material_description+sku strings from Gemini/manual entry into
    resolved vendor_id / material_id, ready to pass into the ORM model.
    """
    updated = dict(fields)

    if fields.get("vendor_name"):
        updated["vendor_id"] = resolve_vendor(db, fields["vendor_name"])

    if fields.get("material_description") or fields.get("sku"):
        updated["material_id"] = resolve_material(
            db, project_id, fields.get("material_description"), fields.get("sku")
        )

    return updated
