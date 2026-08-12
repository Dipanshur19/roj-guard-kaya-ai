"""Layer 3 API: train, compute, persist and retrieve *active* ROJ risks."""

from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database_layer1 import get_db
from models_layer1 import Material, Vendor, ShippingLogistics, PurchaseOrder
from models_layer3 import MaterialRiskScore
from train_models_layer3 import train_and_save_models
from inference_layer3 import compute_risk_for_material

router = APIRouter(prefix="/api/risk", tags=["Layer 3 - Prediction & Risk Scoring"])


def _latest_shipment(db: Session, material_id: str):
    return (
        db.query(ShippingLogistics)
        .filter(ShippingLogistics.material_id == material_id)
        .order_by(ShippingLogistics.created_at.desc())
        .first()
    )


def _is_active_material(db: Session, material_id: str) -> bool:
    shipment = _latest_shipment(db, material_id)
    return not shipment or (shipment.status or "").lower() != "delivered"


def _store_result(db: Session, material_id: str, result: dict):
    record = db.query(MaterialRiskScore).filter(MaterialRiskScore.material_id == material_id).first()
    if record is None:
        record = MaterialRiskScore(material_id=material_id)
        db.add(record)

    record.project_id = result.get("project_id")
    record.vendor_id = result.get("vendor_id")
    record.predicted_lead_time_days = result.get("predicted_lead_time_days")
    record.predicted_arrival_date = result.get("predicted_arrival_date")
    record.roj_date = result.get("roj_date")
    record.predicted_delay_days = result.get("predicted_delay_days")
    record.miss_roj_probability = result.get("miss_roj_probability")
    record.risk_level = result.get("risk_level")
    record.explanation = result.get("explanation")
    record.is_critical_path = result.get("is_critical_path", False)
    record.computed_at = datetime.utcnow()
    db.commit()
    db.refresh(record)
    return record


@router.post("/train")
def trigger_training():
    return train_and_save_models()


@router.post("/compute/{material_id}")
def compute_and_store_risk(material_id: str, db: Session = Depends(get_db)):
    result = compute_risk_for_material(db, material_id)
    if result.get("inactive"):
        db.query(MaterialRiskScore).filter(MaterialRiskScore.material_id == material_id).delete()
        db.commit()
        return result
    if "error" in result:
        return result
    _store_result(db, material_id, result)
    return result


@router.post("/compute-all")
def compute_risk_for_all_materials(db: Session = Depends(get_db)):
    """Score only live procurement lines. Delivered history is training-only."""
    materials = db.query(Material).all()
    active = [m for m in materials if _is_active_material(db, m.material_id)]
    delivered_ids = [m.material_id for m in materials if m not in active]
    if delivered_ids:
        db.query(MaterialRiskScore).filter(MaterialRiskScore.material_id.in_(delivered_ids)).delete(
            synchronize_session=False
        )
        db.commit()

    results = []
    for m in active:
        result = compute_risk_for_material(db, m.material_id)
        if "error" not in result and not result.get("inactive"):
            _store_result(db, m.material_id, result)
        results.append(result)
    return {
        "status": "computed",
        "materials_processed": len(results),
        "historical_delivered_excluded": len(delivered_ids),
        "results": results,
    }


def _serialize_score(db: Session, r: MaterialRiskScore):
    material = db.query(Material).filter(Material.material_id == r.material_id).first()
    vendor = db.query(Vendor).filter(Vendor.vendor_id == r.vendor_id).first() if r.vendor_id else None
    po = (
        db.query(PurchaseOrder)
        .filter(PurchaseOrder.material_id == r.material_id)
        .order_by(PurchaseOrder.created_at.desc())
        .first()
    )
    return {
        "material_id": r.material_id,
        "material_name": material.description if material else r.material_id,
        "sku": material.sku if material else None,
        "po_number": po.po_number if po else None,
        "vendor_id": r.vendor_id,
        "vendor_name": vendor.vendor_name if vendor else None,
        "risk_level": r.risk_level,
        "miss_roj_probability": r.miss_roj_probability,
        "predicted_lead_time_days": r.predicted_lead_time_days,
        "predicted_arrival_date": r.predicted_arrival_date,
        "roj_date": r.roj_date,
        "predicted_delay_days": r.predicted_delay_days,
        "is_critical_path": r.is_critical_path,
        "explanation": r.explanation,
        "computed_at": r.computed_at,
    }


@router.get("/scores")
def get_all_risk_scores(db: Session = Depends(get_db)):
    order_map = {"High": 0, "Medium": 1, "Low": 2}
    records = [
        r for r in db.query(MaterialRiskScore).all()
        if _is_active_material(db, r.material_id)
    ]
    records.sort(key=lambda r: (order_map.get(r.risk_level, 3), -(r.miss_roj_probability or 0)))
    return [_serialize_score(db, r) for r in records]


@router.get("/scores/{material_id}")
def get_risk_score(material_id: str, db: Session = Depends(get_db)):
    if not _is_active_material(db, material_id):
        return {"inactive": True, "status": "delivered", "material_id": material_id}
    record = db.query(MaterialRiskScore).filter(MaterialRiskScore.material_id == material_id).first()
    if not record:
        return {"error": "No risk score computed yet for this material"}
    return _serialize_score(db, record)
