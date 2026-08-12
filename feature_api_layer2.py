"""
Layer 2 API endpoints: trigger feature computation and read results.
Mounted into main_layer1.py via `app.include_router(feature_router)`.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database_layer1 import get_db
from models_layer2 import ShipmentFeature, VendorReliability
from feature_engineering_layer2 import compute_lead_time_features, compute_vendor_reliability

router = APIRouter(prefix="/api/features", tags=["Layer 2 - Feature Engineering"])


@router.post("/compute/lead-time")
def trigger_lead_time_computation(db: Session = Depends(get_db)):
    features = compute_lead_time_features(db)
    return {"status": "computed", "shipments_processed": len(features)}


@router.post("/compute/vendor-reliability")
def trigger_vendor_reliability_computation(db: Session = Depends(get_db)):
    records = compute_vendor_reliability(db)
    return {
        "status": "computed",
        "vendors_processed": len(records),
        "results": [
            {
                "vendor_id": r.vendor_id,
                "reliability_score": r.reliability_score,
                "total_requests_accepted": r.total_requests_accepted,
            }
            for r in records
        ],
    }


@router.get("/shipments")
def get_shipment_features(db: Session = Depends(get_db)):
    features = db.query(ShipmentFeature).all()
    return [
        {
            "shipment_id": f.shipment_id,
            "vendor_id": f.vendor_id,
            "material_id": f.material_id,
            "is_ongoing": f.is_ongoing,
            "lead_time_days": f.lead_time_days,
            "current_transit_days": f.current_transit_days,
            "delay_vs_roj_days": f.delay_vs_roj_days,
            "delay_bucket": f.delay_bucket,
            "computed_at": f.computed_at,
        }
        for f in features
    ]


@router.get("/vendor-reliability")
def get_vendor_reliability(db: Session = Depends(get_db)):
    records = db.query(VendorReliability).all()
    return [
        {
            "vendor_id": r.vendor_id,
            "reliability_score": r.reliability_score,
            "total_requests_accepted": r.total_requests_accepted,
            "breakdown": {
                "no_delivery": r.count_no_delivery,
                "very_late": r.count_very_late,
                "quite_late": r.count_quite_late,
                "not_too_late": r.count_not_too_late,
            },
            "computed_at": r.computed_at,
        }
        for r in records
    ]


@router.get("/vendor-reliability/{vendor_id}")
def get_vendor_reliability_single(vendor_id: str, db: Session = Depends(get_db)):
    record = db.query(VendorReliability).filter(VendorReliability.vendor_id == vendor_id).first()
    if not record:
        return {"error": "No reliability record found for this vendor"}
    return {
        "vendor_id": record.vendor_id,
        "reliability_score": record.reliability_score,
        "total_requests_accepted": record.total_requests_accepted,
        "breakdown": {
            "no_delivery": record.count_no_delivery,
            "very_late": record.count_very_late,
            "quite_late": record.count_quite_late,
            "not_too_late": record.count_not_too_late,
        },
    }
