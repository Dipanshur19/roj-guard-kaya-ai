"""
Layer 2: Feature Engineering
Computes two core features from Layer 1 data and stores them for Layer 3:

1. LEAD TIME
   lead_time_days = actual_delivered_date - shipped_date   (if delivered)
   current_transit_days = today - shipped_date             (if still in transit)
   "shipped_date" is used as the dispatch time reference, since that is the
   only reliably-captured "time of dispatch" field in shipping_logistics.

2. VENDOR RELIABILITY
   f(vendor) = (0*no_delivery + 1*very_late + 3*quite_late + 5*not_too_late)
               / (5 * total_delivery_requests_accepted)

   Delay buckets are computed by comparing the shipment's actual (or, if still
   in transit, estimated) arrival date against the material's ROJ date:
     - no_delivery : PO accepted but shipment never dispatched/delivered
     - very_late   : delivered more than 7 days after ROJ date
     - quite_late  : delivered 3-7 days after ROJ date
     - not_too_late: delivered within 0-2 days after ROJ date (or on time/early)
"""

from datetime import date, datetime
from time_utils import current_date
from sqlalchemy.orm import Session

from models_layer1 import PurchaseOrder, ShippingLogistics, ScheduleROJ, Material
from models_layer2 import ShipmentFeature, VendorReliability

WEIGHTS = {
    "no_delivery": 0,
    "very_late": 1,
    "quite_late": 3,
    "not_too_late": 5,
}

VERY_LATE_THRESHOLD_DAYS = 7
QUITE_LATE_THRESHOLD_DAYS = 3


def _classify_delay(delay_days: float) -> str:
    """delay_days = (actual or estimated arrival) - roj_date, in days. Positive = late."""
    if delay_days is None:
        return "no_delivery"
    if delay_days > VERY_LATE_THRESHOLD_DAYS:
        return "very_late"
    if delay_days > QUITE_LATE_THRESHOLD_DAYS:
        return "quite_late"
    return "not_too_late"  # covers on-time, early, and up to 3 days late


def _get_roj_date_for_material(db: Session, material_id: str):
    if not material_id:
        return None
    roj = (
        db.query(ScheduleROJ)
        .filter(ScheduleROJ.material_id == material_id)
        .order_by(ScheduleROJ.roj_date.asc())
        .first()
    )
    return roj.roj_date if roj else None


def compute_lead_time_features(db: Session):
    """
    Iterates every shipment record, computes lead time (or current transit
    duration) and delay-vs-ROJ classification, and upserts into shipment_features.
    """
    shipments = db.query(ShippingLogistics).all()
    today = current_date()
    results = []

    for shipment in shipments:
        roj_date = _get_roj_date_for_material(db, shipment.material_id)

        is_ongoing = shipment.status in ("in_transit", "delayed")
        lead_time_days = None
        current_transit_days = None
        delay_days = None

        if shipment.status == "delivered" and shipment.actual_delivered_date and shipment.shipped_date:
            lead_time_days = (shipment.actual_delivered_date - shipment.shipped_date).days
            if roj_date:
                delay_days = (shipment.actual_delivered_date - roj_date).days

        elif is_ongoing and shipment.shipped_date:
            current_transit_days = (today - shipment.shipped_date).days
            if roj_date and shipment.estimated_arrival:
                delay_days = (shipment.estimated_arrival - roj_date).days

        delay_bucket = _classify_delay(delay_days) if (lead_time_days is not None or delay_days is not None) else None

        feature = (
            db.query(ShipmentFeature)
            .filter(ShipmentFeature.shipment_id == shipment.shipment_id)
            .first()
        )
        if feature is None:
            feature = ShipmentFeature(shipment_id=shipment.shipment_id)
            db.add(feature)

        feature.vendor_id = getattr(shipment, "vendor_id", None)
        feature.material_id = shipment.material_id
        feature.is_ongoing = is_ongoing
        feature.lead_time_days = lead_time_days
        feature.current_transit_days = current_transit_days
        feature.delay_vs_roj_days = delay_days
        feature.delay_bucket = delay_bucket
        feature.computed_at = datetime.utcnow()

        results.append(feature)

    db.commit()
    return results


def compute_vendor_reliability(db: Session, vendor_id: str = None):
    """Compute reliability from *matured* outcomes only.

    Open POs whose ROJ date is still in the future are excluded instead of being
    incorrectly counted as no-delivery failures.
    """
    po_query = db.query(PurchaseOrder)
    if vendor_id:
        po_query = po_query.filter(PurchaseOrder.vendor_id == vendor_id)
    vendor_ids = {vendor_id} if vendor_id else {po.vendor_id for po in po_query.all() if po.vendor_id}
    results = []
    today = current_date()

    for vid in vendor_ids:
        pos = db.query(PurchaseOrder).filter(PurchaseOrder.vendor_id == vid).all()
        counts = {"no_delivery": 0, "very_late": 0, "quite_late": 0, "not_too_late": 0}
        evaluated = 0

        for po in pos:
            shipment = (
                db.query(ShippingLogistics)
                .filter(ShippingLogistics.material_id == po.material_id)
                .order_by(ShippingLogistics.created_at.desc())
                .first()
            )
            schedule = (
                db.query(ScheduleROJ)
                .filter(ScheduleROJ.material_id == po.material_id)
                .order_by(ScheduleROJ.created_at.desc())
                .first()
            )

            if shipment and (shipment.status or "").lower() == "delivered":
                feature = (
                    db.query(ShipmentFeature)
                    .filter(ShipmentFeature.shipment_id == shipment.shipment_id)
                    .first()
                )
                bucket = feature.delay_bucket if feature and feature.delay_bucket else "no_delivery"
                counts[bucket] += 1
                evaluated += 1
            elif schedule and schedule.roj_date and schedule.roj_date < today:
                counts["no_delivery"] += 1
                evaluated += 1
            else:
                # Still-open but not yet due: not a reliability failure.
                continue

        if evaluated == 0:
            continue

        numerator = sum(WEIGHTS[bucket] * count for bucket, count in counts.items())
        denominator = 5 * evaluated
        reliability_score = numerator / denominator if denominator > 0 else 0.0

        record = db.query(VendorReliability).filter(VendorReliability.vendor_id == vid).first()
        if record is None:
            record = VendorReliability(vendor_id=vid)
            db.add(record)

        record.reliability_score = round(reliability_score, 4)
        record.total_requests_accepted = evaluated
        record.count_no_delivery = counts["no_delivery"]
        record.count_very_late = counts["very_late"]
        record.count_quite_late = counts["quite_late"]
        record.count_not_too_late = counts["not_too_late"]
        record.computed_at = datetime.utcnow()
        results.append(record)

    db.commit()
    return results
