"""
Layer 3 feature preparation.

The model now uses *current operational signals*, not only static PO/vendor
history. A new vendor email, shipment ETA/status update, submittal change, or
ROJ/schedule change therefore changes the feature row used for inference.

Historical training rows are built from a pre-delivery snapshot and vendor
history available *before* that snapshot to reduce target leakage.
"""

from datetime import date, timedelta
from time_utils import current_date
import pandas as pd
from sqlalchemy.orm import Session

from models_layer1 import (
    PurchaseOrder, ShippingLogistics, Material, ScheduleROJ,
    VendorCommunication, Submittal,
)
from models_layer2 import ShipmentFeature

CATEGORICAL_COLS = [
    "material_class",
    "carrier",
    "shipment_status",
    "submittal_status",
    "latest_comm_type",
]

NUMERIC_COLS = [
    "vendor_reliability_score",
    "quantity",
    "unit_price",
    "order_to_promised_days",
    "vendor_avg_lead_time_days",
    "float_days",
    "days_until_roj",
    "current_transit_days",
    "estimated_delay_vs_roj_days",
    "latest_vendor_delay_days",
    "days_since_latest_vendor_comm",
    "submittal_approval_days",
    "days_past_promised_ship_date",
    "is_critical_path_int",
]


def _get_schedule(db: Session, material_id: str):
    return (
        db.query(ScheduleROJ)
        .filter(ScheduleROJ.material_id == material_id)
        .order_by(ScheduleROJ.created_at.desc(), ScheduleROJ.roj_date.asc())
        .first()
    )


def _latest_shipment(db: Session, material_id: str):
    return (
        db.query(ShippingLogistics)
        .filter(ShippingLogistics.material_id == material_id)
        .order_by(ShippingLogistics.created_at.desc())
        .first()
    )


def _latest_comm(db: Session, material_id: str, as_of_date: date):
    return (
        db.query(VendorCommunication)
        .filter(
            VendorCommunication.material_id == material_id,
            VendorCommunication.message_date.isnot(None),
            VendorCommunication.message_date <= as_of_date,
        )
        .order_by(VendorCommunication.message_date.desc(), VendorCommunication.created_at.desc())
        .first()
    )


def _latest_submittal(db: Session, material_id: str):
    return (
        db.query(Submittal)
        .filter(Submittal.material_id == material_id)
        .order_by(Submittal.created_at.desc())
        .first()
    )


def _class_avg_lead_time(db: Session, material_class: str, as_of_date: date, exclude_shipment_id: str = None):
    query = (
        db.query(ShippingLogistics, Material)
        .join(Material, Material.material_id == ShippingLogistics.material_id)
        .filter(
            Material.material_class == material_class,
            ShippingLogistics.actual_delivered_date.isnot(None),
            ShippingLogistics.shipped_date.isnot(None),
            ShippingLogistics.actual_delivered_date < as_of_date,
        )
    )
    if exclude_shipment_id:
        query = query.filter(ShippingLogistics.shipment_id != exclude_shipment_id)
    values = [
        (sh.actual_delivered_date - sh.shipped_date).days
        for sh, _ in query.all()
        if sh.actual_delivered_date and sh.shipped_date
    ]
    return sum(values) / len(values) if values else 30.0


def _vendor_history_stats(
    db: Session,
    vendor_id: str,
    as_of_date: date,
    material_class: str,
    exclude_shipment_id: str = None,
):
    """Return leakage-safe prior average lead time and reliability.

    Only outcomes completed before ``as_of_date`` are considered and the row
    currently being predicted can be excluded.
    """
    query = db.query(ShippingLogistics).filter(
        ShippingLogistics.vendor_id == vendor_id,
        ShippingLogistics.actual_delivered_date.isnot(None),
        ShippingLogistics.shipped_date.isnot(None),
        ShippingLogistics.actual_delivered_date < as_of_date,
    )
    if exclude_shipment_id:
        query = query.filter(ShippingLogistics.shipment_id != exclude_shipment_id)

    shipments = query.all()
    lead_times = []
    reliability_points = []

    for sh in shipments:
        lead_times.append((sh.actual_delivered_date - sh.shipped_date).days)
        sched = _get_schedule(db, sh.material_id)
        if sched and sched.roj_date:
            delay = (sh.actual_delivered_date - sched.roj_date).days
            if delay <= 3:
                reliability_points.append(5)
            elif delay <= 7:
                reliability_points.append(3)
            else:
                reliability_points.append(1)

    avg_lead = sum(lead_times) / len(lead_times) if lead_times else _class_avg_lead_time(
        db, material_class, as_of_date, exclude_shipment_id
    )
    reliability = (
        sum(reliability_points) / (5 * len(reliability_points))
        if reliability_points else 0.65
    )
    return float(avg_lead), float(reliability)


def _historical_snapshot_date(db: Session, shipment: ShippingLogistics) -> date:
    """Create a realistic pre-outcome observation point for a completed shipment."""
    shipped = shipment.shipped_date
    delivered = shipment.actual_delivered_date
    if not shipped or not delivered:
        return shipped or delivered or current_date()

    snapshot = shipped + timedelta(days=7)
    latest_comm = (
        db.query(VendorCommunication)
        .filter(
            VendorCommunication.material_id == shipment.material_id,
            VendorCommunication.message_date.isnot(None),
            VendorCommunication.message_date >= shipped,
            VendorCommunication.message_date < delivered,
        )
        .order_by(VendorCommunication.message_date.desc())
        .first()
    )
    if latest_comm and latest_comm.message_date:
        snapshot = max(snapshot, latest_comm.message_date)

    # Keep the snapshot strictly before the actual outcome.
    if snapshot >= delivered:
        snapshot = max(shipped, delivered - timedelta(days=1))
    return snapshot


def _build_snapshot_row(
    db: Session,
    po: PurchaseOrder,
    shipment: ShippingLogistics,
    as_of_date: date,
    historical: bool,
) -> dict:
    material = db.query(Material).filter(Material.material_id == po.material_id).first()
    schedule = _get_schedule(db, po.material_id)
    submittal = _latest_submittal(db, po.material_id)
    comm = _latest_comm(db, po.material_id, as_of_date)

    material_class = material.material_class if material else "unknown"
    vendor_avg_lead, vendor_reliability = _vendor_history_stats(
        db,
        po.vendor_id,
        as_of_date,
        material_class,
        exclude_shipment_id=shipment.shipment_id if historical and shipment else None,
    )

    order_to_promised = (
        (po.promised_ship_date - po.order_date).days
        if po.order_date and po.promised_ship_date else 0
    )

    roj_date = schedule.roj_date if schedule else None
    days_until_roj = (roj_date - as_of_date).days if roj_date else 60

    shipped_date = shipment.shipped_date if shipment else None
    current_transit_days = (
        max(0, (as_of_date - shipped_date).days)
        if shipped_date and shipped_date <= as_of_date else 0
    )

    # For a completed historical row, the status at the pre-outcome snapshot was
    # still operationally active, not "delivered".
    if shipment:
        if historical and shipment.actual_delivered_date and as_of_date < shipment.actual_delivered_date:
            shipment_status = "delayed" if (comm and comm.comm_type == "delay_notice") else "in_transit"
        else:
            shipment_status = (shipment.status or "unknown").lower()
    else:
        shipment_status = "not_shipped"

    estimated_arrival = shipment.estimated_arrival if shipment else None
    estimated_delay = (
        (estimated_arrival - roj_date).days
        if estimated_arrival and roj_date else 0
    )

    latest_vendor_delay = max(0, int(comm.delay_days_mentioned or 0)) if comm else 0
    days_since_comm = (
        max(0, (as_of_date - comm.message_date).days)
        if comm and comm.message_date else 999
    )

    if submittal:
        if submittal.approved_date and submittal.approved_date <= as_of_date:
            submittal_status = (submittal.approval_status or "approved").lower()
            end_date = submittal.approved_date
        elif submittal.submitted_date and submittal.submitted_date <= as_of_date:
            submittal_status = "pending"
            end_date = as_of_date
        else:
            submittal_status = "not_submitted"
            end_date = as_of_date
        submittal_approval_days = max(0, (end_date - submittal.submitted_date).days) if submittal.submitted_date else 0
    else:
        submittal_status = "unknown"
        submittal_approval_days = 0

    days_past_promised = (
        max(0, (as_of_date - po.promised_ship_date).days)
        if po.promised_ship_date else 0
    )

    is_critical = bool(schedule.is_critical_path) if schedule else False

    if shipped_date:
        prediction_anchor = shipped_date
    elif po.promised_ship_date:
        prediction_anchor = max(as_of_date, po.promised_ship_date)
    else:
        prediction_anchor = as_of_date

    return {
        "material_id": po.material_id,
        "vendor_id": po.vendor_id,
        "project_id": po.project_id,
        "material_class": material_class,
        "carrier": shipment.carrier if shipment and shipment.carrier else "unknown",
        "shipment_status": shipment_status,
        "submittal_status": submittal_status,
        "latest_comm_type": (comm.comm_type or "none").lower() if comm else "none",
        "quantity": float(po.quantity) if po.quantity else 0.0,
        "unit_price": float(po.unit_price) if po.unit_price else 0.0,
        "order_to_promised_days": order_to_promised,
        "vendor_avg_lead_time_days": vendor_avg_lead,
        "vendor_reliability_score": vendor_reliability,
        "float_days": schedule.float_days if schedule and schedule.float_days is not None else 0,
        "days_until_roj": days_until_roj,
        "current_transit_days": current_transit_days,
        "estimated_delay_vs_roj_days": estimated_delay,
        "latest_vendor_delay_days": latest_vendor_delay,
        "days_since_latest_vendor_comm": days_since_comm,
        "submittal_approval_days": submittal_approval_days,
        "days_past_promised_ship_date": days_past_promised,
        "is_critical_path_int": 1 if is_critical else 0,
        "is_critical_path": is_critical,
        "roj_date": roj_date,
        "shipped_date": shipped_date,
        "estimated_arrival": estimated_arrival,
        "prediction_anchor_date": prediction_anchor,
        "latest_vendor_comm_summary": comm.extracted_summary if comm else None,
        "latest_vendor_comm_date": comm.message_date if comm else None,
    }


def build_training_dataframe(db: Session) -> pd.DataFrame:
    """One leakage-reduced pre-delivery snapshot per completed shipment."""
    rows = []
    shipments = (
        db.query(ShippingLogistics)
        .filter(
            ShippingLogistics.status == "delivered",
            ShippingLogistics.actual_delivered_date.isnot(None),
            ShippingLogistics.shipped_date.isnot(None),
        )
        .all()
    )

    for shipment in shipments:
        po = db.query(PurchaseOrder).filter(PurchaseOrder.material_id == shipment.material_id).first()
        if not po:
            continue
        snapshot = _historical_snapshot_date(db, shipment)
        feature_row = _build_snapshot_row(db, po, shipment, snapshot, historical=True)
        lead_time_days = (shipment.actual_delivered_date - shipment.shipped_date).days
        schedule = _get_schedule(db, shipment.material_id)
        missed_roj = 1 if schedule and shipment.actual_delivered_date > schedule.roj_date else 0
        feature_row["lead_time_days"] = lead_time_days
        feature_row["missed_roj"] = missed_roj
        feature_row["snapshot_date"] = snapshot
        rows.append(feature_row)

    return pd.DataFrame(rows)


def build_inference_row(db: Session, material_id: str) -> dict:
    """Build the current live feature row for an active material."""
    po = (
        db.query(PurchaseOrder)
        .filter(PurchaseOrder.material_id == material_id)
        .order_by(PurchaseOrder.created_at.desc())
        .first()
    )
    if not po:
        return None

    shipment = _latest_shipment(db, material_id)
    if shipment and (shipment.status or "").lower() == "delivered":
        return {"inactive": True, "material_id": material_id, "status": "delivered"}

    return _build_snapshot_row(db, po, shipment, current_date(), historical=False)
