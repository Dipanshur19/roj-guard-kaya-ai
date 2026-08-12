"""
Layer 2 schema additions: feature store tables for computed metrics
(lead time, vendor reliability).

Note: `vendor_id` and `actual_delivered_date` now live directly on
ShippingLogistics in models_layer1.py, so patch_schema() is kept only as a
safe no-op fallback for older databases created before that column existed.
"""

from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from models_layer1 import Base, gen_uuid


class ShipmentFeature(Base):
    """
    Computed lead-time metrics per shipment.
    - lead_time_days: actual time taken (dispatch -> delivery) if completed
    - current_transit_days: elapsed time so far (dispatch -> now) if still in transit
    """
    __tablename__ = "shipment_features"
    feature_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    shipment_id = Column(UUID(as_uuid=False), ForeignKey("shipping_logistics.shipment_id"))
    vendor_id = Column(UUID(as_uuid=False), ForeignKey("vendors.vendor_id"), nullable=True)
    material_id = Column(UUID(as_uuid=False), ForeignKey("materials.material_id"), nullable=True)
    is_ongoing = Column(Boolean, default=False)  # True if still in transit
    lead_time_days = Column(Float, nullable=True)          # filled when delivered
    current_transit_days = Column(Float, nullable=True)    # filled when ongoing
    delay_vs_roj_days = Column(Float, nullable=True)       # actual/estimated - roj_date
    delay_bucket = Column(String(30), nullable=True)  # not_too_late, quite_late, very_late, no_delivery
    computed_at = Column(DateTime, default=datetime.utcnow)


class VendorReliability(Base):
    """
    Computed reliability score per vendor using the weighted formula:
    f(vendor) = (0*no_delivery + 1*very_late + 3*quite_late + 5*not_too_late)
                / (5 * total_delivery_requests_accepted)
    """
    __tablename__ = "vendor_reliability"
    record_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    vendor_id = Column(UUID(as_uuid=False), ForeignKey("vendors.vendor_id"))
    reliability_score = Column(Float, nullable=False)  # 0.0 - 1.0
    total_requests_accepted = Column(Integer)
    count_no_delivery = Column(Integer, default=0)
    count_very_late = Column(Integer, default=0)
    count_quite_late = Column(Integer, default=0)
    count_not_too_late = Column(Integer, default=0)
    computed_at = Column(DateTime, default=datetime.utcnow)


def patch_schema(engine):
    """Safe no-op fallback: adds actual_delivered_date/vendor_id to
    shipping_logistics if running against an older DB missing them."""
    with engine.connect() as conn:
        for stmt in [
            "ALTER TABLE shipping_logistics ADD COLUMN actual_delivered_date DATE",
            "ALTER TABLE shipping_logistics ADD COLUMN vendor_id VARCHAR",
        ]:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass  # column already exists -- safe to ignore
