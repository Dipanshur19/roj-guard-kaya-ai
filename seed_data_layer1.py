"""Generate leakage-safer historical training data + a realistic live demo project.

Default dataset:
- 360 completed historical material deliveries for model training
- 30 active current material lines (10 low-like, 10 medium-like, 10 high-like)
- all five Layer-1 sources represented

Running this script RESETs the local demo database and then computes Layer-2
features, trains Layer-3 models, and populates active risk scores.
"""

import random
from datetime import date, timedelta
from time_utils import current_date

from database_layer1 import SessionLocal, init_db, engine
from models_layer1 import (
    Base, Project, Vendor, Material, SourceDocument,
    PurchaseOrder, Submittal, VendorCommunication, ShippingLogistics, ScheduleROJ,
)
from models_layer2 import patch_schema, ShipmentFeature, VendorReliability  # noqa: F401
from models_layer3 import MaterialRiskScore  # noqa: F401
from models_layer4 import AgentAction, ActionExecution  # noqa: F401

random.seed(42)
TODAY = current_date()
NUM_HISTORICAL = 360
NUM_ACTIVE_PER_BUCKET = 10

MATERIAL_TEMPLATES = [
    ("Structural steel beams - Grade A", "STL-A", "structural_steel", 12),
    ("Precast concrete panels", "CON-P", "concrete", 7),
    ("UPS backup system 500kVA", "ELEC-UPS", "electrical", 15),
    ("HVAC rooftop unit", "MECH-HVAC", "mechanical", 14),
    ("Cable tray assembly", "ELEC-CT", "electrical", 8),
    ("Rebar reinforcement bundles", "STL-REBAR", "structural_steel", 9),
    ("Transformer 2000kVA", "ELEC-TX", "electrical", 20),
    ("Fire suppression piping", "MECH-FIRE", "mechanical", 10),
    ("Curtain wall glazing panels", "ARCH-CW", "architectural", 13),
    ("Generator set 1MW", "ELEC-GEN", "electrical", 18),
    ("Structural steel columns", "STL-COL", "structural_steel", 11),
    ("Concrete mix - high strength", "CON-HS", "concrete", 5),
    ("Chiller unit 300 ton", "MECH-CHILL", "mechanical", 17),
    ("Switchgear panel", "ELEC-SG", "electrical", 14),
    ("Roofing membrane rolls", "ARCH-ROOF", "architectural", 9),
]

VENDOR_PROFILES = [
    ("Apex Structural Systems", 0.90, {"structural_steel", "architectural"}),
    ("Bharat Steel & Fabrication", 0.78, {"structural_steel"}),
    ("Metro Civil Products", 0.86, {"concrete", "structural_steel"}),
    ("BuildRight Precast", 0.68, {"concrete", "architectural"}),
    ("Prime Electricals India", 0.91, {"electrical"}),
    ("VoltEdge Power Systems", 0.73, {"electrical", "mechanical"}),
    ("RapidGrid Equipment", 0.56, {"electrical"}),
    ("Summit Mechanical Works", 0.84, {"mechanical"}),
    ("ThermoFlow Systems", 0.66, {"mechanical", "electrical"}),
    ("Horizon Facades", 0.88, {"architectural"}),
    ("Urban Envelope Supply", 0.64, {"architectural", "concrete"}),
    ("Fastline Industrial Supply", 0.48, {"structural_steel", "concrete", "electrical", "mechanical", "architectural"}),
]

CARRIERS = ["BlueDart Freight", "TCI Logistics", "Gati Cargo", "DHL Industrial"]


def _make_doc(db, project_id, doc_type, raw_text=None):
    doc = SourceDocument(
        project_id=project_id, doc_type=doc_type, input_method="MANUAL",
        raw_text=raw_text, uploaded_by="seed_script",
    )
    db.add(doc)
    db.flush()
    return doc


def _choose_vendor(vendors, material_class, quality="any"):
    eligible = [(v, perf, caps) for v, perf, caps in vendors if material_class in caps]
    if quality == "high":
        pool = [x for x in eligible if x[1] >= 0.82] or eligible
    elif quality == "mid":
        pool = [x for x in eligible if 0.62 <= x[1] < 0.82] or eligible
    elif quality == "low":
        pool = [x for x in eligible if x[1] < 0.62] or eligible
    else:
        pool = eligible
    return random.choice(pool)


def _new_material(db, project_id, idx, prefix):
    desc, sku_prefix, mclass, base_transit = MATERIAL_TEMPLATES[idx % len(MATERIAL_TEMPLATES)]
    material = Material(
        project_id=project_id,
        sku=f"{sku_prefix}-{prefix}-{idx:04d}",
        description=f"{desc} [{prefix}-{idx:04d}]",
        material_class=mclass,
    )
    db.add(material)
    db.flush()
    return material, base_transit


def _add_common_records(db, project, material, vendor, idx, order_date, promised_ship_date, roj_date, critical, float_days, submittal_mode="approved"):
    po_doc = _make_doc(db, project.project_id, "PO")
    po = PurchaseOrder(
        doc_id=po_doc.doc_id, project_id=project.project_id, material_id=material.material_id,
        vendor_id=vendor.vendor_id, po_number=f"PO-{idx:05d}", quantity=random.randint(5, 500),
        order_date=order_date, promised_ship_date=promised_ship_date,
        unit_price=round(random.uniform(800, 120000), 2), incoterms=random.choice(["FOB", "CIF", "DAP"]),
    )
    db.add(po)

    submitted = order_date - timedelta(days=random.randint(5, 18))
    if submittal_mode == "pending":
        approved = None
        approval_status = "pending"
    else:
        approved = submitted + timedelta(days=random.randint(3, 14))
        approval_status = "approved"
    sub_doc = _make_doc(db, project.project_id, "SUBMITTAL")
    db.add(Submittal(
        doc_id=sub_doc.doc_id, project_id=project.project_id, material_id=material.material_id,
        spec_section=f"Section {random.randint(1, 26):02d}", approval_status=approval_status,
        submitted_date=submitted, approved_date=approved,
    ))

    sched_doc = _make_doc(db, project.project_id, "SCHEDULE")
    db.add(ScheduleROJ(
        doc_id=sched_doc.doc_id, project_id=project.project_id, material_id=material.material_id,
        task_name=f"Install / commission {material.description.split(' [')[0]}",
        roj_date=roj_date, float_days=float_days, is_critical_path=critical,
    ))
    return po


def _seed_historical(db, project, vendors):
    misses = 0
    for i in range(NUM_HISTORICAL):
        material, base_transit = _new_material(db, project.project_id, i, "HIST")
        vendor, performance, _ = _choose_vendor(vendors, material.material_class)

        order_date = TODAY - timedelta(days=random.randint(100, 700))
        manufacturing = random.randint(18, 55)
        promised_ship = order_date + timedelta(days=manufacturing)

        # Better vendors usually dispatch closer to promise and have less transit variance.
        if random.random() < performance:
            ship_slip = random.randint(-2, 4)
            transit_extra = random.randint(-2, 3)
        else:
            ship_slip = random.randint(5, 16)
            transit_extra = random.randint(3, 12)
        shipped = promised_ship + timedelta(days=ship_slip)
        transit = max(3, base_transit + transit_extra + random.randint(-2, 2))
        actual = shipped + timedelta(days=transit)

        planning_buffer = random.randint(5, 16)
        roj = promised_ship + timedelta(days=base_transit + planning_buffer)
        critical = random.random() < 0.42
        float_days = random.randint(0, 3) if critical else random.randint(4, 14)
        _add_common_records(
            db, project, material, vendor, i, order_date, promised_ship, roj,
            critical, float_days, "approved",
        )

        eta_error = random.randint(-2, 5) if performance >= 0.7 else random.randint(0, 8)
        estimated = shipped + timedelta(days=max(2, base_transit + eta_error))
        ship_doc = _make_doc(db, project.project_id, "SHIPPING")
        db.add(ShippingLogistics(
            doc_id=ship_doc.doc_id, project_id=project.project_id, material_id=material.material_id,
            vendor_id=vendor.vendor_id, carrier=random.choice(CARRIERS),
            tracking_number=f"H-{i:06d}", current_location="Site Delivered",
            shipped_date=shipped, estimated_arrival=estimated, actual_delivered_date=actual,
            status="delivered",
        ))

        actual_delay = (actual - roj).days
        if actual_delay > 0:
            misses += 1
        projected_delay = (estimated - roj).days
        comm_date = min(actual - timedelta(days=1), shipped + timedelta(days=random.randint(4, 10)))
        if comm_date > shipped:
            is_delay_notice = projected_delay > 0 or (actual_delay > 5 and random.random() < 0.7)
            delay_mentioned = max(1, projected_delay if projected_delay > 0 else min(actual_delay, 12)) if is_delay_notice else None
            comm_doc = _make_doc(db, project.project_id, "VENDOR_EMAIL")
            db.add(VendorCommunication(
                doc_id=comm_doc.doc_id, project_id=project.project_id, vendor_id=vendor.vendor_id,
                material_id=material.material_id,
                comm_type="delay_notice" if is_delay_notice else "status_update",
                message_date=comm_date,
                extracted_summary=(
                    f"{vendor.vendor_name} reported a {delay_mentioned}-day schedule risk for {material.description}."
                    if is_delay_notice else f"{vendor.vendor_name} confirmed progress remains on the current logistics plan."
                ),
                delay_days_mentioned=delay_mentioned,
            ))
    return misses


def _seed_active_bucket(db, project, vendors, bucket, start_idx):
    for j in range(NUM_ACTIVE_PER_BUCKET):
        idx = start_idx + j
        material, base_transit = _new_material(db, project.project_id, idx, bucket.upper())
        quality = {"low": "high", "medium": "mid", "high": "low"}[bucket]
        vendor, _, _ = _choose_vendor(vendors, material.material_class, quality)

        if bucket == "low":
            roj = TODAY + timedelta(days=random.randint(30, 55))
            promised_ship = TODAY - timedelta(days=random.randint(2, 12))
            order_date = promised_ship - timedelta(days=random.randint(20, 45))
            critical = random.random() < 0.25
            float_days = random.randint(5, 12)
            sub_mode = "approved"
            status = "in_transit"
            shipped = TODAY - timedelta(days=random.randint(2, 7))
            eta = roj - timedelta(days=random.randint(8, 16))
            comm_type, delay_mentioned = "status_update", None
        elif bucket == "medium":
            roj = TODAY + timedelta(days=random.randint(18, 38))
            promised_ship = TODAY - timedelta(days=random.randint(1, 8))
            order_date = promised_ship - timedelta(days=random.randint(25, 50))
            critical = random.random() < 0.50
            float_days = random.randint(2, 6)
            sub_mode = "pending" if j % 3 == 0 else "approved"
            status = "in_transit" if j % 4 else "not_shipped"
            shipped = None if status == "not_shipped" else TODAY - timedelta(days=random.randint(2, 8))
            eta = roj + timedelta(days=random.randint(0, 3))
            delay_mentioned = random.randint(1, 3) if j % 2 == 0 else None
            comm_type = "delay_notice" if delay_mentioned else "status_update"
        else:
            roj = TODAY + timedelta(days=random.randint(8, 28))
            promised_ship = TODAY - timedelta(days=random.randint(5, 18))
            order_date = promised_ship - timedelta(days=random.randint(28, 60))
            critical = random.random() < 0.75
            float_days = random.randint(0, 2)
            sub_mode = "pending" if j % 2 == 0 else "approved"
            status = "delayed"
            shipped = TODAY - timedelta(days=random.randint(3, 12))
            delay_mentioned = random.randint(8, 15)
            eta = roj + timedelta(days=delay_mentioned)
            comm_type = "delay_notice"

        _add_common_records(
            db, project, material, vendor, idx, order_date, promised_ship, roj,
            critical, float_days, sub_mode,
        )
        ship_doc = _make_doc(db, project.project_id, "SHIPPING")
        db.add(ShippingLogistics(
            doc_id=ship_doc.doc_id, project_id=project.project_id, material_id=material.material_id,
            vendor_id=vendor.vendor_id, carrier=random.choice(CARRIERS),
            tracking_number=f"LIVE-{idx:06d}",
            current_location="Fabrication" if status == "not_shipped" else ("Exception Hub" if status == "delayed" else "Regional Transit Hub"),
            shipped_date=shipped, estimated_arrival=eta, actual_delivered_date=None, status=status,
        ))

        comm_doc = _make_doc(db, project.project_id, "VENDOR_EMAIL")
        db.add(VendorCommunication(
            doc_id=comm_doc.doc_id, project_id=project.project_id, vendor_id=vendor.vendor_id,
            material_id=material.material_id, comm_type=comm_type,
            message_date=TODAY - timedelta(days=random.randint(0, 3)),
            extracted_summary=(
                f"Vendor confirmed an expected {delay_mentioned}-day delay against the current required date."
                if delay_mentioned else "Vendor confirmed fabrication/logistics remains aligned with the current plan."
            ),
            delay_days_mentioned=delay_mentioned,
        ))


def seed(reset=True):
    if reset:
        # Importing all models above registers every table on Base.metadata.
        Base.metadata.drop_all(bind=engine)
    init_db()
    patch_schema(engine)
    db = SessionLocal()
    try:
        project = Project(project_name="Bangalore Hyperscale Data Center - Phase 1", location="Bengaluru, India")
        db.add(project)
        db.flush()

        vendors = []
        for name, performance, capabilities in VENDOR_PROFILES:
            vendor = Vendor(
                vendor_name=name,
                contact_email=name.lower().replace(" ", ".").replace("&", "and") + "@example.com",
                reliability_score=performance,
            )
            db.add(vendor)
            db.flush()
            vendors.append((vendor, performance, capabilities))

        misses = _seed_historical(db, project, vendors)
        base = NUM_HISTORICAL
        _seed_active_bucket(db, project, vendors, "low", base)
        _seed_active_bucket(db, project, vendors, "medium", base + NUM_ACTIVE_PER_BUCKET)
        _seed_active_bucket(db, project, vendors, "high", base + 2 * NUM_ACTIVE_PER_BUCKET)
        db.commit()

        print(f"Seeded {NUM_HISTORICAL} historical completed deliveries; miss rate={misses/NUM_HISTORICAL:.1%}")
        print(f"Seeded {3*NUM_ACTIVE_PER_BUCKET} active materials across low/medium/high demo scenarios")
    finally:
        db.close()

    # Build the prototype into a ready-to-run state in one command.
    from feature_engineering_layer2 import compute_lead_time_features, compute_vendor_reliability
    from train_models_layer3 import train_and_save_models
    from api_layer3 import compute_risk_for_all_materials

    db = SessionLocal()
    try:
        compute_lead_time_features(db)
        compute_vendor_reliability(db)
    finally:
        db.close()

    train_result = train_and_save_models()
    db = SessionLocal()
    try:
        risk_result = compute_risk_for_all_materials(db)
    finally:
        db.close()
    print("Training:", train_result)
    print("Live scoring:", {k: v for k, v in risk_result.items() if k != "results"})
    return train_result, risk_result


if __name__ == "__main__":
    seed(reset=True)
    print("\nReady. Start backend: uvicorn main_layer1:app --reload")
    print("Then dashboard: streamlit run dashboard_layer3.py")
