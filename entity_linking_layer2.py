"""
Layer 2: Entity Linking
Resolves raw extracted values (vendor_name strings, material descriptions/SKUs
from Gemini or manual entry) into consistent vendor_id / material_id references,
creating new records only when no reasonable match exists.

Uses fuzzy string matching (rapidfuzz) since the same vendor or material can
appear with slightly different spellings across a PO, a vendor email, and a
shipping manifest (e.g. "Acme Steel Pvt Ltd" vs "ACME Steel").
"""

from rapidfuzz import fuzz
from sqlalchemy.orm import Session

from models_layer1 import Vendor, Material

FUZZY_MATCH_THRESHOLD = 85  # 0-100 scale; below this, treat as a new entity


def resolve_vendor(db: Session, vendor_name: str, contact_email: str = None) -> str:
    """
    Finds an existing vendor by fuzzy name match, or creates a new one.
    Returns the vendor_id.
    """
    if not vendor_name:
        return None

    vendor_name_clean = vendor_name.strip().lower()
    existing_vendors = db.query(Vendor).all()

    best_match = None
    best_score = 0
    for v in existing_vendors:
        score = fuzz.token_sort_ratio(vendor_name_clean, v.vendor_name.strip().lower())
        if score > best_score:
            best_score = score
            best_match = v

    if best_match and best_score >= FUZZY_MATCH_THRESHOLD:
        return best_match.vendor_id

    new_vendor = Vendor(vendor_name=vendor_name.strip(), contact_email=contact_email)
    db.add(new_vendor)
    db.flush()
    return new_vendor.vendor_id


def resolve_material(db: Session, project_id: str, description: str, sku: str = None, vendor_id: str = None, active_only: bool = False) -> str:
    """
    Finds an existing material within the same project, preferring exact SKU
    match, falling back to fuzzy description match, or creates a new one.
    Returns the material_id.
    """
    if not description and not sku:
        return None

    # 1. Exact SKU match takes priority (most reliable identifier)
    if sku:
        exact = (
            db.query(Material)
            .filter(Material.project_id == project_id, Material.sku == sku)
            .first()
        )
        if exact:
            return exact.material_id

    # 2. Fuzzy match on description within the same project
    if description:
        description_clean = description.strip().lower()
        candidates = db.query(Material).filter(Material.project_id == project_id).all()

        # Operational updates should link to the vendor's ACTIVE project line, not
        # an older historical material with a near-identical description.
        if vendor_id or active_only:
            from models_layer1 import PurchaseOrder, ShippingLogistics
            filtered = []
            for candidate in candidates:
                if vendor_id:
                    po_match = (
                        db.query(PurchaseOrder)
                        .filter(PurchaseOrder.material_id == candidate.material_id, PurchaseOrder.vendor_id == vendor_id)
                        .first()
                    )
                    if not po_match:
                        continue
                if active_only:
                    latest_ship = (
                        db.query(ShippingLogistics)
                        .filter(ShippingLogistics.material_id == candidate.material_id)
                        .order_by(ShippingLogistics.created_at.desc())
                        .first()
                    )
                    if latest_ship and (latest_ship.status or "").lower() == "delivered":
                        continue
                filtered.append(candidate)
            if filtered:
                candidates = filtered

        best_match = None
        best_score = 0
        for m in candidates:
            if not m.description:
                continue
            score = fuzz.token_sort_ratio(description_clean, m.description.strip().lower())
            if score > best_score:
                best_score = score
                best_match = m

        if best_match and best_score >= FUZZY_MATCH_THRESHOLD:
            return best_match.material_id

    # 3. No match found -- create new material
    new_material = Material(
        project_id=project_id,
        sku=sku,
        description=description,
    )
    db.add(new_material)
    db.flush()
    return new_material.material_id


def resolve_entities_from_fields(db: Session, project_id: str, doc_type: str, fields: dict) -> dict:
    """
    Convenience wrapper: given Gemini's (or manual) extracted_fields dict,
    resolves vendor_name / material_description+sku into IDs and returns
    an updated fields dict with vendor_id / material_id injected.
    """
    resolved = dict(fields)

    vendor_name = fields.get("vendor_name")
    if vendor_name:
        resolved["vendor_id"] = resolve_vendor(db, vendor_name)

    material_desc = fields.get("material_description")
    sku = fields.get("sku")
    if material_desc or sku:
        operational_update = doc_type in ("VENDOR_EMAIL", "SHIPPING")
        resolved["material_id"] = resolve_material(
            db, project_id, material_desc, sku,
            vendor_id=resolved.get("vendor_id") if operational_update else None,
            active_only=operational_update,
        )

    return resolved
