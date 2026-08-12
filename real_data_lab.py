"""Real Data Lab for ROJ Guard.

This module provides an isolated BYOD (bring-your-own-data) pipeline for CSV/XLSX
procurement exports. It deliberately does not mutate the main ROJ Guard demo
SQLite database or overwrite the baseline model artifacts.

Flow:
    upload -> auto-map -> validate -> optional real-history retraining -> score active rows

The feature engineering and live evidence overlays mirror the main ROJ Guard
Layer-3 contract closely enough to make real-data evaluation meaningful while
preserving the deterministic hackathon demo as a separate workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

from feature_engineering_layer3 import CATEGORICAL_COLS, NUMERIC_COLS
from train_models_layer3 import LEAD_TIME_MODEL_PATH, RISK_CLASSIFIER_MODEL_PATH, ENCODERS_PATH


# ---------------------------------------------------------------------------
# Canonical input contract and automatic column mapping
# ---------------------------------------------------------------------------

CANONICAL_FIELDS = {
    "project_name": {"label": "Project name", "required": False, "group": "Identity"},
    "material_description": {"label": "Material description", "required": True, "group": "Identity"},
    "material_class": {"label": "Material class", "required": False, "group": "Identity"},
    "sku": {"label": "SKU / material code", "required": False, "group": "Identity"},
    "vendor_name": {"label": "Vendor / supplier", "required": True, "group": "Identity"},
    "po_number": {"label": "PO number", "required": False, "group": "PO & Commercial"},
    "quantity": {"label": "Quantity", "required": False, "group": "PO & Commercial"},
    "unit_price": {"label": "Unit price", "required": False, "group": "PO & Commercial"},
    "order_date": {"label": "Order / PO date", "required": False, "group": "PO & Commercial"},
    "promised_ship_date": {"label": "Promised ship date", "required": False, "group": "PO & Commercial"},
    "carrier": {"label": "Carrier / logistics provider", "required": False, "group": "Shipment"},
    "tracking_number": {"label": "Tracking number", "required": False, "group": "Shipment"},
    "shipped_date": {"label": "Shipped / dispatch date", "required": False, "group": "Shipment"},
    "estimated_arrival": {"label": "Estimated arrival / ETA", "required": False, "group": "Shipment"},
    "actual_delivered_date": {"label": "Actual delivered date", "required": False, "group": "Shipment"},
    "shipment_status": {"label": "Shipment status", "required": False, "group": "Shipment"},
    "roj_date": {"label": "Required-On-Job (ROJ) date", "required": True, "group": "Schedule"},
    "float_days": {"label": "Schedule float (days)", "required": False, "group": "Schedule"},
    "is_critical_path": {"label": "Critical-path flag", "required": False, "group": "Schedule"},
    "snapshot_date": {"label": "Historical prediction snapshot date", "required": False, "group": "Operational Signals"},
    "latest_vendor_delay_days": {"label": "Vendor-reported delay (days)", "required": False, "group": "Operational Signals"},
    "latest_comm_type": {"label": "Latest communication type", "required": False, "group": "Operational Signals"},
    "latest_vendor_comm_date": {"label": "Latest vendor communication date", "required": False, "group": "Operational Signals"},
    "submittal_status": {"label": "Submittal status", "required": False, "group": "Operational Signals"},
    "submitted_date": {"label": "Submittal submitted date", "required": False, "group": "Operational Signals"},
    "approved_date": {"label": "Submittal approved date", "required": False, "group": "Operational Signals"},
}

SYNONYMS: Dict[str, List[str]] = {
    "project_name": ["project", "project name", "job", "job name", "project title"],
    "material_description": ["material", "material description", "item", "item description", "description", "material name", "product"],
    "material_class": ["material class", "category", "material category", "item class", "commodity", "discipline"],
    "sku": ["sku", "material code", "item code", "product code", "material id", "item id"],
    "vendor_name": ["vendor", "vendor name", "supplier", "supplier name", "manufacturer", "seller"],
    "po_number": ["po", "po number", "purchase order", "purchase order number", "po no", "order number"],
    "quantity": ["quantity", "qty", "ordered quantity", "order qty"],
    "unit_price": ["unit price", "price", "rate", "unit cost", "cost per unit"],
    "order_date": ["order date", "po date", "purchase date", "ordered date", "date ordered"],
    "promised_ship_date": ["promised ship date", "promised date", "committed ship date", "planned ship date", "promised dispatch", "commit date"],
    "carrier": ["carrier", "logistics provider", "freight provider", "courier", "transporter"],
    "tracking_number": ["tracking number", "tracking no", "awb", "lr number", "consignment number", "tracking id"],
    "shipped_date": ["shipped date", "ship date", "dispatch date", "despatch date", "actual ship date", "date shipped"],
    "estimated_arrival": ["estimated arrival", "eta", "expected delivery", "expected arrival", "forecast arrival", "estimated delivery date"],
    "actual_delivered_date": ["actual delivered date", "actual delivery", "delivery date", "actual arrival", "received date", "date delivered"],
    "shipment_status": ["shipment status", "delivery status", "logistics status", "status", "shipping status"],
    "roj_date": ["roj", "roj date", "required on job", "required on job date", "required date", "need by date", "required at site", "site required date"],
    "float_days": ["float", "float days", "schedule float", "total float", "slack", "slack days"],
    "is_critical_path": ["critical path", "is critical", "critical", "critical path flag", "on critical path"],
    "snapshot_date": ["snapshot date", "as of date", "prediction date", "status date", "data date", "cutoff date"],
    "latest_vendor_delay_days": ["vendor delay", "delay days", "vendor delay days", "reported delay", "reported delay days", "delay mentioned"],
    "latest_comm_type": ["communication type", "comm type", "vendor message type", "update type", "signal type"],
    "latest_vendor_comm_date": ["vendor communication date", "comm date", "email date", "latest update date", "vendor update date"],
    "submittal_status": ["submittal status", "approval status", "drawing status", "technical submittal status"],
    "submitted_date": ["submitted date", "submittal date", "submission date"],
    "approved_date": ["approved date", "approval date", "submittal approved date"],
}

DATE_FIELDS = {
    "order_date", "promised_ship_date", "shipped_date", "estimated_arrival",
    "actual_delivered_date", "roj_date", "snapshot_date", "latest_vendor_comm_date",
    "submitted_date", "approved_date",
}
NUMERIC_FIELDS = {"quantity", "unit_price", "float_days", "latest_vendor_delay_days"}


def _norm_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def suggest_column_mapping(columns: Iterable[str]) -> Tuple[Dict[str, Optional[str]], Dict[str, float]]:
    """Suggest a one-to-one schema mapping.

    Exact synonym matches win first. Fuzzy matches are deliberately conservative
    and a source column is never auto-assigned to two canonical fields.
    """
    columns = [str(c) for c in columns]
    norm_cols = {_norm_name(c): c for c in columns}
    mapping: Dict[str, Optional[str]] = {field: None for field in CANONICAL_FIELDS}
    confidence: Dict[str, float] = {field: 0.0 for field in CANONICAL_FIELDS}
    used_sources = set()

    # Pass 1: exact canonical/synonym matches.
    for canonical, aliases in SYNONYMS.items():
        targets = [_norm_name(canonical.replace("_", " "))] + [_norm_name(a) for a in aliases]
        for target in targets:
            src = norm_cols.get(target)
            if src is not None and src not in used_sources:
                mapping[canonical] = src
                confidence[canonical] = 1.0
                used_sources.add(src)
                break

    # Pass 2: globally rank conservative fuzzy candidates so each raw column is
    # assigned at most once. This avoids mappings such as Required Date -> both
    # ROJ and Actual Delivery.
    candidates = []
    for canonical, aliases in SYNONYMS.items():
        if mapping[canonical] is not None:
            continue
        targets = [_norm_name(canonical.replace("_", " "))] + [_norm_name(a) for a in aliases]
        for col in columns:
            if col in used_sources:
                continue
            ncol = _norm_name(col)
            score = max(fuzz.token_set_ratio(ncol, target) for target in targets) / 100.0
            # Require meaningful lexical overlap as well as fuzzy similarity.
            col_tokens = set(ncol.split())
            target_overlap = max((len(col_tokens & set(t.split())) for t in targets), default=0)
            if score >= 0.84 and target_overlap >= 1:
                candidates.append((score, canonical, col))

    for score, canonical, col in sorted(candidates, reverse=True):
        if mapping[canonical] is None and col not in used_sources:
            mapping[canonical] = col
            confidence[canonical] = score
            used_sources.add(col)

    return mapping, confidence


def excel_sheet_names(file_bytes: bytes) -> List[str]:
    xls = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    return list(xls.sheet_names)


def read_tabular_bytes(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    bio = BytesIO(file_bytes)
    if suffix == ".csv":
        # sep=None handles comma/semicolon/tab exports reasonably well.
        return pd.read_csv(bio, sep=None, engine="python")
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(bio, sheet_name=sheet_name or 0, engine="openpyxl")
    raise ValueError("Real Data Lab accepts CSV, XLSX, or XLSM files.")


def _coerce_bool(v: Any) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.number)):
        return float(v) != 0
    return _norm_name(v) in {"yes", "y", "true", "1", "critical", "on critical path", "cp"}


def _infer_material_class(description: Any) -> str:
    s = _norm_name(description)
    if any(k in s for k in ["steel", "rebar", "beam", "column"]):
        return "structural_steel"
    if "concrete" in s or "precast" in s:
        return "concrete"
    if any(k in s for k in ["hvac", "chiller", "mechanical", "pipe", "piping", "pump"]):
        return "mechanical"
    if any(k in s for k in ["transformer", "switchgear", "cable", "generator", "ups", "electrical"]):
        return "electrical"
    if any(k in s for k in ["glazing", "roof", "architectural", "curtain wall"]):
        return "architectural"
    tokens = s.split()
    return "_".join(tokens[:2]) if tokens else "unknown"


def normalize_dataset(raw_df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = pd.DataFrame(index=raw_df.index)
    for canonical in CANONICAL_FIELDS:
        src = mapping.get(canonical)
        out[canonical] = raw_df[src] if src and src in raw_df.columns else pd.NA

    for col in DATE_FIELDS:
        if out[col].notna().any():
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.date
        else:
            out[col] = pd.Series([pd.NaT] * len(out), index=out.index).dt.date
    for col in NUMERIC_FIELDS:
        cleaned = out[col].astype("string").str.replace(",", "", regex=False).str.replace(r"[^0-9.\-]", "", regex=True)
        out[col] = pd.to_numeric(cleaned, errors="coerce")

    for col in ["project_name", "material_description", "material_class", "sku", "vendor_name",
                "po_number", "carrier", "tracking_number", "shipment_status", "latest_comm_type",
                "submittal_status"]:
        out[col] = out[col].astype("string").str.strip()
        out.loc[out[col].isin(["", "nan", "None", "<NA>"]), col] = pd.NA

    out["is_critical_path"] = out["is_critical_path"].apply(_coerce_bool)
    out["material_class"] = out.apply(
        lambda r: r["material_class"] if pd.notna(r["material_class"]) else _infer_material_class(r["material_description"]), axis=1
    )
    out["carrier"] = out["carrier"].fillna("unknown")
    out["latest_comm_type"] = out["latest_comm_type"].fillna("none").astype(str).str.lower()
    out["submittal_status"] = out["submittal_status"].fillna("unknown").astype(str).str.lower()
    out["float_days"] = out["float_days"].fillna(0)
    out["latest_vendor_delay_days"] = out["latest_vendor_delay_days"].fillna(0).clip(lower=0)
    out["quantity"] = out["quantity"].fillna(0)
    out["unit_price"] = out["unit_price"].fillna(0)

    def status_for(r):
        if pd.notna(r["actual_delivered_date"]):
            return "delivered"
        value = _norm_name(r["shipment_status"])
        if value:
            aliases = {
                "in transit": "in_transit", "intransit": "in_transit", "transit": "in_transit",
                "late": "delayed", "delay": "delayed", "delayed": "delayed",
                "not shipped": "not_shipped", "pending": "not_shipped", "ordered": "not_shipped",
                "fabrication": "not_shipped", "manufacturing": "not_shipped",
                "delivered": "delivered", "received": "delivered", "complete": "delivered",
            }
            return aliases.get(value, value.replace(" ", "_"))
        return "in_transit" if pd.notna(r["shipped_date"]) else "not_shipped"

    out["shipment_status"] = out.apply(status_for, axis=1)
    out["is_historical_completed"] = (
        out["actual_delivered_date"].notna()
        & out["shipped_date"].notna()
        & out["roj_date"].notna()
    )
    out["is_active"] = ~out["is_historical_completed"] & out["roj_date"].notna()

    # Preserve source row for traceability/downloads.
    out["source_row"] = np.arange(2, len(out) + 2)
    return out


@dataclass
class ValidationReport:
    rows: int
    historical_rows: int
    active_rows: int
    vendors: int
    materials: int
    quality_score: int
    missing_required: Dict[str, int]
    warnings: List[str]
    errors: List[str]
    on_time_rows: int
    missed_roj_rows: int
    class_balance_ok: bool

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    missing_required: Dict[str, int] = {}

    for field, meta in CANONICAL_FIELDS.items():
        if meta["required"]:
            count = int(df[field].isna().sum())
            missing_required[field] = count
            if count == len(df):
                errors.append(f"Required field '{meta['label']}' is entirely missing.")
            elif count:
                warnings.append(f"{count} row(s) are missing {meta['label']}.")

    hist = df[df["is_historical_completed"]].copy()
    active = df[df["is_active"]].copy()
    if len(active) == 0:
        warnings.append("No active rows detected. Prediction output will be empty until active/non-delivered rows are supplied.")
    if len(hist) < 50:
        warnings.append(f"Only {len(hist)} completed historical rows detected. Real-data retraining requires at least 50 usable completed rows; use Prototype Model mode otherwise.")

    missed = 0
    on_time = 0
    if len(hist):
        valid = hist.dropna(subset=["actual_delivered_date", "roj_date"])
        missed = int(sum(a > r for a, r in zip(valid["actual_delivered_date"], valid["roj_date"])))
        on_time = len(valid) - missed
        if missed == 0 or on_time == 0:
            warnings.append("Historical ROJ outcomes contain only one class; classifier retraining needs both on-time and missed-ROJ examples.")

    # Data plausibility checks.
    if ((df["promised_ship_date"].notna()) & (df["order_date"].notna())).any():
        bad = df[df["promised_ship_date"].notna() & df["order_date"].notna()].apply(
            lambda r: r["promised_ship_date"] < r["order_date"], axis=1
        ).sum()
        if bad:
            warnings.append(f"{int(bad)} row(s) have promised ship date before order date.")
    if len(hist):
        bad_lead = hist.apply(lambda r: (r["actual_delivered_date"] - r["shipped_date"]).days < 0, axis=1).sum()
        if bad_lead:
            warnings.append(f"{int(bad_lead)} completed row(s) have delivery before ship date and will be excluded from retraining.")

    total_cells = max(len(df) * 6, 1)
    key_missing = sum(int(df[f].isna().sum()) for f in ["material_description", "vendor_name", "roj_date", "order_date", "promised_ship_date", "po_number"])
    quality = int(round(max(0, min(100, 100 * (1 - key_missing / total_cells)))))
    if "snapshot_date" in df.columns and df["snapshot_date"].notna().sum() == 0 and len(hist):
        warnings.append("No historical snapshot/status date was mapped. Retraining will derive a conservative pre-delivery snapshot from ship/promised dates; supplying true snapshot dates is better for leakage control.")

    return ValidationReport(
        rows=len(df), historical_rows=len(hist), active_rows=len(active),
        vendors=int(df["vendor_name"].nunique(dropna=True)),
        materials=int(df["material_description"].nunique(dropna=True)),
        quality_score=quality, missing_required=missing_required, warnings=warnings, errors=errors,
        on_time_rows=on_time, missed_roj_rows=missed, class_balance_ok=(missed > 0 and on_time > 0),
    )


# ---------------------------------------------------------------------------
# Real-data feature engineering
# ---------------------------------------------------------------------------


def _safe_days(a, b, default=0):
    if pd.isna(a) or pd.isna(b):
        return default
    try:
        return (a - b).days
    except Exception:
        return default


def _historical_snapshot(row: pd.Series) -> date:
    if pd.notna(row.get("snapshot_date")):
        return row["snapshot_date"]
    # A conservative pre-outcome proxy. True project status dates are preferable.
    candidates = [row.get("shipped_date"), row.get("promised_ship_date"), row.get("order_date")]
    candidates = [d for d in candidates if pd.notna(d)]
    if candidates:
        return max(candidates)
    delivered = row.get("actual_delivered_date")
    return delivered - timedelta(days=1) if pd.notna(delivered) else date.today()


def _base_feature_row(row: pd.Series, as_of: date, vendor_avg: float, reliability: float, historical: bool, has_true_snapshot: bool) -> Dict[str, Any]:
    order_date = row.get("order_date")
    promised = row.get("promised_ship_date")
    roj = row.get("roj_date")
    shipped = row.get("shipped_date")
    eta = row.get("estimated_arrival")
    comm_date = row.get("latest_vendor_comm_date")
    submitted = row.get("submitted_date")
    approved = row.get("approved_date")

    order_to_promised = max(0, _safe_days(promised, order_date, 0))
    days_until_roj = _safe_days(roj, as_of, 60)
    current_transit = max(0, _safe_days(as_of, shipped, 0)) if pd.notna(shipped) and shipped <= as_of else 0

    # Historical ETA/delay signals are only trusted when a true snapshot date is supplied;
    # otherwise the final ETA in an export may be post-outcome leakage.
    historical_live_ok = (not historical) or has_true_snapshot
    estimated_delay = _safe_days(eta, roj, 0) if historical_live_ok else 0
    vendor_delay = int(row.get("latest_vendor_delay_days") or 0) if historical_live_ok else 0
    latest_comm_type = str(row.get("latest_comm_type") or "none").lower() if historical_live_ok else "none"
    days_since_comm = max(0, _safe_days(as_of, comm_date, 999)) if historical_live_ok and pd.notna(comm_date) else 999

    if pd.notna(submitted):
        end = approved if pd.notna(approved) and approved <= as_of else as_of
        submittal_days = max(0, _safe_days(end, submitted, 0))
    else:
        submittal_days = 0

    days_past_promised = max(0, _safe_days(as_of, promised, 0)) if pd.notna(promised) else 0
    shipment_status = str(row.get("shipment_status") or "unknown").lower()
    if historical:
        shipment_status = "delayed" if latest_comm_type == "delay_notice" else ("in_transit" if pd.notna(shipped) else "not_shipped")

    prediction_anchor = shipped if pd.notna(shipped) else (promised if pd.notna(promised) else as_of)
    if pd.notna(prediction_anchor) and prediction_anchor < as_of and not pd.notna(shipped):
        prediction_anchor = as_of

    return {
        "source_row": int(row.get("source_row") or 0),
        "project_name": row.get("project_name"),
        "material_description": row.get("material_description"),
        "sku": row.get("sku"),
        "vendor_name": row.get("vendor_name"),
        "po_number": row.get("po_number"),
        "material_class": str(row.get("material_class") or "unknown"),
        "carrier": str(row.get("carrier") or "unknown"),
        "shipment_status": shipment_status,
        "submittal_status": str(row.get("submittal_status") or "unknown").lower(),
        "latest_comm_type": latest_comm_type,
        "vendor_reliability_score": float(reliability),
        "quantity": float(row.get("quantity") or 0),
        "unit_price": float(row.get("unit_price") or 0),
        "order_to_promised_days": float(order_to_promised),
        "vendor_avg_lead_time_days": float(vendor_avg),
        "float_days": float(row.get("float_days") or 0),
        "days_until_roj": float(days_until_roj),
        "current_transit_days": float(current_transit),
        "estimated_delay_vs_roj_days": float(estimated_delay),
        "latest_vendor_delay_days": float(vendor_delay),
        "days_since_latest_vendor_comm": float(days_since_comm),
        "submittal_approval_days": float(submittal_days),
        "days_past_promised_ship_date": float(days_past_promised),
        "is_critical_path_int": 1 if bool(row.get("is_critical_path")) else 0,
        "is_critical_path": bool(row.get("is_critical_path")),
        "roj_date": roj,
        "shipped_date": shipped,
        "estimated_arrival": eta,
        "prediction_anchor_date": prediction_anchor,
        "snapshot_date": as_of,
    }


def build_real_feature_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hist = df[df["is_historical_completed"]].copy()
    active = df[df["is_active"]].copy()
    hist["_snapshot"] = hist.apply(_historical_snapshot, axis=1)
    hist = hist.sort_values(["_snapshot", "source_row"]).reset_index(drop=True)

    vendor_history: Dict[str, List[Tuple[float, int]]] = {}
    class_history: Dict[str, List[float]] = {}
    global_leads: List[float] = []
    training_rows: List[Dict[str, Any]] = []
    for _, row in hist.iterrows():
        lead = _safe_days(row["actual_delivered_date"], row["shipped_date"], -1)
        if lead < 0 or lead > 730:
            continue
        missed = 1 if row["actual_delivered_date"] > row["roj_date"] else 0
        vendor = str(row.get("vendor_name") or "unknown")
        mclass = str(row.get("material_class") or "unknown")
        prior_vendor = vendor_history.get(vendor, [])
        prior_class = class_history.get(mclass, [])
        vendor_avg = (
            float(np.mean([x[0] for x in prior_vendor])) if prior_vendor
            else float(np.mean(prior_class)) if prior_class
            else float(np.mean(global_leads)) if global_leads else 30.0
        )
        reliability = (
            float(np.mean([1 - x[1] for x in prior_vendor])) if prior_vendor else 0.65
        )
        fr = _base_feature_row(row, row["_snapshot"], vendor_avg, reliability, historical=True, has_true_snapshot=pd.notna(row.get("snapshot_date")))
        fr["lead_time_days"] = float(lead)
        fr["missed_roj"] = int(missed)
        training_rows.append(fr)

        vendor_history.setdefault(vendor, []).append((lead, missed))
        class_history.setdefault(mclass, []).append(lead)
        global_leads.append(lead)

    # Active rows use all completed history available in the uploaded dataset.
    global_avg = float(np.mean(global_leads)) if global_leads else 30.0
    active_rows: List[Dict[str, Any]] = []
    today = date.today()
    for _, row in active.iterrows():
        vendor = str(row.get("vendor_name") or "unknown")
        mclass = str(row.get("material_class") or "unknown")
        vhist = vendor_history.get(vendor, [])
        chist = class_history.get(mclass, [])
        vendor_avg = float(np.mean([x[0] for x in vhist])) if vhist else (float(np.mean(chist)) if chist else global_avg)
        reliability = float(np.mean([1 - x[1] for x in vhist])) if vhist else 0.65
        active_rows.append(_base_feature_row(row, today, vendor_avg, reliability, historical=False, has_true_snapshot=True))

    return pd.DataFrame(training_rows), pd.DataFrame(active_rows)


# ---------------------------------------------------------------------------
# Isolated real-history model training
# ---------------------------------------------------------------------------


def train_real_data_models(training_df: pd.DataFrame, min_rows: int = 50) -> Dict[str, Any]:
    if len(training_df) < min_rows:
        return {
            "status": "insufficient_data", "rows_available": len(training_df), "rows_required": min_rows,
            "note": "Not enough completed history for isolated real-data retraining. Use Prototype Model mode or provide more completed deliveries.",
        }
    if training_df["missed_roj"].nunique() < 2:
        return {
            "status": "insufficient_class_variation", "rows_available": len(training_df),
            "note": "Historical rows contain only one ROJ outcome class. Both on-time and missed-ROJ examples are required.",
        }

    df = training_df.sort_values(["snapshot_date", "source_row"]).reset_index(drop=True).copy()
    split = min(max(int(len(df) * 0.8), 1), len(df) - 1)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    if train["missed_roj"].nunique() < 2:
        return {
            "status": "insufficient_class_variation", "rows_available": len(df),
            "note": "The temporal training window contains only one ROJ class. Supply broader history or use Prototype Model mode.",
        }

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train[CATEGORICAL_COLS].fillna("unknown").astype(str))

    def matrix(frame: pd.DataFrame) -> pd.DataFrame:
        enc = encoder.transform(frame[CATEGORICAL_COLS].fillna("unknown").astype(str))
        enc_df = pd.DataFrame(enc, columns=[c + "_enc" for c in CATEGORICAL_COLS], index=frame.index)
        num = frame[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
        return pd.concat([enc_df, num], axis=1)

    X_train = matrix(train)
    X_test = matrix(test)
    y_reg_train, y_reg_test = train["lead_time_days"], test["lead_time_days"]

    lead_model = XGBRegressor(
        n_estimators=260, max_depth=4, learning_rate=0.04, subsample=0.85,
        colsample_bytree=0.85, random_state=42, objective="reg:squarederror",
    )
    lead_model.fit(X_train, y_reg_train)
    test_lead_pred = lead_model.predict(X_test)
    mae = mean_absolute_error(y_reg_test, test_lead_pred)

    X_train_c = X_train.copy()
    X_test_c = X_test.copy()
    X_train_c["predicted_lead_time"] = lead_model.predict(X_train)
    X_test_c["predicted_lead_time"] = test_lead_pred
    y_train_c, y_test_c = train["missed_roj"], test["missed_roj"]
    neg = max(int((y_train_c == 0).sum()), 1)
    pos = max(int((y_train_c == 1).sum()), 1)
    clf = XGBClassifier(
        n_estimators=260, max_depth=4, learning_rate=0.04, subsample=0.85,
        colsample_bytree=0.85, random_state=42, eval_metric="logloss",
        scale_pos_weight=neg / pos,
    )
    clf.fit(X_train_c, y_train_c)
    probs = clf.predict_proba(X_test_c)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test_c, probs) if y_test_c.nunique() > 1 else None

    metrics = {
        "status": "trained",
        "rows_used": len(df),
        "train_rows": len(train),
        "holdout_rows": len(test),
        "lead_time_mae_days": round(float(mae), 2),
        "risk_auc": round(float(auc), 3) if auc is not None else None,
        "precision": round(float(precision_score(y_test_c, preds, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test_c, preds, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test_c, preds, zero_division=0)), 3),
        "historical_miss_rate": round(float(df["missed_roj"].mean()), 3),
        "model_grade": "strong demo history" if len(df) >= 500 else ("usable demo history" if len(df) >= 200 else "experimental history"),
    }
    return {
        **metrics,
        "bundle": {
            "lead_model": lead_model,
            "risk_model": clf,
            "encoder": encoder,
            "feature_cols": [c + "_enc" for c in CATEGORICAL_COLS] + NUMERIC_COLS,
            "feature_cols_clf": [c + "_enc" for c in CATEGORICAL_COLS] + NUMERIC_COLS + ["predicted_lead_time"],
            "metrics": metrics,
        },
    }


def _operational_score(row: Dict[str, Any], delay: float) -> float:
    if delay >= 10:
        score = 0.94
    elif delay >= 4:
        score = 0.78
    elif delay > 0:
        score = 0.58
    elif delay >= -7:
        score = 0.34
    else:
        score = 0.16
    reliability = float(row.get("vendor_reliability_score") or 0.65)
    if reliability < 0.45:
        score += 0.12
    elif reliability > 0.80:
        score -= 0.08
    if row.get("is_critical_path") and (row.get("float_days") or 0) <= 2:
        score += 0.08
    if row.get("submittal_status") not in ("approved", "approved_with_comments") and row.get("days_until_roj", 99) < 21:
        score += 0.08
    return max(0.02, min(0.98, score))


def _evidence_floor(row: Dict[str, Any], delay: float) -> float:
    floor = 0.0
    vendor_delay = float(row.get("latest_vendor_delay_days") or 0)
    eta_delay = float(row.get("estimated_delay_vs_roj_days") or 0)
    status = str(row.get("shipment_status") or "").lower()
    if vendor_delay >= 8: floor = max(floor, 0.90)
    elif vendor_delay >= 3: floor = max(floor, 0.68)
    elif vendor_delay > 0: floor = max(floor, 0.58)
    if status == "delayed": floor = max(floor, 0.80)
    if eta_delay >= 8: floor = max(floor, 0.88)
    elif eta_delay > 0: floor = max(floor, 0.62)
    if delay >= 8: floor = max(floor, 0.86)
    if row.get("days_until_roj", 999) < 0: floor = max(floor, 0.92)
    return floor


def _arrival(row: Dict[str, Any], predicted_lead: float) -> date:
    anchor = row.get("prediction_anchor_date") or date.today()
    if isinstance(anchor, pd.Timestamp): anchor = anchor.date()
    baseline = anchor + timedelta(days=max(0, int(round(predicted_lead))))
    eta = row.get("estimated_arrival")
    if pd.notna(eta): baseline = max(baseline, eta)
    roj = row.get("roj_date")
    vendor_delay = int(row.get("latest_vendor_delay_days") or 0)
    if pd.notna(roj) and vendor_delay > 0:
        baseline = max(baseline, roj + timedelta(days=vendor_delay))
    return baseline


def _risk_level(prob: float) -> str:
    return "High" if prob >= 0.70 else ("Medium" if prob >= 0.38 else "Low")


def _final_prediction(row: Dict[str, Any], predicted_lead: float, model_prob: Optional[float], model_source: str) -> Dict[str, Any]:
    arrival = _arrival(row, predicted_lead)
    roj = row.get("roj_date")
    delay = _safe_days(arrival, roj, 0) if pd.notna(roj) else 0
    ops = _operational_score(row, delay)
    if model_prob is None:
        prob = max(ops, _evidence_floor(row, delay))
    else:
        prob = max(0.55 * float(model_prob) + 0.45 * ops, _evidence_floor(row, delay))
    prob = float(max(0.02, min(0.98, prob)))
    signals = []
    if row.get("latest_vendor_delay_days", 0) > 0: signals.append(f"vendor reported +{int(row['latest_vendor_delay_days'])}d")
    if row.get("estimated_delay_vs_roj_days", 0) > 0: signals.append(f"ETA +{int(row['estimated_delay_vs_roj_days'])}d vs ROJ")
    if row.get("shipment_status") == "delayed": signals.append("shipment marked delayed")
    if row.get("is_critical_path"): signals.append(f"critical path / {int(row.get('float_days') or 0)}d float")
    if not signals: signals.append("statistical lead-time + schedule context")
    return {
        "Source row": row.get("source_row"),
        "Project": row.get("project_name") if pd.notna(row.get("project_name")) else "—",
        "Material": row.get("material_description"),
        "SKU": row.get("sku") if pd.notna(row.get("sku")) else "—",
        "Vendor": row.get("vendor_name"),
        "PO": row.get("po_number") if pd.notna(row.get("po_number")) else "—",
        "Risk": _risk_level(prob),
        "Miss ROJ probability": round(prob, 3),
        "Predicted lead time days": round(float(predicted_lead), 1),
        "Forecast arrival": str(arrival),
        "ROJ date": str(roj) if pd.notna(roj) else None,
        "Forecast delay days": int(delay),
        "Critical path": bool(row.get("is_critical_path")),
        "Vendor reliability": round(float(row.get("vendor_reliability_score") or 0.65), 3),
        "Model source": model_source,
        "Why": "; ".join(signals),
    }


def _bundle_matrix(active_df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    enc = bundle["encoder"].transform(active_df[CATEGORICAL_COLS].fillna("unknown").astype(str))
    enc_df = pd.DataFrame(enc, columns=[c + "_enc" for c in CATEGORICAL_COLS], index=active_df.index)
    num = active_df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    return pd.concat([enc_df, num], axis=1)


def score_with_real_model(active_df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    if active_df.empty:
        return pd.DataFrame()
    X = _bundle_matrix(active_df, bundle)
    lead = bundle["lead_model"].predict(X[bundle["feature_cols"]])
    Xc = X.copy()
    Xc["predicted_lead_time"] = lead
    probs = bundle["risk_model"].predict_proba(Xc[bundle["feature_cols_clf"]])[:, 1]
    rows = []
    for (_, row), lt, p in zip(active_df.iterrows(), lead, probs):
        rows.append(_final_prediction(row.to_dict(), float(lt), float(p), "Retrained on supplied historical data"))
    return pd.DataFrame(rows).sort_values(["Miss ROJ probability"], ascending=False).reset_index(drop=True)


def score_with_prototype_model(active_df: pd.DataFrame) -> pd.DataFrame:
    if active_df.empty:
        return pd.DataFrame()
    if not all(Path(p).exists() for p in [LEAD_TIME_MODEL_PATH, RISK_CLASSIFIER_MODEL_PATH, ENCODERS_PATH]):
        rows = []
        for _, row in active_df.iterrows():
            lt = float(row.get("vendor_avg_lead_time_days") or 30)
            rows.append(_final_prediction(row.to_dict(), lt, None, "Operational rules (prototype model artifacts unavailable)"))
        return pd.DataFrame(rows).sort_values("Miss ROJ probability", ascending=False).reset_index(drop=True)

    lead_model = joblib.load(LEAD_TIME_MODEL_PATH)
    risk_model = joblib.load(RISK_CLASSIFIER_MODEL_PATH)
    meta = joblib.load(ENCODERS_PATH)
    frame = active_df.copy()
    for col in CATEGORICAL_COLS:
        le = meta["encoders"][col]
        frame[col] = frame[col].fillna("unknown").astype(str)
        frame[col + "_enc"] = frame[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    frame[NUMERIC_COLS] = frame[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    lead = lead_model.predict(frame[meta["feature_cols"]])
    frame["predicted_lead_time"] = lead
    probs = risk_model.predict_proba(frame[meta["feature_cols_clf"]])[:, 1]
    rows = []
    for (_, row), lt, p in zip(active_df.iterrows(), lead, probs):
        rows.append(_final_prediction(row.to_dict(), float(lt), float(p), "Prototype model + supplied live project data"))
    return pd.DataFrame(rows).sort_values("Miss ROJ probability", ascending=False).reset_index(drop=True)


def template_csv_bytes() -> bytes:
    template = pd.DataFrame([
        {
            "Project Name": "Example Data Center", "Material Description": "2000 kVA Transformer",
            "Material Class": "electrical", "SKU": "TX-2000-01", "Vendor Name": "Example Electricals",
            "PO Number": "PO-1001", "Quantity": 1, "Unit Price": 2500000, "Order Date": "2026-04-01",
            "Promised Ship Date": "2026-07-15", "Carrier": "Example Freight", "Shipped Date": "2026-07-18",
            "Estimated Arrival": "2026-08-29", "Actual Delivered Date": "", "Shipment Status": "in_transit",
            "ROJ Date": "2026-08-25", "Float Days": 1, "Critical Path": "Yes", "Vendor Delay Days": 4,
            "Latest Comm Type": "delay_notice", "Latest Vendor Comm Date": "2026-08-11",
            "Submittal Status": "approved", "Submitted Date": "2026-03-01", "Approved Date": "2026-03-10",
        },
        {
            "Project Name": "Example Data Center", "Material Description": "Cable tray assembly",
            "Material Class": "electrical", "SKU": "CT-101", "Vendor Name": "Example Electricals",
            "PO Number": "PO-HIST-01", "Quantity": 120, "Unit Price": 800, "Order Date": "2025-06-01",
            "Promised Ship Date": "2025-06-28", "Carrier": "Example Freight", "Shipped Date": "2025-06-30",
            "Estimated Arrival": "2025-07-12", "Actual Delivered Date": "2025-07-12", "Shipment Status": "delivered",
            "ROJ Date": "2025-07-15", "Float Days": 5, "Critical Path": "No", "Vendor Delay Days": 0,
            "Latest Comm Type": "status_update", "Latest Vendor Comm Date": "2025-07-02",
            "Submittal Status": "approved", "Submitted Date": "2025-05-10", "Approved Date": "2025-05-16",
        },
    ])
    return template.to_csv(index=False).encode("utf-8")
