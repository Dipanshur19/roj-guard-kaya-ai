"""Layer 3 live inference with ML + explicit construction-domain overlays."""

import os
from datetime import date, timedelta
from time_utils import current_date
import joblib
import pandas as pd
from sqlalchemy.orm import Session

from feature_engineering_layer3 import build_inference_row, CATEGORICAL_COLS, NUMERIC_COLS
from train_models_layer3 import LEAD_TIME_MODEL_PATH, RISK_CLASSIFIER_MODEL_PATH, ENCODERS_PATH

HIGH_RISK_PROB_THRESHOLD = 0.70
MEDIUM_RISK_PROB_THRESHOLD = 0.38


def _models_available() -> bool:
    return all(os.path.exists(p) for p in (
        LEAD_TIME_MODEL_PATH, RISK_CLASSIFIER_MODEL_PATH, ENCODERS_PATH
    ))


def _operational_score(row: dict, predicted_delay_days: float) -> float:
    """Transparent score from current schedule/vendor/shipping evidence."""
    d = predicted_delay_days or 0
    if d >= 10:
        score = 0.94
    elif d >= 4:
        score = 0.78
    elif d > 0:
        score = 0.58
    elif d >= -7:
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


def _evidence_floor(row: dict, predicted_delay_days: float) -> float:
    """Strong explicit evidence should never be washed out by the statistical model."""
    floor = 0.0
    vendor_delay = row.get("latest_vendor_delay_days") or 0
    eta_delay = row.get("estimated_delay_vs_roj_days") or 0
    status = (row.get("shipment_status") or "").lower()

    if vendor_delay >= 8:
        floor = max(floor, 0.90)
    elif vendor_delay >= 3:
        floor = max(floor, 0.68)
    elif vendor_delay > 0:
        floor = max(floor, 0.58)

    if status == "delayed":
        floor = max(floor, 0.80)
    if eta_delay >= 8:
        floor = max(floor, 0.88)
    elif eta_delay > 0:
        floor = max(floor, 0.62)
    if predicted_delay_days >= 8:
        floor = max(floor, 0.86)
    if row.get("days_until_roj", 999) < 0:
        floor = max(floor, 0.92)
    return floor


def _predicted_arrival(row: dict, predicted_lead_time: float):
    anchor = row.get("prediction_anchor_date") or current_date()
    baseline = anchor + timedelta(days=max(0, int(round(predicted_lead_time))))

    # Operational ETA and explicit vendor delay notices are first-class signals.
    eta = row.get("estimated_arrival")
    if eta:
        baseline = max(baseline, eta)

    roj = row.get("roj_date")
    vendor_delay = row.get("latest_vendor_delay_days") or 0
    if roj and vendor_delay > 0:
        baseline = max(baseline, roj + timedelta(days=int(vendor_delay)))
    return baseline


def _rule_based_fallback(row: dict) -> dict:
    predicted_lead_time = float(row.get("vendor_avg_lead_time_days") or 30.0)
    arrival = _predicted_arrival(row, predicted_lead_time)
    roj = row.get("roj_date")
    delay = (arrival - roj).days if roj else 0
    prob = max(_operational_score(row, delay), _evidence_floor(row, delay))
    return _finalize(row, predicted_lead_time, arrival, delay, prob, model_prob=None, fallback=True)


def _finalize(row, predicted_lead_time, arrival, delay, prob, model_prob=None, fallback=False):
    if prob >= HIGH_RISK_PROB_THRESHOLD:
        risk_level = "High"
    elif prob >= MEDIUM_RISK_PROB_THRESHOLD:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    signals = []
    if row.get("latest_vendor_delay_days", 0) > 0:
        signals.append(f"vendor reported {row['latest_vendor_delay_days']} day delay")
    if (row.get("shipment_status") or "") == "delayed":
        signals.append("shipment status is delayed")
    if row.get("estimated_delay_vs_roj_days", 0) > 0:
        signals.append(f"current ETA is {row['estimated_delay_vs_roj_days']} day(s) after ROJ")
    if row.get("days_past_promised_ship_date", 0) > 0 and not row.get("shipped_date"):
        signals.append(f"promised ship date passed {row['days_past_promised_ship_date']} day(s) ago")
    if row.get("submittal_status") not in ("approved", "approved_with_comments"):
        signals.append(f"submittal is {row.get('submittal_status', 'unknown')}")
    if row.get("is_critical_path"):
        signals.append(f"critical path with {row.get('float_days', 0)} day(s) float")

    source = "rule fallback" if fallback else f"ML {model_prob*100:.0f}% + live evidence"
    signal_text = "; ".join(signals) if signals else "no explicit delay notice"
    explanation = (
        f"{source}. Final miss-ROJ risk {prob*100:.0f}%. "
        f"Predicted lead time {predicted_lead_time:.1f} days; operational arrival {arrival}; "
        f"ROJ {row.get('roj_date')}; predicted delay {delay} day(s). "
        f"Vendor reliability {row.get('vendor_reliability_score', 0.65):.2f}. Signals: {signal_text}."
    )

    return {
        "predicted_lead_time_days": round(float(predicted_lead_time), 1),
        "predicted_arrival_date": str(arrival),
        "predicted_delay_days": float(delay),
        "miss_roj_probability": round(float(prob), 3),
        "risk_level": risk_level,
        "explanation": explanation,
        "model_probability": round(float(model_prob), 3) if model_prob is not None else None,
        "shipment_status": row.get("shipment_status"),
        "days_until_roj": row.get("days_until_roj"),
        "latest_vendor_delay_days": row.get("latest_vendor_delay_days"),
        "estimated_delay_vs_roj_days": row.get("estimated_delay_vs_roj_days"),
        "latest_vendor_comm_summary": row.get("latest_vendor_comm_summary"),
    }


def compute_risk_for_material(db: Session, material_id: str) -> dict:
    row = build_inference_row(db, material_id)
    if row is None:
        return {"error": "No purchase order found for this material_id"}
    if row.get("inactive"):
        return {"inactive": True, "material_id": material_id, "status": row.get("status", "delivered")}

    if not _models_available():
        result = _rule_based_fallback(row)
    else:
        lead_time_model = joblib.load(LEAD_TIME_MODEL_PATH)
        risk_classifier = joblib.load(RISK_CLASSIFIER_MODEL_PATH)
        meta = joblib.load(ENCODERS_PATH)
        encoders = meta["encoders"]

        df = pd.DataFrame([row])
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            df[col] = df[col].fillna("unknown").astype(str)
            df[col + "_enc"] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

        X = df[meta["feature_cols"]]
        predicted_lead_time = float(lead_time_model.predict(X)[0])
        df["predicted_lead_time"] = predicted_lead_time
        model_prob = float(risk_classifier.predict_proba(df[meta["feature_cols_clf"]])[:, 1][0])

        arrival = _predicted_arrival(row, predicted_lead_time)
        roj = row.get("roj_date")
        delay = (arrival - roj).days if roj else 0
        ops_prob = _operational_score(row, delay)
        blended = 0.55 * model_prob + 0.45 * ops_prob
        final_prob = max(blended, _evidence_floor(row, delay))
        result = _finalize(row, predicted_lead_time, arrival, delay, final_prob, model_prob=model_prob)

    result.update({
        "material_id": material_id,
        "vendor_id": row.get("vendor_id"),
        "project_id": row.get("project_id"),
        "roj_date": str(row.get("roj_date")) if row.get("roj_date") else None,
        "is_critical_path": row.get("is_critical_path", False),
    })
    return result
