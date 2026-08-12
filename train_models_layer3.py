"""Layer 3 model training for lead-time forecasting and ROJ miss risk."""

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

from database_layer1 import SessionLocal
from feature_engineering_layer3 import build_training_dataframe, CATEGORICAL_COLS, NUMERIC_COLS

LEAD_TIME_MODEL_PATH = "lead_time_model_layer3.joblib"
RISK_CLASSIFIER_MODEL_PATH = "risk_classifier_layer3.joblib"
ENCODERS_PATH = "categorical_encoders_layer3.joblib"
MIN_TRAINING_ROWS = 80


def _encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    encoders = encoders or {}
    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = df[col].fillna("unknown").astype(str)
            df[col + "_enc"] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].fillna("unknown").astype(str)
            df[col + "_enc"] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    return df, encoders


def train_and_save_models():
    db = SessionLocal()
    try:
        df = build_training_dataframe(db)
    finally:
        db.close()

    if len(df) < MIN_TRAINING_ROWS:
        return {
            "status": "insufficient_data",
            "rows_available": len(df),
            "rows_required": MIN_TRAINING_ROWS,
            "note": "Models not trained. Inference will use transparent rules until enough delivery history exists.",
        }

    if df["missed_roj"].nunique() < 2:
        return {
            "status": "insufficient_class_variation",
            "rows_available": len(df),
            "note": "Historical data contains only one ROJ outcome class; regenerate/broaden training history.",
        }

    # Temporal holdout is closer to the real use case than a random split.
    df = df.sort_values("snapshot_date").reset_index(drop=True)
    df, encoders = _encode_categoricals(df, fit=True)
    feature_cols = [c + "_enc" for c in CATEGORICAL_COLS] + NUMERIC_COLS
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

    split_idx = max(int(len(df) * 0.8), 1)
    split_idx = min(split_idx, len(df) - 1)

    X = df[feature_cols]
    y_reg = df["lead_time_days"]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

    lead_time_model = XGBRegressor(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        objective="reg:squarederror",
    )
    lead_time_model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, lead_time_model.predict(X_test))

    # Stacked classifier: predicted lead time becomes an additional signal.
    df["predicted_lead_time"] = lead_time_model.predict(X)
    feature_cols_clf = feature_cols + ["predicted_lead_time"]
    X_clf = df[feature_cols_clf]
    y_clf = df["missed_roj"]

    X_train_c, X_test_c = X_clf.iloc[:split_idx], X_clf.iloc[split_idx:]
    y_train_c, y_test_c = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]

    neg = max(int((y_train_c == 0).sum()), 1)
    pos = max(int((y_train_c == 1).sum()), 1)
    scale_pos_weight = neg / pos

    risk_classifier = XGBClassifier(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    risk_classifier.fit(X_train_c, y_train_c)

    auc = None
    if y_test_c.nunique() > 1:
        auc = roc_auc_score(y_test_c, risk_classifier.predict_proba(X_test_c)[:, 1])

    joblib.dump(lead_time_model, LEAD_TIME_MODEL_PATH)
    joblib.dump(risk_classifier, RISK_CLASSIFIER_MODEL_PATH)
    joblib.dump(
        {
            "encoders": encoders,
            "feature_cols": feature_cols,
            "feature_cols_clf": feature_cols_clf,
            "trained_rows": len(df),
            "trained_at_snapshot_max": str(df["snapshot_date"].max()),
        },
        ENCODERS_PATH,
    )

    return {
        "status": "trained",
        "rows_used": len(df),
        "holdout_rows": len(y_test),
        "lead_time_model_mae_days": round(float(mae), 2),
        "risk_classifier_auc": round(float(auc), 3) if auc is not None else "N/A (one class in temporal holdout)",
        "historical_miss_rate": round(float(df["missed_roj"].mean()), 3),
    }


if __name__ == "__main__":
    print(train_and_save_models())
