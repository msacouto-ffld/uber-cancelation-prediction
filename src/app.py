"""
Uber Ride Cancellation Prediction API
Based on Week 9 Demo pattern.

Usage (local):
    python src/app.py

Usage (Docker):
    docker build -t uber-cancellation-api .
    docker run -p 5000:5000 uber-cancellation-api

Usage (test):
    python src/test_api.py
    python src/test_api.py https://your-app.onrender.com

Feature notes
-------------
Booking-time features (required):
    hour, month, is_weekend, vtat_missing, Avg VTAT,
    hour_bucket, day_of_week, Vehicle Type,
    pickup_grouped, drop_grouped

Derived internally (do NOT send — computed from required fields):
    is_peak       — 1 if hour in {8, 9, 17, 18, 19}
    vtat_bucket   — pd.cut of Avg VTAT into [low/medium/high/very_high]

Historical / behavioural features (optional):
    customer_cancel_rate  — caller's pre-computed per-customer cancel rate
    pickup_risk           — historical cancel rate for the pickup location
    drop_risk             — historical cancel rate for the drop location
    route_risk            — historical cancel rate for the pickup→drop route
    customer_prior_rides  — how many rides the customer had before this booking

    If omitted, all historical rates default to HISTORICAL_DEFAULT (≈ overall
    training cancel rate) and customer_prior_rides defaults to 0.
    New or anonymous customers should omit these fields entirely.
"""

import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

# ── Initialize app and load model ────────────────────────────────────────────
app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.environ.get(
    "MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "best_pipeline.pkl")
)

print(f"Loading model from {MODEL_PATH}...")
pipeline = joblib.load(MODEL_PATH)
print(f"Model loaded: {type(pipeline).__name__}")

# ── Feature definitions (must match notebook exactly) ────────────────────────
NUMERIC_FEATURES = [
    "hour",
    "month",
    "is_weekend",
    "is_peak",                # derived internally
    "vtat_missing",
    "Avg VTAT",
    "customer_cancel_rate",   # optional historical
    "pickup_risk",            # optional historical
    "drop_risk",              # optional historical
    "route_risk",             # optional historical
    "customer_prior_rides",   # optional historical
]

CATEGORICAL_FEATURES = [
    "hour_bucket",
    "day_of_week",
    "Vehicle Type",
    "pickup_grouped",
    "drop_grouped",
    "vtat_bucket",            # derived internally
]

# Fields the caller must provide
REQUIRED_FIELDS = [
    "hour", "month", "is_weekend", "vtat_missing", "Avg VTAT",
    "hour_bucket", "day_of_week", "Vehicle Type",
    "pickup_grouped", "drop_grouped",
]

# Historical features — optional, fallback to overall training cancel rate
HISTORICAL_FIELDS  = [
    "customer_cancel_rate", "pickup_risk", "drop_risk", "route_risk",
]
HISTORICAL_DEFAULT = float(os.environ.get("HISTORICAL_DEFAULT", 0.38))

# Numeric fields that can also be None/null (Avg VTAT is null when vtat_missing=1)
NULLABLE_NUMERIC   = {"Avg VTAT"}

VALID_HOUR_BUCKETS  = {"night", "morning", "afternoon", "evening"}
VALID_DAYS          = {"Monday", "Tuesday", "Wednesday", "Thursday",
                       "Friday", "Saturday", "Sunday"}
VALID_VEHICLE_TYPES = {"Auto", "Bike", "eBike", "Go Mini",
                       "Go Sedan", "Premier Sedan", "Uber XL"}

# ── Internal feature derivation ───────────────────────────────────────────────
def derive_features(data: dict) -> dict:
    """
    Compute is_peak and vtat_bucket from raw inputs.
    Fills historical features with their defaults when absent.
    Returns a new dict ready to be turned into a single-row DataFrame.
    """
    row = dict(data)

    # is_peak: morning and evening rush hours
    row["is_peak"] = 1 if row.get("hour") in {8, 9, 17, 18, 19} else 0

    # vtat_bucket: mirrors pd.cut(bins=[0,5,10,15,100])
    vtat = row.get("Avg VTAT")
    if vtat is None or (isinstance(vtat, float) and np.isnan(vtat)):
        row["vtat_bucket"] = np.nan   # imputed to most_frequent inside pipeline
    elif vtat <= 5:
        row["vtat_bucket"] = "low"
    elif vtat <= 10:
        row["vtat_bucket"] = "medium"
    elif vtat <= 15:
        row["vtat_bucket"] = "high"
    else:
        row["vtat_bucket"] = "very_high"

    # Historical features — use caller value or fall back to default
    for field in HISTORICAL_FIELDS:
        row.setdefault(field, HISTORICAL_DEFAULT)

    row.setdefault("customer_prior_rides", 0)

    return row


# ── Input validation ──────────────────────────────────────────────────────────
def validate_input(data: dict):
    errors = {}

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        errors["missing_fields"] = missing
        return False, errors

    # Numeric type checks (skip None only for nullable fields)
    for field in ["hour", "month", "is_weekend", "vtat_missing"]:
        val = data.get(field)
        if val is None:
            errors[field] = "must not be null"
        else:
            try:
                float(val)
            except (ValueError, TypeError):
                errors[field] = f"expected numeric, got {type(val).__name__}"

    vtat = data.get("Avg VTAT")
    if vtat is not None:
        try:
            float(vtat)
        except (ValueError, TypeError):
            errors["Avg VTAT"] = f"expected numeric or null, got {type(vtat).__name__}"

    # Categorical validation
    if data.get("hour_bucket") not in VALID_HOUR_BUCKETS:
        errors["hour_bucket"] = f"must be one of {sorted(VALID_HOUR_BUCKETS)}"

    if data.get("day_of_week") not in VALID_DAYS:
        errors["day_of_week"] = f"must be one of {sorted(VALID_DAYS)}"

    # Vehicle type: warn but don't reject — pipeline uses handle_unknown='ignore'
    if data.get("Vehicle Type") not in VALID_VEHICLE_TYPES:
        errors["Vehicle Type"] = (
            f"unknown value '{data.get('Vehicle Type')}'; "
            f"known types: {sorted(VALID_VEHICLE_TYPES)}"
        )

    # Optional historical fields — validate if provided
    for field in HISTORICAL_FIELDS:
        if field in data and data[field] is not None:
            try:
                v = float(data[field])
                if not (0.0 <= v <= 1.0):
                    errors[field] = "must be a rate between 0.0 and 1.0"
            except (ValueError, TypeError):
                errors[field] = f"expected float in [0, 1], got {type(data[field]).__name__}"

    if "customer_prior_rides" in data and data["customer_prior_rides"] is not None:
        try:
            v = int(data["customer_prior_rides"])
            if v < 0:
                errors["customer_prior_rides"] = "must be >= 0"
        except (ValueError, TypeError):
            errors["customer_prior_rides"] = "expected non-negative integer"

    return len(errors) == 0, errors


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Derive features and assemble a DataFrame in the exact column order the pipeline expects."""
    rows = [derive_features(r) for r in records]
    df = pd.DataFrame(rows, columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    return df


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":             "healthy",
        "model":              "loaded",
        "model_type":         type(pipeline.named_steps["classifier"]).__name__,
        "required_features":  REQUIRED_FIELDS,
        "optional_features":  HISTORICAL_FIELDS + ["customer_prior_rides"],
        "derived_internally": ["is_peak", "vtat_bucket"],
    })


# ── Single prediction ─────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    is_valid, errors = validate_input(data)
    if not is_valid:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    try:
        df    = build_dataframe([data])
        label = int(pipeline.predict(df)[0])
        prob  = float(pipeline.predict_proba(df)[0][1])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    used_defaults = [f for f in HISTORICAL_FIELDS + ["customer_prior_rides"]
                     if f not in data]

    return jsonify({
        "cancelled":                label,
        "cancellation_probability": round(prob, 4),
        "label":                    "Cancelled" if label == 1 else "Completed",
        "historical_defaults_used": used_defaults or None,
    })


# ── Batch prediction ──────────────────────────────────────────────────────────
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array of bookings"}), 400
    if len(data) > 100:
        return jsonify({"error": "Batch size limited to 100"}), 400

    for i, record in enumerate(data):
        is_valid, errors = validate_input(record)
        if not is_valid:
            return jsonify({"error": f"Invalid input at index {i}", "details": errors}), 400

    try:
        df     = build_dataframe(data)
        labels = pipeline.predict(df)
        probas = pipeline.predict_proba(df)[:, 1]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    predictions = [
        {
            "cancelled":                int(label),
            "cancellation_probability": round(float(prob), 4),
            "label":                    "Cancelled" if label == 1 else "Completed",
        }
        for label, prob in zip(labels, probas)
    ]
    return jsonify({"predictions": predictions, "count": len(predictions)})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
