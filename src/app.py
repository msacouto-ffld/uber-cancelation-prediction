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
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# ── Initialize app and load model ────────────────────────────────────────────
app = Flask(__name__)

# Model loads ONCE at startup, not per request
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.environ.get(
    "MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "best_pipeline.pkl")
)

print(f"Loading model from {MODEL_PATH}...")
pipeline = joblib.load(MODEL_PATH)
print(f"Model loaded: {type(pipeline).__name__}")

REQUIRED_FEATURES = [
    "hour", "month", "is_weekend", "vtat_missing", "Avg VTAT",
    "hour_bucket", "day_of_week", "Vehicle Type",
    "pickup_grouped", "drop_grouped"
]

NUMERIC_FEATURES    = ["hour", "month", "is_weekend", "vtat_missing", "Avg VTAT"]
VALID_HOUR_BUCKETS  = {"night", "morning", "afternoon", "evening"}
VALID_DAYS          = {"Monday", "Tuesday", "Wednesday", "Thursday",
                       "Friday", "Saturday", "Sunday"}
VALID_VEHICLE_TYPES = {"Auto", "Bike", "eBike", "Go Mini",
                       "Go Sedan", "Premier Sedan", "Uber XL"}


# ── Input validation ──────────────────────────────────────────────────────────
def validate_input(data):
    errors = {}
    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        errors["missing_fields"] = missing
        return False, errors

    for field in NUMERIC_FEATURES:
        if data[field] is not None:
            try:
                float(data[field])
            except (ValueError, TypeError):
                errors[field] = f"expected numeric, got {type(data[field]).__name__}"

    if data.get("hour_bucket") not in VALID_HOUR_BUCKETS:
        errors["hour_bucket"] = f"must be one of {sorted(VALID_HOUR_BUCKETS)}"

    if data.get("day_of_week") not in VALID_DAYS:
        errors["day_of_week"] = f"must be one of {sorted(VALID_DAYS)}"

    return len(errors) == 0, errors


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "healthy",
        "model":      "loaded",
        "model_type": type(pipeline.named_steps["classifier"]).__name__,
        "n_features": len(REQUIRED_FEATURES)
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
        df    = pd.DataFrame([data])
        label = int(pipeline.predict(df)[0])
        prob  = float(pipeline.predict_proba(df)[0][1])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "cancelled":                label,
        "cancellation_probability": round(prob, 4),
        "label":                    "Cancelled" if label == 1 else "Completed"
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
        df     = pd.DataFrame(data)
        labels = pipeline.predict(df)
        probas = pipeline.predict_proba(df)[:, 1]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    predictions = [
        {
            "cancelled":                int(label),
            "cancellation_probability": round(float(prob), 4),
            "label":                    "Cancelled" if label == 1 else "Completed"
        }
        for label, prob in zip(labels, probas)
    ]
    return jsonify({"predictions": predictions, "count": len(predictions)})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)