"""
Uber Ride Cancellation Prediction — Flask API
Usage:
    python src/app.py
Then POST to http://localhost:5000/predict with a JSON body.
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# ── Load model ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "best_pipeline.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run the notebook through Section 10 first to generate best_pipeline.pkl."
    )

pipeline = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

app = Flask(__name__)

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "HistGradientBoosting"})


# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body with one or more bookings.

    Example single booking:
    {
        "hour": 22,
        "month": 8,
        "is_weekend": 1,
        "vtat_missing": 0,
        "Avg VTAT": 12.5,
        "hour_bucket": "evening",
        "day_of_week": "Friday",
        "Vehicle Type": "Go Sedan",
        "pickup_grouped": "Other",
        "drop_grouped": "Other"
    }

    Returns:
    {
        "predictions": [
            {
                "cancelled": 1,
                "cancellation_probability": 0.712
            }
        ]
    }
    """
    data = request.get_json(force=True)

    # Accept either a single dict or a list of dicts
    if isinstance(data, dict):
        data = [data]

    try:
        df = pd.DataFrame(data)
    except Exception as e:
        return jsonify({"error": f"Could not parse input: {str(e)}"}), 400

    # Validate required fields
    required = [
        "hour", "month", "is_weekend", "vtat_missing", "Avg VTAT",
        "hour_bucket", "day_of_week", "Vehicle Type",
        "pickup_grouped", "drop_grouped"
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return jsonify({"error": f"Missing fields: {missing_cols}"}), 400

    try:
        labels = pipeline.predict(df)
        probas = pipeline.predict_proba(df)[:, 1]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    results = [
        {
            "cancelled": int(label),
            "cancellation_probability": round(float(prob), 4)
        }
        for label, prob in zip(labels, probas)
    ]

    return jsonify({"predictions": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
