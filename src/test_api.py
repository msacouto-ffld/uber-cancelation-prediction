#!/usr/bin/env python3
"""
Test script for the Uber Ride Cancellation Prediction API.
Tests all endpoints: /health, /predict, /predict/batch, and error handling.

Usage:
    python src/test_api.py                               # test localhost:5001
    python src/test_api.py https://your-app.onrender.com # test deployed API
"""

import sys
import requests

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:5001"

# ── Base sample — only required booking-time fields ───────────────────────────
# is_peak and vtat_bucket are derived by the API; do not send them.
SAMPLE_REQUIRED = {
    "hour":           22,
    "month":          8,
    "is_weekend":     1,
    "vtat_missing":   0,
    "Avg VTAT":       12.5,
    "hour_bucket":    "evening",
    "day_of_week":    "Friday",
    "Vehicle Type":   "Go Sedan",
    "pickup_grouped": "Other",
    "drop_grouped":   "Other",
}

# Same booking, but caller also provides historical features
SAMPLE_WITH_HISTORY = {
    **SAMPLE_REQUIRED,
    "customer_cancel_rate": 0.55,   # this customer cancels often
    "pickup_risk":          0.42,
    "drop_risk":            0.35,
    "route_risk":           0.40,
    "customer_prior_rides": 12,
}

# High-risk booking: no driver found + late night
SAMPLE_HIGH_RISK = {
    "hour":           2,
    "month":          12,
    "is_weekend":     0,
    "vtat_missing":   1,
    "Avg VTAT":       None,         # null when no driver found
    "hour_bucket":    "night",
    "day_of_week":    "Wednesday",
    "Vehicle Type":   "eBike",
    "pickup_grouped": "Airport",
    "drop_grouped":   "Other",
}

# Low-risk booking: morning commute, fast driver assignment
SAMPLE_LOW_RISK = {
    "hour":           8,
    "month":          3,
    "is_weekend":     0,
    "vtat_missing":   0,
    "Avg VTAT":       3.0,
    "hour_bucket":    "morning",
    "day_of_week":    "Monday",
    "Vehicle Type":   "Go Sedan",
    "pickup_grouped": "Other",
    "drop_grouped":   "Other",
    "customer_cancel_rate": 0.05,
    "customer_prior_rides": 47,
}


# ── Test helpers ──────────────────────────────────────────────────────────────
def ok(msg):
    print(f"  PASS  {msg}")

def fail(msg):
    print(f"  FAIL  {msg}")
    raise AssertionError(msg)

def check(condition, msg):
    if condition:
        ok(msg)
    else:
        fail(msg)


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_health():
    print("\n--- /health ---")
    r = requests.get(f"{BASE_URL}/health", timeout=60)
    check(r.status_code == 200, f"status 200 (got {r.status_code})")
    data = r.json()
    check(data["status"] == "healthy", "status == healthy")
    check("required_features" in data, "required_features listed")
    check("derived_internally" in data, "derived_internally listed")
    check(set(data["derived_internally"]) == {"is_peak", "vtat_bucket"},
          "derived fields = {is_peak, vtat_bucket}")
    print(f"  model_type: {data.get('model_type')}")


def test_predict_required_only():
    print("\n--- /predict — required fields only (no historical) ---")
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE_REQUIRED, timeout=60)
    check(r.status_code == 200, f"status 200 (got {r.status_code}: {r.text})")
    data = r.json()
    check("cancelled" in data, "cancelled in response")
    check("cancellation_probability" in data, "probability in response")
    check(0 <= data["cancellation_probability"] <= 1, "probability in [0, 1]")
    check(data["historical_defaults_used"] is not None,
          "historical_defaults_used non-null when caller omits history")
    print(f"  cancelled={data['cancelled']}, "
          f"prob={data['cancellation_probability']}, "
          f"defaults_used={data['historical_defaults_used']}")


def test_predict_with_history():
    print("\n--- /predict — with historical features ---")
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE_WITH_HISTORY, timeout=60)
    check(r.status_code == 200, f"status 200 (got {r.status_code}: {r.text})")
    data = r.json()
    check(data["historical_defaults_used"] is None,
          "historical_defaults_used is null when all history provided")
    print(f"  cancelled={data['cancelled']}, prob={data['cancellation_probability']}")


def test_predict_high_risk():
    print("\n--- /predict — high-risk booking (no driver, late night) ---")
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE_HIGH_RISK, timeout=60)
    check(r.status_code == 200, f"status 200 (got {r.status_code}: {r.text})")
    data = r.json()
    print(f"  cancelled={data['cancelled']}, prob={data['cancellation_probability']}")
    # vtat_missing=1 should push strongly toward cancellation
    check(data["cancellation_probability"] > 0.5,
          f"high-risk prob > 0.5 (got {data['cancellation_probability']})")


def test_predict_low_risk():
    print("\n--- /predict — low-risk booking (peak morning, fast driver) ---")
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE_LOW_RISK, timeout=60)
    check(r.status_code == 200, f"status 200 (got {r.status_code}: {r.text})")
    data = r.json()
    print(f"  cancelled={data['cancelled']}, prob={data['cancellation_probability']}")
    check(data["cancellation_probability"] < 0.5,
          f"low-risk prob < 0.5 (got {data['cancellation_probability']})")


def test_derived_fields_rejected():
    """Sending is_peak or vtat_bucket explicitly should still work (pipeline ignores extra
    columns via ColumnTransformer), but the API derives them itself, so the test confirms
    the prediction is identical whether or not the caller sends them."""
    print("\n--- /predict — derived fields sent by caller are ignored ---")
    with_derived = {**SAMPLE_REQUIRED, "is_peak": 99, "vtat_bucket": "fake_value"}
    r1 = requests.post(f"{BASE_URL}/predict", json=SAMPLE_REQUIRED, timeout=60)
    r2 = requests.post(f"{BASE_URL}/predict", json=with_derived, timeout=60)
    check(r1.status_code == 200 and r2.status_code == 200, "both return 200")
    check(r1.json()["cancellation_probability"] == r2.json()["cancellation_probability"],
          "probability identical regardless of extra fields sent")
    ok("derived fields are overwritten correctly")


def test_batch_prediction():
    print("\n--- /predict/batch — 4 records (varied scenarios) ---")
    batch = [
        SAMPLE_REQUIRED,
        SAMPLE_WITH_HISTORY,
        SAMPLE_HIGH_RISK,
        SAMPLE_LOW_RISK,
    ]
    r = requests.post(f"{BASE_URL}/predict/batch", json=batch, timeout=60)
    check(r.status_code == 200, f"status 200 (got {r.status_code}: {r.text})")
    data = r.json()
    check(data["count"] == 4, f"count == 4 (got {data['count']})")
    for i, pred in enumerate(data["predictions"]):
        check(0 <= pred["cancellation_probability"] <= 1,
              f"record {i}: prob in [0, 1]")
        print(f"  [{i}] cancelled={pred['cancelled']}, "
              f"prob={pred['cancellation_probability']}, label={pred['label']}")


def test_batch_too_large():
    print("\n--- /predict/batch — >100 records (should return 400) ---")
    batch = [SAMPLE_REQUIRED] * 101
    r = requests.post(f"{BASE_URL}/predict/batch", json=batch, timeout=60)
    check(r.status_code == 400, f"status 400 (got {r.status_code})")
    ok(r.json().get("error", ""))


def test_missing_required_fields():
    print("\n--- /predict — missing required fields (should return 400) ---")
    r = requests.post(f"{BASE_URL}/predict", json={"hour": 10}, timeout=60)
    check(r.status_code == 400, f"status 400 (got {r.status_code})")
    err = r.json()
    check("missing_fields" in err.get("details", {}), "missing_fields listed")
    ok(str(err))


def test_invalid_hour_bucket():
    print("\n--- /predict — invalid hour_bucket (should return 400) ---")
    bad = {**SAMPLE_REQUIRED, "hour_bucket": "noon"}
    r = requests.post(f"{BASE_URL}/predict", json=bad, timeout=60)
    check(r.status_code == 400, f"status 400 (got {r.status_code})")
    ok(r.json().get("details", {}).get("hour_bucket", ""))


def test_invalid_historical_rate():
    print("\n--- /predict — historical rate out of [0,1] (should return 400) ---")
    bad = {**SAMPLE_REQUIRED, "customer_cancel_rate": 1.5}
    r = requests.post(f"{BASE_URL}/predict", json=bad, timeout=60)
    check(r.status_code == 400, f"status 400 (got {r.status_code})")
    ok(r.json().get("details", {}).get("customer_cancel_rate", ""))


def test_empty_body():
    print("\n--- /predict — empty body (should return 400) ---")
    r = requests.post(f"{BASE_URL}/predict",
                      headers={"Content-Type": "application/json"},
                      timeout=60)
    check(r.status_code == 400, f"status 400 (got {r.status_code})")
    ok(r.json().get("error", ""))


# ── Runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Testing API at: {BASE_URL}")
    print("(Render free tier may need ~30s to wake on first request)\n")

    tests = [
        test_health,
        test_predict_required_only,
        test_predict_with_history,
        test_predict_high_risk,
        test_predict_low_risk,
        test_derived_fields_rejected,
        test_batch_prediction,
        test_batch_too_large,
        test_missing_required_fields,
        test_invalid_hour_bucket,
        test_invalid_historical_rate,
        test_empty_body,
    ]

    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {e}")
            failed += 1
        except requests.exceptions.ConnectionError:
            print(f"\nERROR: Could not connect to {BASE_URL}")
            print("Is the server running? Is the URL correct?")
            break
        except requests.exceptions.Timeout:
            print("\nERROR: Request timed out.")
            print("Render free tier may need more time to wake up.")
            break

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 55)
