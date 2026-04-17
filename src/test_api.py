#!/usr/bin/env python3
"""
Test script for the Uber Ride Cancellation Prediction API.
Tests all endpoints: /health, /predict, /predict/batch, and error handling.

Usage:
    python src/test_api.py                               # test localhost:5000
    python src/test_api.py https://your-app.onrender.com # test deployed API
"""

import sys
import requests
import json

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:5000"

SAMPLE = {
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


def test_health():
    print("\n--- Test: Health Check ---")
    r = requests.get(f"{BASE_URL}/health", timeout=60)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] == "healthy"
    print(f"  PASS: {data}")


def test_single_prediction():
    print("\n--- Test: Single Prediction ---")
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE, timeout=60)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "cancelled" in data
    assert "cancellation_probability" in data
    assert 0 <= data["cancellation_probability"] <= 1
    print(f"  PASS: cancelled={data['cancelled']}, "
          f"probability={data['cancellation_probability']}, "
          f"label={data['label']}")


def test_batch_prediction():
    print("\n--- Test: Batch Prediction (3 records) ---")
    batch = [SAMPLE, {**SAMPLE, "hour": 8, "hour_bucket": "morning",
                      "is_weekend": 0, "day_of_week": "Monday"},
             {**SAMPLE, "vtat_missing": 1, "Avg VTAT": None}]
    r = requests.post(f"{BASE_URL}/predict/batch", json=batch, timeout=60)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["count"] == 3
    for i, pred in enumerate(data["predictions"]):
        print(f"  Record {i}: cancelled={pred['cancelled']}, "
              f"prob={pred['cancellation_probability']}, label={pred['label']}")
    print("  PASS")


def test_missing_fields():
    print("\n--- Test: Missing Fields (should return 400) ---")
    r = requests.post(f"{BASE_URL}/predict", json={"hour": 10}, timeout=60)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print(f"  PASS: {r.json()}")


def test_empty_body():
    print("\n--- Test: Empty Body (should return 400) ---")
    r = requests.post(f"{BASE_URL}/predict",
                      headers={"Content-Type": "application/json"},
                      timeout=60)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print(f"  PASS: {r.json()}")


if __name__ == "__main__":
    print(f"Testing API at: {BASE_URL}")
    print("(If Render free tier is sleeping, first request may take ~30s)\n")

    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_missing_fields()
        test_empty_body()
        print("\n" + "=" * 55)
        print("All tests passed!")
        print("=" * 55)
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Could not connect to {BASE_URL}")
        print("Is the server running? Is the URL correct?")
    except requests.exceptions.Timeout:
        print("\nERROR: Request timed out.")
        print("The Render free tier may need more time to wake up.")
    except AssertionError as e:
        print(f"\nFAIL: {e}")