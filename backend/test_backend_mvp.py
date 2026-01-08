import requests
import json
import pandas as pd
from datetime import datetime

BASE = "http://127.0.0.1:8000/api"

LOG_FILE = "backend_test_report.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def print_header(title):
    line = "=" * 80
    log(f"\n{line}\n{title}\n{line}")

# -------------------------------------------------------
# 1) TEST FILE UPLOAD
# -------------------------------------------------------

def test_upload():
    print_header("TEST 1 — FILE UPLOAD")

    files = {"file": open("test_data.csv", "rb")}
    resp = requests.post(f"{BASE}/upload/file", files=files)
    
    if resp.status_code != 200:
        log(f"❌ Upload failed: {resp.text}")
        return None

    data = resp.json()
    file_id = data.get("file_id")

    log(f"✔ Upload Success — File ID: {file_id}")
    return file_id

# -------------------------------------------------------
# 2) TEST SCHEMA VALIDATION
# -------------------------------------------------------

def test_schema_validation(file_id):
    print_header("TEST 2 — SCHEMA VALIDATION")

    resp = requests.get(f"{BASE}/upload/schema/{file_id}")
    if resp.status_code != 200:
        log(f"❌ Schema validation failed: {resp.text}")
        return None

    data = resp.json()
    log(f"✔ Schema Validated — {len(data['columns'])} columns detected")

    return data

# -------------------------------------------------------
# 3) TEST CLEANING DETECTION + AUTOCLEAN
# -------------------------------------------------------

def test_cleaning(file_id):
    print_header("TEST 3 — CLEANING ENGINE")

    # Detect issues
    resp = requests.post(f"{BASE}/cleaning/detect-issues", json={"file_id": file_id})
    if resp.status_code != 200:
        log(f"❌ Cleaning detect-issues failed: {resp.text}")
        return None

    log("✔ Cleaning issues detected")

    # Auto-clean
    resp = requests.post(f"{BASE}/cleaning/auto-clean", json={"file_id": file_id})
    if resp.status_code != 200:
        log(f"❌ Auto-clean failed: {resp.text}")
        return None

    cleaned_id = resp.json().get("cleaned_file_id")
    log(f"✔ Auto-Clean Success — Cleaned File ID: {cleaned_id}")

    return cleaned_id

# -------------------------------------------------------
# 4) WEIGHTING ENGINE
# -------------------------------------------------------

def test_weighting(cleaned_id):
    print_header("TEST 4 — WEIGHTING ENGINE")

    payload = {
        "file_id": cleaned_id,
        "weight_column": None,
        "targets": {}
    }

    resp = requests.post(f"{BASE}/weighting/calculate", json=payload)
    if resp.status_code != 200:
        log(f"❌ Weighting failed: {resp.text}")
        return None

    weighted_id = resp.json().get("weighted_file_id")
    log(f"✔ Weighting Success — Weighted File ID: {weighted_id}")

    return weighted_id

# -------------------------------------------------------
# 5) ANALYSIS ENGINE
# -------------------------------------------------------

def test_analysis(file_id):
    print_header("TEST 5 — ANALYSIS ENGINE")

    payload = {
        "file_id": file_id,
        "columns": []
    }

    resp = requests.post(f"{BASE}/analysis/descriptive", json=payload)
    if resp.status_code != 200:
        log(f"❌ Descriptive stats failed: {resp.text}")
        return None

    log(f"✔ Descriptive stats OK")

# -------------------------------------------------------
# 6) FORECASTING ENGINE
# -------------------------------------------------------

def test_forecast(file_id):
    print_header("TEST 6 — FORECASTING ENGINE")

    payload = {
        "file_id": file_id,
        "time_column": "Date",
        "periods": 30
    }

    resp = requests.post(f"{BASE}/forecasting/run", json=payload)
    if resp.status_code != 200:
        log(f"❌ Forecasting failed: {resp.text}")
    else:
        log("✔ Forecasting success")

# -------------------------------------------------------
# 7) ML ENGINE
# -------------------------------------------------------

def test_ml(file_id):
    print_header("TEST 7 — ML ENGINE")

    payload = {
        "file_id": file_id,
        "target": "RainTomorrow",
        "features": []
    }

    resp = requests.post(f"{BASE}/ml/classify", json=payload)
    if resp.status_code != 200:
        log(f"❌ ML classification failed: {resp.text}")
    else:
        log("✔ ML Classification OK")

# -------------------------------------------------------
# 8) INSIGHT ENGINE
# -------------------------------------------------------

def test_insight(file_id):
    print_header("TEST 8 — INSIGHT ENGINE")

    resp = requests.get(f"{BASE}/insight/overview/{file_id}")
    if resp.status_code != 200:
        log(f"❌ Insight overview failed: {resp.text}")
    else:
        log("✔ Insight overview OK")

# -------------------------------------------------------
# 9) REPORT GENERATION
# -------------------------------------------------------

def test_report(file_id):
    print_header("TEST 9 — REPORT GENERATION")

    resp = requests.post(f"{BASE}/report/generate", json={"file_id": file_id})
    if resp.status_code != 200:
        log(f"❌ Report generation failed: {resp.text}")
        return

    log("✔ PDF Report generated successfully")

# -------------------------------------------------------
# MAIN TEST SEQUENCE
# -------------------------------------------------------

print_header("STARTING FULL BACKEND MVP TEST")

file_id = test_upload()
schema = test_schema_validation(file_id)
cleaned_id = test_cleaning(file_id)
weighted_id = test_weighting(cleaned_id)
test_analysis(cleaned_id)
test_forecast(cleaned_id)
test_ml(cleaned_id)
test_insight(cleaned_id)
test_report(cleaned_id)

print_header("TESTING COMPLETE — Check backend_test_report.txt for results")
