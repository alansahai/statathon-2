import requests
import json
import time
from pathlib import Path

BASE = "http://127.0.0.1:8000/api"
TEST_FILE = "test_data.csv"

LOG_FILE = "backend_pipeline_test_report.txt"


def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def run():
    # Reset log file
    open(LOG_FILE, "w").close()

    log("="*70)
    log("      STATFLOW — FULL BACKEND PIPELINE TEST")
    log("="*70)

    # -------------------------------------------------------------
    # STEP 1 — UPLOAD CSV
    # -------------------------------------------------------------
    log("\n=== STEP 1: UPLOAD CSV ===")
    files = {"file": ("test_data.csv", open(TEST_FILE, "rb"), "text/csv")}
    resp = requests.post(f"{BASE}/upload/single", files=files)

    log(f"Status: {resp.status_code}")
    log(f"Response: {resp.text}")

    if resp.status_code != 200:
        log("\nUpload failed. Stopping test.")
        return

    data = resp.json()
    file_id = data["file_ids"][0]
    log(f"\nFile ID: {file_id}")

    # -------------------------------------------------------------
    # STEP 2 — GET SCHEMA
    # -------------------------------------------------------------
    log("\n=== STEP 2: GET SCHEMA ===")
    resp = requests.get(f"{BASE}/schema/columns/{file_id}")
    log(f"Status: {resp.status_code}")
    log(f"Response: {resp.text}")

    if resp.status_code != 200:
        log("\nSchema retrieval failed.")
        return

    schema = resp.json()["columns"]

    # AUTO GENERATE MAPPING - ALL NUMERIC/CATEGORICAL
    mapping = {column: "numeric" for column in schema}  # safe default
    mapping["name"] = "categorical"
    mapping["education"] = "categorical"

    # -------------------------------------------------------------
    # STEP 3 — SAVE SCHEMA MAPPING
    # -------------------------------------------------------------
    log("\n=== STEP 3: SAVE SCHEMA MAPPING ===")
    resp = requests.post(
        f"{BASE}/schema/save/{file_id}",
        json={"mapping": mapping}
    )
    log(f"Status: {resp.status_code}")
    log(f"Response: {resp.text}")

    if resp.status_code != 200:
        log("\nSave schema failed.")
        return

    # -------------------------------------------------------------
    # STEP 4 — APPLY SCHEMA
    # -------------------------------------------------------------
    log("\n=== STEP 4: APPLY SCHEMA ===")
    resp = requests.post(f"{BASE}/schema/apply/{file_id}")
    log(f"Status: {resp.status_code}")
    log(f"Response: {resp.text}")

    if resp.status_code != 200:
        log("\nSchema application failed.")
        return

    # -------------------------------------------------------------
    # STEP 5 — CLEANING (DETECT)
    # -------------------------------------------------------------
    log("\n=== STEP 5: CLEANING — DETECT ISSUES ===")
    resp = requests.post(f"{BASE}/cleaning/detect-issues", json={"file_ids": [file_id]})
    log(f"Status: {resp.status_code}")
    log(resp.text)

    if resp.status_code != 200:
        return

    # -------------------------------------------------------------
    # STEP 6 — AUTO CLEAN
    # -------------------------------------------------------------
    log("\n=== STEP 6: AUTO CLEAN DATA ===")
    resp = requests.post(f"{BASE}/cleaning/auto-clean", json={"file_ids": [file_id]})
    log(f"Status: {resp.status_code}")
    log(resp.text)

    if resp.status_code != 200:
        return

    # -------------------------------------------------------------
    # STEP 7 — WEIGHTING
    # -------------------------------------------------------------
    log("\n=== STEP 7: WEIGHTING ENGINE ===")
    resp = requests.post(
        f"{BASE}/weighting/calculate",
        json={
            "file_ids": [file_id],
            "method": "base",
            "targets": {}
        }
    )
    log(f"Status: {resp.status_code}")
    log(resp.text)

    # -------------------------------------------------------------
    # STEP 8 — DESCRIPTIVE ANALYSIS
    # -------------------------------------------------------------
    log("\n=== STEP 8: DESCRIPTIVE ANALYSIS ===")
    resp = requests.post(
        f"{BASE}/analysis/descriptive",
        json={
            "file_ids": [file_id],
            "columns": ["age", "income", "satisfaction"]
        }
    )
    log(f"Status: {resp.status_code}")
    log(resp.text)

    # -------------------------------------------------------------
    # STEP 9 — INSIGHT OVERVIEW
    # -------------------------------------------------------------
    log("\n=== STEP 9: INSIGHT OVERVIEW ===")
    resp = requests.get(f"{BASE}/insight/overview/{file_id}")
    log(f"Status: {resp.status_code}")
    log(resp.text)

    # -------------------------------------------------------------
    # STEP 10 — GENERATE PDF REPORT
    # -------------------------------------------------------------
    log("\n=== STEP 10: GENERATE PDF REPORT ===")
    resp = requests.post(
        f"{BASE}/report/generate",
        json={
            "file_ids": [file_id],
            "report_type": "full"
        }
    )

    if resp.status_code == 200:
        pdf_path = f"generated_report_{file_id}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(resp.content)
        log(f"PDF saved as {pdf_path}")
    else:
        log(f"Report generation failed: {resp.text}")

    log("\n=== TEST COMPLETE ===")


if __name__ == "__main__":
    run()
