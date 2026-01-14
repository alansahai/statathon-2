import requests
import json
import time
import os

BASE_URL = "http://127.0.0.1:8000"
TEST_FILE = "test_data.csv"  # update if needed

def pretty(data):
    return json.dumps(data, indent=4)

def print_section(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

# 1. UPLOAD FILE
def test_upload():
    print_section("STEP 1: UPLOAD CSV FILE")
    with open(TEST_FILE, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/upload/single", files={"file": f})
    print(res.text)
    data = res.json()
    return data["file_ids"][0]

# 2. AUTO SCHEMA DETECTION
def test_auto_schema(file_id):
    print_section("STEP 2: AUTO SCHEMA DETECTION")
    res = requests.post(f"{BASE_URL}/api/schema/auto/{file_id}")
    print(res.text)
    return res.json()["auto_mapping"]

# 3. SAVE MANUAL MAPPING
def test_save_manual(file_id, mapping):
    print_section("STEP 3: SAVE MANUAL SCHEMA MAPPING")
    res = requests.post(f"{BASE_URL}/api/schema/save/{file_id}", json={"mapping": mapping})
    print(res.text)

# 4. APPLY MAPPING
def test_apply_mapping(file_id):
    print_section("STEP 4: APPLY MAPPING")
    res = requests.post(f"{BASE_URL}/api/schema/apply/{file_id}")
    print(res.text)

# 5. DETECT CLEANING ISSUES
def test_cleaning(file_id):
    print_section("STEP 5: DETECT CLEANING ISSUES")
    res = requests.post(f"{BASE_URL}/api/cleaning/detect-issues", json={"file_ids": [file_id]})
    print(res.text)

# 6. RUN AUTO CLEAN
def test_auto_clean(file_id):
    print_section("STEP 6: AUTO CLEAN DATA")
    res = requests.post(f"{BASE_URL}/api/cleaning/auto-clean", json={"file_ids": [file_id]})
    print(res.text)

# 7. DESCRIPTIVE ANALYSIS
def test_analysis(file_id, columns):
    print_section("STEP 7: DESCRIPTIVE ANALYSIS")
    res = requests.post(f"{BASE_URL}/api/analysis/descriptive", json={
        "file_ids": [file_id],
        "columns": columns
    })
    print(res.text)

# 8. INSIGHT OVERVIEW
def test_insight(file_id):
    print_section("STEP 8: INSIGHT OVERVIEW")
    res = requests.get(f"{BASE_URL}/api/insight/overview/{file_id}")
    print(res.text)

# 9. GENERATE PDF REPORT
def test_report(file_id):
    print_section("STEP 9: GENERATE PDF REPORT")
    res = requests.post(f"{BASE_URL}/api/report/generate", json={
        "file_ids": [file_id],
        "report_type": "full"
    })
    
    filename = "generated_pipeline_report.pdf"
    with open(filename, "wb") as f:
        f.write(res.content)
    print(f"PDF saved as {filename}")

# MAIN RUNNER
def run_all():
    start = time.time()
    file_id = test_upload()
    auto_map = test_auto_schema(file_id)
    test_save_manual(file_id, auto_map)
    test_apply_mapping(file_id)
    test_cleaning(file_id)
    test_auto_clean(file_id)
    test_analysis(file_id, list(auto_map.keys()))
    test_insight(file_id)
    test_report(file_id)
    print_section("TEST COMPLETE")
    print(f"Total time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    run_all()
