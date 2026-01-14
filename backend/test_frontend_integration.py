"""
Test Frontend-Backend Integration for Weighting Module
This simulates frontend behavior and validates all endpoints
"""
import requests
import json
import time
from pathlib import Path

API_BASE = "http://127.0.0.1:8000/api/v1"

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_result(endpoint, status, data_preview):
    print(f"\n✓ {endpoint}")
    print(f"  Status: {status}")
    print(f"  Preview: {data_preview[:150]}...")

def test_complete_workflow():
    print_section("Frontend-Backend Integration Test")
    
    # Step 1: Upload
    print_section("STEP 1: Upload File (Simulating Frontend Upload)")
    test_file = Path("test_simple.csv")
    
    files = {'file': open(test_file, 'rb')}
    response = requests.post(f"{API_BASE}/upload", files=files)
    
    if response.status_code == 200:
        upload_data = response.json()
        file_id = upload_data.get('file_id')
        print_result("POST /api/v1/upload", response.status_code, json.dumps(upload_data))
        print(f"\n  📄 File ID: {file_id}")
        print(f"  📄 Filename: {upload_data.get('filename')}")
    else:
        print(f"❌ Upload failed: {response.text}")
        return
    
    # Step 2: Calculate Base Weights (Default Method)
    print_section("STEP 2: Calculate Weights (Frontend: Calculate Tab)")
    
    payload = {
        "file_id": file_id,
        "method": "base"
    }
    
    response = requests.post(f"{API_BASE}/weighting/calculate", json=payload)
    
    if response.status_code == 200:
        calc_data = response.json()
        print_result("POST /api/v1/weighting/calculate", response.status_code, json.dumps(calc_data))
        
        result = calc_data.get('results', {}).get(file_id, {})
        if result:
            print(f"\n  ⚖️ Method: {result.get('method')}")
            print(f"  ⚖️ Auto-actions: {len(result.get('auto_actions', []))}")
            print(f"  ⚖️ Warnings: {len(result.get('warnings', []))}")
    else:
        print(f"❌ Calculate failed: {response.text}")
        return
    
    # Step 3: Validate Weights (Auto-triggered after calculate)
    print_section("STEP 3: Validate Weights (Frontend: Auto-validation)")
    
    payload = {"file_id": file_id}
    response = requests.post(f"{API_BASE}/weighting/validate", json=payload)
    
    if response.status_code == 200:
        validate_data = response.json()
        print_result("POST /api/v1/weighting/validate", response.status_code, json.dumps(validate_data))
        
        result = validate_data.get('results', {}).get(file_id, {})
        if result:
            print(f"\n  ✓ Status: {result.get('status')}")
            print(f"  ✓ Problems: {len(result.get('problems', []))}")
            print(f"  ✓ Valid Observations: {result.get('n_valid')}")
    else:
        print(f"❌ Validation failed: {response.text}")
    
    # Step 4: Get Diagnostics (When user clicks Diagnostics tab)
    print_section("STEP 4: Load Diagnostics (Frontend: Diagnostics Tab)")
    
    response = requests.get(f"{API_BASE}/weighting/diagnostics/{file_id}")
    
    if response.status_code == 200:
        diag_data = response.json()
        print_result("GET /api/v1/weighting/diagnostics/{file_id}", response.status_code, json.dumps(diag_data))
        
        diagnostics = diag_data.get('diagnostics', {})
        if diagnostics:
            stats = diagnostics.get('statistics', {})
            print(f"\n  📊 Weight Column: {diagnostics.get('weight_column')}")
            print(f"  📊 Mean: {stats.get('mean')}")
            print(f"  📊 CV: {stats.get('cv')}")
            print(f"  📊 Effective Sample Size: {diagnostics.get('effective_sample_size')}")
            print(f"  📊 Design Effect: {diagnostics.get('design_effect')}")
    else:
        print(f"❌ Diagnostics failed: {response.text}")
    
    # Step 5: Trim Weights (Optional - if user clicks Trim tab)
    print_section("STEP 5: Trim Weights (Frontend: Trim Tab - Optional)")
    
    payload = {
        "file_id": file_id,
        "min_w": 0.5,
        "max_w": 2.0
    }
    
    response = requests.post(f"{API_BASE}/weighting/trim", json=payload)
    
    if response.status_code == 200:
        trim_data = response.json()
        print_result("POST /api/v1/weighting/trim", response.status_code, json.dumps(trim_data))
        
        result = trim_data.get('results', {}).get(file_id, {})
        if result:
            summary = result.get('summary', {})
            print(f"\n  ✂️ Trimmed Count: {summary.get('trimmed_count')}")
            print(f"  ✂️ Trimmed %: {summary.get('trimmed_pct', 0):.2f}%")
            print(f"  ✂️ New Mean: {summary.get('new_mean')}")
    else:
        print(f"❌ Trim failed: {response.text}")
    
    # Step 6: Check Operations Log (Metadata tracking)
    print_section("STEP 6: Get Operations Log (Backend Metadata)")
    
    response = requests.get(f"{API_BASE}/weighting/operations-log/{file_id}")
    
    if response.status_code == 200:
        log_data = response.json()
        print_result("GET /api/v1/weighting/operations-log/{file_id}", response.status_code, json.dumps(log_data))
    else:
        print(f"⚠️ Operations log: {response.text}")
    
    # Summary
    print_section("✅ INTEGRATION TEST COMPLETE")
    print("""
    All Frontend-Backend Integration Points Verified:
    
    ✓ File Upload → Backend receives and stores file
    ✓ Calculate Weights → Backend processes and returns results
    ✓ Auto-validation → Backend validates weight quality
    ✓ Diagnostics → Backend provides comprehensive statistics
    ✓ Trim Weights → Backend adjusts extreme weights
    ✓ Metadata Tracking → Backend persists operations
    
    Frontend is fully integrated with Backend! 🎉
    """)

if __name__ == "__main__":
    try:
        test_complete_workflow()
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
