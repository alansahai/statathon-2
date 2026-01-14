"""
Test weighting flow - simulate frontend behavior
"""
import requests
import json
from pathlib import Path

API_BASE = "http://127.0.0.1:8000/api/v1"

def test_weighting_flow():
    print("="*60)
    print("Testing Weighting Flow")
    print("="*60)
    
    # Step 1: Upload a file
    print("\n1. Uploading test file...")
    test_file = Path("test_simple.csv")
    
    files = {'file': open(test_file, 'rb')}
    response = requests.post(f"{API_BASE}/upload", files=files)
    print(f"Upload Status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Upload failed: {response.text}")
        return
    
    upload_data = response.json()
    print(f"Upload Response: {json.dumps(upload_data, indent=2)}")
    
    file_id = upload_data.get('file_id')
    if not file_id:
        print("ERROR: No file_id in response")
        return
    
    print(f"File ID: {file_id}")
    
    # Step 2: Calculate base weights
    print("\n2. Calculating base weights...")
    payload = {
        "file_id": file_id,
        "method": "base"
    }
    
    response = requests.post(
        f"{API_BASE}/weighting/calculate",
        json=payload
    )
    print(f"Calculate Status: {response.status_code}")
    calc_data = response.json()
    print(f"Calculate Response: {json.dumps(calc_data, indent=2)[:500]}")
    
    if response.status_code != 200:
        print(f"ERROR: Calculate failed: {calc_data}")
        return
    
    # Step 3: Validate weights
    print("\n3. Validating weights...")
    payload = {
        "file_id": file_id
    }
    
    response = requests.post(
        f"{API_BASE}/weighting/validate",
        json=payload
    )
    print(f"Validate Status: {response.status_code}")
    validate_data = response.json()
    print(f"Validate Response: {json.dumps(validate_data, indent=2)[:500]}")
    
    # Step 4: Get diagnostics
    print("\n4. Getting diagnostics...")
    response = requests.get(
        f"{API_BASE}/weighting/diagnostics/{file_id}"
    )
    print(f"Diagnostics Status: {response.status_code}")
    
    if response.status_code == 200:
        diag_data = response.json()
        print(f"Diagnostics Response: {json.dumps(diag_data, indent=2)[:800]}")
    else:
        print(f"ERROR: Diagnostics failed: {response.text}")
    
    # Step 5: Trim weights
    print("\n5. Trimming weights...")
    payload = {
        "file_id": file_id,
        "min_w": 0.5,
        "max_w": 2.0
    }
    
    response = requests.post(
        f"{API_BASE}/weighting/trim",
        json=payload
    )
    print(f"Trim Status: {response.status_code}")
    trim_data = response.json()
    print(f"Trim Response: {json.dumps(trim_data, indent=2)[:500]}")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

if __name__ == "__main__":
    test_weighting_flow()
