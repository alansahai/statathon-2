from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

from utils.file_manager import FileManager
from utils.schema_validator import SchemaValidator

router = APIRouter(tags=["01 Upload"])

file_manager = FileManager(base_storage_path="temp_uploads")
schema_validator = SchemaValidator()


@router.get("/ping")
async def ping():
    """Health check endpoint for upload router"""
    return {"status": "ok", "message": "Upload router is operational"}

MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
PREVIEW_ROWS = 10
MAX_MULTIPLE_FILES = 5


def make_json_safe(data):
    """Recursively convert NaN/INF values to None."""
    if isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
        return data
    if isinstance(data, dict):
        return {k: make_json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [make_json_safe(x) for x in data]
    return data


@router.post("/single")
async def upload_single_file(file: UploadFile = File(...)):
    """
    STEP 01 UPLOAD:
    Upload a single CSV/Excel file for analysis.
    
    This is the entry point for the entire pipeline. Upload your dataset here
    to receive a unique file_id for use in all subsequent operations.
    
    **Accepted Formats:** .csv, .xlsx, .xls
    **Max Size:** 50MB
    
    **Returns:**
    ```json
    {
        "status": "success",
        "step": "upload",
        "file_ids": ["uuid-123"],
        "results": {
            "uuid-123": {
                "file_id": "uuid-123",
                "filename": "survey_data.csv",
                "row_count": 1000,
                "column_count": 25,
                "columns": ["age", "gender", ...],
                "preview": [{...}, {...}]
            }
        }
    }
    ```
    """
    try:
        result = await _process_upload(file, expected_type="auto")
        file_id = result["file_info"]["file_id"]
        
        return {
            "status": "success",
            "step": "upload",
            "file_ids": [file_id],
            "results": {
                file_id: result
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    STEP 01 UPLOAD:
    Upload multiple CSV files (up to 5 files at once).
    
    Use this for batch processing. All uploaded files will receive unique file_ids
    that can be passed together to subsequent pipeline endpoints.
    
    **Max Files:** 5
    **Accepted Formats:** .csv, .xlsx, .xls
    **Max Size per File:** 50MB
    
    **Input:**
    ```
    files: array of file uploads
    ```
    
    **Returns:**
    ```json
    {
        "status": "success",
        "step": "upload",
        "file_ids": ["uuid-1", "uuid-2"],
        "results": {
            "uuid-1": {...},
            "uuid-2": {...}
        }
    }
    ```
    """
    try:
        # Validate file count
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > MAX_MULTIPLE_FILES:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many files. Maximum {MAX_MULTIPLE_FILES} files allowed"
            )
        
        uploaded_files = []
        errors = []
        results_dict = {}
        
        for idx, file in enumerate(files):
            try:
                # Process each file
                result = await _process_upload(file, expected_type="auto")
                file_id = result["file_info"]["file_id"]
                
                # Store result
                results_dict[file_id] = result
                
                # Track uploaded file info
                uploaded_files.append({
                    "file_id": file_id,
                    "filename": result["file_info"]["filename"],
                    "row_count": result["file_info"]["row_count"],
                    "column_count": result["file_info"]["column_count"]
                })
                
            except HTTPException as e:
                errors.append({
                    "file_index": idx,
                    "filename": file.filename if file.filename else f"file_{idx}",
                    "error": e.detail
                })
            except Exception as e:
                errors.append({
                    "file_index": idx,
                    "filename": file.filename if file.filename else f"file_{idx}",
                    "error": str(e)
                })
        
        # If no files were successfully uploaded
        if not uploaded_files:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to upload any files. Errors: {errors}"
            )
        
        # Return standardized response
        response = {
            "status": "success" if not errors else "partial_success",
            "step": "upload",
            "file_ids": [f["file_id"] for f in uploaded_files],
            "results": results_dict
        }
        
        if errors:
            response["errors"] = errors
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _process_upload(file: UploadFile, expected_type: str = "auto") -> Dict[str, Any]:
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Invalid extension {file_ext}")

        # Expected type check
        if expected_type == "csv" and file_ext != ".csv":
            raise HTTPException(status_code=400, detail="Expected CSV file")
        if expected_type == "excel" and file_ext not in [".xlsx", ".xls"]:
            raise HTTPException(status_code=400, detail="Expected Excel file")

        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds limit")

        # Save file to disk
        file_info = file_manager.save_uploaded_file(
            file_content=file_bytes,
            file_name=file.filename,
            user_id="default_user"
        )

        # Load DataFrame safely
        try:
            if file_ext == ".csv":
                df = pd.read_csv(
                    file_info["file_path"],
                    encoding="utf-8",
                    low_memory=False,
                    on_bad_lines="skip"
                )
            else:
                df = pd.read_excel(file_info["file_path"], engine="openpyxl")

        except Exception as e:
            file_manager.delete_file_by_path(file_info["file_path"])
            raise HTTPException(status_code=400, detail=f"Failed to parse data: {str(e)}")

        if df.empty:
            file_manager.delete_file_by_path(file_info["file_path"])
            raise HTTPException(status_code=400, detail="Parsed file contains no data")

        # Schema
        schema = schema_validator.infer_schema(df)
        schema = make_json_safe(schema)

        # Preview rows
        preview_df = df.head(PREVIEW_ROWS)
        preview_df = preview_df.replace([np.inf, -np.inf], None)
        preview_df = preview_df.where(pd.notnull(preview_df), None)
        preview = preview_df.to_dict(orient="records")

        # Return sanitized JSON
        return make_json_safe({
            "status": "success",
            "file_info": {
                "file_id": file_info["file_id"],
                "filename": file.filename,
                "filepath": str(Path(file_info["file_path"]).as_posix()),
                "file_size": file_info["file_size"],
                "upload_timestamp": file_info["upload_timestamp"],
                "row_count": len(df),
                "column_count": len(df.columns)
            },
            "schema": schema,
            "columns": list(df.columns),
            "preview": preview,
            "preview_row_count": len(preview)
        })

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/file/{file_id}")
async def delete_uploaded_file(file_id: str):
    success = file_manager.delete_file(file_id)

    if not success:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "status": "success",
        "message": "File deleted successfully",
        "file_id": file_id
    }


@router.get("/status/{file_id}")
async def get_upload_status(file_id: str):
    file_path = file_manager.get_file_path(file_id)

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    file_stats = os.stat(file_path)

    return {
        "status": "success",
        "file_id": file_id,
        "filepath": str(Path(file_path).as_posix()),
        "file_size": file_stats.st_size,
        "exists": True
    }
