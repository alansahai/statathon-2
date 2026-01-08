from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

from utils.file_manager import FileManager
from utils.schema_validator import SchemaValidator

router = APIRouter(tags=["Upload"])

file_manager = FileManager(base_storage_path="temp_uploads")
schema_validator = SchemaValidator()

MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
PREVIEW_ROWS = 10


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


@router.post("/csv")
async def upload_csv(file: UploadFile = File(...)):
    return await _process_upload(file, expected_type="csv")


@router.post("/excel")
async def upload_excel(file: UploadFile = File(...)):
    return await _process_upload(file, expected_type="excel")


@router.post("/file")
async def upload_file(file: UploadFile = File(...)):
    return await _process_upload(file, expected_type="auto")


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
