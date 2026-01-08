from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from pathlib import Path

from services.cleaning_engine import CleaningEngine
from services.schema_mapping_engine import SchemaMappingEngine
from utils.file_manager import FileManager
from utils.schema_validator import SchemaValidator

router = APIRouter(
    prefix="/api/cleaning",
    tags=["Cleaning"]
)

file_manager = FileManager(base_storage_path="temp_uploads")
schema_validator = SchemaValidator()


# -----------------------------
# REQUEST MODELS
# -----------------------------

class ManualCleanRequest(BaseModel):
    file_id: str
    range_rules: dict | None = None
    regex_rules: dict | None = None
    conditional_rules: list | None = None


class AutoCleanRequest(BaseModel):
    file_id: str


class DetectIssuesRequest(BaseModel):
    file_id: str


# -----------------------------
# ENDPOINTS
# -----------------------------


@router.post("/detect-issues")
async def detect_issues(req: DetectIssuesRequest):
    # Try to get from registry first, if not found, construct path
    file_path = file_manager.get_file_path(req.file_id)
    
    if not file_path:
        # Construct path: temp_uploads/uploads/default_user/file_id.csv
        user_dir = file_manager.uploads_dir / "default_user"
        file_path = user_dir / f"{req.file_id}.csv"
        
        # Also try .xlsx extension if csv not found
        if not file_path.exists():
            file_path = user_dir / f"{req.file_id}.xlsx"
            
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path = str(file_path)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Use file_manager to load dataframe with proper encoding handling
    try:
        df = file_manager.load_dataframe(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {str(e)}")

    engine = CleaningEngine(df)

    issues = engine.detect_issues()

    return {
        "status": "success",
        "file_id": req.file_id,
        "issues": issues
    }


@router.post("/auto-clean")
async def auto_clean(req: AutoCleanRequest):
    file_path = file_manager.get_file_path(req.file_id)

    if not file_path:
        user_dir = file_manager.uploads_dir / "default_user"
        file_path = user_dir / f"{req.file_id}.csv"
        if not file_path.exists():
            file_path = user_dir / f"{req.file_id}.xlsx"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        file_path = str(file_path)

    try:
        df = file_manager.load_dataframe(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {str(e)}")
    
    # Apply schema mapping if configured
    mapping_engine = SchemaMappingEngine(req.file_id)
    mapping = mapping_engine.load_mapping()
    if mapping:
        df = mapping_engine.apply_mapping(df, mapping)

    engine = CleaningEngine(df)

    # Perform automatic cleaning
    summary = engine.auto_clean()

    # Get cleaned dataframe from engine
    cleaned_df = engine.df

    # Save cleaned version
    cleaned_path = file_manager.save_cleaned_file(cleaned_df, req.file_id)

    return {
        "status": "success",
        "message": "Auto cleaning completed",
        "file_id": req.file_id,
        "cleaned_file_path": cleaned_path,
        "summary": summary
    }


@router.post("/manual-clean")
async def manual_clean(req: ManualCleanRequest):
    # Try to get from registry first, if not found, construct path
    file_path = file_manager.get_file_path(req.file_id)
    
    if not file_path:
        # Construct path: temp_uploads/uploads/default_user/file_id.csv
        user_dir = file_manager.uploads_dir / "default_user"
        file_path = user_dir / f"{req.file_id}.csv"
        
        # Also try .xlsx extension if csv not found
        if not file_path.exists():
            file_path = user_dir / f"{req.file_id}.xlsx"
            
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path = str(file_path)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Use file_manager to load dataframe with proper encoding handling
    try:
        df = file_manager.load_dataframe(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {str(e)}")
    
    # Apply schema mapping if configured
    mapping_engine = SchemaMappingEngine(req.file_id)
    mapping = mapping_engine.load_mapping()
    if mapping:
        df = mapping_engine.apply_mapping(df, mapping)

    engine = CleaningEngine(df)

    # Apply rules
    rules = {
        "range_rules": req.range_rules,
        "regex_rules": req.regex_rules,
        "conditional_rules": req.conditional_rules
    }

    issue_summary = engine.detect_issues()
    rules_summary = engine.apply_rules(rules)
    outlier_summary = engine.detect_outliers()
    cleaned_df = engine.fix_outliers()

    cleaned_path = file_manager.save_cleaned_file(cleaned_df, req.file_id)

    return {
        "status": "success",
        "message": "Manual cleaning completed",
        "file_id": req.file_id,
        "cleaned_file_path": cleaned_path,
        "issue_summary": issue_summary,
        "rules_summary": rules_summary,
        "outlier_summary": outlier_summary
    }


@router.get("/summary/{file_id}")
async def cleaning_summary(file_id: str):
    cleaned_path = file_manager.get_cleaned_file_path(file_id)

    if not cleaned_path or not os.path.exists(cleaned_path):
        raise HTTPException(status_code=404, detail="Cleaned file not found")

    # Use file_manager to load dataframe with proper encoding handling
    try:
        df = file_manager.load_dataframe(cleaned_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {str(e)}")

    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "preview": df.head(10).to_dict(orient="records")
    }

    return {
        "status": "success",
        "file_id": file_id,
        "summary": summary
    }
