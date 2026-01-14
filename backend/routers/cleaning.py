from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import os
from pathlib import Path

from services.cleaning_engine import CleaningEngine
from services.schema_mapping_engine import SchemaMappingEngine
from utils.file_manager import FileManager
from utils.schema_validator import SchemaValidator

router = APIRouter(
    prefix="/cleaning",
    tags=["03 Cleaning"]
)

file_manager = FileManager(base_storage_path="temp_uploads")
schema_validator = SchemaValidator()


# -----------------------------
# REQUEST MODELS
# -----------------------------

class ManualCleanRequest(BaseModel):
    file_id: Optional[str] = None
    file_ids: Optional[List[str]] = None
    range_rules: dict | None = None
    regex_rules: dict | None = None
    conditional_rules: list | None = None


class AutoCleanRequest(BaseModel):
    file_id: Optional[str] = None
    file_ids: Optional[List[str]] = None


class DetectIssuesRequest(BaseModel):
    file_id: Optional[str] = None
    file_ids: Optional[List[str]] = None


# -----------------------------
# ENDPOINTS
# -----------------------------


@router.post("/detect-issues")
async def detect_issues(req: DetectIssuesRequest):
    # Normalize file_ids
    file_ids = req.file_ids or ([req.file_id] if req.file_id else None)
    
    if not file_ids or len(file_ids) == 0:
        raise HTTPException(status_code=400, detail="No file_ids provided")
    if len(file_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
    
    # Process each file
    results_per_file = {}
    errors = {}
    
    for fid in file_ids:
        try:
            # Try to get from registry first, if not found, construct path
            file_path = file_manager.get_file_path(fid)
            
            if not file_path:
                # Construct path: temp_uploads/uploads/default_user/file_id.csv
                user_dir = file_manager.uploads_dir / "default_user"
                file_path = user_dir / f"{fid}.csv"
                
                # Also try .xlsx extension if csv not found
                if not file_path.exists():
                    file_path = user_dir / f"{fid}.xlsx"
                    
                if not file_path.exists():
                    errors[fid] = "File not found"
                    continue
                
                file_path = str(file_path)

            if not os.path.exists(file_path):
                errors[fid] = "File not found"
                continue

            # Use file_manager to load dataframe with proper encoding handling
            try:
                df = file_manager.load_dataframe(file_path)
            except Exception as e:
                errors[fid] = f"Failed to load file: {str(e)}"
                continue

            engine = CleaningEngine(df)
            issues = engine.detect_issues()

            results_per_file[fid] = {"issues": issues}
            
        except Exception as e:
            errors[fid] = str(e)
    
    # Determine status
    status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
    
    response = {
        "status": status,
        "file_ids": file_ids,
        "results": results_per_file
    }
    
    if errors:
        response["errors"] = errors
    
    return response


@router.post("/auto-clean")
async def auto_clean(req: AutoCleanRequest):
    # Normalize file_ids
    file_ids = req.file_ids or ([req.file_id] if req.file_id else None)
    
    if not file_ids or len(file_ids) == 0:
        raise HTTPException(status_code=400, detail="No file_ids provided")
    if len(file_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
    
    # Process each file
    results_per_file = {}
    errors = {}
    
    for fid in file_ids:
        try:
            # Try to get the best available file path
            try:
                file_path = file_manager.get_best_available_file(fid)
            except FileNotFoundError as e:
                errors[fid] = f"File not found: {str(e)}"
                continue

            try:
                df = file_manager.load_dataframe(file_path)
            except Exception as e:
                errors[fid] = f"Failed to load file: {str(e)}"
                continue
            
            # Apply schema mapping if configured
            try:
                mapping_engine = SchemaMappingEngine(fid)
                mapping = mapping_engine.load_mapping()
                if mapping and mapping.get('columns'):
                    print(f"[cleaning] Applying schema mapping for file {fid}")
                    df = mapping_engine.apply_mapping(df, mapping)
                else:
                    print(f"[cleaning] No schema mapping found for file {fid}, continuing with original data")
            except Exception as e:
                print(f"[cleaning] Schema mapping failed: {str(e)}, continuing with original data")

            engine = CleaningEngine(df)

            # Perform automatic cleaning
            summary = engine.auto_clean()

            # Get cleaned dataframe from engine
            cleaned_df = engine.df

            # Save cleaned version
            cleaned_path = file_manager.save_cleaned_file(cleaned_df, fid)

            results_per_file[fid] = {
                "cleaned_file_path": cleaned_path,
                "summary": summary
            }
            
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[cleaning] Error processing file {fid}: {error_detail}")
            errors[fid] = str(e)
    
    # Determine status
    if len(errors) > 0 and len(results_per_file) == 0:
        raise HTTPException(status_code=500, detail=f"All files failed: {errors}")
    
    status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
    
    response = {
        "status": status,
        "message": "Auto cleaning completed",
        "file_ids": file_ids,
        "results": results_per_file
    }
    
    if errors:
        response["errors"] = errors
    
    return response


@router.post("/manual-clean")
async def manual_clean(req: ManualCleanRequest):
    # Normalize file_ids
    file_ids = req.file_ids or ([req.file_id] if req.file_id else None)
    
    if not file_ids or len(file_ids) == 0:
        raise HTTPException(status_code=400, detail="No file_ids provided")
    if len(file_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
    
    # Process each file
    results_per_file = {}
    errors = {}
    
    for fid in file_ids:
        try:
            # Try to get from registry first, if not found, construct path
            file_path = file_manager.get_file_path(fid)
            
            if not file_path:
                # Construct path: temp_uploads/uploads/default_user/file_id.csv
                user_dir = file_manager.uploads_dir / "default_user"
                file_path = user_dir / f"{fid}.csv"
                
                # Also try .xlsx extension if csv not found
                if not file_path.exists():
                    file_path = user_dir / f"{fid}.xlsx"
                    
                if not file_path.exists():
                    errors[fid] = "File not found"
                    continue
                
                file_path = str(file_path)

            if not os.path.exists(file_path):
                errors[fid] = "File not found"
                continue

            # Use file_manager to load dataframe with proper encoding handling
            try:
                df = file_manager.load_dataframe(file_path)
            except Exception as e:
                errors[fid] = f"Failed to load file: {str(e)}"
                continue
            
            # Apply schema mapping if configured
            mapping_engine = SchemaMappingEngine(fid)
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

            cleaned_path = file_manager.save_cleaned_file(cleaned_df, fid)

            results_per_file[fid] = {
                "cleaned_file_path": cleaned_path,
                "issue_summary": issue_summary,
                "rules_summary": rules_summary,
                "outlier_summary": outlier_summary
            }
            
        except Exception as e:
            errors[fid] = str(e)
    
    # Determine status
    status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
    
    response = {
        "status": status,
        "message": "Manual cleaning completed",
        "file_ids": file_ids,
        "results": results_per_file
    }
    
    if errors:
        response["errors"] = errors
    
    return response


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
