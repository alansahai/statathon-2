"""
Weighting Router - MoSPI-ready automated survey weighting
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

from services.weighting_engine import WeightingEngine
from services.file_manager import FileManager

router = APIRouter(
    prefix="/weighting",
    tags=["04 Weighting"]
)


# -----------------------------
# REQUEST MODELS (SIMPLIFIED)
# -----------------------------

class CalculateWeightsRequest(BaseModel):
    file_id: Optional[str] = None
    file_ids: Optional[List[str]] = None
    method: str = "base"  # Default to "base" - "base", "poststrat", "raking"
    # All other fields are optional - auto-detection handles them
    inclusion_prob_column: Optional[str] = None
    strata_column: Optional[str] = None
    population_totals: Optional[Dict[str, float]] = None
    control_totals: Optional[Dict[str, Dict[str, float]]] = None
    max_iterations: Optional[int] = 50
    tolerance: Optional[float] = 0.001


class ValidateWeightsRequest(BaseModel):
    file_id: Optional[str] = None
    file_ids: Optional[List[str]] = None
    weight_column: Optional[str] = None


class TrimWeightsRequest(BaseModel):
    file_id: Optional[str] = None
    file_ids: Optional[List[str]] = None
    min_w: float = 0.3
    max_w: float = 3.0
    weight_column: Optional[str] = None


# -----------------------------
# ENDPOINTS
# -----------------------------


@router.post("/calculate")
async def calculate_weights(req: CalculateWeightsRequest):
    """
    MoSPI-ready automated weight calculation
    
    Supports:
    - base: Base weights (uniform if no inclusion_prob_column)
    - poststrat: Post-stratification (auto-creates age_group if needed)
    - raking: Iterative proportional fitting (auto-validates control columns)
    
    Returns detailed results with auto-actions and warnings
    """
    try:
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
                # Load file using FileManager
                file_path = FileManager.get_best_available_file(f"{fid}.csv")
                
                if not os.path.exists(file_path):
                    errors[fid] = "File not found"
                    continue
                
                # Load DataFrame
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                except:
                    try:
                        df = pd.read_excel(file_path)
                    except Exception as e:
                        errors[fid] = f"Cannot read file: {str(e)}"
                        continue
                
                # Initialize weighting engine (auto-creates base weights)
                engine = WeightingEngine(df)
                
                # Execute weighting method
                result = {}
                
                if req.method == "base":
                    log_entry = engine.calculate_base_weights(req.inclusion_prob_column)
                    result = {
                        "method": "base",
                        "log_entry": log_entry,
                        "summary": log_entry["details"]["summary"]
                    }
                    
                elif req.method == "poststrat":
                    _, log_entry = engine.apply_poststrat_weights(
                        req.strata_column,
                        req.population_totals
                    )
                    result = {
                        "method": "poststrat",
                        "log_entry": log_entry,
                        "summary": log_entry["details"]
                    }
                    
                elif req.method == "raking":
                    if not req.control_totals:
                        errors[fid] = "control_totals required for raking method"
                        continue
                    
                    log_entry = engine.raking(
                        req.control_totals,
                        max_iterations=req.max_iterations,
                        tolerance=req.tolerance
                    )
                    result = {
                        "method": "raking",
                        "log_entry": log_entry,
                        "summary": log_entry["details"]
                    }
                    
                else:
                    errors[fid] = f"Invalid method '{req.method}'. Use 'base', 'poststrat', or 'raking'"
                    continue
                
                # Save weighted file
                weighted_path = FileManager.get_weighted_path(f"{fid}.csv")
                engine.export_weighted(weighted_path)
                
                # Save metadata
                metadata_path = FileManager.WEIGHTED_DIR / f"{fid}_weighting_metadata.json"
                metadata = {
                    "method": req.method,
                    "weight_column": engine.weight_column,
                    "auto_actions": engine.get_auto_actions(),
                    "warnings": engine.get_warnings(),
                    "operations_log": engine.get_operations_log()
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Build comprehensive response
                results_per_file[fid] = engine.make_json_safe({
                    "method": req.method,
                    "result": result,
                    "weighted_file_path": str(weighted_path),
                    "auto_actions": engine.get_auto_actions(),
                    "warnings": engine.get_warnings(),
                    "operations_log": engine.get_operations_log()
                })
                
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weight calculation failed: {str(e)}")


@router.post("/validate")
async def validate_weights(req: ValidateWeightsRequest):
    """
    Validate calculated weights for quality
    
    Checks:weight quality with auto-detection
    
    Checks:
    - Weight column exists (auto-detects if not specified)
    - No zeros, negatives, NaN, or Inf values
    - Distribution characteristics
    """
    try:
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
                # Load weighted file
                weighted_path = FileManager.get_weighted_path(f"{fid}.csv")
                
                if not os.path.exists(weighted_path):
                    errors[fid] = "Weighted file not found. Run /calculate first."
                    continue
                
                try:
                    df = pd.read_csv(weighted_path, low_memory=False)
                except:
                    df = pd.read_excel(weighted_path)
                
                # Initialize engine for auto-detection
                engine = WeightingEngine(df)
                
                # Auto-detect weight column
                weight_column = req.weight_column
                if weight_column is None:
                    weight_column = engine._get_active_weight_column()
                
                if weight_column not in df.columns:
                    errors[fid] = f"Weight column '{weight_column}' not found"
                    continue
                
                weights = df[weight_column].copy()
                
                # Validation checks
                problems = []
                
                # Check for NaN
                if weights.isna().any():
                    nan_count = int(weights.isna().sum())
                    problems.append(f"Contains {nan_count} NaN values")
                
                # Check for zeros
                if (weights == 0).any():
                    zero_count = int((weights == 0).sum())
                    problems.append(f"Contains {zero_count} zero values")
                
                # Check for negatives
                if (weights < 0).any():
                    neg_count = int((weights < 0).sum())
                    problems.append(f"Contains {neg_count} negative values")
                
                # Check for infinite
                if np.isinf(weights).any():
                    inf_count = int(np.isinf(weights).sum())
                    problems.append(f"Contains {inf_count} infinite values")
                
                # Get clean weights for statistics
                clean_weights = weights.replace([np.inf, -np.inf], np.nan).dropna()
                
                validation_result = {
                    "status": "pass" if len(problems) == 0 else "fail",
                    "weight_column": weight_column,
                    "problems": problems,
                    "n_observations": int(len(weights)),
                    "n_valid": int(len(clean_weights)),
                    "statistics": {
                        "count": int(len(clean_weights)),
                        "min": float(clean_weights.min()) if len(clean_weights) > 0 else None,
                        "max": float(clean_weights.max()) if len(clean_weights) > 0 else None,
                        "mean": float(clean_weights.mean()) if len(clean_weights) > 0 else None,
                        "median": float(clean_weights.median()) if len(clean_weights) > 0 else None,
                        "std": float(clean_weights.std()) if len(clean_weights) > 0 else None,
                        "cv": float(clean_weights.std() / clean_weights.mean()) if (len(clean_weights) > 0 and clean_weights.mean() > 0) else None
                    }
                }
                
                # Create engine instance to use make_json_safe
                temp_engine = WeightingEngine()
                results_per_file[fid] = temp_engine.make_json_safe(validation_result)
                
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
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "error",
            "file_id": req.file_id,
            "message": f"Validation failed: {str(e)}"
        }


@router.post("/trim")
async def trim_weights(req: TrimWeightsRequest):
    """
    Trim extreme weights with auto-detection and graceful NaN/Inf handling
    
    Caps weights at minimum and maximum thresholds
    """
    try:
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
                # Load weighted file
                weighted_path = FileManager.get_weighted_path(f"{fid}.csv")
                
                if not os.path.exists(weighted_path):
                    errors[fid] = "Weighted file not found. Run /calculate first."
                    continue
                
                try:
                    df = pd.read_csv(weighted_path, low_memory=False)
                except:
                    df = pd.read_excel(weighted_path)
                
                # Initialize engine
                engine = WeightingEngine(df)
                
                # Trim weights (auto-detects column, handles NaN/Inf)
                summary = engine.trim_weights(
                    min_w=req.min_w,
                    max_w=req.max_w,
                    weight_column=req.weight_column
                )
                
                # Save trimmed file
                base_name = Path(weighted_path).stem
                trimmed_path = FileManager.WEIGHTED_DIR / f"{fid}_weighted_trimmed.csv"
                engine.export_weighted(str(trimmed_path))
                
                results_per_file[fid] = engine.make_json_safe({
                    "trimmed_file_path": str(trimmed_path),
                    "summary": summary,
                    "auto_actions": engine.get_auto_actions(),
                    "warnings": engine.get_warnings(),
                    "operations_log": engine.get_operations_log()
                })
                
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trimming failed: {str(e)}")


@router.get("/diagnostics/{file_id}")
async def get_weight_diagnostics(file_id: str, weight_column: Optional[str] = None):
    """
    Get comprehensive diagnostic information about calculated weights
    
    Returns:
    - Mean, median, std, min, max
    - Coefficient of variation
    - Effective sample size
    - Design effect
    - Distribution percentiles
    """
    try:
        # Load weighted file
        weighted_path = FileManager.get_weighted_path(f"{file_id}.csv")
        
        # Also try trimmed file
        if not os.path.exists(weighted_path):
            weighted_path = FileManager.WEIGHTED_DIR / f"{file_id}_weighted_trimmed.csv"
        
        if not os.path.exists(str(weighted_path)):
            raise HTTPException(status_code=404, detail="Weighted file not found")
        
        try:
            df = pd.read_csv(weighted_path, low_memory=False)
        except:
            df = pd.read_excel(weighted_path)
        
        # Initialize engine
        engine = WeightingEngine(df)
        
        # Get diagnostics
        diagnostics = engine.diagnostics(weight_column=weight_column)
        
        return {
            "status": "success",
            "file_id": file_id,
            "diagnostics": diagnostics
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostics failed: {str(e)}")


@router.get("/operations-log/{file_id}")
async def get_operations_log(file_id: str):
    """
    Get the operations log for a weighted file
    
    Returns all operations performed with timestamps
    """
    try:
        # Try to load metadata first
        metadata_path = FileManager.WEIGHTED_DIR / f"{file_id}_weighting_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return {
                "status": "success",
                "file_id": file_id,
                "operations_log": metadata.get("operations_log", []),
                "auto_actions": metadata.get("auto_actions", []),
                "warnings": metadata.get("warnings", [])
            }
        
        # Fallback to file check
        weighted_path = FileManager.get_weighted_path(f"{file_id}.csv")
        
        if not os.path.exists(weighted_path):
            weighted_path = FileManager.WEIGHTED_DIR / f"{file_id}_weighted_trimmed.csv"
        
        if not os.path.exists(str(weighted_path)):
            raise HTTPException(status_code=404, detail="Weighted file not found")
        
        try:
            df = pd.read_csv(weighted_path, low_memory=False)
        except:
            df = pd.read_excel(weighted_path)
        
        # Initialize engine (operations log won't be available from saved file)
        # This is a limitation - we'd need to save logs separately in production
        engine = WeightingEngine(df)
        
        return {
            "status": "success",
            "file_id": file_id,
            "message": "Operations log only available during active session",
            "available_weight_columns": [col for col in df.columns if 'weight' in col.lower()]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve log: {str(e)}")
