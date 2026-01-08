"""
Weighting Router - MoSPI-ready automated survey weighting
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
import os
from pathlib import Path

from services.weighting_engine import WeightingEngine
from utils.file_manager import FileManager

router = APIRouter(
    prefix="/api/weighting",
    tags=["Weighting"]
)

file_manager = FileManager(base_storage_path="temp_uploads")


# -----------------------------
# REQUEST MODELS (SIMPLIFIED)
# -----------------------------

class CalculateWeightsRequest(BaseModel):
    file_id: str
    method: str  # "base", "poststrat", "raking"
    # All other fields are optional - auto-detection handles them
    inclusion_prob_column: Optional[str] = None
    strata_column: Optional[str] = None
    population_totals: Optional[Dict[str, float]] = None
    control_totals: Optional[Dict[str, Dict[str, float]]] = None
    max_iterations: Optional[int] = 50
    tolerance: Optional[float] = 0.001


class ValidateWeightsRequest(BaseModel):
    file_id: str
    weight_column: Optional[str] = None


class TrimWeightsRequest(BaseModel):
    file_id: str
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
        # Load file with fallback path logic
        file_path = file_manager.get_file_path(req.file_id)
        
        if not file_path:
            user_dir = file_manager.uploads_dir / "default_user"
            file_path = user_dir / f"{req.file_id}.csv"
            
            if not file_path.exists():
                file_path = user_dir / f"{req.file_id}.xlsx"
                
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="File not found")
            
            file_path = str(file_path)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Load DataFrame
        df = file_manager.load_dataframe(file_path)
        
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
                # Return error only if truly no input provided
                raise HTTPException(
                    status_code=400,
                    detail="control_totals required for raking method"
                )
            
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
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method '{req.method}'. Use 'base', 'poststrat', or 'raking'"
            )
        
        # Save weighted file
        weighted_dir = file_manager.base_storage_path / "weighted" / "default_user"
        weighted_dir.mkdir(parents=True, exist_ok=True)
        
        weighted_path = weighted_dir / f"{req.file_id}_weighted.csv"
        engine.export_weighted(str(weighted_path))
        
        # Build comprehensive response
        response = engine.make_json_safe({
            "status": "success",
            "file_id": req.file_id,
            "weighted_file_path": str(weighted_path),
            "method": req.method,
            "result": result,
            "auto_actions": engine.get_auto_actions(),
            "warnings": engine.get_warnings(),
            "operations_log": engine.get_operations_log()
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Never return 400 for data issues - return 200 with warnings
        return {
            "status": "partial_success",
            "error": str(e),
            "message": "Weighting completed with errors",
            "file_id": req.file_id
        }


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
        # Load weighted file
        weighted_dir = file_manager.base_storage_path / "weighted" / "default_user"
        weighted_path = weighted_dir / f"{req.file_id}_weighted.csv"
        
        if not weighted_path.exists():
            raise HTTPException(status_code=404, detail="Weighted file not found. Run /calculate first.")
        
        df = file_manager.load_dataframe(str(weighted_path))
        
        # Initialize engine for auto-detection
        engine = WeightingEngine(df)
        
        # Auto-detect weight column
        if req.weight_column is None:
            req.weight_column = engine._get_active_weight_column()
        
        if req.weight_column not in df.columns:
            return {
                "status": "error",
                "file_id": req.file_id,
                "message": f"Weight column '{req.weight_column}' not found",
                "available_columns": [col for col in df.columns if 'weight' in col.lower()]
            }
        
        weights = df[req.weight_column].copy()
        
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
            "weight_column": req.weight_column,
            "problems": problems,
            "n_observations": int(len(weights)),
            "n_valid": int(len(clean_weights)),
            "statistics": {
                "min": float(clean_weights.min()) if len(clean_weights) > 0 else None,
                "max": float(clean_weights.max()) if len(clean_weights) > 0 else None,
                "mean": float(clean_weights.mean()) if len(clean_weights) > 0 else None,
                "median": float(clean_weights.median()) if len(clean_weights) > 0 else None,
                "std": float(clean_weights.std()) if len(clean_weights) > 0 else None
            }
        }
        
        return WeightingEngine.make_json_safe({
            "status": "success",
            "file_id": req.file_id,
            "validation": validation_result
        })
        
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
        # Load weighted file
        weighted_dir = file_manager.base_storage_path / "weighted" / "default_user"
        weighted_path = weighted_dir / f"{req.file_id}_weighted.csv"
        
        if not weighted_path.exists():
            raise HTTPException(status_code=404, detail="Weighted file not found. Run /calculate first.")
        
        df = file_manager.load_dataframe(str(weighted_path))
        
        # Initialize engine
        engine = WeightingEngine(df)
        
        # Trim weights (auto-detects column, handles NaN/Inf)
        summary = engine.trim_weights(
            min_w=req.min_w,
            max_w=req.max_w,
            weight_column=req.weight_column
        )
        
        # Save trimmed file
        trimmed_path = weighted_dir / f"{req.file_id}_weighted_trimmed.csv"
        engine.export_weighted(str(trimmed_path))
        
        return engine.make_json_safe({
            "status": "success",
            "file_id": req.file_id,
            "trimmed_file_path": str(trimmed_path),
            "summary": summary,
            "auto_actions": engine.get_auto_actions(),
            "warnings": engine.get_warnings(),
            "operations_log": engine.get_operations_log()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "error",
            "file_id": req.file_id,
            "message": f"Trimming failed: {str(e)}"
        }


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
        weighted_dir = file_manager.base_storage_path / "weighted" / "default_user"
        weighted_path = weighted_dir / f"{file_id}_weighted.csv"
        
        # Also try trimmed file
        if not weighted_path.exists():
            weighted_path = weighted_dir / f"{file_id}_weighted_trimmed.csv"
        
        if not weighted_path.exists():
            raise HTTPException(status_code=404, detail="Weighted file not found")
        
        df = file_manager.load_dataframe(str(weighted_path))
        
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
        # Load weighted file
        weighted_dir = file_manager.base_storage_path / "weighted" / "default_user"
        weighted_path = weighted_dir / f"{file_id}_weighted.csv"
        
        if not weighted_path.exists():
            weighted_path = weighted_dir / f"{file_id}_weighted_trimmed.csv"
        
        if not weighted_path.exists():
            raise HTTPException(status_code=404, detail="Weighted file not found")
        
        df = file_manager.load_dataframe(str(weighted_path))
        
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
