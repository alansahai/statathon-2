"""
StatFlow AI - Insight Router
API endpoints for automated insights.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import pandas as pd
import os

from services.insight_engine import InsightEngine
from models.insight_models import InsightOverviewRequest, FullInsightRequest
from utils.file_manager import FileManager

router = APIRouter(prefix="/insight", tags=["08 Insight Engine"])

# File manager instance
file_manager = FileManager()


def _load_dataframe(file_id: str, use_weighted: bool = False) -> pd.DataFrame:
    """Load DataFrame from file_id."""
    user_id = "default_user"
    
    if use_weighted:
        # Try weighted file first
        weighted_path = file_manager.get_weighted_path(file_id)
        if os.path.exists(weighted_path):
            return pd.read_csv(weighted_path)
    
    # Use best available file (cleaned → mapped → uploaded)
    try:
        path = file_manager.get_best_available_file(file_id)
        return pd.read_csv(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/overview/{file_id}", summary="Get Overview", description="Get data overview insights")
async def get_overview(file_id: str, use_weighted: bool = False) -> Dict[str, Any]:
    """
    Get a high-level overview of the data.
    
    Returns:
    - Data shape and column types
    - Data quality metrics
    - Quick numeric summaries
    """
    try:
        df = _load_dataframe(file_id, use_weighted)
        engine = InsightEngine(df)
        
        insights = engine.generate_overview()
        
        return {
            "status": "success",
            "file_id": file_id,
            "insights": insights
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overview error: {str(e)}")


@router.post("/full", summary="Get Full Insights", description="Get comprehensive insights")
async def get_full_insights(request: FullInsightRequest) -> Dict[str, Any]:
    """
    Get comprehensive insights combining descriptive, forecast, and risk analysis.
    
    Returns:
    - Overview: Basic data stats and quality
    - Descriptive findings: Correlations, distributions, missing data
    - Forecast signals: Trends and seasonality (if time/value columns provided)
    - Risks: Subgroup data quality flags
    - Recommended actions: Prioritized suggestions
    """
    try:
        # Normalize file_ids
        file_ids = request.file_ids
        if not file_ids or len(file_ids) == 0:
            raise HTTPException(status_code=400, detail="No file_ids provided")
        if len(file_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
        
        # Process each file
        results_per_file = {}
        errors = {}
        
        for fid in file_ids:
            try:
                df = _load_dataframe(fid, request.use_weighted)
                engine = InsightEngine(df)
                
                result = engine.generate_full_insights(
                    time_column=request.time_column,
                    value_column=request.value_column,
                    group_column=request.group_column
                )
                
                results_per_file[fid] = {"insights": result}
                
            except HTTPException as e:
                errors[fid] = e.detail
            except Exception as e:
                errors[fid] = str(e)
        
        # Determine status
        status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
        
        response = {
            "success": status == "success",
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
        raise HTTPException(status_code=500, detail=f"Insight error: {str(e)}")


@router.get("/correlations/{file_id}", summary="Get Correlations", description="Get correlation analysis")
async def get_correlations(file_id: str, use_weighted: bool = False) -> Dict[str, Any]:
    """
    Get high correlations between numeric variables.
    """
    try:
        df = _load_dataframe(file_id, use_weighted)
        engine = InsightEngine(df)
        
        result = engine._analyze_correlations()
        
        return {
            "success": True,
            "file_id": file_id,
            "correlations": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correlation error: {str(e)}")


@router.get("/distributions/{file_id}", summary="Get Distributions", description="Get distribution analysis")
async def get_distributions(file_id: str, use_weighted: bool = False) -> Dict[str, Any]:
    """
    Get analysis of unusual distributions.
    """
    try:
        df = _load_dataframe(file_id, use_weighted)
        engine = InsightEngine(df)
        
        result = engine._analyze_distributions()
        
        return {
            "success": True,
            "file_id": file_id,
            "distributions": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Distribution error: {str(e)}")


@router.get("/missing/{file_id}", summary="Get Missing Data", description="Get missing data analysis")
async def get_missing_data(file_id: str, use_weighted: bool = False) -> Dict[str, Any]:
    """
    Get analysis of missing data patterns.
    """
    try:
        df = _load_dataframe(file_id, use_weighted)
        engine = InsightEngine(df)
        
        result = engine._analyze_missing_data()
        
        return {
            "success": True,
            "file_id": file_id,
            "missing_data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Missing data error: {str(e)}")


@router.get("/risks/{file_id}", summary="Get Risk Flags", description="Get subgroup risk analysis")
async def get_risk_flags(
    file_id: str, 
    group_column: str = None,
    use_weighted: bool = False
) -> Dict[str, Any]:
    """
    Get risk flags for subgroups in the data.
    """
    try:
        df = _load_dataframe(file_id, use_weighted)
        engine = InsightEngine(df)
        
        result = engine._analyze_subgroup_risks(group_column)
        
        return {
            "success": True,
            "file_id": file_id,
            "risks": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk analysis error: {str(e)}")


@router.get("/operations-log/{file_id}", summary="Get Operations Log", description="Get insight operations log")
async def get_operations_log(file_id: str, use_weighted: bool = False) -> Dict[str, Any]:
    """
    Get the operations log for an insight session.
    """
    try:
        df = _load_dataframe(file_id, use_weighted)
        engine = InsightEngine(df)
        
        return {
            "success": True,
            "file_id": file_id,
            "operations_log": engine.get_operations_log()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving log: {str(e)}")
