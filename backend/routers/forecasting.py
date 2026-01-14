"""
StatFlow AI - Forecasting Router
API endpoints for time series forecasting.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import pandas as pd
import os

from services.forecasting_engine import ForecastingEngine
from models.forecasting_models import ForecastRequest, DecomposeRequest, TimeSeriesAnalysisRequest
from utils.file_manager import FileManager

router = APIRouter(prefix="/forecasting", tags=["06 Forecasting"])

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
    
    # Try cleaned file
    cleaned_path = file_manager.get_cleaned_path(file_id)
    if os.path.exists(cleaned_path):
        return pd.read_csv(cleaned_path)
    
    # Fall back to original upload
    upload_path = file_manager.get_upload_path(file_id)
    if os.path.exists(upload_path):
        return pd.read_csv(upload_path)
    
    raise HTTPException(status_code=404, detail=f"File not found: {file_id}")


@router.post("/run", summary="Run Forecast", description="Run a forecast using specified method")
async def run_forecast(request: ForecastRequest) -> Dict[str, Any]:
    """
    Run a forecast on a time series.
    
    Supported methods:
    - moving_average: Simple moving average
    - exponential_smoothing: Single exponential smoothing
    - holt_winters: Triple exponential smoothing with trend and seasonality
    - arima: ARIMA with auto-selected or manual p,d,q parameters
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
                engine = ForecastingEngine(df)
                
                result = engine.run_forecast(
                    value_column=request.value_column,
                    method=request.method,
                    time_column=request.time_column,
                    forecast_periods=request.forecast_periods,
                    window_size=request.window_size,
                    alpha=request.alpha,
                    beta=request.beta,
                    gamma=request.gamma,
                    seasonal_period=request.seasonal_period,
                    p=request.p,
                    d=request.d,
                    q=request.q,
                    include_confidence_interval=request.include_confidence_interval,
                    confidence_level=request.confidence_level
                )
                
                results_per_file[fid] = {
                    "method": request.method,
                    "result": result
                }
                
            except ValueError as e:
                errors[fid] = str(e)
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
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


@router.post("/decompose", summary="Seasonal Decomposition", description="Decompose time series into components")
async def decompose_series(request: DecomposeRequest) -> Dict[str, Any]:
    """
    Decompose a time series into trend, seasonal, and residual components.
    
    Models:
    - additive: y = trend + seasonal + residual
    - multiplicative: y = trend * seasonal * residual
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
                engine = ForecastingEngine(df)
                
                result = engine.seasonal_decompose(
                    value_column=request.value_column,
                    time_column=request.time_column,
                    period=request.period,
                    model=request.model
                )
                
                results_per_file[fid] = {
                    "model": request.model,
                    "result": result
                }
                
            except ValueError as e:
                errors[fid] = str(e)
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
        raise HTTPException(status_code=500, detail=f"Decomposition error: {str(e)}")


@router.post("/analyze", summary="Time Series Analysis", description="Analyze time series properties")
async def analyze_time_series(request: TimeSeriesAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze a time series for stationarity, trends, and seasonality.
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
                engine = ForecastingEngine(df)
                
                # Detect time column if not provided
                time_col = request.time_column
                if time_col is None:
                    detected = engine.detect_time_column()
                    if detected["detected"]:
                        time_col = detected["column"]
                
                # Get series
                values = df[request.value_column].dropna().values
                n = len(values)
                
                # Basic statistics
                from scipy import stats
                import numpy as np
                
                # Trend test (linear regression)
                x = np.arange(n)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                trend_direction = "stable"
                if p_value < 0.05:
                    trend_direction = "increasing" if slope > 0 else "decreasing"
                
                # Seasonality check via autocorrelation
                detected_seasonality = {"detected": False}
                for period in [4, 7, 12, 24, 52]:
                    if n > 2 * period:
                        vals_centered = values - np.mean(values)
                        autocorr = np.corrcoef(vals_centered[:-period], vals_centered[period:])[0, 1]
                        if not np.isnan(autocorr) and abs(autocorr) > 0.3:
                            detected_seasonality = {
                                "detected": True,
                                "period": period,
                                "autocorrelation": float(autocorr)
                            }
                            break
                
                result = {
                    "n_observations": n,
                    "time_column": time_col,
                    "value_column": request.value_column,
                    "basic_stats": {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values))
                    },
                    "trend": {
                        "direction": trend_direction,
                        "slope": float(slope),
                        "r_squared": float(r_value ** 2),
                        "p_value": float(p_value)
                    },
                    "seasonality": detected_seasonality
                }
                
                results_per_file[fid] = {"result": result}
                
            except ValueError as e:
                errors[fid] = str(e)
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
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/operations-log/{file_id}", summary="Get Operations Log", description="Get forecasting operations log")
async def get_operations_log(file_id: str, use_weighted: bool = False) -> Dict[str, Any]:
    """
    Get the operations log for a forecasting session.
    """
    try:
        df = _load_dataframe(file_id, use_weighted)
        engine = ForecastingEngine(df)
        
        return {
            "success": True,
            "file_id": file_id,
            "operations_log": engine.get_operations_log()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving log: {str(e)}")
