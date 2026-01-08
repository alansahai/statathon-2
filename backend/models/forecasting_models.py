"""
StatFlow AI - Forecasting Models
Pydantic models for forecasting API endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal


class ForecastRequest(BaseModel):
    """Request model for running forecasts."""
    file_id: Optional[str] = Field(None, description="Single file identifier (legacy support)")
    file_ids: Optional[List[str]] = Field(None, description="List of file identifiers (1-5 files)")
    value_column: str = Field(..., description="Column to forecast")
    time_column: Optional[str] = Field(None, description="Time column (auto-detected if not provided)")
    method: Literal["moving_average", "exponential_smoothing", "holt_winters", "arima"] = Field(
        "exponential_smoothing", 
        description="Forecasting method to use"
    )
    forecast_periods: int = Field(10, ge=1, le=100, description="Number of periods to forecast")
    
    # Moving Average params
    window_size: Optional[int] = Field(None, ge=2, description="Window size for moving average")
    
    # Exponential Smoothing params
    alpha: Optional[float] = Field(None, ge=0.01, le=0.99, description="Smoothing factor")
    
    # Holt-Winters params
    beta: Optional[float] = Field(None, ge=0.01, le=0.99, description="Trend smoothing factor")
    gamma: Optional[float] = Field(None, ge=0.01, le=0.99, description="Seasonal smoothing factor")
    seasonal_period: Optional[int] = Field(None, ge=2, description="Seasonal period")
    
    # ARIMA params
    p: Optional[int] = Field(None, ge=0, le=5, description="AR order (auto-selected if None)")
    d: Optional[int] = Field(None, ge=0, le=2, description="Differencing order (auto-selected if None)")
    q: Optional[int] = Field(None, ge=0, le=5, description="MA order (auto-selected if None)")
    
    # Options
    use_weighted: bool = Field(False, description="Use weighted data if available")
    include_confidence_interval: bool = Field(True, description="Include confidence intervals")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99, description="Confidence level for intervals")
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id/file_ids to always use file_ids list."""
        file_id = values.get('file_id')
        if file_id and v:
            raise ValueError("Provide either file_id or file_ids, not both")
        if file_id and not v:
            return [file_id]
        if v and not file_id:
            if len(v) > 5:
                raise ValueError("Maximum 5 files allowed")
            if len(v) != len(set(v)):
                raise ValueError("Duplicate file_ids not allowed")
            return v
        raise ValueError("Either file_id or file_ids must be provided")


class DecomposeRequest(BaseModel):
    """Request model for seasonal decomposition."""
    file_id: Optional[str] = Field(None, description="Single file identifier (legacy support)")
    file_ids: Optional[List[str]] = Field(None, description="List of file identifiers (1-5 files)")
    value_column: str = Field(..., description="Column to decompose")
    time_column: Optional[str] = Field(None, description="Time column (auto-detected if not provided)")
    period: Optional[int] = Field(None, ge=2, description="Seasonal period (auto-detected if not provided)")
    model: Literal["additive", "multiplicative"] = Field("additive", description="Decomposition model")
    use_weighted: bool = Field(False, description="Use weighted data if available")
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id/file_ids to always use file_ids list."""
        file_id = values.get('file_id')
        if file_id and v:
            raise ValueError("Provide either file_id or file_ids, not both")
        if file_id and not v:
            return [file_id]
        if v and not file_id:
            if len(v) > 5:
                raise ValueError("Maximum 5 files allowed")
            if len(v) != len(set(v)):
                raise ValueError("Duplicate file_ids not allowed")
            return v
        raise ValueError("Either file_id or file_ids must be provided")


class TimeSeriesAnalysisRequest(BaseModel):
    """Request model for time series analysis."""
    file_id: Optional[str] = Field(None, description="Single file identifier (legacy support)")
    file_ids: Optional[List[str]] = Field(None, description="List of file identifiers (1-5 files)")
    value_column: str = Field(..., description="Column to analyze")
    time_column: Optional[str] = Field(None, description="Time column (auto-detected if not provided)")
    use_weighted: bool = Field(False, description="Use weighted data if available")
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id/file_ids to always use file_ids list."""
        file_id = values.get('file_id')
        if file_id and v:
            raise ValueError("Provide either file_id or file_ids, not both")
        if file_id and not v:
            return [file_id]
        if v and not file_id:
            if len(v) > 5:
                raise ValueError("Maximum 5 files allowed")
            if len(v) != len(set(v)):
                raise ValueError("Duplicate file_ids not allowed")
            return v
        raise ValueError("Either file_id or file_ids must be provided")
