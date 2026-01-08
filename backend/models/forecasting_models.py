"""
StatFlow AI - Forecasting Models
Pydantic models for forecasting API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class ForecastRequest(BaseModel):
    """Request model for running forecasts."""
    file_id: str = Field(..., description="Unique file identifier")
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


class DecomposeRequest(BaseModel):
    """Request model for seasonal decomposition."""
    file_id: str = Field(..., description="Unique file identifier")
    value_column: str = Field(..., description="Column to decompose")
    time_column: Optional[str] = Field(None, description="Time column (auto-detected if not provided)")
    period: Optional[int] = Field(None, ge=2, description="Seasonal period (auto-detected if not provided)")
    model: Literal["additive", "multiplicative"] = Field("additive", description="Decomposition model")
    use_weighted: bool = Field(False, description="Use weighted data if available")


class TimeSeriesAnalysisRequest(BaseModel):
    """Request model for time series analysis."""
    file_id: str = Field(..., description="Unique file identifier")
    value_column: str = Field(..., description="Column to analyze")
    time_column: Optional[str] = Field(None, description="Time column (auto-detected if not provided)")
    use_weighted: bool = Field(False, description="Use weighted data if available")
