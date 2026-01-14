"""
Dashboard API router for StatFlow AI
Provides chart-ready JSON data for frontend dashboard visualization
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path

from services.dashboard_engine import DashboardEngine

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


class DashboardResponse(BaseModel):
    """Standard response model for dashboard endpoints"""
    status: str
    chart_type: str
    data: Dict[str, Any]
    message: Optional[str] = None


def load_dataframe(file_id: str) -> pd.DataFrame:
    """
    Load dataframe from temp_uploads directory.
    Tries weighted -> cleaned -> uploads in that order.
    
    Args:
        file_id: Unique identifier for the dataset
        
    Returns:
        pandas DataFrame
        
    Raises:
        HTTPException: If file not found
    """
    data_dir = Path("temp_uploads")
    
    # Try weighted data first (most processed)
    weighted_path = data_dir / "weighted" / "default_user" / f"{file_id}_weighted.csv"
    if weighted_path.exists():
        return pd.read_csv(weighted_path)
    
    # Try cleaned data
    cleaned_path = data_dir / "cleaned" / "default_user" / f"{file_id}_cleaned.csv"
    if cleaned_path.exists():
        return pd.read_csv(cleaned_path)
    
    # Try original upload
    upload_path = data_dir / "uploads" / "default_user" / f"{file_id}.csv"
    if upload_path.exists():
        return pd.read_csv(upload_path)
    
    # File not found
    raise HTTPException(
        status_code=404,
        detail=f"Dataset not found for file_id: {file_id}"
    )


@router.get("/bar/{file_id}/{column}", response_model=DashboardResponse)
async def get_bar_chart_data(file_id: str, column: str):
    """
    Get chart-ready bar chart data for a categorical column.
    
    Returns value counts with labels (categories) and values (counts).
    
    Args:
        file_id: Unique identifier for the dataset
        column: Name of the categorical column
        
    Returns:
        DashboardResponse with bar chart data structure
        
    Raises:
        HTTPException: If dataset not found, column missing, or data invalid
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize dashboard engine
        dashboard = DashboardEngine(df)
        
        # Generate bar chart data
        chart_data = dashboard.to_chart_ready_bar(column)
        
        return DashboardResponse(
            status="success",
            chart_type="bar",
            data=chart_data,
            message=f"Bar chart data for column '{column}'"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating bar chart data: {str(e)}"
        )


@router.get("/hist/{file_id}/{column}", response_model=DashboardResponse)
async def get_histogram_data(file_id: str, column: str, bins: int = 10):
    """
    Get chart-ready histogram data for a numeric column.
    
    Returns bin ranges as labels and counts as values.
    
    Args:
        file_id: Unique identifier for the dataset
        column: Name of the numeric column
        bins: Number of histogram bins (default: 10)
        
    Returns:
        DashboardResponse with histogram data structure
        
    Raises:
        HTTPException: If dataset not found, column missing, or column not numeric
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize dashboard engine
        dashboard = DashboardEngine(df)
        
        # Generate histogram data
        chart_data = dashboard.to_chart_ready_hist(column, bins=bins)
        
        return DashboardResponse(
            status="success",
            chart_type="histogram",
            data=chart_data,
            message=f"Histogram data for column '{column}' with {bins} bins"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating histogram data: {str(e)}"
        )


@router.get("/pie/{file_id}/{column}", response_model=DashboardResponse)
async def get_pie_chart_data(file_id: str, column: str, max_categories: int = 6):
    """
    Get chart-ready pie chart data for a categorical column.
    
    Returns proportions with labels (categories) and values (proportions 0-1).
    
    Args:
        file_id: Unique identifier for the dataset
        column: Name of the categorical column
        max_categories: Maximum number of slices (default: 6)
        
    Returns:
        DashboardResponse with pie chart data structure
        
    Raises:
        HTTPException: If dataset not found, column missing, or data invalid
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize dashboard engine
        dashboard = DashboardEngine(df)
        
        # Generate pie chart data
        chart_data = dashboard.to_chart_ready_pie(column, max_categories=max_categories)
        
        return DashboardResponse(
            status="success",
            chart_type="pie",
            data=chart_data,
            message=f"Pie chart data for column '{column}'"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating pie chart data: {str(e)}"
        )


@router.get("/timeseries/{file_id}/{time_col}/{value_col}", response_model=DashboardResponse)
async def get_timeseries_data(file_id: str, time_col: str, value_col: str):
    """
    Get chart-ready time series data.
    
    Returns ISO timestamp strings as labels and numeric values.
    
    Args:
        file_id: Unique identifier for the dataset
        time_col: Name of the time/date column
        value_col: Name of the numeric value column
        
    Returns:
        DashboardResponse with time series data structure
        
    Raises:
        HTTPException: If dataset not found, columns missing, or data invalid
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize dashboard engine
        dashboard = DashboardEngine(df)
        
        # Generate time series data
        chart_data = dashboard.to_chart_ready_timeseries(time_col, value_col)
        
        return DashboardResponse(
            status="success",
            chart_type="timeseries",
            data=chart_data,
            message=f"Time series data for '{value_col}' over '{time_col}'"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating time series data: {str(e)}"
        )


@router.get("/scatter/{file_id}/{x_col}/{y_col}", response_model=DashboardResponse)
async def get_scatter_data(file_id: str, x_col: str, y_col: str):
    """
    Get chart-ready scatter plot data.
    
    Returns x_values and y_values arrays with correlation metadata.
    
    Args:
        file_id: Unique identifier for the dataset
        x_col: Name of the x-axis column
        y_col: Name of the y-axis column
        
    Returns:
        DashboardResponse with scatter plot data structure
        
    Raises:
        HTTPException: If dataset not found, columns missing, or columns not numeric
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize dashboard engine
        dashboard = DashboardEngine(df)
        
        # Generate scatter plot data
        chart_data = dashboard.to_chart_ready_scatter(x_col, y_col)
        
        return DashboardResponse(
            status="success",
            chart_type="scatter",
            data=chart_data,
            message=f"Scatter plot data for '{x_col}' vs '{y_col}'"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating scatter plot data: {str(e)}"
        )


@router.get("/boxplot/{file_id}/{column}", response_model=DashboardResponse)
async def get_boxplot_data(file_id: str, column: str):
    """
    Get chart-ready boxplot data.
    
    Returns quartiles (min, Q1, median, Q3, max) and outliers.
    
    Args:
        file_id: Unique identifier for the dataset
        column: Name of the numeric column
        
    Returns:
        DashboardResponse with boxplot data structure
        
    Raises:
        HTTPException: If dataset not found, column missing, or column not numeric
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize dashboard engine
        dashboard = DashboardEngine(df)
        
        # Generate boxplot data
        chart_data = dashboard.to_chart_ready_boxplot(column)
        
        return DashboardResponse(
            status="success",
            chart_type="boxplot",
            data=chart_data,
            message=f"Boxplot data for column '{column}'"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating boxplot data: {str(e)}"
        )


@router.get("/columns/{file_id}")
async def get_columns_info(file_id: str):
    """
    Get information about available columns in the dataset.
    
    Returns column names, dtypes, and suggested chart types.
    
    Args:
        file_id: Unique identifier for the dataset
        
    Returns:
        Dictionary with column information
        
    Raises:
        HTTPException: If dataset not found
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Analyze columns
        columns_info = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
            unique_count = df[col].nunique()
            null_count = df[col].isna().sum()
            
            # Suggest chart types
            suggested_charts = []
            if is_numeric:
                suggested_charts.extend(["histogram", "boxplot"])
                if unique_count <= 10:
                    suggested_charts.append("bar")
            else:
                suggested_charts.append("bar")
                if unique_count <= 6:
                    suggested_charts.append("pie")
            
            if is_datetime:
                suggested_charts.append("timeseries")
            
            columns_info.append({
                "column": col,
                "dtype": dtype,
                "is_numeric": is_numeric,
                "is_datetime": is_datetime,
                "unique_count": int(unique_count),
                "null_count": int(null_count),
                "suggested_charts": suggested_charts
            })
        
        return {
            "file_id": file_id,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": columns_info
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving column information: {str(e)}"
        )


@router.get("/kpi/{file_id}")
async def get_kpi_metrics(file_id: str):
    """
    Get high-level KPI metrics for the dataset.
    
    Returns overview statistics including row/column counts, missing data percentage,
    and column type distribution.
    
    Args:
        file_id: Unique identifier for the dataset
        
    Returns:
        Dictionary with KPI metrics
        
    Raises:
        HTTPException: If dataset not found or error occurs
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize dashboard engine
        dashboard = DashboardEngine(df)
        
        # Get KPI summary
        kpi_data = dashboard.get_kpi_summary()
        
        return {
            "status": "success",
            "file_id": file_id,
            "kpi_metrics": kpi_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving KPI metrics: {str(e)}"
        )


@router.get("/trend/{file_id}/{column}")
async def get_trend_analysis(file_id: str, column: str):
    """
    Get trend analysis for a numeric column.
    
    Analyzes trend direction (increasing/decreasing/stable) based on
    percent change from first to last value.
    
    Args:
        file_id: Unique identifier for the dataset
        column: Name of the numeric column
        
    Returns:
        Dictionary with trend direction and percent change
        
    Raises:
        HTTPException: If dataset not found, column missing, or column not numeric
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize dashboard engine
        dashboard = DashboardEngine(df)
        
        # Get trend summary
        trend_data = dashboard.get_trend_summary(column)
        
        return {
            "status": "success",
            "trend": trend_data
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing trend: {str(e)}"
        )


@router.get("/risks/{file_id}")
async def get_risk_analysis(file_id: str):
    """
    Get data quality risk indicators for the dataset.
    
    Identifies columns with potential quality issues including:
    - High missing rates (>30%)
    - High outlier rates (>10%)
    - Skewed distributions (|skew| > 1.0)
    - High cardinality (>50 unique values)
    
    Args:
        file_id: Unique identifier for the dataset
        
    Returns:
        Dictionary with risk indicators by category
        
    Raises:
        HTTPException: If dataset not found or error occurs
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize dashboard engine
        dashboard = DashboardEngine(df)
        
        # Get risk indicators
        risk_data = dashboard.get_risk_indicators()
        
        return {
            "status": "success",
            "file_id": file_id,
            "risk_indicators": risk_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing risks: {str(e)}"
        )


@router.get("/summary/{file_id}")
async def get_dashboard_summary(file_id: str):
    """
    Get comprehensive dashboard summary with KPIs, trends, risks, and recommendations.
    
    Provides a unified view combining:
    - High-level KPI metrics
    - Risk indicators and data quality issues
    - Top 3 trends by absolute percent change
    - Recommended charts by column type
    - Auto-generated narrative highlights
    
    Args:
        file_id: Unique identifier for the dataset
        
    Returns:
        Dictionary with comprehensive dashboard summary
        
    Raises:
        HTTPException: If dataset not found or error occurs
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize dashboard engine
        dashboard = DashboardEngine(df)
        
        # Get comprehensive summary
        summary_data = dashboard.get_dashboard_summary()
        
        return {
            "status": "success",
            "file_id": file_id,
            "summary": summary_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating dashboard summary: {str(e)}"
        )


@router.get("/augmented/{file_id}")
async def get_augmented_dashboard_summary(file_id: str):
    """
    Get enhanced dashboard summary with integrated AI recommendations.
    
    Combines standard dashboard analytics (KPIs, trends, risks) with
    intelligent recommendations for analysis methods, transformations,
    statistical tests, and machine learning models.
    
    Args:
        file_id: Unique identifier for the dataset
        
    Returns:
        Dictionary with enhanced dashboard summary including recommendations
        
    Raises:
        HTTPException: If dataset not found or error occurs
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize dashboard engine
        dashboard = DashboardEngine(df)
        
        # Get augmented summary with recommendations
        summary_data = dashboard.get_augmented_dashboard_summary()
        
        return {
            "status": "success",
            "file_id": file_id,
            "summary": summary_data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating augmented dashboard: {str(e)}"
        )
