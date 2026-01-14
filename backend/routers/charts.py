"""
Chart generation API router for StatFlow AI
Provides endpoints for generating various types of statistical charts
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import pandas as pd
from pathlib import Path
import json

from services.chart_engine import ChartEngine

router = APIRouter(prefix="/charts", tags=["charts"])


class ChartGenerationRequest(BaseModel):
    """Request model for chart generation"""
    file_id: str = Field(..., description="Unique identifier for the dataset")
    chart_type: str = Field(..., description="Type of chart to generate")
    
    # Single column charts (histogram, boxplot, bar, pie, frequency)
    column: Optional[str] = Field(None, description="Column name for single-column charts")
    
    # Two-column charts (scatter, regression)
    x_col: Optional[str] = Field(None, description="X-axis column for scatter/regression plots")
    y_col: Optional[str] = Field(None, description="Y-axis column for scatter/regression plots")
    
    # Time series charts
    time_col: Optional[str] = Field(None, description="Time column for time series plots")
    value_col: Optional[str] = Field(None, description="Value column for time series plots")
    
    # Decomposition data
    decomposition: Optional[Dict[str, List[float]]] = Field(
        None, 
        description="Decomposition data with trend, seasonal, residual components"
    )
    
    # Residual data
    residuals: Optional[List[float]] = Field(None, description="Residual values for diagnostics")


class ChartGenerationResponse(BaseModel):
    """Response model for chart generation"""
    success: bool
    chart_type: str
    path: str
    metadata: Dict[str, Any]
    message: Optional[str] = None


@router.post("/generate", response_model=ChartGenerationResponse)
async def generate_chart(request: ChartGenerationRequest):
    """
    Generate a statistical chart for the specified dataset.
    
    Loads the dataset, initializes ChartEngine, and generates the requested chart type.
    Returns the path to the saved image and metadata about the generated chart.
    
    Args:
        request: Chart generation request with file_id, chart_type, and parameters
        
    Returns:
        ChartGenerationResponse with success status, path to image, and metadata
        
    Raises:
        HTTPException: If dataset not found, chart generation fails, or parameters invalid
    """
    try:
        # Load dataset from cleaned or weighted data
        data_dir = Path("temp_uploads")
        
        # Try weighted data first
        weighted_path = data_dir / "weighted" / "default_user" / f"{request.file_id}_weighted.csv"
        if weighted_path.exists():
            df = pd.read_csv(weighted_path)
        else:
            # Fall back to cleaned data
            cleaned_path = data_dir / "cleaned" / "default_user" / f"{request.file_id}_cleaned.csv"
            if cleaned_path.exists():
                df = pd.read_csv(cleaned_path)
            else:
                # Fall back to original upload
                upload_path = data_dir / "uploads" / "default_user" / f"{request.file_id}.csv"
                if upload_path.exists():
                    df = pd.read_csv(upload_path)
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Dataset not found for file_id: {request.file_id}"
                    )
        
        # Initialize ChartEngine
        chart_engine = ChartEngine(dataframe=df, file_id=request.file_id)
        
        # Build kwargs from request
        kwargs = {}
        
        if request.column:
            kwargs['column'] = request.column
        if request.x_col:
            kwargs['x_col'] = request.x_col
        if request.y_col:
            kwargs['y_col'] = request.y_col
        if request.time_col:
            kwargs['time_col'] = request.time_col
        if request.value_col:
            kwargs['value_col'] = request.value_col
        if request.decomposition:
            kwargs['decomposition'] = request.decomposition
        if request.residuals:
            kwargs['residuals'] = request.residuals
        
        # Generate chart using unified dispatcher
        result = chart_engine.generate_chart(request.chart_type, **kwargs)
        
        return ChartGenerationResponse(
            success=True,
            chart_type=result['type'],
            path=result['path'],
            metadata=result,
            message=f"{result['type']} generated successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating chart: {str(e)}"
        )


@router.get("/list/{file_id}")
async def list_charts(file_id: str):
    """
    List all generated charts for a specific file_id.
    
    Args:
        file_id: Unique identifier for the dataset
        
    Returns:
        Dictionary with list of chart files and their metadata
    """
    try:
        charts_dir = Path("temp_uploads/charts") / file_id
        
        if not charts_dir.exists():
            return {
                "file_id": file_id,
                "charts": [],
                "message": "No charts found for this file_id"
            }
        
        # List all PNG files
        chart_files = list(charts_dir.glob("*.png"))
        
        charts = []
        for chart_file in chart_files:
            charts.append({
                "filename": chart_file.name,
                "path": str(chart_file.absolute()),
                "size": chart_file.stat().st_size,
                "created": chart_file.stat().st_mtime
            })
        
        return {
            "file_id": file_id,
            "charts": charts,
            "count": len(charts)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing charts: {str(e)}"
        )


@router.delete("/delete/{file_id}")
async def delete_charts(file_id: str):
    """
    Delete all generated charts for a specific file_id.
    
    Args:
        file_id: Unique identifier for the dataset
        
    Returns:
        Success message with count of deleted charts
    """
    try:
        charts_dir = Path("temp_uploads/charts") / file_id
        
        if not charts_dir.exists():
            return {
                "success": True,
                "message": "No charts directory found",
                "deleted_count": 0
            }
        
        # Delete all PNG files
        chart_files = list(charts_dir.glob("*.png"))
        deleted_count = 0
        
        for chart_file in chart_files:
            chart_file.unlink()
            deleted_count += 1
        
        # Remove directory if empty
        if not any(charts_dir.iterdir()):
            charts_dir.rmdir()
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} chart(s)",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting charts: {str(e)}"
        )
