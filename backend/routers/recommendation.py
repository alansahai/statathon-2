"""
Recommendation API router for StatFlow AI
Provides intelligent recommendations for analysis methods, tests, transformations, and models
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path

from services.recommendation_engine import RecommendationEngine

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoints"""
    status: str
    file_id: str
    recommendations: Dict[str, Any]
    message: Optional[str] = None


class NarrativeResponse(BaseModel):
    """Response model for narrative endpoint"""
    status: str
    file_id: str
    narrative: str


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


@router.get("/{file_id}", response_model=RecommendationResponse)
async def get_recommendations(file_id: str):
    """
    Get comprehensive analysis recommendations for a dataset.
    
    Provides intelligent suggestions for:
    - Analysis methods (PCA, correlation, forecasting, etc.)
    - Data transformations (log transform, scaling, grouping, etc.)
    - Statistical tests (correlation, ANOVA, chi-square, etc.)
    - Machine learning models (classifiers, regressors, clustering, etc.)
    
    Args:
        file_id: Unique identifier for the dataset
        
    Returns:
        RecommendationResponse with all recommendations and metadata
        
    Raises:
        HTTPException: If dataset not found or error occurs
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize recommendation engine
        engine = RecommendationEngine(dataframe=df)
        
        # Generate recommendations
        recommendations = engine.build_summary()
        
        return RecommendationResponse(
            status="success",
            file_id=file_id,
            recommendations=recommendations,
            message="Recommendations generated successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@router.get("/narrative/{file_id}", response_model=NarrativeResponse)
async def get_recommendations_narrative(file_id: str):
    """
    Get human-readable narrative of analysis recommendations.
    
    Converts structured recommendations into a professional narrative
    suitable for reports or user interfaces.
    
    Args:
        file_id: Unique identifier for the dataset
        
    Returns:
        NarrativeResponse with narrative string
        
    Raises:
        HTTPException: If dataset not found or error occurs
    """
    try:
        # Load dataframe
        df = load_dataframe(file_id)
        
        # Initialize recommendation engine
        engine = RecommendationEngine(dataframe=df)
        
        # Generate recommendations
        summary = engine.build_summary()
        
        # Generate narrative
        narrative = engine.generate_narrative(summary)
        
        return NarrativeResponse(
            status="success",
            file_id=file_id,
            narrative=narrative
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating narrative: {str(e)}"
        )
