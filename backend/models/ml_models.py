"""
StatFlow AI - Machine Learning Models
Pydantic models for ML API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class ClassificationRequest(BaseModel):
    """Request model for classification tasks."""
    file_id: str = Field(..., description="Unique file identifier")
    target_column: str = Field(..., description="Target column for classification")
    feature_columns: Optional[List[str]] = Field(
        None, 
        description="Feature columns (auto-detected if not provided)"
    )
    method: Literal["logistic_regression", "random_forest"] = Field(
        "logistic_regression",
        description="Classification method"
    )
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, description="Random seed for reproducibility")
    
    # Random Forest params
    n_trees: int = Field(100, ge=10, le=500, description="Number of trees for Random Forest")
    max_depth: Optional[int] = Field(None, ge=1, le=50, description="Maximum tree depth")
    
    # Options
    use_weighted: bool = Field(False, description="Use weighted data if available")
    check_class_balance: bool = Field(True, description="Check for class imbalance")


class RegressionRequest(BaseModel):
    """Request model for regression tasks."""
    file_id: str = Field(..., description="Unique file identifier")
    target_column: str = Field(..., description="Target column for regression")
    feature_columns: Optional[List[str]] = Field(
        None, 
        description="Feature columns (auto-detected if not provided)"
    )
    method: Literal["linear_regression", "random_forest"] = Field(
        "linear_regression",
        description="Regression method"
    )
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, description="Random seed for reproducibility")
    
    # Random Forest params
    n_trees: int = Field(100, ge=10, le=500, description="Number of trees for Random Forest")
    max_depth: Optional[int] = Field(None, ge=1, le=50, description="Maximum tree depth")
    
    # Options
    use_weighted: bool = Field(False, description="Use weighted data if available")
    check_multicollinearity: bool = Field(True, description="Check for multicollinearity")


class ClusteringRequest(BaseModel):
    """Request model for clustering tasks."""
    file_id: str = Field(..., description="Unique file identifier")
    feature_columns: Optional[List[str]] = Field(
        None, 
        description="Feature columns (auto-detected if not provided)"
    )
    n_clusters: int = Field(3, ge=2, le=20, description="Number of clusters")
    max_iterations: int = Field(100, ge=10, le=500, description="Maximum K-Means iterations")
    random_state: int = Field(42, description="Random seed for reproducibility")
    
    # Options
    use_weighted: bool = Field(False, description="Use weighted data if available")
    standardize: bool = Field(True, description="Standardize features before clustering")


class PCARequest(BaseModel):
    """Request model for PCA dimensionality reduction."""
    file_id: str = Field(..., description="Unique file identifier")
    feature_columns: Optional[List[str]] = Field(
        None, 
        description="Feature columns (auto-detected if not provided)"
    )
    n_components: int = Field(2, ge=1, le=20, description="Number of principal components")
    
    # Options
    use_weighted: bool = Field(False, description="Use weighted data if available")
    standardize: bool = Field(True, description="Standardize features before PCA")


class FeatureImportanceRequest(BaseModel):
    """Request model for feature importance analysis."""
    file_id: str = Field(..., description="Unique file identifier")
    target_column: str = Field(..., description="Target column")
    feature_columns: Optional[List[str]] = Field(
        None, 
        description="Feature columns (auto-detected if not provided)"
    )
    method: Literal["correlation", "random_forest"] = Field(
        "correlation",
        description="Method for computing importance"
    )
    use_weighted: bool = Field(False, description="Use weighted data if available")
