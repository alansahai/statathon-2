"""
StatFlow AI - Machine Learning Models
Pydantic models for ML API endpoints.
Updated to support multi-file operations with backward compatibility.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal


class ClassificationRequest(BaseModel):
    """Request model for classification tasks."""
    file_id: Optional[str] = Field(None, description="Single file ID (legacy)")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs (1-5 files)")
    
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
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id or file_ids into file_ids list"""
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


class RegressionRequest(BaseModel):
    """Request model for regression tasks."""
    file_id: Optional[str] = Field(None, description="Single file ID (legacy)")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs (1-5 files)")
    
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
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id or file_ids into file_ids list"""
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


class ClusteringRequest(BaseModel):
    """Request model for clustering tasks."""
    file_id: Optional[str] = Field(None, description="Single file ID (legacy)")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs (1-5 files)")
    
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
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id or file_ids into file_ids list"""
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


class PCARequest(BaseModel):
    """Request model for PCA dimensionality reduction."""
    file_id: Optional[str] = Field(None, description="Single file ID (legacy)")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs (1-5 files)")
    
    feature_columns: Optional[List[str]] = Field(
        None, 
        description="Feature columns (auto-detected if not provided)"
    )
    n_components: int = Field(2, ge=1, le=20, description="Number of principal components")
    
    # Options
    use_weighted: bool = Field(False, description="Use weighted data if available")
    standardize: bool = Field(True, description="Standardize features before PCA")
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id or file_ids into file_ids list"""
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


class FeatureImportanceRequest(BaseModel):
    """Request model for feature importance analysis."""
    file_id: Optional[str] = Field(None, description="Single file ID (legacy)")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs (1-5 files)")
    
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
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id or file_ids into file_ids list"""
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
