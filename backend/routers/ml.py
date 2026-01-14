"""
StatFlow AI - Machine Learning Router
API endpoints for ML operations.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import pandas as pd
import os

from services.ml_engine import MLEngine
from models.ml_models import (
    ClassificationRequest,
    RegressionRequest,
    ClusteringRequest,
    PCARequest,
    FeatureImportanceRequest
)
from utils.file_manager import FileManager

router = APIRouter(prefix="/ml", tags=["07 Machine Learning"])

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


@router.post("/classify", summary="Run Classification", description="Run classification model")
async def run_classification(request: ClassificationRequest) -> Dict[str, Any]:
    """
    Run a classification model.
    
    Supported methods:
    - logistic_regression: Logistic regression classifier
    - random_forest: Random forest classifier
    
    Returns accuracy, confusion matrix, classification report, and feature importances.
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
                engine = MLEngine(df)
                
                if request.method == "logistic_regression":
                    result = engine.logistic_regression(
                        target_column=request.target_column,
                        feature_columns=request.feature_columns,
                        test_size=request.test_size,
                        random_state=request.random_state
                    )
                else:  # random_forest
                    result = engine.random_forest_classifier(
                        target_column=request.target_column,
                        feature_columns=request.feature_columns,
                        n_trees=request.n_trees,
                        max_depth=request.max_depth,
                        test_size=request.test_size,
                        random_state=request.random_state
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
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@router.post("/regress", summary="Run Regression", description="Run regression model")
async def run_regression(request: RegressionRequest) -> Dict[str, Any]:
    """
    Run a regression model.
    
    Supported methods:
    - linear_regression: Ordinary least squares regression
    - random_forest: Random forest regressor
    
    Returns R², MSE, MAE, coefficients, and predictions.
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
                engine = MLEngine(df)
                
                if request.method == "linear_regression":
                    result = engine.linear_regression(
                        target_column=request.target_column,
                        feature_columns=request.feature_columns,
                        test_size=request.test_size,
                        random_state=request.random_state
                    )
                else:  # random_forest
                    result = engine.random_forest_regressor(
                        target_column=request.target_column,
                        feature_columns=request.feature_columns,
                        n_trees=request.n_trees,
                        max_depth=request.max_depth,
                        test_size=request.test_size,
                        random_state=request.random_state
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
        raise HTTPException(status_code=500, detail=f"Regression error: {str(e)}")


@router.post("/cluster", summary="Run Clustering", description="Run K-Means clustering")
async def run_clustering(request: ClusteringRequest) -> Dict[str, Any]:
    """
    Run K-Means clustering.
    
    Returns cluster assignments, centroids, inertia, and silhouette score.
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
                engine = MLEngine(df)
                
                result = engine.kmeans(
                    feature_columns=request.feature_columns,
                    n_clusters=request.n_clusters,
                    max_iterations=request.max_iterations,
                    random_state=request.random_state
                )
                
                results_per_file[fid] = {
                    "n_clusters": request.n_clusters,
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
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")


@router.post("/pca", summary="Run PCA", description="Run Principal Component Analysis")
async def run_pca(request: PCARequest) -> Dict[str, Any]:
    """
    Run Principal Component Analysis for dimensionality reduction.
    
    Returns principal components, explained variance ratios, and loadings.
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
                engine = MLEngine(df)
                
                result = engine.pca(
                    feature_columns=request.feature_columns,
                    n_components=request.n_components
                )
                
                results_per_file[fid] = {
                    "n_components": request.n_components,
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
        raise HTTPException(status_code=500, detail=f"PCA error: {str(e)}")


@router.post("/feature-importance", summary="Feature Importance", description="Analyze feature importance")
async def analyze_feature_importance(request: FeatureImportanceRequest) -> Dict[str, Any]:
    """
    Analyze feature importance for a target variable.
    
    Methods:
    - correlation: Correlation-based importance
    - random_forest: Random forest-based importance
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
                engine = MLEngine(df)
                
                # Get feature columns
                feature_cols = request.feature_columns
                if feature_cols is None:
                    feature_cols = [col for col in df.columns 
                                  if col != request.target_column and 
                                  pd.api.types.is_numeric_dtype(df[col])]
                
                if request.method == "correlation":
                    # Correlation-based importance
                    import numpy as np
                    importances = {}
                    target = df[request.target_column].values
                    
                    for col in feature_cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            mask = ~(np.isnan(df[col].values) | np.isnan(target))
                            if np.sum(mask) > 2:
                                corr = np.corrcoef(df[col].values[mask], target[mask])[0, 1]
                                importances[col] = float(abs(corr)) if not np.isnan(corr) else 0
                            else:
                                importances[col] = 0
                    
                    # Sort by importance
                    sorted_importance = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                    
                    result = {
                        "method": "correlation",
                        "importances": [{"feature": k, "importance": v} for k, v in sorted_importance],
                        "n_features": len(sorted_importance)
                    }
                else:  # random_forest
                    # Use random forest to get importance
                    rf_result = engine.random_forest_regressor(
                        target_column=request.target_column,
                        feature_columns=feature_cols,
                        n_trees=50,
                        test_size=0.2
                    )
                    
                    result = {
                        "method": "random_forest",
                        "importances": rf_result.get("feature_importance", []),
                        "n_features": len(feature_cols)
                    }
                
                results_per_file[fid] = {
                    "target_column": request.target_column,
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
        raise HTTPException(status_code=500, detail=f"Feature importance error: {str(e)}")


@router.get("/operations-log/{file_id}", summary="Get Operations Log", description="Get ML operations log")
async def get_operations_log(file_id: str, use_weighted: bool = False) -> Dict[str, Any]:
    """
    Get the operations log for an ML session.
    """
    try:
        df = _load_dataframe(file_id, use_weighted)
        engine = MLEngine(df)
        
        return {
            "success": True,
            "file_id": file_id,
            "operations_log": engine.get_operations_log()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving log: {str(e)}")
