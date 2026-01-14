"""
Analysis Router - Production-ready statistical analysis endpoints
Matches conventions from cleaning.py and weighting.py routers
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional, Dict, List, Any
import pandas as pd
import os
from pathlib import Path

from services.analysis_engine import AnalysisEngine
from utils.file_manager import FileManager
from models.analysis_models import StatisticalTestRequest, AutoTestRequest, WelchANOVARequest, ShapiroWilkRequest, TukeyHSDRequest

router = APIRouter(
    prefix="/analysis",
    tags=["05 Analysis"]
)

file_manager = FileManager(base_storage_path="temp_uploads")


# ==========================================
# REQUEST MODELS
# ==========================================

class DescriptiveStatsRequest(BaseModel):
    file_id: Optional[str] = None
    file_ids: Optional[List[str]] = None
    columns: List[str]
    weight_column: Optional[str] = None
    
    @field_validator('file_ids', mode='before')
    @classmethod
    def validate_file_ids(cls, v, info):
        """Ensure at least one of file_id or file_ids is provided"""
        file_id = info.data.get('file_id')
        if not v and not file_id:
            raise ValueError('Either file_id or file_ids must be provided')
        return v


class CrosstabRequest(BaseModel):
    file_id: Optional[str] = None
    file_ids: Optional[List[str]] = None
    row_var: str
    col_var: str
    layer_var: Optional[str] = None
    normalize: str = "none"  # "none", "row", "col", "all"
    weight_column: Optional[str] = None


class RegressionRequest(BaseModel):
    file_id: Optional[str] = None
    file_ids: Optional[List[str]] = None
    dependent: str
    independents: List[str]
    regression_type: str = "ols"  # "ols" or "logistic"
    weight_column: Optional[str] = None


class SubgroupRequest(BaseModel):
    file_id: Optional[str] = None
    file_ids: Optional[List[str]] = None
    group_by: str
    target: str
    metrics: List[str]  # ["mean", "median", "min", "max", "std", "count"]
    weight_column: Optional[str] = None
    min_n: int = 30


class AnovaRequest(BaseModel):
    file_id: str
    dependent_var: str
    group_var: str
    weight_column: Optional[str] = None


class ManovaRequest(BaseModel):
    file_id: str
    dependent_vars: List[str]
    group_var: str
    weight_column: Optional[str] = None


class KruskalRequest(BaseModel):
    file_id: str
    dependent_var: str
    group_var: str


class LeveneRequest(BaseModel):
    file_id: str
    dependent_var: str
    group_var: str


class ShapiroRequest(BaseModel):
    file_id: str
    column: str


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _load_dataframe(file_id: str, user_id: str = "default_user") -> pd.DataFrame:
    """
    Load DataFrame from file system with fallback logic.
    Matches pattern from cleaning.py and weighting.py
    """
    # Try to get from registry first
    file_info = file_manager.get_file_info(file_id)
    
    if file_info and "file_path" in file_info:
        file_path = Path(file_info["file_path"])
        if file_path.exists():
            return file_manager.load_dataframe(str(file_path))
    
    # Fallback: construct path manually
    uploads_dir = Path("temp_uploads/uploads") / user_id
    
    # Try .csv first
    csv_path = uploads_dir / f"{file_id}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    # Try .xlsx
    xlsx_path = uploads_dir / f"{file_id}.xlsx"
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    
    # Try weighted directory
    weighted_dir = Path("temp_uploads/weighted") / user_id
    csv_path_weighted = weighted_dir / f"{file_id}.csv"
    if csv_path_weighted.exists():
        return pd.read_csv(csv_path_weighted)
    
    xlsx_path_weighted = weighted_dir / f"{file_id}.xlsx"
    if xlsx_path_weighted.exists():
        return pd.read_excel(xlsx_path_weighted)
    
    raise FileNotFoundError(f"File not found for file_id: {file_id}")


# ==========================================
# ANALYSIS ENDPOINTS
# ==========================================

@router.post("/descriptive")
async def compute_descriptive_stats(request: DescriptiveStatsRequest):
    """
    Compute comprehensive descriptive statistics.
    
    Returns:
    - For numeric columns: mean, median, std, min, max, IQR, percentiles, skewness, kurtosis
    - For categorical columns: frequencies, percentages, mode
    - Missing value counts and zero/negative detection
    - Distribution warnings (high skew, high kurtosis, near-constant)
    - Weighted variants if weight_column provided
    """
    try:
        # Normalize file_ids
        file_ids = request.file_ids or ([request.file_id] if request.file_id else None)
        
        # Validate file_ids
        if not file_ids or len(file_ids) == 0:
            raise HTTPException(status_code=400, detail="No file_ids provided")
        if len(file_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
        
        # Process each file
        results_per_file = {}
        errors = {}
        
        for fid in file_ids:
            try:
                # Load data
                df = _load_dataframe(fid)
                
                # Validate columns exist
                missing_cols = [col for col in request.columns if col not in df.columns]
                if missing_cols:
                    errors[fid] = f"Columns not found: {missing_cols}"
                    continue
                
                # Validate weight column if provided
                if request.weight_column and request.weight_column not in df.columns:
                    errors[fid] = f"Weight column '{request.weight_column}' not found"
                    continue
                
                # Use the static method to generate statistics
                file_results = AnalysisEngine.generate_statistics(fid)
                
                # Filter to requested columns if specified
                if request.columns:
                    if 'descriptive_stats' in file_results:
                        file_results['descriptive_stats'] = {
                            k: v for k, v in file_results['descriptive_stats'].items() 
                            if k in request.columns
                        }
                
                results_per_file[fid] = {
                    "results": file_results,
                    "operations_log": []
                }
                
            except FileNotFoundError:
                errors[fid] = "File not found"
            except Exception as e:
                errors[fid] = str(e)
        
        # Determine status
        status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
        
        response = {
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


@router.post("/crosstab")
async def create_crosstab(request: CrosstabRequest):
    """
    Create cross-tabulation (2D or 3D) with chi-square test.
    
    Returns:
    - Count table and percentage tables (row%, col%, total%)
    - Chi-square test results with expected frequencies
    - Row/column margins and grand total
    - Warnings for low cell counts and unreliable tests
    - Support for 3D crosstabs via layer_var
    """
    try:
        # Validate normalize parameter
        valid_normalize = ["none", "row", "col", "all"]
        if request.normalize not in valid_normalize:
            raise HTTPException(
                status_code=400,
                detail=f"normalize must be one of {valid_normalize}"
            )
        
        # Normalize file_ids
        file_ids = request.file_ids or ([request.file_id] if request.file_id else None)
        if not file_ids or len(file_ids) == 0:
            raise HTTPException(status_code=400, detail="No file_ids provided")
        if len(file_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
        
        # Process each file
        results_per_file = {}
        errors = {}
        
        for fid in file_ids:
            try:
                # Load data
                df = _load_dataframe(fid)
                
                # Validate variables exist
                required_vars = [request.row_var, request.col_var]
                if request.layer_var:
                    required_vars.append(request.layer_var)
                if request.weight_column:
                    required_vars.append(request.weight_column)
                
                missing_vars = [var for var in required_vars if var not in df.columns]
                if missing_vars:
                    errors[fid] = f"Variables not found: {missing_vars}"
                    continue
                
                # Initialize engine and compute crosstab
                engine = AnalysisEngine(df)
                file_results = engine.crosstab(
                    row_var=request.row_var,
                    col_var=request.col_var,
                    layer_var=request.layer_var,
                    weight_column=request.weight_column,
                    normalize=request.normalize
                )
                
                results_per_file[fid] = {
                    "results": file_results,
                    "operations_log": engine.get_operations_log()
                }
                
            except FileNotFoundError:
                errors[fid] = "File not found"
            except Exception as e:
                errors[fid] = str(e)
        
        # Determine status
        status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
        
        response = {
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
        raise HTTPException(status_code=500, detail=f"Crosstab error: {str(e)}")


@router.post("/regression")
async def run_regression(request: RegressionRequest):
    """
    Run OLS or Logistic regression with full diagnostics.
    
    Returns:
    - Coefficients with robust standard errors (HC3)
    - t-statistics / z-statistics and p-values
    - R² / pseudo-R² and adjusted R²
    - VIF (Variance Inflation Factors) for multicollinearity detection
    - Correlation matrix of predictors
    - Warnings for high VIF, singular matrices, etc.
    - Support for weighted regression
    """
    try:
        # Load data
        df = _load_dataframe(request.file_id)
        
        # Validate variables exist
        required_vars = [request.dependent] + request.independents
        if request.weight_column:
            required_vars.append(request.weight_column)
        
        missing_vars = [var for var in required_vars if var not in df.columns]
        if missing_vars:
            raise HTTPException(
                status_code=400,
                detail=f"Variables not found: {missing_vars}"
            )
        
        # Validate regression type
        valid_types = ["ols", "logistic"]
        if request.regression_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"regression_type must be one of {valid_types}"
            )
        
        # Initialize engine and run regression
        engine = AnalysisEngine(df)
        results = engine.run_regression(
            dependent=request.dependent,
            independents=request.independents,
            regression_type=request.regression_type,
            weight_column=request.weight_column
        )
        
        # Check if regression returned an error
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return {
            "status": "success",
            "file_id": request.file_id,
            "results": results,
            "operations_log": engine.get_operations_log()
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regression error: {str(e)}")


@router.post("/subgroup")
async def subgroup_analysis(request: SubgroupRequest):
    """
    Perform subgroup analysis with confidence intervals and risk tagging.
    
    Returns:
    - Per-group metrics (mean, median, std, min, max, count, sum)
    - 95% confidence intervals for means
    - Risk classification: "stable", "caution", "high_risk"
    - Risk factors: small sample size, high variance, extreme outliers
    - Support for weighted metrics
    """
    try:
        # Load data
        df = _load_dataframe(request.file_id)
        
        # Validate variables exist
        required_vars = [request.group_by, request.target]
        if request.weight_column:
            required_vars.append(request.weight_column)
        
        missing_vars = [var for var in required_vars if var not in df.columns]
        if missing_vars:
            raise HTTPException(
                status_code=400,
                detail=f"Variables not found: {missing_vars}"
            )
        
        # Validate metrics
        valid_metrics = ["mean", "median", "min", "max", "std", "count", "sum"]
        invalid_metrics = [m for m in request.metrics if m not in valid_metrics]
        if invalid_metrics:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metrics: {invalid_metrics}. Valid: {valid_metrics}"
            )
        
        # Initialize engine and perform analysis
        engine = AnalysisEngine(df)
        results = engine.subgroup_analysis(
            group_by=request.group_by,
            target=request.target,
            metrics=request.metrics,
            weight_column=request.weight_column,
            min_n=request.min_n
        )
        
        # Check if analysis returned an error
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return {
            "status": "success",
            "file_id": request.file_id,
            "results": results,
            "operations_log": engine.get_operations_log()
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subgroup analysis error: {str(e)}")


@router.get("/report/{file_id}")
async def generate_analysis_report(
    file_id: str,
    weight_column: Optional[str] = None
):
    """
    Generate comprehensive automated analysis report.
    
    Auto-runs:
    - Descriptive statistics for all numeric/categorical columns
    - Correlation analysis (top patterns)
    - Chi-square tests for categorical pairs (crosstab signals)
    - Simple regression (if sufficient numeric columns)
    - Subgroup risk assessment
    - Anomaly detection
    
    Returns:
    - descriptive_summary: Key stats for each column
    - top_patterns: High correlations and strong associations
    - crosstab_signals: Significant chi-square tests
    - regression_summary: Key regression results (if applicable)
    - subgroup_risks: Groups with risk flags
    - anomalies: Distribution warnings and outliers
    - recommended_actions: Automated recommendations
    """
    try:
        # Load data
        df = _load_dataframe(file_id)
        
        # Validate weight column if provided
        if weight_column and weight_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Weight column '{weight_column}' not found"
            )
        
        # Initialize engine and generate report
        engine = AnalysisEngine(df)
        report = engine.generate_analysis_report(
            numeric_columns=None,  # Auto-detect
            categorical_columns=None,  # Auto-detect
            weight_column=weight_column
        )
        
        return {
            "status": "success",
            "file_id": file_id,
            "report": report,
            "operations_log": engine.get_operations_log()
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")


@router.get("/operations-log/{file_id}")
async def get_operations_log(file_id: str):
    """
    Retrieve operations log for the analysis session.
    
    Note: This creates a new engine instance, so only current operation will be logged.
    For persistent logging, store engine instance in session/cache.
    """
    try:
        df = _load_dataframe(file_id)
        engine = AnalysisEngine(df)
        
        return {
            "status": "success",
            "file_id": file_id,
            "operations_log": engine.get_operations_log()
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving log: {str(e)}")


# ==========================================
# ADVANCED STATISTICAL TESTS
# ==========================================

@router.post("/anova")
async def run_anova(request: AnovaRequest):
    """
    Perform one-way ANOVA test.
    
    Tests whether group means differ significantly across categories.
    Supports weighted and unweighted variants.
    
    Returns:
    - F-statistic and p-value
    - Effect sizes (η² and ω²)
    - Group means and variances
    - Sum of squares breakdown
    - Significance interpretation
    - Diagnostic warnings (variance imbalance, small groups)
    """
    try:
        # Load data
        df = _load_dataframe(request.file_id)
        
        # Validate columns
        if request.dependent_var not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Dependent variable '{request.dependent_var}' not found"
            )
        if request.group_var not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Group variable '{request.group_var}' not found"
            )
        
        # Validate weight column if provided
        if request.weight_column and request.weight_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Weight column '{request.weight_column}' not found"
            )
        
        # Run ANOVA
        engine = AnalysisEngine(df)
        result = engine.run_anova(
            dependent_var=request.dependent_var,
            group_var=request.group_var,
            weight_column=request.weight_column
        )
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "file_id": request.file_id,
            "result": result,
            "operations_log": engine.get_operations_log()
        }
    
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ANOVA error: {str(e)}")


@router.post("/manova")
async def run_manova(request: ManovaRequest):
    """
    Perform multivariate ANOVA (MANOVA) test.
    
    Tests whether multiple dependent variable means differ across groups simultaneously.
    
    Returns:
    - Pillai's Trace (most robust)
    - Wilks' Lambda (most common)
    - Hotelling-Lawley Trace
    - Roy's Largest Root
    - p-values for each statistic
    - Overall significance interpretation
    - Group means for all dependent variables
    - Diagnostic warnings
    """
    try:
        # Load data
        df = _load_dataframe(request.file_id)
        
        # Validate columns
        missing_vars = [v for v in request.dependent_vars if v not in df.columns]
        if missing_vars:
            raise HTTPException(
                status_code=400,
                detail=f"Dependent variables not found: {missing_vars}"
            )
        
        if request.group_var not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Group variable '{request.group_var}' not found"
            )
        
        # Validate weight column if provided
        if request.weight_column and request.weight_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Weight column '{request.weight_column}' not found"
            )
        
        # Run MANOVA
        engine = AnalysisEngine(df)
        result = engine.run_manova(
            dependent_vars=request.dependent_vars,
            group_var=request.group_var,
            weight_column=request.weight_column
        )
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "file_id": request.file_id,
            "result": result,
            "operations_log": engine.get_operations_log()
        }
    
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MANOVA error: {str(e)}")


@router.post("/kruskal")
async def run_kruskal(request: KruskalRequest):
    """
    Perform Kruskal-Wallis H-test (non-parametric alternative to ANOVA).
    
    Tests whether group medians differ significantly.
    Use when:
    - Data is non-normal (fails Shapiro-Wilk test)
    - Data is ordinal
    - Variance assumptions are violated
    
    Returns:
    - H-statistic (chi-square approximation)
    - p-value
    - Degrees of freedom
    - Ties correction indicator
    - Significance interpretation
    - Group sizes
    - Diagnostic warnings
    """
    try:
        # Load data
        df = _load_dataframe(request.file_id)
        
        # Validate columns
        if request.dependent_var not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Dependent variable '{request.dependent_var}' not found"
            )
        if request.group_var not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Group variable '{request.group_var}' not found"
            )
        
        # Run Kruskal-Wallis test
        engine = AnalysisEngine(df)
        result = engine.run_kruskal(
            dependent_var=request.dependent_var,
            group_var=request.group_var
        )
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "file_id": request.file_id,
            "result": result,
            "operations_log": engine.get_operations_log()
        }
    
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kruskal-Wallis error: {str(e)}")


@router.post("/levene")
async def run_levene(request: LeveneRequest):
    """
    Perform Levene's test for homogeneity of variances.
    
    Tests whether group variances are equal (ANOVA assumption).
    Use before ANOVA to check if equal variance assumption holds.
    
    Returns:
    - W-statistic
    - p-value
    - Group variances
    - Variance ratio (max/min)
    - Significance interpretation
    - Diagnostic warnings (high variance imbalance)
    
    Interpretation:
    - p < 0.05: Variances differ (consider Welch's ANOVA or transformations)
    - p ≥ 0.05: Variances equal (safe to proceed with standard ANOVA)
    """
    try:
        # Load data
        df = _load_dataframe(request.file_id)
        
        # Validate columns
        if request.dependent_var not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Dependent variable '{request.dependent_var}' not found"
            )
        if request.group_var not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Group variable '{request.group_var}' not found"
            )
        
        # Run Levene's test
        engine = AnalysisEngine(df)
        result = engine.run_levene(
            dependent_var=request.dependent_var,
            group_var=request.group_var
        )
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "file_id": request.file_id,
            "result": result,
            "operations_log": engine.get_operations_log()
        }
    
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Levene's test error: {str(e)}")


@router.post("/shapiro")
async def run_shapiro(request: ShapiroRequest):
    """
    Perform Shapiro-Wilk normality test.
    
    Tests whether data follows a normal distribution (assumption for parametric tests).
    Use before ANOVA, regression, or t-tests to validate normality assumption.
    
    Returns:
    - W-statistic
    - p-value
    - Skewness and kurtosis
    - Normality interpretation
    - Diagnostic warnings
    
    Interpretation:
    - p < 0.05: Data is non-normal (consider non-parametric tests)
    - p ≥ 0.05: Data is approximately normal (safe for parametric tests)
    
    Note: Very sensitive for large samples (n > 5000).
    """
    try:
        # Load data
        df = _load_dataframe(request.file_id)
        
        # Validate column
        if request.column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{request.column}' not found"
            )
        
        # Run Shapiro-Wilk test
        engine = AnalysisEngine(df)
        result = engine.run_shapiro(column=request.column)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "file_id": request.file_id,
            "result": result,
            "operations_log": engine.get_operations_log()
        }
    
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shapiro-Wilk error: {str(e)}")


# ==========================================
# STATISTICAL TEST ENDPOINTS
# ==========================================

@router.post("/test")
async def run_statistical_test(request: StatisticalTestRequest):
    """
    Run a specific statistical test.
    
    Available test types:
    - one_sample_t: One-sample t-test (requires var1, tests against mean=0)
    - independent_t: Independent samples t-test (requires var1, group)
    - paired_t: Paired samples t-test (requires var1, var2)
    - anova: One-way ANOVA (requires var1, group)
    - chi_square: Chi-square test of independence (requires var1, var2)
    - f_test: F-test for equality of variances (requires var1, var2)
    - levene_test: Levene's test for equality of variances (requires var1, var2)
    - bartlett_test: Bartlett's test for equality of variances (requires var1, var2)
    - kruskal_test: Kruskal-Wallis H-test (requires var1, group)
    - pearson_corr: Pearson correlation (requires var1, var2)
    - spearman_corr: Spearman correlation (requires var1, var2)
    - kendall_corr: Kendall's tau correlation (requires var1, var2)
    - shapiro_test: Shapiro-Wilk normality test (requires var1)
    - ks_test: Kolmogorov-Smirnov normality test (requires var1)
    - anderson_test: Anderson-Darling normality test (requires var1)
    - jb_test: Jarque-Bera normality test (requires var1)
    - proportion_test: One-sample proportion test (requires var1)
    - two_proportion_test: Two-sample proportion test (requires var1, group)
    
    Returns:
    - test: Name of the test performed
    - statistic: Test statistic value
    - p_value: P-value of the test
    - df: Degrees of freedom (if applicable)
    - warnings: List of warnings about assumptions or data issues
    - details: Additional test-specific information
    """
    try:
        # Normalize file_ids
        file_ids = request.file_ids or ([request.file_id] if request.file_id else None)
        if not file_ids or len(file_ids) == 0:
            raise HTTPException(status_code=400, detail="No file_ids provided")
        if len(file_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
        
        # Process each file
        results_per_file = {}
        errors = {}
        
        for fid in file_ids:
            try:
                # Load the dataframe
                df = _load_dataframe(fid)
                
                # Create analysis engine
                engine = AnalysisEngine(df)
                
                # Run the statistical test
                result = engine.run_stat_test(
                    test_type=request.test_type,
                    var1=request.var1,
                    var2=request.var2,
                    group=request.group,
                    weights=request.weights
                )
                
                results_per_file[fid] = {
                    "result": result,
                    "logs": engine.get_operations_log()
                }
                
            except ValueError as e:
                errors[fid] = str(e)
            except FileNotFoundError:
                errors[fid] = "File not found"
            except Exception as e:
                errors[fid] = str(e)
        
        # Determine status
        status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
        
        response = {
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
        raise HTTPException(status_code=500, detail=f"Statistical test error: {str(e)}")


@router.post("/auto-test")
async def auto_select_and_run_test(request: AutoTestRequest):
    """
    Automatically select and run the appropriate statistical test based on variable types.
    
    Decision logic:
    - Single numeric variable → Normality tests (Shapiro-Wilk)
    - Single categorical variable → Proportion test
    - Two numeric variables → Correlation (Pearson)
    - Two categorical variables → Chi-square test
    - Numeric + binary group → Independent t-test
    - Numeric + multi-level group → ANOVA
    - Categorical + group → Two-proportion test or Chi-square
    
    Returns:
    - selection: Recommended test and reasoning
    - test_executed: Actual test that was run
    - result: Test results (statistic, p_value, etc.)
    """
    try:
        # Normalize file_ids
        file_ids = request.file_ids or ([request.file_id] if request.file_id else None)
        if not file_ids or len(file_ids) == 0:
            raise HTTPException(status_code=400, detail="No file_ids provided")
        if len(file_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
        
        # Process each file
        results_per_file = {}
        errors = {}
        
        for fid in file_ids:
            try:
                # Load the dataframe
                df = _load_dataframe(fid)
                
                # Create analysis engine
                engine = AnalysisEngine(df)
                
                # Auto-select and run test
                result = engine.auto_test(
                    var1=request.var1,
                    var2=request.var2,
                    group=request.group
                )
                
                results_per_file[fid] = {
                    "result": result,
                    "logs": engine.get_operations_log()
                }
                
            except ValueError as e:
                errors[fid] = str(e)
            except FileNotFoundError:
                errors[fid] = "File not found"
            except Exception as e:
                errors[fid] = str(e)
        
        # Determine status
        status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
        
        response = {
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
        raise HTTPException(status_code=500, detail=f"Auto-test error: {str(e)}")


# ==========================================
# WELCH ANOVA, SHAPIRO-WILK, TUKEY HSD ENDPOINTS
# ==========================================

@router.post("/welch-anova")
async def run_welch_anova(request: WelchANOVARequest):
    """
    Perform Welch's ANOVA test (robust to unequal variances).
    
    Welch's ANOVA is preferred over standard ANOVA when:
    - Group variances are unequal (heteroscedasticity)
    - Sample sizes are unequal across groups
    
    Returns:
    - F_statistic: Welch's F statistic
    - p_value: P-value for the test
    - df_between: Degrees of freedom (between groups)
    - df_within: Welch-Satterthwaite degrees of freedom
    - effect_size: Eta-squared and omega-squared
    - group_statistics: Mean, variance, std, n for each group
    - warnings: Variance ratio warnings, sample size issues
    - interpretation: Plain-text interpretation of results
    """
    try:
        # Normalize file_ids
        file_ids = request.file_ids or ([request.file_id] if request.file_id else None)
        if not file_ids or len(file_ids) == 0:
            raise HTTPException(status_code=400, detail="No file_ids provided")
        if len(file_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
        
        # Process each file
        results_per_file = {}
        errors = {}
        
        for fid in file_ids:
            try:
                # Load the dataframe
                df = _load_dataframe(fid)
                
                # Validate columns exist
                if request.group_col not in df.columns:
                    errors[fid] = f"Group column '{request.group_col}' not found"
                    continue
                
                if request.value_col not in df.columns:
                    errors[fid] = f"Value column '{request.value_col}' not found"
                    continue
                
                if request.weight_column and request.weight_column not in df.columns:
                    errors[fid] = f"Weight column '{request.weight_column}' not found"
                    continue
                
                # Create analysis engine and run test
                engine = AnalysisEngine(df)
                result = engine.run_welch_anova(
                    group_col=request.group_col,
                    value_col=request.value_col,
                    weight_col=request.weight_column
                )
                
                # Check for errors in result
                if "error" in result:
                    errors[fid] = result["error"]
                    continue
                
                results_per_file[fid] = {
                    "result": result,
                    "warnings": result.get("warnings", []),
                    "operations_log": engine.get_operations_log()
                }
                
            except FileNotFoundError:
                errors[fid] = "File not found"
            except Exception as e:
                errors[fid] = str(e)
        
        # Determine status
        status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
        
        response = {
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
        raise HTTPException(status_code=500, detail=f"Welch ANOVA error: {str(e)}")


@router.post("/shapiro")
async def run_shapiro_wilk(request: ShapiroWilkRequest):
    """
    Perform Shapiro-Wilk test for normality.
    
    The Shapiro-Wilk test is one of the most powerful normality tests,
    especially for small to medium sample sizes.
    
    Returns:
    - W_statistic: Shapiro-Wilk W statistic (closer to 1 = more normal)
    - p_value: P-value for the test
    - normality: "normal" or "non-normal" based on α=0.05
    - interpretation: Plain-text interpretation
    - distribution_characteristics: Skewness, kurtosis, mean, median, std
    - distribution_flags: Descriptive flags (skewed, heavy_tails, multimodal, etc.)
    - recommended_transformations: Suggested transformations if non-normal
    """
    try:
        # Normalize file_ids
        file_ids = request.file_ids or ([request.file_id] if request.file_id else None)
        if not file_ids or len(file_ids) == 0:
            raise HTTPException(status_code=400, detail="No file_ids provided")
        if len(file_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
        
        # Process each file
        results_per_file = {}
        errors = {}
        
        for fid in file_ids:
            try:
                # Load the dataframe
                df = _load_dataframe(fid)
                
                # Validate column exists
                if request.value_col not in df.columns:
                    errors[fid] = f"Column '{request.value_col}' not found"
                    continue
                
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(df[request.value_col]):
                    errors[fid] = f"Column '{request.value_col}' must be numeric"
                    continue
                
                # Create analysis engine and run test
                engine = AnalysisEngine(df)
                result = engine.run_shapiro_wilk(value_col=request.value_col)
                
                # Check for errors in result
                if "error" in result:
                    errors[fid] = result["error"]
                    continue
                
                results_per_file[fid] = {
                    "result": result,
                    "warnings": result.get("warnings", []),
                    "operations_log": engine.get_operations_log()
                }
                
            except FileNotFoundError:
                errors[fid] = "File not found"
            except Exception as e:
                errors[fid] = str(e)
        
        # Determine status
        status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
        
        response = {
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
        raise HTTPException(status_code=500, detail=f"Shapiro-Wilk error: {str(e)}")


@router.post("/tukey-hsd")
async def run_tukey_hsd(request: TukeyHSDRequest):
    """
    Perform Tukey's HSD (Honestly Significant Difference) post-hoc test.
    
    Tukey's HSD is used after ANOVA to determine which specific group means differ.
    It controls the family-wise error rate while making all pairwise comparisons.
    
    Returns:
    - n_groups: Number of groups compared
    - n_comparisons: Number of pairwise comparisons
    - n_significant: Number of significant differences found
    - pooled_std: Pooled standard deviation
    - df_within: Degrees of freedom (within groups)
    - group_statistics: Mean, variance, std, n for each group
    - pairwise_comparisons: Full table with mean difference, q-statistic, p-value, CI
    - summary: Plain-text summary of results
    """
    try:
        # Normalize file_ids
        file_ids = request.file_ids or ([request.file_id] if request.file_id else None)
        if not file_ids or len(file_ids) == 0:
            raise HTTPException(status_code=400, detail="No file_ids provided")
        if len(file_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
        
        # Process each file
        results_per_file = {}
        errors = {}
        
        for fid in file_ids:
            try:
                # Load the dataframe
                df = _load_dataframe(fid)
                
                # Validate columns exist
                if request.group_col not in df.columns:
                    errors[fid] = f"Group column '{request.group_col}' not found"
                    continue
                
                if request.value_col not in df.columns:
                    errors[fid] = f"Value column '{request.value_col}' not found"
                    continue
                
                if request.weight_column and request.weight_column not in df.columns:
                    errors[fid] = f"Weight column '{request.weight_column}' not found"
                    continue
                
                # Check if value column is numeric
                if not pd.api.types.is_numeric_dtype(df[request.value_col]):
                    errors[fid] = f"Value column '{request.value_col}' must be numeric"
                    continue
                
                # Check for at least 2 groups
                n_groups = df[request.group_col].dropna().nunique()
                if n_groups < 2:
                    errors[fid] = f"Need at least 2 groups, found {n_groups}"
                    continue
                
                # Create analysis engine and run test
                engine = AnalysisEngine(df)
                result = engine.run_tukey_hsd(
                    group_col=request.group_col,
                    value_col=request.value_col,
                    weight_col=request.weight_column
                )
                
                # Check for errors in result
                if "error" in result:
                    errors[fid] = result["error"]
                    continue
                
                results_per_file[fid] = {
                    "result": result,
                    "warnings": result.get("warnings", []),
                    "operations_log": engine.get_operations_log()
                }
                
            except FileNotFoundError:
                errors[fid] = "File not found"
            except Exception as e:
                errors[fid] = str(e)
        
        # Determine status
        status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
        
        response = {
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
        raise HTTPException(status_code=500, detail=f"Tukey HSD error: {str(e)}")

