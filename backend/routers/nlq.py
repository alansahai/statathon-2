"""
Natural Language Query Router
Accepts natural language questions and returns analysis results
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from services.nlq_engine import NLQEngine
from services.analysis_engine import AnalysisEngine
from services.forecasting_engine import ForecastingEngine
from services.ml_engine import MLEngine
from services.insight_engine import InsightEngine


router = APIRouter(prefix="/nlq", tags=["09 NLQ Engine"])


class NLQRequest(BaseModel):
    """Request model for natural language queries"""
    file_id: str
    query: str


class NLQResponse(BaseModel):
    """Response model for NLQ results"""
    status: str
    intent: str
    action: str
    columns: list
    result: Dict[str, Any]
    narrative: str
    confidence: float
    validation: Optional[Dict] = None


def load_dataframe(file_id: str) -> pd.DataFrame:
    """
    Load dataframe for the given file_id.
    Tries weighted → cleaned → uploads directories.
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Pandas DataFrame
        
    Raises:
        HTTPException: If file not found
    """
    base_path = Path("temp_uploads")
    
    # Try weighted first
    weighted_path = base_path / "weighted" / "default_user" / f"{file_id}_weighted.csv"
    if weighted_path.exists():
        return pd.read_csv(weighted_path)
    
    # Try cleaned
    cleaned_path = base_path / "cleaned" / "default_user" / f"{file_id}_cleaned.csv"
    if cleaned_path.exists():
        return pd.read_csv(cleaned_path)
    
    # Try uploads
    upload_path = base_path / "uploads" / "default_user" / f"{file_id}.csv"
    if upload_path.exists():
        return pd.read_csv(upload_path)
    
    raise HTTPException(
        status_code=404,
        detail=f"File not found for file_id: {file_id}"
    )


def generate_narrative(intent: str, columns: list, result: Dict[str, Any], action: str) -> str:
    """
    Generate natural language summary from analysis results.
    
    Converts statistical outputs into readable English sentences
    based on the intent and result structure.
    
    Args:
        intent: Detected intent (compare, trend, distribution, etc.)
        columns: Columns involved in the analysis
        result: Analysis result dictionary
        action: Action taken (e.g., "analysis.crosstab")
        
    Returns:
        Human-readable narrative string
    """
    try:
        if intent == "compare":
            # Crosstab comparison narrative
            if "crosstab" in result:
                col1 = columns[0] if len(columns) > 0 else "first variable"
                col2 = columns[1] if len(columns) > 1 else "second variable"
                
                narrative = f"Comparison analysis between {col1} and {col2}. "
                
                # Try to extract some statistics
                if "statistics" in result:
                    stats = result["statistics"]
                    if "chi_square" in stats:
                        chi_val = stats["chi_square"]
                        p_val = stats.get("p_value", 0)
                        if p_val < 0.05:
                            narrative += f"Significant association detected (χ²={chi_val:.2f}, p<0.05). "
                        else:
                            narrative += f"No significant association found (p={p_val:.3f}). "
                
                narrative += "See detailed crosstab results for group breakdowns."
                return narrative
            
            return f"Compared {' vs '.join(columns[:2])} showing distinct group patterns."
        
        elif intent == "distribution":
            # Descriptive statistics narrative
            col = columns[0] if columns else "variable"
            
            if "numeric" in result and col in result["numeric"]:
                stats = result["numeric"][col]
                mean = stats.get("mean", 0)
                median = stats.get("median", 0)
                std = stats.get("std", 0)
                skewness = stats.get("skewness", 0)
                
                # Determine skewness direction
                if abs(skewness) < 0.5:
                    skew_text = "approximately normal"
                elif skewness > 0:
                    skew_text = "right-skewed"
                else:
                    skew_text = "left-skewed"
                
                return (
                    f"The distribution of {col} is {skew_text} with "
                    f"mean={mean:.2f}, median={median:.2f}, and std={std:.2f}."
                )
            
            elif "categorical" in result and col in result["categorical"]:
                stats = result["categorical"][col]
                unique = stats.get("unique_count", 0)
                mode = stats.get("mode", "N/A")
                
                return (
                    f"The variable {col} has {unique} unique categories. "
                    f"The most common value is '{mode}'."
                )
            
            return f"Statistical summary of {col} completed. Check detailed results for full breakdown."
        
        elif intent == "trend":
            # Time series forecast narrative
            col = columns[1] if len(columns) > 1 else columns[0] if columns else "value"
            
            if "forecast" in result:
                forecast_data = result["forecast"]
                if "trend" in forecast_data:
                    trend = forecast_data["trend"]
                    if trend > 0:
                        direction = "upward"
                    elif trend < 0:
                        direction = "downward"
                    else:
                        direction = "stable"
                    
                    return (
                        f"Time series analysis of {col} shows a {direction} trend. "
                        f"Forecast generated for future periods."
                    )
            
            return f"Trend analysis of {col} over time completed with forecasting projections."
        
        elif intent == "relationship":
            # Correlation narrative
            if len(columns) >= 2:
                col1, col2 = columns[0], columns[1]
                
                if "correlation" in result:
                    corr = result["correlation"]
                    
                    # Try to find correlation value
                    corr_val = None
                    if isinstance(corr, dict):
                        if "matrix" in corr and isinstance(corr["matrix"], dict):
                            # Navigate nested dict structure
                            if col1 in corr["matrix"] and col2 in corr["matrix"][col1]:
                                corr_val = corr["matrix"][col1][col2]
                        elif col1 in corr and col2 in corr[col1]:
                            corr_val = corr[col1][col2]
                    
                    if corr_val is not None:
                        # Classify correlation strength
                        abs_corr = abs(corr_val)
                        if abs_corr > 0.7:
                            strength = "strong"
                        elif abs_corr > 0.4:
                            strength = "moderate"
                        else:
                            strength = "weak"
                        
                        direction = "positive" if corr_val > 0 else "negative"
                        
                        return (
                            f"{col1} and {col2} show {strength} {direction} "
                            f"correlation (r={corr_val:.3f})."
                        )
                
                return f"Correlation analysis between {col1} and {col2} completed."
            
            return "Correlation analysis performed across multiple variables."
        
        elif intent == "predict":
            # ML model prediction narrative
            target = columns[-1] if columns else "target variable"
            
            if "recommended_model" in result:
                model = result["recommended_model"]
                return f"A {model} model is recommended to predict {target}."
            
            elif "models" in result and result["models"]:
                models = result["models"]
                if isinstance(models, list) and len(models) > 0:
                    best_model = models[0].get("name", "machine learning")
                    accuracy = models[0].get("accuracy", 0)
                    return (
                        f"Predictive model for {target}: {best_model} "
                        f"with {accuracy:.2%} accuracy."
                    )
            
            return f"Machine learning model selection completed for predicting {target}."
        
        elif intent == "risk":
            # Risk group identification narrative
            if "risk_groups" in result:
                groups = result["risk_groups"]
                num_groups = len(groups) if isinstance(groups, list) else 0
                
                if num_groups > 0:
                    return (
                        f"Identified {num_groups} risk groups requiring attention. "
                        f"See detailed results for group characteristics."
                    )
            
            if "high_risk_count" in result:
                count = result["high_risk_count"]
                return f"Risk analysis detected {count} high-risk observations."
            
            return "Risk group analysis completed. Check detailed results for flagged segments."
        
        elif intent == "summary":
            # General summary narrative
            if "row_count" in result:
                rows = result["row_count"]
                cols = result.get("column_count", 0)
                return (
                    f"Dataset summary: {rows} observations across {cols} variables. "
                    f"See full report for statistical overview."
                )
            
            return "General dataset summary completed with key statistics."
        
        else:
            return f"Analysis completed for query intent: {intent}."
    
    except Exception as e:
        # Fallback narrative
        return f"Analysis completed. Results available in detailed output."


def execute_analysis(routing: Dict, df: pd.DataFrame, file_id: str) -> Dict[str, Any]:
    """
    Execute the appropriate analysis based on routing decision.
    
    Args:
        routing: Routing dictionary from NLQEngine
        df: Pandas DataFrame
        file_id: File identifier
        
    Returns:
        Analysis result dictionary
        
    Raises:
        HTTPException: If execution fails
    """
    action = routing.get("action", "")
    columns = routing.get("columns", [])
    params = routing.get("params", {})
    
    try:
        # Route to analysis engine
        if action == "analysis.crosstab":
            engine = AnalysisEngine(file_id)
            
            # Need categorical and numeric columns
            cat_col = params.get("categorical_col")
            num_col = params.get("numeric_col")
            
            if not cat_col or not num_col:
                # Try to use first two columns
                if len(columns) >= 2:
                    cat_col = columns[0]
                    num_col = columns[1]
                else:
                    raise ValueError("Crosstab requires at least 2 columns")
            
            result = engine.crosstab_analysis(cat_col, num_col)
            return result
        
        elif action == "analysis.descriptive":
            engine = AnalysisEngine(file_id)
            
            # Use specified columns or first column
            target_cols = params.get("columns", columns)
            if not target_cols:
                raise ValueError("No columns specified for descriptive analysis")
            
            # Run descriptive stats on first column
            result = engine.descriptive_statistics(target_cols[0])
            return result
        
        elif action == "analysis.correlation":
            engine = AnalysisEngine(file_id)
            
            # Use specified columns or all numeric
            target_cols = params.get("columns", columns)
            if len(target_cols) < 2:
                # Use all numeric columns
                target_cols = df.select_dtypes(include=['number']).columns.tolist()[:5]
            
            result = engine.correlation_analysis(target_cols)
            return result
        
        elif action == "analysis.summary":
            engine = AnalysisEngine(file_id)
            
            # General summary doesn't need specific columns
            result = engine.generate_summary()
            return result
        
        elif action == "forecasting.timeseries":
            engine = ForecastingEngine(file_id)
            
            # Need time and value columns
            time_col = params.get("time_col")
            value_col = params.get("value_col")
            
            if not time_col or not value_col:
                if len(columns) >= 2:
                    time_col = columns[0]
                    value_col = columns[1]
                else:
                    raise ValueError("Time series requires time and value columns")
            
            result = engine.forecast_timeseries(time_col, value_col)
            return result
        
        elif action == "ml.autoselect":
            engine = MLEngine(file_id)
            
            # Need target column
            target_col = params.get("target_col")
            feature_cols = params.get("feature_cols", [])
            
            if not target_col:
                if columns:
                    target_col = columns[-1]
                else:
                    raise ValueError("ML prediction requires target column")
            
            result = engine.auto_model_selection(target_col, feature_cols)
            return result
        
        elif action == "insight.risk_groups":
            engine = InsightEngine(file_id)
            
            # Use specified columns or detect automatically
            target_cols = params.get("columns", columns)
            
            result = engine.identify_risk_groups(target_cols if target_cols else None)
            return result
        
        else:
            raise ValueError(f"Unsupported action: {action}")
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis execution failed: {str(e)}"
        )


@router.post("/query", response_model=NLQResponse)
async def natural_language_query(request: NLQRequest):
    """
    Process natural language query and return analysis results.
    
    **Flow:**
    1. Load dataset for file_id
    2. Initialize NLQEngine to parse query
    3. Route to appropriate analysis engine
    4. Execute analysis
    5. Generate natural language narrative
    6. Return structured response
    
    **Examples:**
    ```
    Query: "compare male and female income"
    → Executes crosstab analysis on gender vs income
    
    Query: "show trend of sales over time"
    → Executes time series forecasting on sales
    
    Query: "distribution of age"
    → Executes descriptive statistics on age
    
    Query: "relationship between height and weight"
    → Executes correlation analysis
    ```
    
    **Error Codes:**
    - 400: Missing or invalid columns
    - 404: File not found
    - 422: Unsupported intent or query format
    - 500: Analysis execution failed
    """
    try:
        # STEP 1: Load dataframe
        df = load_dataframe(request.file_id)
        
        # STEP 2: Initialize NLQEngine
        nlq_engine = NLQEngine(dataframe=df)
        
        # STEP 3: Route query
        routing = nlq_engine.route(request.query)
        
        # Validate routing
        routing = nlq_engine.validate_routing(routing)
        
        # Check validation status
        if not routing.get("validation", {}).get("valid", True):
            warnings = routing["validation"].get("warnings", [])
            raise HTTPException(
                status_code=400,
                detail=f"Invalid query: {'; '.join(warnings)}"
            )
        
        # Check confidence threshold
        if routing.get("confidence", 0) < 0.3:
            raise HTTPException(
                status_code=422,
                detail="Query too ambiguous. Please rephrase with more specific terms."
            )
        
        # Check if action is supported
        if not routing.get("action"):
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported query intent: {routing.get('intent', 'unknown')}"
            )
        
        # STEP 4: Execute analysis
        result = execute_analysis(routing, df, request.file_id)
        
        # STEP 5: Generate narrative
        narrative = generate_narrative(
            intent=routing["intent"],
            columns=routing["columns"],
            result=result,
            action=routing["action"]
        )
        
        # Log query using NLQEngine method
        nlq_engine.log_query(
            file_id=request.file_id,
            query=request.query,
            intent_data=routing,
            narrative=narrative
        )
        
        # STEP 6: Return response
        return NLQResponse(
            status="success",
            intent=routing["intent"],
            action=routing["action"],
            columns=routing["columns"],
            result=result,
            narrative=narrative,
            confidence=routing.get("confidence", 0.0),
            validation=routing.get("validation")
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@router.get("/examples")
async def get_query_examples():
    """
    Get example natural language queries for testing.
    
    Returns dictionary of example queries organized by intent.
    """
    return {
        "examples": {
            "compare": [
                "compare male and female income",
                "difference between urban and rural education levels",
                "group by gender and show average salary"
            ],
            "trend": [
                "show trend of sales over time",
                "forecast revenue for next quarter",
                "historical progression of unemployment rate"
            ],
            "distribution": [
                "distribution of age",
                "histogram of income levels",
                "describe the statistics for height"
            ],
            "relationship": [
                "relationship between height and weight",
                "correlation between education and income",
                "how does age relate to salary"
            ],
            "predict": [
                "predict customer churn",
                "classify loan default risk",
                "estimate house prices"
            ],
            "risk": [
                "identify high risk groups",
                "find vulnerable populations",
                "detect anomalies in spending"
            ],
            "summary": [
                "give me a summary of the dataset",
                "overview of all variables",
                "describe the data"
            ]
        },
        "tips": [
            "Use specific column names when possible",
            "Include action words like 'compare', 'show', 'predict'",
            "For time series, mention time-related terms",
            "For comparisons, use 'vs', 'between', or 'by group'"
        ]
    }
