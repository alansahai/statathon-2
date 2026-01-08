"""
StatFlow AI - Insight Engine
Provides automated insights combining descriptive, forecast, and ML findings.
Production-ready implementation matching existing engine conventions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from scipy import stats
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class InsightEngine:
    """
    Automated insight generation engine.
    
    Combines:
    - Descriptive patterns (correlations, distributions, missing data)
    - Forecast insights (trends, seasonality)
    - ML insights (feature importance, clusters, risk segments)
    
    All outputs are JSON-safe (NaN/Inf converted to None).
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the insight engine with a DataFrame.
        
        Args:
            df: Input DataFrame containing data
        """
        self.df = df.copy()
        self.operations_log = []
        self._log_operation("initialized", {"rows": len(df), "columns": len(df.columns)})
    
    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Helper method to log operations."""
        self.operations_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            **details
        })
    
    def get_operations_log(self) -> List[Dict[str, Any]]:
        """Return the operations log."""
        return self.operations_log
    
    def _make_json_safe(self, obj: Any) -> Any:
        """Recursively convert NaN, Inf, and -Inf to None for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, np.ndarray):
            return self._make_json_safe(obj.tolist())
        elif isinstance(obj, pd.Series):
            return self._make_json_safe(obj.tolist())
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj
    
    # ==========================================
    # DESCRIPTIVE INSIGHTS
    # ==========================================
    
    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns."""
        return [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
    
    def _get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns."""
        return [col for col in self.df.columns if not pd.api.types.is_numeric_dtype(self.df[col])]
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Find high correlations between numeric variables."""
        numeric_cols = self._get_numeric_columns()
        
        if len(numeric_cols) < 2:
            return {"high_correlations": [], "message": "Insufficient numeric columns"}
        
        # Compute correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        high_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    corr = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr) and abs(corr) > 0.7:
                        high_correlations.append({
                            "variable1": col1,
                            "variable2": col2,
                            "correlation": float(corr),
                            "strength": "strong" if abs(corr) > 0.8 else "moderate",
                            "direction": "positive" if corr > 0 else "negative"
                        })
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return {
            "high_correlations": high_correlations[:10],  # Top 10
            "total_found": len(high_correlations)
        }
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """Analyze unusual distributions."""
        numeric_cols = self._get_numeric_columns()
        unusual_distributions = []
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) < 10:
                continue
            
            # Calculate statistics
            skewness = float(stats.skew(data))
            kurtosis = float(stats.kurtosis(data))
            
            issues = []
            
            if abs(skewness) > 2:
                issues.append(f"highly {'right' if skewness > 0 else 'left'} skewed (skew={skewness:.2f})")
            
            if kurtosis > 7:
                issues.append(f"heavy tails (kurtosis={kurtosis:.2f})")
            elif kurtosis < -1:
                issues.append(f"flat distribution (kurtosis={kurtosis:.2f})")
            
            # Check for potential outliers (IQR method)
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            outlier_count = np.sum((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr))
            outlier_pct = outlier_count / len(data) * 100
            
            if outlier_pct > 5:
                issues.append(f"{outlier_pct:.1f}% potential outliers")
            
            if issues:
                unusual_distributions.append({
                    "column": col,
                    "issues": issues,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "outlier_percentage": float(outlier_pct)
                })
        
        return {
            "unusual_distributions": unusual_distributions,
            "total_analyzed": len(numeric_cols)
        }
    
    def _analyze_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_info = []
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = missing_count / len(self.df) * 100
            
            if missing_pct > 0:
                missing_info.append({
                    "column": col,
                    "missing_count": int(missing_count),
                    "missing_percentage": float(missing_pct),
                    "severity": "high" if missing_pct > 20 else ("medium" if missing_pct > 5 else "low")
                })
        
        missing_info.sort(key=lambda x: x["missing_percentage"], reverse=True)
        
        # Check for patterns
        high_missing_cols = [m["column"] for m in missing_info if m["missing_percentage"] > 20]
        
        return {
            "columns_with_missing": missing_info,
            "total_columns_affected": len(missing_info),
            "high_missing_columns": high_missing_cols,
            "overall_completeness": float(100 - self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns)) * 100)
        }
    
    # ==========================================
    # FORECAST INSIGHTS
    # ==========================================
    
    def _analyze_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in a series."""
        n = len(values)
        if n < 3:
            return {"direction": "unknown", "strength": 0}
        
        # Linear trend
        x = np.arange(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Direction and strength
        if p_value > 0.05:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Calculate percentage change
        first_val = np.mean(values[:max(1, n//4)])
        last_val = np.mean(values[-max(1, n//4):])
        
        if first_val != 0:
            pct_change = (last_val - first_val) / abs(first_val) * 100
        else:
            pct_change = 0
        
        return {
            "direction": direction,
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "percentage_change": float(pct_change),
            "strength": "strong" if r_value ** 2 > 0.7 else ("moderate" if r_value ** 2 > 0.3 else "weak")
        }
    
    def _analyze_seasonality(self, values: np.ndarray, period: int = 12) -> Dict[str, Any]:
        """Analyze seasonality in a series."""
        n = len(values)
        
        if n < 2 * period:
            return {"has_seasonality": False, "strength": 0, "reason": "Insufficient data for seasonality analysis"}
        
        # Simple seasonality detection using autocorrelation
        values_centered = values - np.mean(values)
        
        # Autocorrelation at seasonal lag
        if n > period:
            autocorr = np.corrcoef(values_centered[:-period], values_centered[period:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0
        else:
            autocorr = 0
        
        has_seasonality = abs(autocorr) > 0.3
        
        return {
            "has_seasonality": has_seasonality,
            "autocorrelation": float(autocorr),
            "strength": float(abs(autocorr)),
            "period": period,
            "interpretation": "Strong seasonal pattern" if abs(autocorr) > 0.6 else (
                "Moderate seasonal pattern" if abs(autocorr) > 0.3 else "Weak or no seasonal pattern"
            )
        }
    
    def _forecast_next_period(self, values: np.ndarray) -> Dict[str, Any]:
        """Simple forecast for next period."""
        n = len(values)
        
        if n < 3:
            return {"forecast": None, "confidence": "low", "reason": "Insufficient data"}
        
        # Use exponential smoothing
        alpha = 0.3
        smoothed = values[0]
        for v in values[1:]:
            smoothed = alpha * v + (1 - alpha) * smoothed
        
        next_value = alpha * values[-1] + (1 - alpha) * smoothed
        
        # Calculate growth
        if values[-1] != 0:
            growth_pct = (next_value - values[-1]) / abs(values[-1]) * 100
        else:
            growth_pct = 0
        
        return {
            "next_period_forecast": float(next_value),
            "current_value": float(values[-1]),
            "expected_change": float(next_value - values[-1]),
            "growth_percentage": float(growth_pct),
            "confidence": "moderate" if n > 10 else "low"
        }
    
    # ==========================================
    # SUBGROUP RISK FLAGS
    # ==========================================
    
    def _analyze_subgroup_risks(self, group_column: Optional[str] = None) -> Dict[str, Any]:
        """Analyze risk flags for subgroups."""
        if group_column is None:
            # Try to find a suitable grouping column
            categorical_cols = self._get_categorical_columns()
            if not categorical_cols:
                return {"message": "No categorical columns found for subgroup analysis"}
            group_column = categorical_cols[0]
        
        if group_column not in self.df.columns:
            return {"error": f"Column '{group_column}' not found"}
        
        numeric_cols = self._get_numeric_columns()
        if not numeric_cols:
            return {"message": "No numeric columns found for risk analysis"}
        
        groups = self.df[group_column].dropna().unique()
        risk_flags = []
        
        for group in groups:
            group_data = self.df[self.df[group_column] == group]
            n = len(group_data)
            
            flags = []
            
            # Small sample size
            if n < 20:
                flags.append({
                    "type": "small_sample",
                    "message": f"Small sample size (n={n})",
                    "severity": "high" if n < 10 else "medium"
                })
            
            # High CV for numeric columns
            for col in numeric_cols[:5]:  # Check first 5
                col_data = group_data[col].dropna()
                if len(col_data) > 0 and col_data.mean() != 0:
                    cv = col_data.std() / abs(col_data.mean())
                    if cv > 0.5:
                        flags.append({
                            "type": "high_cv",
                            "column": col,
                            "cv": float(cv),
                            "message": f"High variability in {col} (CV={cv:.2f})",
                            "severity": "high" if cv > 1.0 else "medium"
                        })
            
            # Outlier density
            for col in numeric_cols[:3]:
                col_data = group_data[col].dropna()
                if len(col_data) > 0:
                    q1, q3 = np.percentile(col_data, [25, 75])
                    iqr = q3 - q1
                    outliers = np.sum((col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr))
                    outlier_pct = outliers / len(col_data) * 100
                    
                    if outlier_pct > 15:
                        flags.append({
                            "type": "high_outlier_density",
                            "column": col,
                            "outlier_percentage": float(outlier_pct),
                            "message": f"High outlier density in {col} ({outlier_pct:.1f}%)",
                            "severity": "medium"
                        })
            
            if flags:
                risk_flags.append({
                    "group": str(group),
                    "sample_size": n,
                    "flags": flags,
                    "risk_score": len(flags)
                })
        
        # Sort by risk score
        risk_flags.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return {
            "group_column": group_column,
            "total_groups": len(groups),
            "groups_with_risks": len(risk_flags),
            "risk_flags": risk_flags
        }
    
    # ==========================================
    # MAIN INSIGHT METHODS
    # ==========================================
    
    def generate_overview(self) -> Dict[str, Any]:
        """Generate a high-level overview of the data."""
        n_rows = len(self.df)
        n_cols = len(self.df.columns)
        numeric_cols = self._get_numeric_columns()
        categorical_cols = self._get_categorical_columns()
        
        # Missing data summary
        total_missing = self.df.isna().sum().sum()
        missing_pct = total_missing / (n_rows * n_cols) * 100
        
        # Data quality score (simple heuristic)
        quality_score = 100
        quality_score -= min(30, missing_pct)  # Missing data penalty
        
        # Check for duplicates
        duplicate_rows = self.df.duplicated().sum()
        if duplicate_rows > 0:
            quality_score -= min(10, duplicate_rows / n_rows * 100)
        
        # Quick stats for numeric columns
        numeric_summary = {}
        for col in numeric_cols[:5]:  # First 5
            data = self.df[col].dropna()
            if len(data) > 0:
                numeric_summary[col] = {
                    "mean": float(data.mean()),
                    "median": float(data.median()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max())
                }
        
        result = {
            "data_shape": {
                "rows": n_rows,
                "columns": n_cols,
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols)
            },
            "data_quality": {
                "completeness_percentage": float(100 - missing_pct),
                "duplicate_rows": int(duplicate_rows),
                "quality_score": float(max(0, quality_score))
            },
            "numeric_summary": numeric_summary,
            "column_types": {
                "numeric": numeric_cols,
                "categorical": categorical_cols
            }
        }
        
        self._log_operation("generate_overview", {"rows": n_rows, "columns": n_cols})
        
        return self._make_json_safe(result)
    
    def generate_full_insights(
        self,
        time_column: Optional[str] = None,
        value_column: Optional[str] = None,
        group_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive insights from the data.
        
        Args:
            time_column: Optional time column for forecast insights
            value_column: Optional value column for forecast insights
            group_column: Optional grouping column for subgroup analysis
            
        Returns:
            Dict with organized insights
        """
        # Overview
        overview = self.generate_overview()
        
        # Descriptive findings
        correlations = self._analyze_correlations()
        distributions = self._analyze_distributions()
        missing_data = self._analyze_missing_data()
        
        descriptive_findings = {
            "correlations": correlations,
            "distributions": distributions,
            "missing_data": missing_data,
            "key_insights": []
        }
        
        # Generate key insights from descriptive
        if correlations["high_correlations"]:
            top_corr = correlations["high_correlations"][0]
            descriptive_findings["key_insights"].append(
                f"Strong {top_corr['direction']} correlation ({top_corr['correlation']:.2f}) between {top_corr['variable1']} and {top_corr['variable2']}"
            )
        
        if distributions["unusual_distributions"]:
            top_dist = distributions["unusual_distributions"][0]
            descriptive_findings["key_insights"].append(
                f"Unusual distribution in {top_dist['column']}: {', '.join(top_dist['issues'])}"
            )
        
        if missing_data["high_missing_columns"]:
            descriptive_findings["key_insights"].append(
                f"High missing data in: {', '.join(missing_data['high_missing_columns'][:3])}"
            )
        
        # Forecast signals
        forecast_signals = {"available": False}
        
        if value_column and value_column in self.df.columns:
            values = self.df[value_column].dropna().values
            if len(values) >= 10:
                trend = self._analyze_trend(values)
                seasonality = self._analyze_seasonality(values)
                next_period = self._forecast_next_period(values)
                
                forecast_signals = {
                    "available": True,
                    "value_column": value_column,
                    "trend": trend,
                    "seasonality": seasonality,
                    "next_period": next_period,
                    "key_insights": []
                }
                
                if trend["direction"] != "stable":
                    forecast_signals["key_insights"].append(
                        f"Trend is {trend['direction']} with {abs(trend['percentage_change']):.1f}% change"
                    )
                
                if seasonality["has_seasonality"]:
                    forecast_signals["key_insights"].append(
                        f"Seasonal pattern detected (strength: {seasonality['strength']:.2f})"
                    )
        
        # ML findings (placeholder - would need ML results)
        ml_findings = {
            "available": False,
            "message": "Run ML analysis endpoints for ML insights"
        }
        
        # Risk analysis
        risks = self._analyze_subgroup_risks(group_column)
        
        # Recommended actions
        recommended_actions = []
        
        if missing_data["total_columns_affected"] > 0:
            recommended_actions.append({
                "priority": "high" if any(m["severity"] == "high" for m in missing_data["columns_with_missing"]) else "medium",
                "action": "Address missing data",
                "details": f"{missing_data['total_columns_affected']} columns have missing values"
            })
        
        if distributions["unusual_distributions"]:
            recommended_actions.append({
                "priority": "medium",
                "action": "Review unusual distributions",
                "details": f"{len(distributions['unusual_distributions'])} columns have unusual distributions"
            })
        
        if correlations["high_correlations"]:
            recommended_actions.append({
                "priority": "low",
                "action": "Review correlated variables",
                "details": f"{len(correlations['high_correlations'])} pairs of highly correlated variables found"
            })
        
        if risks.get("groups_with_risks", 0) > 0:
            recommended_actions.append({
                "priority": "medium",
                "action": "Review subgroup quality",
                "details": f"{risks['groups_with_risks']} groups have data quality concerns"
            })
        
        result = {
            "overview": overview,
            "descriptive_findings": descriptive_findings,
            "forecast_signals": forecast_signals,
            "ml_findings": ml_findings,
            "risks": risks,
            "recommended_actions": recommended_actions
        }
        
        self._log_operation("generate_full_insights", {
            "has_forecast": forecast_signals["available"],
            "n_recommendations": len(recommended_actions)
        })
        
        return self._make_json_safe(result)
