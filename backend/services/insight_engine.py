"""
StatFlow AI - Insight Engine
Combines local statistical analysis with GenAI-powered narrative generation.
Provides deterministic analytics with AI-enhanced interpretations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats
from datetime import datetime
import warnings

from services.file_manager import FileManager
from services.analysis_engine import AnalysisEngine
from services.genai_engine import GenAIEngine, GenAIError

warnings.filterwarnings('ignore')


class InsightEngine:
    """
    Automated insight generation engine combining local analytics with GenAI.
    
    Features:
    - Deterministic statistical analysis (MoSPI-compliant)
    - Anomaly and outlier detection
    - Risk indicator identification
    - GenAI-powered narrative generation
    - Fallback to structured summaries if GenAI unavailable
    - Cached GenAI instance (prevents double initialization)
    
    All outputs are JSON-safe (NaN/Inf converted to None).
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize the insight engine.
        
        Args:
            df: Optional DataFrame. If not provided, use generate_insights(filename) method.
        """
        self.df = df.copy() if df is not None else None
        self.operations_log = []
        self.genai: Optional[GenAIEngine] = None  # Cached GenAI instance
        
        if self.df is not None:
            self._log_operation("initialized", {
                "rows": len(self.df),
                "columns": len(self.df.columns)
            })
    
    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Helper method to log operations."""
        self.operations_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            **details
        })
    
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
    # HELPER FUNCTIONS FOR ANOMALY DETECTION
    # ==========================================
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers using Z-score and IQR methods (deterministic).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with outlier information per column
        """
        outliers_info = {}
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 10:
                continue
            
            # Z-score method (threshold = 3)
            z_scores = np.abs(stats.zscore(data))
            z_outliers = np.sum(z_scores > 3)
            
            # IQR method
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_outliers = np.sum((data < lower_bound) | (data > upper_bound))
            
            outlier_pct = (iqr_outliers / len(data)) * 100
            
            if iqr_outliers > 0:
                outliers_info[col] = {
                    "z_score_outliers": int(z_outliers),
                    "iqr_outliers": int(iqr_outliers),
                    "outlier_percentage": float(outlier_pct),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "severity": "high" if outlier_pct > 5 else "medium" if outlier_pct > 1 else "low"
                }
        
        return {
            "outliers_by_column": outliers_info,
            "total_columns_with_outliers": len(outliers_info),
            "columns_with_high_outliers": [
                col for col, info in outliers_info.items()
                if info["severity"] == "high"
            ]
        }
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect various data anomalies (deterministic).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with detected anomalies
        """
        anomalies = []
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                anomalies.append({
                    "type": "constant_column",
                    "column": col,
                    "message": f"Column '{col}' has only one unique value",
                    "severity": "medium"
                })
        
        # Check for high missing value patterns
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 50:
                anomalies.append({
                    "type": "high_missing",
                    "column": col,
                    "missing_percentage": float(missing_pct),
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values",
                    "severity": "high"
                })
        
        # Check for highly skewed distributions
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) >= 10:
                skewness = float(stats.skew(data))
                if abs(skewness) > 3:
                    anomalies.append({
                        "type": "extreme_skewness",
                        "column": col,
                        "skewness": skewness,
                        "message": f"Column '{col}' has extreme skewness ({skewness:.2f})",
                        "severity": "medium"
                    })
        
        # Check for categorical imbalance
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            if len(value_counts) > 1:
                max_pct = (value_counts.iloc[0] / value_counts.sum()) * 100
                if max_pct > 95:
                    anomalies.append({
                        "type": "categorical_imbalance",
                        "column": col,
                        "dominant_percentage": float(max_pct),
                        "message": f"Column '{col}' has one value representing {max_pct:.1f}% of data",
                        "severity": "medium"
                    })
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(df)) * 100
            anomalies.append({
                "type": "duplicate_rows",
                "count": int(duplicate_count),
                "percentage": float(duplicate_pct),
                "message": f"{duplicate_count} duplicate rows found ({duplicate_pct:.1f}%)",
                "severity": "high" if duplicate_pct > 10 else "medium"
            })
        
        # Check for contradictory values (basic logic)
        # Example: negative values where they shouldn't be
        for col in numeric_cols:
            if 'age' in col.lower() or 'count' in col.lower() or 'population' in col.lower():
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    anomalies.append({
                        "type": "logical_contradiction",
                        "column": col,
                        "count": int(negative_count),
                        "message": f"Column '{col}' has {negative_count} negative values (unexpected)",
                        "severity": "high"
                    })
        
        return {
            "anomalies": anomalies,
            "total_anomalies": len(anomalies),
            "high_severity_count": len([a for a in anomalies if a["severity"] == "high"]),
            "medium_severity_count": len([a for a in anomalies if a["severity"] == "medium"])
        }
    
    def detect_risks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data quality risks (deterministic).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with risk indicators
        """
        risks = []
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Small sample size risk
        if len(df) < 30:
            risks.append({
                "type": "small_sample",
                "value": len(df),
                "message": f"Small sample size (n={len(df)}) may limit statistical reliability",
                "severity": "high" if len(df) < 10 else "medium"
            })
        
        # High variance indicators
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                mean_val = data.mean()
                if mean_val != 0:
                    cv = data.std() / abs(mean_val)
                    if cv > 1.0:
                        risks.append({
                            "type": "high_variance",
                            "column": col,
                            "cv": float(cv),
                            "message": f"Column '{col}' has high coefficient of variation ({cv:.2f})",
                            "severity": "high" if cv > 2.0 else "medium"
                        })
        
        # Missing data patterns
        total_missing = df.isna().sum().sum()
        total_cells = len(df) * len(df.columns)
        missing_pct = (total_missing / total_cells) * 100
        
        if missing_pct > 10:
            risks.append({
                "type": "overall_missing_data",
                "percentage": float(missing_pct),
                "message": f"Overall {missing_pct:.1f}% of data is missing",
                "severity": "high" if missing_pct > 25 else "medium"
            })
        
        # Data quality score
        completeness_score = 100 - missing_pct
        outlier_info = self.detect_outliers(df)
        outlier_penalty = min(20, len(outlier_info["outliers_by_column"]) * 2)
        
        quality_score = max(0, completeness_score - outlier_penalty)
        
        return {
            "risks": risks,
            "total_risks": len(risks),
            "high_severity_count": len([r for r in risks if r["severity"] == "high"]),
            "data_quality_score": float(quality_score)
        }
    
    # ==========================================
    # MAIN INSIGHT GENERATION
    # ==========================================
    
    def generate_insights(self, filename: str) -> Dict[str, Any]:
        """
        Generate comprehensive insights combining local analytics with GenAI narrative.
        
        Process Flow:
        1. Resolve file path using FileManager
        2. Load DataFrame
        3. Compute statistical summary using AnalysisEngine
        4. Detect outliers, anomalies, and risks
        5. Build structured summary_dict
        6. Generate AI narrative (with fallback)
        7. Return complete response
        
        Args:
            filename: Name of the file to analyze
            
        Returns:
            Dictionary containing:
                - summary: Structured statistical summary
                - narrative: AI-generated narrative (or fallback)
                - status: Success/error status
                
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If DataFrame is empty or invalid
        """
        try:
            # Step 1: Resolve best file path using FileManager
            self._log_operation("resolve_file", {"filename": filename})
            file_path = FileManager.get_best_available_file(filename)
            
            # Step 2: Load DataFrame (handle different file types)
            self._log_operation("load_data", {"file_path": file_path})
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, low_memory=False)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                # Default to CSV
                df = pd.read_csv(file_path, low_memory=False)
            
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            self.df = df
            self._log_operation("data_loaded", {
                "rows": len(df),
                "columns": len(df.columns)
            })
            
            # Step 3: Compute summary using AnalysisEngine (static method)
            self._log_operation("compute_statistics", {})
            
            try:
                # Use AnalysisEngine static method
                analysis = AnalysisEngine()
                stats_result = analysis.generate_statistics(filename)
            except Exception as e:
                # If AnalysisEngine fails, fall back to manual statistics
                self._log_operation("analysis_engine_fallback", {"reason": str(e)})
                stats_result = {
                    "descriptive_stats": {},
                    "frequencies": {},
                    "crosstabs": {},
                    "weighted_stats": {},
                    "distribution_notes": {}
                }
            
            # Build basic statistics
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            
            key_stats = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "missing_data": df.isna().sum().to_dict(),
                "dtypes": df.dtypes.astype(str).to_dict()
            }
            
            # Add descriptive statistics from AnalysisEngine or compute manually
            if stats_result.get("descriptive_stats"):
                key_stats["numeric_stats"] = stats_result["descriptive_stats"]
            elif numeric_cols:
                # Fallback: compute manually if AnalysisEngine failed
                numeric_df = df[numeric_cols]
                key_stats["numeric_stats"] = {
                    "means": numeric_df.mean().to_dict(),
                    "medians": numeric_df.median().to_dict(),
                    "std_devs": numeric_df.std().to_dict(),
                    "mins": numeric_df.min().to_dict(),
                    "maxs": numeric_df.max().to_dict()
                }
            
            # Add frequency distributions if available
            if stats_result.get("frequencies"):
                key_stats["frequencies"] = stats_result["frequencies"]
            
            # Add distribution notes if available
            if stats_result.get("distribution_notes"):
                key_stats["distribution_notes"] = stats_result["distribution_notes"]
            
            # Step 4: Detect anomalies (outliers, missing patterns, variance, imbalance, contradictions)
            self._log_operation("detect_anomalies", {})
            outliers = self.detect_outliers(df)
            anomalies = self.detect_anomalies(df)
            risk_indicators = self.detect_risks(df)
            
            # Step 5: Build summary_dict with required structure
            summary_dict = {
                "dataset_overview": {
                    "filename": filename,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_variables": len(numeric_cols),
                    "categorical_variables": len(categorical_cols),
                    "total_cells": len(df) * len(df.columns),
                    "generated_at": datetime.now().isoformat()
                },
                "key_stats": key_stats,
                "outliers": outliers,
                "anomalies": anomalies,
                "risk_indicators": risk_indicators
            }
            
            # Add weighted summary if weight column exists (optional)
            potential_weight_cols = [col for col in df.columns if 'weight' in col.lower()]
            if potential_weight_cols:
                summary_dict["weighted_summary"] = {
                    "weight_columns_detected": potential_weight_cols,
                    "message": "Weighted analysis available - use weighting endpoints"
                }
            
            # Add margin of error indicators (optional)
            if len(df) >= 30:
                # Simple MoE calculation for proportions at 95% confidence
                moe_95 = 1.96 * np.sqrt(0.25 / len(df))  # Worst case p=0.5
                summary_dict["moe_indicators"] = {
                    "sample_size": len(df),
                    "moe_95_percent": float(moe_95 * 100),
                    "confidence_level": "95%",
                    "message": f"Margin of error: ±{moe_95 * 100:.2f}% at 95% confidence"
                }
            
            # Make summary JSON-safe (remove NaN/Inf)
            summary_dict = self._make_json_safe(summary_dict)
            
            # Step 6: Initialize GenAIEngine (PATCH 7: Cached Instance Logic)
            # Step 7: Generate narrative with fallback
            narrative = None
            genai_status = "not_attempted"
            
            try:
                self._log_operation("generate_narrative", {"method": "genai"})
                
                # Check if instance exists, else create and cache
                if not getattr(self, "genai", None):
                    self.genai = GenAIEngine()
                
                # Use the cached instance
                narrative = self.genai.generate_narrative(summary_dict)
                genai_status = "success"
                self._log_operation("narrative_generated", {"method": "genai"})
            
            except (ValueError, GenAIError) as e:
                # API key not configured or GenAI error
                genai_status = "api_key_missing" if isinstance(e, ValueError) else "error"
                narrative = "Text generation unavailable. See structured summary instead."
                self._log_operation("narrative_fallback", {"reason": str(e)})
            except Exception as e:
                # Any other error - use fallback
                genai_status = "error"
                narrative = "Text generation unavailable. See structured summary instead."
                self._log_operation("narrative_fallback", {"reason": str(e)})
            
            # Step 8: Return full response
            result = {
                "status": "success",
                "summary": summary_dict,
                "narrative": narrative,
                "genai_status": genai_status,
                "operations_log": self.operations_log
            }
            
            self._log_operation("generate_insights_complete", {
                "genai_status": genai_status
            })
            
            return result
            
        except FileNotFoundError as e:
            self._log_operation("error", {"type": "file_not_found", "message": str(e)})
            raise FileNotFoundError(str(e))
            
        except ValueError as e:
            self._log_operation("error", {"type": "invalid_data", "message": str(e)})
            raise ValueError(str(e))
            
        except Exception as e:
            self._log_operation("error", {"type": "unexpected", "message": str(e)})
            raise Exception(f"Failed to generate insights: {str(e)}")
    
    # ==========================================
    # LEGACY COMPATIBILITY METHODS
    # ==========================================
    
    def generate_overview(self) -> Dict[str, Any]:
        """
        Legacy method: Generate basic overview of the DataFrame.
        Maintained for backward compatibility with existing routers.
        
        Returns:
            Dictionary with dataset overview
        """
        if self.df is None:
            raise ValueError("DataFrame not initialized. Use generate_insights(filename) instead.")
        
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=['number']).columns.tolist()
        
        overview = {
            "shape": {
                "rows": len(self.df),
                "columns": len(self.df.columns)
            },
            "columns": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "all": self.df.columns.tolist()
            },
            "data_types": self.df.dtypes.astype(str).to_dict(),
            "missing_summary": {
                "by_column": self.df.isna().sum().to_dict(),
                "total_missing": int(self.df.isna().sum().sum()),
                "completeness_pct": float(100 - (self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns)) * 100))
            }
        }
        
        # Add quick numeric summary if available
        if numeric_cols:
            overview["numeric_summary"] = self.df[numeric_cols].describe().to_dict()
        
        return self._make_json_safe(overview)
    
    def generate_full_insights(
        self,
        time_column: Optional[str] = None,
        value_column: Optional[str] = None,
        group_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Legacy method: Generate full insights with optional parameters.
        Maintained for backward compatibility with existing routers.
        
        Args:
            time_column: Optional time column for trend analysis
            value_column: Optional value column for analysis
            group_column: Optional grouping column
            
        Returns:
            Dictionary with comprehensive insights
        """
        if self.df is None:
            raise ValueError("DataFrame not initialized. Use generate_insights(filename) instead.")
        
        # Use the helper methods to build insights
        overview = self.generate_overview()
        outliers = self.detect_outliers(self.df)
        anomalies = self.detect_anomalies(self.df)
        risks = self.detect_risks(self.df)
        
        result = {
            "overview": overview,
            "outliers": outliers,
            "anomalies": anomalies,
            "risks": risks,
            "parameters": {
                "time_column": time_column,
                "value_column": value_column,
                "group_column": group_column
            }
        }
        
        return self._make_json_safe(result)
    
    def get_operations_log(self) -> list:
        """Return the operations log."""
        return self.operations_log