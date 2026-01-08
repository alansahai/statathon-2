"""
Data Cleaning Engine - Core business logic for data cleaning operations
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


def make_json_safe(data: Any) -> Any:
    """
    Convert data to JSON-safe format, handling NaN and Inf values
    
    Args:
        data: Data to convert
        
    Returns:
        JSON-safe data
    """
    if isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, dict):
        return {k: make_json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_safe(x) for x in data]
    elif isinstance(data, (np.ndarray, pd.Series)):
        return [make_json_safe(x) for x in data.tolist()]
    elif pd.isna(data):
        return None
    return data


class CleaningEngine:
    """
    Engine for performing automatic and manual data cleaning operations
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the cleaning engine with a DataFrame
        
        Args:
            df: Input DataFrame to clean
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.logs: List[Dict[str, Any]] = []
        self.issues: Dict[str, Any] = {}
        
    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """
        Log a cleaning operation
        
        Args:
            operation: Name of the operation
            details: Details about the operation
        """
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "details": details
        })
    
    def detect_issues(self) -> Dict[str, Any]:
        """
        Detect data quality issues in the dataset
        
        Returns:
            Dictionary containing detected issues
        """
        issues = {
            "missing_summary": {},
            "numeric_summary": {},
            "categorical_summary": {},
            "potential_id_columns": []
        }
        
        # Missing value analysis
        for col in self.df.columns:
            missing_count = int(self.df[col].isnull().sum())
            missing_percent = (missing_count / len(self.df) * 100) if len(self.df) > 0 else 0
            
            if missing_count > 0:
                issues["missing_summary"][col] = {
                    "missing_count": missing_count,
                    "missing_percent": round(missing_percent, 2),
                    "non_missing_count": len(self.df) - missing_count
                }
        
        # Numeric column analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            non_null = self.df[col].dropna()
            if len(non_null) > 0:
                issues["numeric_summary"][col] = {
                    "dtype": str(self.df[col].dtype),
                    "count": int(len(non_null)),
                    "mean": float(non_null.mean()),
                    "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                    "min": float(non_null.min()),
                    "q1": float(non_null.quantile(0.25)),
                    "median": float(non_null.median()),
                    "q3": float(non_null.quantile(0.75)),
                    "max": float(non_null.max()),
                    "unique_count": int(non_null.nunique())
                }
        
        # Categorical column analysis
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            non_null = self.df[col].dropna()
            if len(non_null) > 0:
                value_counts = non_null.value_counts().head(10)
                issues["categorical_summary"][col] = {
                    "dtype": str(self.df[col].dtype),
                    "count": int(len(non_null)),
                    "unique_count": int(non_null.nunique()),
                    "top_values": value_counts.to_dict(),
                    "most_common": str(non_null.mode().iloc[0]) if len(non_null.mode()) > 0 else None
                }
        
        # Detect potential ID columns
        for col in self.df.columns:
            non_null = self.df[col].dropna()
            if len(non_null) > 0 and non_null.nunique() == len(non_null):
                issues["potential_id_columns"].append(col)
        
        self.issues = issues
        self._log_operation("detect_issues", {
            "missing_columns": len(issues["missing_summary"]),
            "numeric_columns": len(issues["numeric_summary"]),
            "categorical_columns": len(issues["categorical_summary"]),
            "potential_id_columns": len(issues["potential_id_columns"])
        })
        
        return make_json_safe(issues)
    
    def impute_missing(
        self, 
        method: str = "auto", 
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Impute missing values in the DataFrame
        
        Args:
            method: Imputation method ("auto", "median", "mean", "mode", "forward", "backward")
            columns: Specific columns to impute (None = all columns with missing values)
            
        Returns:
            DataFrame with imputed values
        """
        if columns is None:
            columns = self.df.columns[self.df.isnull().any()].tolist()
        
        # Get potential ID columns to skip
        if not self.issues:
            self.detect_issues()
        id_columns = self.issues.get("potential_id_columns", [])
        
        imputation_log = []
        
        for col in columns:
            # Skip ID columns
            if col in id_columns:
                imputation_log.append({
                    "column": col,
                    "method": "skipped",
                    "reason": "potential_id_column"
                })
                continue
            
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            try:
                if method == "auto":
                    # Automatic method selection based on dtype
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        # Use median for numeric
                        fill_value = self.df[col].median()
                        self.df[col].fillna(fill_value, inplace=True)
                        imputation_log.append({
                            "column": col,
                            "method": "median",
                            "fill_value": float(fill_value),
                            "imputed_count": int(missing_count)
                        })
                    else:
                        # Use mode for categorical
                        mode_values = self.df[col].mode()
                        if len(mode_values) > 0:
                            fill_value = mode_values.iloc[0]
                            self.df[col].fillna(fill_value, inplace=True)
                            imputation_log.append({
                                "column": col,
                                "method": "mode",
                                "fill_value": str(fill_value),
                                "imputed_count": int(missing_count)
                            })
                
                elif method == "median":
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        fill_value = self.df[col].median()
                        self.df[col].fillna(fill_value, inplace=True)
                        imputation_log.append({
                            "column": col,
                            "method": "median",
                            "fill_value": float(fill_value),
                            "imputed_count": int(missing_count)
                        })
                
                elif method == "mean":
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        fill_value = self.df[col].mean()
                        self.df[col].fillna(fill_value, inplace=True)
                        imputation_log.append({
                            "column": col,
                            "method": "mean",
                            "fill_value": float(fill_value),
                            "imputed_count": int(missing_count)
                        })
                
                elif method == "mode":
                    mode_values = self.df[col].mode()
                    if len(mode_values) > 0:
                        fill_value = mode_values.iloc[0]
                        self.df[col].fillna(fill_value, inplace=True)
                        imputation_log.append({
                            "column": col,
                            "method": "mode",
                            "fill_value": str(fill_value),
                            "imputed_count": int(missing_count)
                        })
                
                elif method == "forward":
                    self.df[col].fillna(method='ffill', inplace=True)
                    imputation_log.append({
                        "column": col,
                        "method": "forward_fill",
                        "imputed_count": int(missing_count)
                    })
                
                elif method == "backward":
                    self.df[col].fillna(method='bfill', inplace=True)
                    imputation_log.append({
                        "column": col,
                        "method": "backward_fill",
                        "imputed_count": int(missing_count)
                    })
                    
            except Exception as e:
                imputation_log.append({
                    "column": col,
                    "method": method,
                    "status": "failed",
                    "error": str(e)
                })
        
        self._log_operation("impute_missing", {
            "method": method,
            "columns_processed": len(imputation_log),
            "details": imputation_log
        })
        
        return self.df
    
    def detect_outliers(
        self, 
        method: str = "iqr", 
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns
        
        Args:
            method: Detection method ("iqr" or "zscore")
            columns: Specific columns to check (None = all numeric columns)
            
        Returns:
            Dictionary containing outlier information
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_summary = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            non_null = self.df[col].dropna()
            if len(non_null) < 4:  # Need at least 4 values for IQR
                continue
            
            try:
                if method == "iqr":
                    Q1 = non_null.quantile(0.25)
                    Q3 = non_null.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
                    outlier_count = len(outliers)
                    
                    outlier_summary[col] = {
                        "method": "iqr",
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "q1": float(Q1),
                        "q3": float(Q3),
                        "iqr": float(IQR),
                        "outlier_count": int(outlier_count),
                        "outlier_percent": round(outlier_count / len(non_null) * 100, 2),
                        "outlier_values": outliers.head(20).tolist() if outlier_count > 0 else []
                    }
                
                elif method == "zscore":
                    mean = non_null.mean()
                    std = non_null.std()
                    if std > 0:
                        z_scores = np.abs((non_null - mean) / std)
                        outliers = non_null[z_scores > 3]
                        outlier_count = len(outliers)
                        
                        outlier_summary[col] = {
                            "method": "zscore",
                            "mean": float(mean),
                            "std": float(std),
                            "threshold": 3.0,
                            "outlier_count": int(outlier_count),
                            "outlier_percent": round(outlier_count / len(non_null) * 100, 2),
                            "outlier_values": outliers.head(20).tolist() if outlier_count > 0 else []
                        }
                        
            except Exception as e:
                outlier_summary[col] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        self._log_operation("detect_outliers", {
            "method": method,
            "columns_analyzed": len(outlier_summary),
            "total_outliers": sum(info.get("outlier_count", 0) for info in outlier_summary.values())
        })
        
        return make_json_safe(outlier_summary)
    
    def fix_outliers(
        self, 
        method: str = "iqr", 
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fix outliers by capping to bounds
        
        Args:
            method: Detection method ("iqr" or "zscore")
            columns: Specific columns to fix (None = all numeric columns)
            
        Returns:
            DataFrame with fixed outliers
        """
        outlier_info = self.detect_outliers(method=method, columns=columns)
        
        fix_log = []
        
        for col, info in outlier_info.items():
            if "status" in info and info["status"] == "failed":
                continue
            
            if info.get("outlier_count", 0) == 0:
                continue
            
            try:
                if method == "iqr":
                    lower_bound = info["lower_bound"]
                    upper_bound = info["upper_bound"]
                    
                    # Count values that will be changed
                    below_count = (self.df[col] < lower_bound).sum()
                    above_count = (self.df[col] > upper_bound).sum()
                    
                    # Cap values
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    fix_log.append({
                        "column": col,
                        "method": "iqr_capping",
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "values_capped_below": int(below_count),
                        "values_capped_above": int(above_count),
                        "total_fixed": int(below_count + above_count)
                    })
                
                elif method == "zscore":
                    mean = info["mean"]
                    std = info["std"]
                    threshold = info["threshold"]
                    
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    
                    below_count = (self.df[col] < lower_bound).sum()
                    above_count = (self.df[col] > upper_bound).sum()
                    
                    # Cap values
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    fix_log.append({
                        "column": col,
                        "method": "zscore_capping",
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "values_capped_below": int(below_count),
                        "values_capped_above": int(above_count),
                        "total_fixed": int(below_count + above_count)
                    })
                    
            except Exception as e:
                fix_log.append({
                    "column": col,
                    "status": "failed",
                    "error": str(e)
                })
        
        self._log_operation("fix_outliers", {
            "method": method,
            "columns_fixed": len(fix_log),
            "details": fix_log
        })
        
        return self.df
    
    def apply_rules(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply manual cleaning rules to the DataFrame
        
        Args:
            rules: Dictionary containing range_rules, regex_rules, and conditional_rules
            
        Returns:
            Summary of rule application results
        """
        results = {
            "range_rules_applied": [],
            "regex_rules_applied": [],
            "conditional_rules_applied": []
        }
        
        # Apply range rules
        if "range_rules" in rules:
            for col, bounds in rules["range_rules"].items():
                if col not in self.df.columns:
                    results["range_rules_applied"].append({
                        "column": col,
                        "status": "failed",
                        "reason": "column_not_found"
                    })
                    continue
                
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    results["range_rules_applied"].append({
                        "column": col,
                        "status": "failed",
                        "reason": "not_numeric"
                    })
                    continue
                
                try:
                    violations = 0
                    min_val = bounds.get("min")
                    max_val = bounds.get("max")
                    
                    if min_val is not None:
                        violation_mask = self.df[col] < min_val
                        violations += violation_mask.sum()
                        self.df.loc[violation_mask, col] = None
                    
                    if max_val is not None:
                        violation_mask = self.df[col] > max_val
                        violations += violation_mask.sum()
                        self.df.loc[violation_mask, col] = None
                    
                    results["range_rules_applied"].append({
                        "column": col,
                        "status": "success",
                        "bounds": bounds,
                        "violations_found": int(violations)
                    })
                    
                except Exception as e:
                    results["range_rules_applied"].append({
                        "column": col,
                        "status": "failed",
                        "error": str(e)
                    })
        
        # Apply regex rules
        if "regex_rules" in rules:
            for col, pattern in rules["regex_rules"].items():
                if col not in self.df.columns:
                    results["regex_rules_applied"].append({
                        "column": col,
                        "status": "failed",
                        "reason": "column_not_found"
                    })
                    continue
                
                try:
                    violations = 0
                    compiled_pattern = re.compile(pattern)
                    
                    for idx, value in self.df[col].items():
                        if pd.notna(value):
                            if not compiled_pattern.match(str(value)):
                                self.df.at[idx, col] = None
                                violations += 1
                    
                    results["regex_rules_applied"].append({
                        "column": col,
                        "status": "success",
                        "pattern": pattern,
                        "violations_found": violations
                    })
                    
                except Exception as e:
                    results["regex_rules_applied"].append({
                        "column": col,
                        "status": "failed",
                        "error": str(e)
                    })
        
        # Apply conditional rules
        if "conditional_rules" in rules:
            for rule_expr in rules["conditional_rules"]:
                try:
                    # Evaluate the condition
                    valid_mask = self.df.eval(rule_expr)
                    violations = (~valid_mask).sum()
                    
                    results["conditional_rules_applied"].append({
                        "rule": rule_expr,
                        "status": "success",
                        "violations_found": int(violations),
                        "valid_rows": int(valid_mask.sum())
                    })
                    
                except Exception as e:
                    results["conditional_rules_applied"].append({
                        "rule": rule_expr,
                        "status": "failed",
                        "error": str(e)
                    })
        
        self._log_operation("apply_rules", {
            "range_rules_count": len(results["range_rules_applied"]),
            "regex_rules_count": len(results["regex_rules_applied"]),
            "conditional_rules_count": len(results["conditional_rules_applied"])
        })
        
        return make_json_safe(results)
    
    def auto_clean(self) -> Dict[str, Any]:
        """
        Perform automatic cleaning pipeline
        
        Returns:
            Comprehensive summary of all cleaning operations
        """
        # Step 1: Detect issues
        issue_summary = self.detect_issues()
        
        # Step 2: Impute missing values
        self.impute_missing(method="auto")
        
        # Step 3: Detect outliers
        outlier_summary = self.detect_outliers(method="iqr")
        
        # Step 4: Fix outliers
        self.fix_outliers(method="iqr")
        
        # Generate preview of cleaned data
        cleaned_preview = self.df.head(10).to_dict(orient="records")
        
        summary = {
            "status": "success",
            "cleaning_logs": self.logs,
            "issue_summary": issue_summary,
            "outlier_summary": outlier_summary,
            "cleaned_preview": cleaned_preview,
            "row_count": len(self.df),
            "column_count": len(self.df.columns),
            "original_row_count": len(self.original_df),
            "rows_dropped": len(self.original_df) - len(self.df)
        }
        
        return make_json_safe(summary)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of current DataFrame state
        
        Returns:
            Summary dictionary with statistics and metadata
        """
        summary = {
            "row_count": len(self.df),
            "column_count": len(self.df.columns),
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": {
                col: int(self.df[col].isnull().sum()) 
                for col in self.df.columns 
                if self.df[col].isnull().sum() > 0
            },
            "total_missing": int(self.df.isnull().sum().sum()),
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / (1024 * 1024)),
            "operations_performed": len(self.logs),
            "cleaning_logs": self.logs[-5:] if len(self.logs) > 5 else self.logs  # Last 5 operations
        }
        
        return make_json_safe(summary)
