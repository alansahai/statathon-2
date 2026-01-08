"""
StatFlow AI - Analysis Engine
Provides comprehensive statistical analysis capabilities for survey data with weighted/unweighted support.
Production-ready implementation matching CleaningEngine and WeightingEngine conventions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from scipy import stats
from scipy.stats import chi2_contingency
from datetime import datetime
import warnings

from services.stats_engine import StatisticalTestEngine

warnings.filterwarnings('ignore')


class AnalysisEngine:
    """
    Comprehensive statistical analysis engine for survey data.
    
    Supports:
    - Descriptive statistics (weighted/unweighted)
    - Cross-tabulations with multiple normalization modes
    - OLS regression with weighted/unweighted variants
    - Subgroup analysis with flexible metrics
    
    All outputs are JSON-safe (NaN/Inf converted to None).
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analysis engine with a DataFrame.
        
        Args:
            df: Input DataFrame containing survey data
        """
        self.df = df.copy()
        self.operations_log = []
        self.stats = StatisticalTestEngine()
        self._log_operation("initialized", {"rows": len(df), "columns": len(df.columns)})
    
    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Helper method to log operations."""
        self.operations_log.append({
            "operation": operation,
            **details
        })
    
    def descriptive_stats(
        self, 
        columns: List[str], 
        weight_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive descriptive statistics for specified columns.
        
        Args:
            columns: List of column names to analyze
            weight_column: Optional weight column for weighted statistics
            
        Returns:
            Dictionary with statistics for each column:
            - For numeric: mean, median, std, min, max, q1, q3, skewness, kurtosis
            - For categorical: frequencies, percentages, mode
            - missing_count for all columns
        """
        results = {}
        
        for col in columns:
            if col not in self.df.columns:
                results[col] = {"error": f"Column '{col}' not found"}
                continue
            
            col_data = self.df[col]
            missing_count = col_data.isna().sum()
            
            # Determine if column is numeric or categorical
            if pd.api.types.is_numeric_dtype(col_data):
                results[col] = self._compute_numeric_stats(col, weight_column, missing_count)
            else:
                results[col] = self._compute_categorical_stats(col, weight_column, missing_count)
        
        self.operations_log.append({
            "operation": "descriptive_stats",
            "columns": columns,
            "weighted": weight_column is not None
        })
        
        return self._make_json_safe(results)
    
    def _compute_numeric_stats(
        self, 
        column: str, 
        weight_column: Optional[str], 
        missing_count: int
    ) -> Dict[str, Any]:
        """Compute comprehensive statistics for numeric columns with warnings."""
        col_data = self.df[column].dropna()
        
        if len(col_data) == 0:
            return {
                "dtype": "numeric",
                "count": 0,
                "missing": int(missing_count),
                "error": "All values are missing"
            }
        
        # Zero and negative detection
        zero_count = int((col_data == 0).sum())
        negative_count = int((col_data < 0).sum())
        warnings_list = []
        
        if weight_column and weight_column in self.df.columns:
            # Weighted statistics
            weights = self.df.loc[col_data.index, weight_column]
            weights = weights.fillna(0)
            
            # Remove rows where weight is 0 or NaN
            valid_mask = (weights > 0) & (~weights.isna())
            col_data = col_data[valid_mask]
            weights = weights[valid_mask]
            
            if len(col_data) == 0:
                return {
                    "type": "numeric",
                    "missing_count": int(missing_count),
                    "error": "No valid weighted observations"
                }
            
            # Normalize weights
            weights = weights / weights.sum()
            
            mean = np.average(col_data, weights=weights)
            variance = np.average((col_data - mean) ** 2, weights=weights)
            std = np.sqrt(variance)
            
            # Weighted quantiles (full percentile distribution)
            sorted_indices = np.argsort(col_data)
            sorted_data = col_data.iloc[sorted_indices].values
            sorted_weights = weights.iloc[sorted_indices].values
            cumsum = np.cumsum(sorted_weights)
            
            percentiles = {
                "p1": float(np.interp(0.01, cumsum, sorted_data)),
                "p5": float(np.interp(0.05, cumsum, sorted_data)),
                "p10": float(np.interp(0.10, cumsum, sorted_data)),
                "p25": float(np.interp(0.25, cumsum, sorted_data)),
                "p50": float(np.interp(0.50, cumsum, sorted_data)),
                "p75": float(np.interp(0.75, cumsum, sorted_data)),
                "p90": float(np.interp(0.90, cumsum, sorted_data)),
                "p95": float(np.interp(0.95, cumsum, sorted_data)),
                "p99": float(np.interp(0.99, cumsum, sorted_data))
            }
            median = percentiles["p50"]
            q1 = percentiles["p25"]
            q3 = percentiles["p75"]
            iqr = q3 - q1
            
            # Weighted skewness and kurtosis
            m3 = np.average((col_data - mean) ** 3, weights=weights)
            m4 = np.average((col_data - mean) ** 4, weights=weights)
            skewness = m3 / (std ** 3) if std > 0 else 0
            kurtosis = (m4 / (std ** 4)) - 3 if std > 0 else 0
            
            weighted_mean = float(mean)
            weighted_std = float(std)
            weighted_median = float(median)
            weighted_variance = float(variance)
            
        else:
            # Unweighted statistics
            weighted_mean = None
            weighted_std = None
            weighted_median = None
            weighted_variance = None
            
            skewness = float(col_data.skew())
            kurtosis = float(col_data.kurtosis())
            
            percentiles = {
                "p1": float(col_data.quantile(0.01)),
                "p5": float(col_data.quantile(0.05)),
                "p10": float(col_data.quantile(0.10)),
                "p25": float(col_data.quantile(0.25)),
                "p50": float(col_data.quantile(0.50)),
                "p75": float(col_data.quantile(0.75)),
                "p90": float(col_data.quantile(0.90)),
                "p95": float(col_data.quantile(0.95)),
                "p99": float(col_data.quantile(0.99))
            }
            iqr = percentiles["p75"] - percentiles["p25"]
        
        # Unweighted basics (always computed)
        mean = float(col_data.mean())
        median = float(col_data.median())
        std = float(col_data.std())
        variance = float(col_data.var())
        
        # Distribution warnings
        if abs(skewness) > 2:
            warnings_list.append(f"High skewness detected ({skewness:.2f})")
        if abs(kurtosis) > 5:
            warnings_list.append(f"High kurtosis detected ({kurtosis:.2f})")
        if std < 0.01 * abs(mean) and mean != 0:
            warnings_list.append("Near-constant column detected (very low variance)")
        if negative_count > len(col_data) * 0.1:
            warnings_list.append(f"{negative_count} negative values detected ({negative_count/len(col_data)*100:.1f}%)")
        if zero_count > len(col_data) * 0.3:
            warnings_list.append(f"{zero_count} zero values detected ({zero_count/len(col_data)*100:.1f}%)")
        
        result = {
            "column": column,
            "dtype": "numeric",
            "count": int(len(col_data)),
            "missing": int(missing_count),
            "zero_count": zero_count,
            "negative_count": negative_count,
            "mean": mean,
            "median": median,
            "std": std,
            "variance": variance,
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "iqr": float(iqr),
            "percentiles": percentiles,
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "warnings": warnings_list
        }
        
        if weight_column:
            result.update({
                "weighted_mean": weighted_mean,
                "weighted_median": weighted_median,
                "weighted_std": weighted_std,
                "weighted_variance": weighted_variance
            })
        
        return result
    
    def _compute_vif(self, X: np.ndarray, var_names: List[str]) -> Dict[str, float]:
        """Compute Variance Inflation Factors for multicollinearity detection."""
        n, k = X.shape
        vif = {}
        
        for i in range(k):
            # Regress X[:, i] on all other X columns
            X_i = X[:, i]
            X_other = np.delete(X, i, axis=1)
            
            try:
                # Add intercept to other predictors
                X_other_with_intercept = np.column_stack([np.ones(n), X_other])
                beta = np.linalg.lstsq(X_other_with_intercept, X_i, rcond=None)[0]
                fitted = X_other_with_intercept @ beta
                
                # Compute R²
                ss_res = np.sum((X_i - fitted) ** 2)
                ss_tot = np.sum((X_i - X_i.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # VIF = 1 / (1 - R²)
                if r_squared < 0.9999:  # Avoid division by near-zero
                    vif[var_names[i]] = 1 / (1 - r_squared)
                else:
                    vif[var_names[i]] = 999.0  # Signal very high multicollinearity
            except:
                vif[var_names[i]] = None  # Could not compute
        
        return vif
    
    def _compute_correlation_matrix(self, X: np.ndarray, var_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute correlation matrix for predictors."""
        corr_matrix = {}
        k = X.shape[1]
        
        for i in range(k):
            corr_matrix[var_names[i]] = {}
            for j in range(k):
                if i == j:
                    corr_matrix[var_names[i]][var_names[j]] = 1.0
                else:
                    corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                    corr_matrix[var_names[i]][var_names[j]] = float(corr) if not np.isnan(corr) else 0.0
        
        return corr_matrix
    
    def _compute_robust_se(self, X: np.ndarray, residuals: np.ndarray, XtX_inv: np.ndarray) -> np.ndarray:
        """Compute HC3 heteroskedasticity-consistent standard errors."""
        n = len(residuals)
        
        # Compute leverage (diagonal of hat matrix)
        leverage = np.sum((X @ XtX_inv) * X, axis=1)
        
        # HC3 adjustment: divide by (1 - h)²
        adjustment = residuals ** 2 / ((1 - leverage) ** 2)
        
        # Sandwich estimator: (X'X)^(-1) X' Ω X (X'X)^(-1)
        Omega = np.diag(adjustment)
        meat = X.T @ Omega @ X
        robust_var = XtX_inv @ meat @ XtX_inv
        
        return np.sqrt(np.diag(robust_var))
    
    def _compute_robust_se_weighted(self, X: np.ndarray, residuals: np.ndarray, 
                                     weights: np.ndarray, XtWX_inv: np.ndarray) -> np.ndarray:
        """Compute weighted HC3 robust standard errors."""
        n = len(residuals)
        W = np.diag(weights)
        
        # Weighted leverage
        leverage = np.sum((X @ XtWX_inv @ (X.T @ W).T) * X, axis=1)
        
        # HC3 adjustment with weights
        adjustment = weights * (residuals ** 2) / ((1 - leverage) ** 2)
        
        Omega = np.diag(adjustment)
        meat = X.T @ Omega @ X
        robust_var = XtWX_inv @ meat @ XtWX_inv
        
        return np.sqrt(np.diag(robust_var))
    
    def _run_logistic(self, X: np.ndarray, y: np.ndarray, var_names: List[str], 
                     max_iter: int = 25, tol: float = 1e-6) -> Dict[str, Any]:
        """Run logistic regression using Newton-Raphson method."""
        n, k = X.shape
        
        # Check if y is binary
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            return {"error": "Logistic regression requires binary dependent variable"}
        
        # Recode y to 0/1 if needed
        if not (set(unique_y) <= {0, 1}):
            y = (y == unique_y[1]).astype(int)
        
        # Initialize coefficients
        beta = np.zeros(k)
        
        # Newton-Raphson iteration
        for iteration in range(max_iter):
            # Predicted probabilities
            linear_pred = X @ beta
            # Clip to avoid overflow in exp
            linear_pred = np.clip(linear_pred, -500, 500)
            probs = 1 / (1 + np.exp(-linear_pred))
            
            # Gradient (score)
            gradient = X.T @ (y - probs)
            
            # Hessian (information matrix)
            W = np.diag(probs * (1 - probs))
            hessian = -X.T @ W @ X
            
            # Update
            try:
                delta = np.linalg.solve(-hessian, gradient)
            except np.linalg.LinAlgError:
                return {"error": "Singular Hessian matrix in logistic regression"}
            
            beta = beta + delta
            
            # Check convergence
            if np.max(np.abs(delta)) < tol:
                break
        
        # Final predictions
        linear_pred = X @ beta
        linear_pred = np.clip(linear_pred, -500, 500)
        probs = 1 / (1 + np.exp(-linear_pred))
        
        # Standard errors (from inverse Hessian)
        W = np.diag(probs * (1 - probs))
        info_matrix = X.T @ W @ X
        try:
            var_beta = np.linalg.inv(info_matrix)
            std_errors = np.sqrt(np.diag(var_beta))
        except:
            std_errors = np.full(k, np.nan)
        
        # z-statistics and p-values
        z_stats = beta / std_errors
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
        
        # Pseudo R² (McFadden)
        null_prob = y.mean()
        null_ll = np.sum(y * np.log(null_prob + 1e-10) + (1 - y) * np.log(1 - null_prob + 1e-10))
        model_ll = np.sum(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
        pseudo_r2 = 1 - (model_ll / null_ll) if null_ll != 0 else 0
        
        return {
            "coefficients": {var_names[i]: float(beta[i]) for i in range(k)},
            "std_errors": {var_names[i]: float(std_errors[i]) for i in range(k)},
            "z_statistics": {var_names[i]: float(z_stats[i]) for i in range(k)},
            "p_values": {var_names[i]: float(p_values[i]) for i in range(k)},
            "pseudo_r_squared": float(pseudo_r2),
            "n_obs": int(n),
            "converged": iteration < max_iter - 1
        }
    
    def _compute_categorical_stats(
        self, 
        column: str, 
        weight_column: Optional[str], 
        missing_count: int
    ) -> Dict[str, Any]:
        """Compute statistics for categorical columns."""
        col_data = self.df[column].dropna()
        
        if len(col_data) == 0:
            return {
                "type": "categorical",
                "missing_count": int(missing_count),
                "error": "All values are missing"
            }
        
        if weight_column and weight_column in self.df.columns:
            # Weighted frequencies
            weights = self.df.loc[col_data.index, weight_column]
            weights = weights.fillna(0)
            
            freq_dict = {}
            for category in col_data.unique():
                mask = col_data == category
                freq_dict[str(category)] = float(weights[mask].sum())
            
            total = sum(freq_dict.values())
            percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in freq_dict.items()}
            mode = max(freq_dict, key=freq_dict.get)
            
        else:
            # Unweighted frequencies
            value_counts = col_data.value_counts()
            freq_dict = {str(k): int(v) for k, v in value_counts.items()}
            total = len(col_data)
            percentages = {k: (v / total * 100) for k, v in freq_dict.items()}
            mode = str(col_data.mode()[0]) if len(col_data.mode()) > 0 else None
        
        return {
            "column": column,
            "dtype": "categorical",
            "count": int(len(col_data)),
            "missing": int(missing_count),
            "unique_values": int(len(freq_dict)),
            "frequencies": freq_dict,
            "percentages": percentages,
            "mode": mode,
            "warnings": []
        }
    
    def crosstab(
        self,
        row_var: str,
        col_var: str,
        weight_column: Optional[str] = None,
        normalize: str = "none"
    ) -> Dict[str, Any]:
        """
        Create cross-tabulation between two variables.
        
        Args:
            row_var: Row variable name
            col_var: Column variable name
            weight_column: Optional weight column for weighted frequencies
            normalize: Normalization mode - "none", "row", "col", "all"
            
        Returns:
            Dictionary containing:
            - table: 2D dictionary of frequencies/percentages
            - row_margins: Row totals
            - col_margins: Column totals
            - grand_total: Overall total
        """
        if row_var not in self.df.columns:
            return {"error": f"Row variable '{row_var}' not found"}
        if col_var not in self.df.columns:
            return {"error": f"Column variable '{col_var}' not found"}
        
        # Remove rows with missing values in either variable
        valid_data = self.df[[row_var, col_var]].dropna()
        
        if len(valid_data) == 0:
            return {"error": "No valid observations after removing missing values"}
        
        if weight_column and weight_column in self.df.columns:
            # Weighted crosstab
            weights = self.df.loc[valid_data.index, weight_column].fillna(0)
            
            # Build weighted frequency table
            table = {}
            row_values = sorted(valid_data[row_var].unique())
            col_values = sorted(valid_data[col_var].unique())
            
            for row_val in row_values:
                table[str(row_val)] = {}
                for col_val in col_values:
                    mask = (valid_data[row_var] == row_val) & (valid_data[col_var] == col_val)
                    table[str(row_val)][str(col_val)] = float(weights[mask].sum())
        else:
            # Unweighted crosstab
            ct = pd.crosstab(valid_data[row_var], valid_data[col_var])
            table = {str(idx): {str(col): int(ct.loc[idx, col]) for col in ct.columns} for idx in ct.index}
            row_values = ct.index
            col_values = ct.columns
        
        # Calculate margins
        row_margins = {}
        for row_val in table:
            row_margins[row_val] = sum(table[row_val].values())
        
        col_margins = {}
        for col_val in [str(c) for c in col_values]:
            col_margins[col_val] = sum(table[row_val][col_val] for row_val in table)
        
        grand_total = sum(row_margins.values())
        
        # Apply normalization
        if normalize == "row":
            for row_val in table:
                row_sum = row_margins[row_val]
                if row_sum > 0:
                    table[row_val] = {k: (v / row_sum * 100) for k, v in table[row_val].items()}
        elif normalize == "col":
            for row_val in table:
                for col_val in table[row_val]:
                    col_sum = col_margins[col_val]
                    if col_sum > 0:
                        table[row_val][col_val] = table[row_val][col_val] / col_sum * 100
        elif normalize == "all":
            if grand_total > 0:
                for row_val in table:
                    table[row_val] = {k: (v / grand_total * 100) for k, v in table[row_val].items()}
        
        # Chi-square test (only on original frequencies before normalization)
        chi_square_result = None
        chi_warnings = []
        
        if normalize == "none" and not weight_column:
            # Build frequency matrix for chi-square test
            freq_matrix = []
            for row_val in table:
                row_freqs = [table[row_val][str(col_val)] for col_val in col_values]
                freq_matrix.append(row_freqs)
            
            freq_matrix = np.array(freq_matrix)
            
            if freq_matrix.min() >= 0 and freq_matrix.sum() > 0:
                try:
                    chi2, p_value, dof, expected = chi2_contingency(freq_matrix)
                    
                    # Check for low expected frequencies
                    low_expected = (expected < 5).sum()
                    if low_expected > 0:
                        chi_warnings.append(f"{low_expected} cells have expected frequency < 5")
                    
                    # Convert expected frequencies to same format as table
                    expected_table = {}
                    for i, row_val in enumerate(table.keys()):
                        expected_table[row_val] = {}
                        for j, col_val in enumerate([str(c) for c in col_values]):
                            expected_table[row_val][col_val] = float(expected[i, j])
                    
                    chi_square_result = {
                        "chi2_statistic": float(chi2),
                        "p_value": float(p_value),
                        "degrees_of_freedom": int(dof),
                        "expected_frequencies": expected_table,
                        "warnings": chi_warnings
                    }
                except Exception as e:
                    chi_warnings.append(f"Chi-square test failed: {str(e)}")
        
        self.operations_log.append({
            "operation": "crosstab",
            "row_var": row_var,
            "col_var": col_var,
            "normalize": normalize,
            "weighted": weight_column is not None
        })
        
        result = {
            "table": table,
            "row_margins": row_margins,
            "col_margins": col_margins,
            "grand_total": grand_total,
            "normalize": normalize,
            "weighted": weight_column is not None,
            "warnings": chi_warnings
        }
        
        if chi_square_result:
            result["chi_square_test"] = chi_square_result
        
        return self._make_json_safe(result)
    
    def run_regression(
        self,
        dependent: str,
        independents: List[str],
        weight_column: Optional[str] = None,
        model_type: str = "ols"
    ) -> Dict[str, Any]:
        """
        Run regression analysis (OLS or Logistic).
        
        Args:
            dependent: Dependent variable name
            independents: List of independent variable names
            weight_column: Optional weight column for weighted regression
            model_type: "ols" for linear regression or "logistic" for binary outcomes
            
        Returns:
            Dictionary with regression results including VIF, correlation matrix, and robust SE
        """
        if dependent not in self.df.columns:
            return {"error": f"Dependent variable '{dependent}' not found"}
        
        missing_vars = [v for v in independents if v not in self.df.columns]
        if missing_vars:
            return {"error": f"Independent variables not found: {missing_vars}"}
        
        # Prepare data
        all_vars = [dependent] + independents
        valid_data = self.df[all_vars].dropna()
        
        if len(valid_data) < len(independents) + 1:
            return {"error": "Insufficient observations for regression"}
        
        y = valid_data[dependent].values
        X = valid_data[independents].values
        
        # Compute VIF and correlation matrix (before adding intercept)
        vif = self._compute_vif(X, independents)
        correlation_matrix = self._compute_correlation_matrix(X, independents)
        
        # Warnings for multicollinearity
        warnings_list = []
        for var, vif_val in vif.items():
            if vif_val is not None and vif_val > 10:
                warnings_list.append(f"High VIF for {var}: {vif_val:.2f} (possible multicollinearity)")
        
        # Logistic regression path
        if model_type.lower() == "logistic":
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            var_names = ["intercept"] + independents
            
            logistic_result = self._run_logistic(X_with_intercept, y, var_names)
            
            if "error" in logistic_result:
                return logistic_result
            
            self.operations_log.append({
                "operation": "logistic_regression",
                "dependent": dependent,
                "independents": independents,
                "weighted": False  # Logistic doesn't support weights in this implementation
            })
            
            return self._make_json_safe({
                "model_type": "logistic",
                "dependent": dependent,
                "independents": independents,
                "coefficients": logistic_result["coefficients"],
                "std_errors": logistic_result["std_errors"],
                "z_statistics": logistic_result["z_statistics"],
                "p_values": logistic_result["p_values"],
                "pseudo_r_squared": logistic_result["pseudo_r_squared"],
                "n_obs": logistic_result["n_obs"],
                "converged": logistic_result["converged"],
                "vif": vif,
                "correlation_matrix": correlation_matrix,
                "warnings": warnings_list
            })
        
        # OLS regression path
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        var_names = ["intercept"] + independents
        
        n = len(y)
        k = X_with_intercept.shape[1]
        
        if weight_column and weight_column in self.df.columns:
            # Weighted regression
            weights = self.df.loc[valid_data.index, weight_column].fillna(0)
            weights = weights.values
            
            # Remove zero weights
            valid_mask = weights > 0
            if valid_mask.sum() < k:
                return {"error": "Insufficient non-zero weights for regression"}
            
            X_with_intercept = X_with_intercept[valid_mask]
            y = y[valid_mask]
            weights = weights[valid_mask]
            n = len(y)
            
            # Weight matrix
            W = np.diag(weights)
            
            # Compute β = (X'WX)^(-1) X'Wy
            XtWX = X_with_intercept.T @ W @ X_with_intercept
            XtWy = X_with_intercept.T @ W @ y
            
            try:
                beta = np.linalg.solve(XtWX, XtWy)
                XtWX_inv = np.linalg.inv(XtWX)
            except np.linalg.LinAlgError:
                return {"error": "Singular matrix - regression cannot be computed"}
            
            # Predictions and residuals
            y_pred = X_with_intercept @ beta
            residuals = y - y_pred
            
            # Weighted sum of squares
            ss_res = residuals.T @ W @ residuals
            y_mean = np.average(y, weights=weights)
            ss_tot = ((y - y_mean) ** 2).T @ W @ np.ones(n)
            
            # Robust standard errors (HC3)
            std_errors = self._compute_robust_se_weighted(X_with_intercept, residuals, weights, XtWX_inv)
            
        else:
            # Unweighted regression
            # β = (X'X)^(-1) X'y
            XtX = X_with_intercept.T @ X_with_intercept
            Xty = X_with_intercept.T @ y
            
            try:
                beta = np.linalg.solve(XtX, Xty)
                XtX_inv = np.linalg.inv(XtX)
            except np.linalg.LinAlgError:
                return {"error": "Singular matrix - regression cannot be computed"}
            
            # Predictions and residuals
            y_pred = X_with_intercept @ beta
            residuals = y - y_pred
            
            # Sum of squares
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            
            # Robust standard errors (HC3)
            std_errors = self._compute_robust_se(X_with_intercept, residuals, XtX_inv)
        
        # R² and adjusted R²
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k) if n > k else 0
        
        # t-statistics and p-values using robust SE
        t_stats = beta / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
        
        # Compute MSE for residual std
        mse = ss_res / (n - k)
        
        # Build results
        coefficients = {var_names[i]: float(beta[i]) for i in range(k)}
        std_errs = {var_names[i]: float(std_errors[i]) for i in range(k)}
        t_statistics = {var_names[i]: float(t_stats[i]) for i in range(k)}
        p_vals = {var_names[i]: float(p_values[i]) for i in range(k)}
        
        self.operations_log.append({
            "operation": "ols_regression",
            "dependent": dependent,
            "independents": independents,
            "weighted": weight_column is not None
        })
        
        return self._make_json_safe({
            "model_type": "ols",
            "dependent": dependent,
            "independents": independents,
            "coefficients": coefficients,
            "std_errors": std_errs,
            "t_statistics": t_statistics,
            "p_values": p_vals,
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared),
            "n_obs": int(n),
            "residuals_std": float(np.sqrt(mse)),
            "weighted": weight_column is not None,
            "vif": vif,
            "correlation_matrix": correlation_matrix,
            "warnings": warnings_list,
            "robust_se": True
        })
    
    def subgroup_analysis(
        self,
        group_by: str,
        target: str,
        metrics: List[str],
        weight_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform analysis by subgroups.
        
        Args:
            group_by: Column to group by
            target: Target column to analyze
            metrics: List of metrics to compute - "mean", "median", "min", "max", "std", "count"
            weight_column: Optional weight column for weighted metrics
            
        Returns:
            Dictionary with group values as keys and metric results as values
        """
        if group_by not in self.df.columns:
            return {"error": f"Group variable '{group_by}' not found"}
        if target not in self.df.columns:
            return {"error": f"Target variable '{target}' not found"}
        
        valid_data = self.df[[group_by, target]].dropna()
        
        if len(valid_data) == 0:
            return {"error": "No valid observations"}
        
        results = {}
        
        for group_val in valid_data[group_by].unique():
            group_mask = valid_data[group_by] == group_val
            group_data = valid_data.loc[group_mask, target]
            
            if len(group_data) == 0:
                continue
            
            group_results = {}
            
            if weight_column and weight_column in self.df.columns:
                # Weighted metrics
                weights = self.df.loc[group_data.index, weight_column].fillna(0)
                valid_weight_mask = weights > 0
                group_data = group_data[valid_weight_mask]
                weights = weights[valid_weight_mask]
                
                if len(group_data) == 0:
                    continue
                
                weights_norm = weights / weights.sum()
                
                if "mean" in metrics:
                    mean = float(np.average(group_data, weights=weights_norm))
                    group_results["mean"] = mean
                    
                    # 95% Confidence interval for weighted mean
                    variance = np.average((group_data - mean) ** 2, weights=weights_norm)
                    std = np.sqrt(variance)
                    se = std / np.sqrt(len(group_data))
                    ci_lower = mean - 1.96 * se
                    ci_upper = mean + 1.96 * se
                    group_results["ci_95"] = {"lower": float(ci_lower), "upper": float(ci_upper)}
                    
                if "median" in metrics:
                    sorted_indices = np.argsort(group_data)
                    sorted_data = group_data.iloc[sorted_indices].values
                    sorted_weights = weights_norm.iloc[sorted_indices].values
                    cumsum = np.cumsum(sorted_weights)
                    group_results["median"] = float(np.interp(0.5, cumsum, sorted_data))
                if "std" in metrics:
                    if "mean" not in group_results:
                        mean = np.average(group_data, weights=weights_norm)
                    else:
                        mean = group_results["mean"]
                    variance = np.average((group_data - mean) ** 2, weights=weights_norm)
                    std = float(np.sqrt(variance))
                    group_results["std"] = std
                    
                    # Coefficient of Variation (CV)
                    if mean != 0:
                        cv = (std / abs(mean)) * 100
                        group_results["cv"] = float(cv)
                    else:
                        group_results["cv"] = None
                if "min" in metrics:
                    group_results["min"] = float(group_data.min())
                if "max" in metrics:
                    group_results["max"] = float(group_data.max())
                if "count" in metrics:
                    group_results["count"] = int(len(group_data))
                if "sum" in metrics:
                    group_results["sum"] = float(weights.sum())
                
            else:
                # Unweighted metrics
                if "mean" in metrics:
                    mean = float(group_data.mean())
                    group_results["mean"] = mean
                    
                    # 95% Confidence interval for mean
                    std = float(group_data.std())
                    se = std / np.sqrt(len(group_data))
                    ci_lower = mean - 1.96 * se
                    ci_upper = mean + 1.96 * se
                    group_results["ci_95"] = {"lower": float(ci_lower), "upper": float(ci_upper)}
                    
                if "median" in metrics:
                    group_results["median"] = float(group_data.median())
                if "std" in metrics:
                    std = float(group_data.std())
                    group_results["std"] = std
                    
                    # Coefficient of Variation (CV)
                    if "mean" in group_results and group_results["mean"] != 0:
                        cv = (std / abs(group_results["mean"])) * 100
                        group_results["cv"] = float(cv)
                    elif (mean := float(group_data.mean())) != 0:
                        cv = (std / abs(mean)) * 100
                        group_results["cv"] = float(cv)
                    else:
                        group_results["cv"] = None
                if "min" in metrics:
                    group_results["min"] = float(group_data.min())
                if "max" in metrics:
                    group_results["max"] = float(group_data.max())
                if "count" in metrics:
                    group_results["count"] = int(len(group_data))
                if "sum" in metrics:
                    group_results["sum"] = float(group_data.sum())
            
            # Outlier detection using IQR method
            q1 = group_data.quantile(0.25)
            q3 = group_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            outliers = group_data[(group_data < lower_bound) | (group_data > upper_bound)]
            group_results["outlier_count"] = int(len(outliers))
            
            # Risk tagging: stable / caution / high_risk
            risk_level = "stable"
            risk_reasons = []
            
            sample_size = len(group_data)
            cv = group_results.get("cv")
            outlier_pct = (len(outliers) / sample_size * 100) if sample_size > 0 else 0
            
            if sample_size < 30:
                risk_reasons.append("Small sample size (< 30)")
                risk_level = "caution"
            
            if cv is not None and cv > 50:
                risk_reasons.append(f"High CV ({cv:.1f}%)")
                if risk_level == "stable":
                    risk_level = "caution"
            
            if cv is not None and cv > 100:
                risk_reasons.append(f"Very high CV ({cv:.1f}%)")
                risk_level = "high_risk"
            
            if outlier_pct > 5:
                risk_reasons.append(f"{outlier_pct:.1f}% outliers")
                if risk_level == "stable":
                    risk_level = "caution"
            
            if outlier_pct > 10:
                risk_reasons.append(f"Excessive outliers ({outlier_pct:.1f}%)")
                risk_level = "high_risk"
            
            if sample_size < 10:
                risk_reasons.append("Very small sample (< 10)")
                risk_level = "high_risk"
            
            group_results["risk_level"] = risk_level
            group_results["risk_reasons"] = risk_reasons
            
            results[str(group_val)] = group_results
        
        self.operations_log.append({
            "operation": "subgroup_analysis",
            "group_by": group_by,
            "target": target,
            "metrics": metrics,
            "weighted": weight_column is not None
        })
        
        return self._make_json_safe({
            "group_by": group_by,
            "target": target,
            "metrics": metrics,
            "groups": results,
            "weighted": weight_column is not None
        })
    
    def run_anova(
        self,
        dependent_var: str,
        group_var: str,
        weight_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform one-way ANOVA with weighted and unweighted variants.
        
        Args:
            dependent_var: Continuous dependent variable
            group_var: Grouping variable (categorical)
            weight_column: Optional weight column for weighted ANOVA
            
        Returns:
            Dictionary containing:
            - group_means: Mean for each group
            - ss_between: Sum of squares between groups
            - ss_within: Sum of squares within groups
            - df_between: Degrees of freedom between
            - df_within: Degrees of freedom within
            - ms_between: Mean square between
            - ms_within: Mean square within
            - f_statistic: F-statistic
            - p_value: p-value
            - eta_squared: Effect size (η²)
            - omega_squared: Effect size (ω²)
            - significance: "significant" or "not significant"
            - warnings: List of diagnostic warnings
        """
        if dependent_var not in self.df.columns:
            return {"error": f"Dependent variable '{dependent_var}' not found"}
        if group_var not in self.df.columns:
            return {"error": f"Group variable '{group_var}' not found"}
        
        # Check if dependent is numeric
        if not pd.api.types.is_numeric_dtype(self.df[dependent_var]):
            return {"error": f"Dependent variable '{dependent_var}' must be numeric"}
        
        valid_data = self.df[[dependent_var, group_var]].dropna()
        
        if len(valid_data) < 3:
            return {"error": "Insufficient observations for ANOVA (need at least 3)"}
        
        groups = valid_data[group_var].unique()
        k = len(groups)  # Number of groups
        
        if k < 2:
            return {"error": "Need at least 2 groups for ANOVA"}
        
        warnings_list = []
        group_means = {}
        group_sizes = {}
        group_vars = {}
        
        if weight_column and weight_column in self.df.columns:
            # Weighted ANOVA
            weights = self.df.loc[valid_data.index, weight_column].fillna(0)
            valid_mask = weights > 0
            valid_data = valid_data[valid_mask]
            weights = weights[valid_mask]
            
            if len(valid_data) < 3:
                return {"error": "Insufficient non-zero weights for ANOVA"}
            
            # Compute weighted grand mean
            grand_mean = np.average(valid_data[dependent_var], weights=weights)
            
            # Compute group statistics
            ss_between = 0
            ss_within = 0
            n_total = 0
            
            for group in groups:
                mask = valid_data[group_var] == group
                group_data = valid_data.loc[mask, dependent_var]
                group_weights = weights[mask]
                
                if len(group_data) < 2:
                    warnings_list.append(f"Group '{group}' has fewer than 2 observations")
                    continue
                
                n_group = group_weights.sum()
                n_total += n_group
                
                # Weighted mean
                mean_group = np.average(group_data, weights=group_weights)
                group_means[str(group)] = float(mean_group)
                group_sizes[str(group)] = int(len(group_data))
                
                # Weighted variance
                variance_group = np.average((group_data - mean_group) ** 2, weights=group_weights)
                group_vars[str(group)] = float(variance_group)
                
                # SS between
                ss_between += n_group * (mean_group - grand_mean) ** 2
                
                # SS within
                ss_within += (group_weights * (group_data - mean_group) ** 2).sum()
            
            n = n_total
            
        else:
            # Unweighted ANOVA
            grand_mean = valid_data[dependent_var].mean()
            
            ss_between = 0
            ss_within = 0
            n = len(valid_data)
            
            for group in groups:
                group_data = valid_data[valid_data[group_var] == group][dependent_var]
                
                if len(group_data) < 2:
                    warnings_list.append(f"Group '{group}' has fewer than 2 observations")
                    continue
                
                n_group = len(group_data)
                mean_group = group_data.mean()
                var_group = group_data.var()
                
                group_means[str(group)] = float(mean_group)
                group_sizes[str(group)] = int(n_group)
                group_vars[str(group)] = float(var_group)
                
                ss_between += n_group * (mean_group - grand_mean) ** 2
                ss_within += ((group_data - mean_group) ** 2).sum()
        
        # Check for variance imbalance
        if group_vars:
            max_var = max(group_vars.values())
            min_var = min(group_vars.values())
            if min_var > 0 and max_var / min_var > 4:
                warnings_list.append(f"High variance imbalance detected (ratio: {max_var/min_var:.2f})")
        
        # Degrees of freedom
        df_between = k - 1
        df_within = n - k
        
        if df_within <= 0:
            return {"error": "Insufficient degrees of freedom for ANOVA"}
        
        # Mean squares
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        
        # F-statistic
        f_statistic = ms_between / ms_within if ms_within > 0 else 0
        
        # p-value
        p_value = 1 - stats.f.cdf(f_statistic, df_between, df_within)
        
        # Effect sizes
        ss_total = ss_between + ss_within
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within) if ss_total + ms_within > 0 else 0
        
        significance = "significant" if p_value < 0.05 else "not significant"
        
        self.operations_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": "anova",
            "details": {
                "dependent_var": dependent_var,
                "group_var": group_var,
                "F": float(f_statistic),
                "p": float(p_value),
                "effect_size": f"η²={eta_squared:.4f}, ω²={omega_squared:.4f}",
                "warnings": warnings_list
            }
        })
        
        return self._make_json_safe({
            "test": "one_way_anova",
            "dependent_var": dependent_var,
            "group_var": group_var,
            "n_groups": int(k),
            "n_total": int(n),
            "group_means": group_means,
            "group_sizes": group_sizes,
            "group_variances": group_vars,
            "ss_between": float(ss_between),
            "ss_within": float(ss_within),
            "ss_total": float(ss_total),
            "df_between": int(df_between),
            "df_within": int(df_within),
            "ms_between": float(ms_between),
            "ms_within": float(ms_within),
            "f_statistic": float(f_statistic),
            "p_value": float(p_value),
            "eta_squared": float(eta_squared),
            "omega_squared": float(omega_squared),
            "significance": significance,
            "weighted": weight_column is not None,
            "warnings": warnings_list
        })
    
    def run_manova(
        self,
        dependent_vars: List[str],
        group_var: str,
        weight_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform multivariate ANOVA (MANOVA).
        
        Args:
            dependent_vars: List of continuous dependent variables
            group_var: Grouping variable (categorical)
            weight_column: Optional weight column (warning: limited support)
            
        Returns:
            Dictionary containing:
            - pillai_trace: Pillai's Trace statistic
            - wilks_lambda: Wilks' Lambda statistic
            - hotelling_lawley_trace: Hotelling-Lawley Trace
            - roy_largest_root: Roy's Largest Root
            - p_values: p-value for each statistic
            - overall_significance: Overall test result
            - warnings: List of diagnostic warnings
        """
        if group_var not in self.df.columns:
            return {"error": f"Group variable '{group_var}' not found"}
        
        missing_vars = [v for v in dependent_vars if v not in self.df.columns]
        if missing_vars:
            return {"error": f"Dependent variables not found: {missing_vars}"}
        
        # Check all dependent vars are numeric
        for var in dependent_vars:
            if not pd.api.types.is_numeric_dtype(self.df[var]):
                return {"error": f"Dependent variable '{var}' must be numeric"}
        
        all_vars = [group_var] + dependent_vars
        valid_data = self.df[all_vars].dropna()
        
        if len(valid_data) < len(dependent_vars) + 2:
            return {"error": "Insufficient observations for MANOVA"}
        
        groups = valid_data[group_var].unique()
        k = len(groups)  # Number of groups
        p = len(dependent_vars)  # Number of dependent variables
        n = len(valid_data)
        
        if k < 2:
            return {"error": "Need at least 2 groups for MANOVA"}
        
        warnings_list = []
        
        if weight_column:
            warnings_list.append("Weighted MANOVA has limited support; using unweighted computation")
        
        # Check group sizes
        group_sizes = {}
        for group in groups:
            group_size = len(valid_data[valid_data[group_var] == group])
            group_sizes[str(group)] = group_size
            if group_size < p:
                warnings_list.append(f"Group '{group}' has size {group_size} < {p} (number of variables)")
        
        # Create design matrix and response matrix
        Y = valid_data[dependent_vars].values
        
        # Compute group means
        grand_mean = Y.mean(axis=0)
        group_means_dict = {}
        
        # Between-group and within-group matrices
        H = np.zeros((p, p))  # Hypothesis (between-group) matrix
        E = np.zeros((p, p))  # Error (within-group) matrix
        
        for group in groups:
            mask = valid_data[group_var] == group
            Y_group = Y[mask]
            n_group = len(Y_group)
            
            if n_group == 0:
                continue
            
            mean_group = Y_group.mean(axis=0)
            group_means_dict[str(group)] = {
                dependent_vars[i]: float(mean_group[i]) for i in range(p)
            }
            
            # Contribution to H (between-group)
            diff = (mean_group - grand_mean).reshape(-1, 1)
            H += n_group * (diff @ diff.T)
            
            # Contribution to E (within-group)
            for i in range(n_group):
                resid = (Y_group[i] - mean_group).reshape(-1, 1)
                E += resid @ resid.T
        
        # Compute test statistics
        try:
            E_inv = np.linalg.inv(E)
            HE_inv = H @ E_inv
            eigenvalues = np.linalg.eigvals(HE_inv)
            eigenvalues = eigenvalues.real  # Take real part
            
            # Pillai's Trace
            pillai = np.sum(eigenvalues / (1 + eigenvalues))
            
            # Wilks' Lambda
            wilks = np.prod(1 / (1 + eigenvalues))
            
            # Hotelling-Lawley Trace
            hotelling = np.sum(eigenvalues)
            
            # Roy's Largest Root
            roy = np.max(eigenvalues)
            
            # Approximate p-values (using standard formulas)
            s = min(p, k - 1)
            m = (abs(p - k + 1) - 1) / 2
            n_stat = (n - k - p - 1) / 2
            
            # Pillai's p-value (F-approximation)
            df1_pillai = s * (2 * m + s + 1)
            df2_pillai = s * (2 * n_stat + s + 1)
            F_pillai = (pillai / s) / ((s - pillai) / s) * (df2_pillai / df1_pillai) if s > 0 else 0
            p_pillai = 1 - stats.f.cdf(F_pillai, df1_pillai, df2_pillai) if df1_pillai > 0 and df2_pillai > 0 else 1.0
            
            # Wilks' p-value (chi-square approximation)
            df_wilks = p * (k - 1)
            chi2_wilks = -(n - 1 - (p + k) / 2) * np.log(wilks) if wilks > 0 else 0
            p_wilks = 1 - stats.chi2.cdf(chi2_wilks, df_wilks) if df_wilks > 0 else 1.0
            
            # Hotelling p-value (F-approximation)
            b = (p ** 2 + (k - 1) ** 2 - 5) / (p ** 2 + (k - 1) ** 2)
            df1_hotelling = p * (k - 1)
            df2_hotelling = 4 + (p * (k - 1) + 2) / b if b > 0 else 1
            F_hotelling = hotelling * df2_hotelling / df1_hotelling if df1_hotelling > 0 else 0
            p_hotelling = 1 - stats.f.cdf(F_hotelling, df1_hotelling, df2_hotelling) if df2_hotelling > 0 else 1.0
            
            # Roy p-value (complex, use Pillai as proxy)
            p_roy = p_pillai
            
            # Overall significance (use Wilks' Lambda as primary test)
            overall_significance = "significant" if p_wilks < 0.05 else "not significant"
            
        except np.linalg.LinAlgError:
            return {"error": "Singular matrix encountered; check for multicollinearity"}
        
        self.operations_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": "manova",
            "details": {
                "dependent_vars": dependent_vars,
                "group_var": group_var,
                "wilks_lambda": float(wilks),
                "p": float(p_wilks),
                "warnings": warnings_list
            }
        })
        
        return self._make_json_safe({
            "test": "manova",
            "dependent_vars": dependent_vars,
            "group_var": group_var,
            "n_groups": int(k),
            "n_total": int(n),
            "n_vars": int(p),
            "group_sizes": group_sizes,
            "group_means": group_means_dict,
            "pillai_trace": float(pillai),
            "pillai_p_value": float(p_pillai),
            "wilks_lambda": float(wilks),
            "wilks_p_value": float(p_wilks),
            "hotelling_lawley_trace": float(hotelling),
            "hotelling_p_value": float(p_hotelling),
            "roy_largest_root": float(roy),
            "roy_p_value": float(p_roy),
            "overall_significance": overall_significance,
            "warnings": warnings_list
        })
    
    def run_kruskal(
        self,
        dependent_var: str,
        group_var: str
    ) -> Dict[str, Any]:
        """
        Perform Kruskal-Wallis H-test (non-parametric alternative to ANOVA).
        
        Args:
            dependent_var: Continuous or ordinal dependent variable
            group_var: Grouping variable (categorical)
            
        Returns:
            Dictionary containing:
            - h_statistic: H-statistic (chi-square approximation)
            - df: Degrees of freedom
            - p_value: p-value
            - ties_corrected: Whether ties correction was applied
            - significance: "significant" or "not significant"
            - warnings: List of diagnostic warnings
        """
        if dependent_var not in self.df.columns:
            return {"error": f"Dependent variable '{dependent_var}' not found"}
        if group_var not in self.df.columns:
            return {"error": f"Group variable '{group_var}' not found"}
        
        valid_data = self.df[[dependent_var, group_var]].dropna()
        
        if len(valid_data) < 3:
            return {"error": "Insufficient observations for Kruskal-Wallis test"}
        
        groups = valid_data[group_var].unique()
        k = len(groups)
        
        if k < 2:
            return {"error": "Need at least 2 groups for Kruskal-Wallis test"}
        
        warnings_list = []
        group_sizes = {}
        
        # Prepare data for scipy
        group_samples = []
        for group in groups:
            group_data = valid_data[valid_data[group_var] == group][dependent_var].values
            group_samples.append(group_data)
            group_sizes[str(group)] = len(group_data)
            
            if len(group_data) < 5:
                warnings_list.append(f"Group '{group}' has small sample size ({len(group_data)})")
        
        # Run Kruskal-Wallis test
        h_statistic, p_value = stats.kruskal(*group_samples)
        
        # Check for ties
        all_data = valid_data[dependent_var].values
        unique_count = len(np.unique(all_data))
        has_ties = unique_count < len(all_data)
        
        if has_ties:
            warnings_list.append("Ties detected in data; H-statistic is corrected for ties")
        
        df = k - 1
        significance = "significant" if p_value < 0.05 else "not significant"
        
        self.operations_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": "kruskal_wallis",
            "details": {
                "dependent_var": dependent_var,
                "group_var": group_var,
                "H": float(h_statistic),
                "p": float(p_value),
                "warnings": warnings_list
            }
        })
        
        return self._make_json_safe({
            "test": "kruskal_wallis",
            "dependent_var": dependent_var,
            "group_var": group_var,
            "n_groups": int(k),
            "n_total": int(len(valid_data)),
            "group_sizes": group_sizes,
            "h_statistic": float(h_statistic),
            "df": int(df),
            "p_value": float(p_value),
            "ties_corrected": has_ties,
            "significance": significance,
            "warnings": warnings_list
        })
    
    def run_levene(
        self,
        dependent_var: str,
        group_var: str
    ) -> Dict[str, Any]:
        """
        Perform Levene's test for homogeneity of variances.
        
        Args:
            dependent_var: Continuous dependent variable
            group_var: Grouping variable (categorical)
            
        Returns:
            Dictionary containing:
            - w_statistic: W-statistic
            - p_value: p-value
            - group_variances: Variance for each group
            - variance_ratio: Max variance / min variance
            - significance: "significant" (variances differ) or "not significant" (equal variances)
            - warnings: List of diagnostic warnings
        """
        if dependent_var not in self.df.columns:
            return {"error": f"Dependent variable '{dependent_var}' not found"}
        if group_var not in self.df.columns:
            return {"error": f"Group variable '{group_var}' not found"}
        
        if not pd.api.types.is_numeric_dtype(self.df[dependent_var]):
            return {"error": f"Dependent variable '{dependent_var}' must be numeric"}
        
        valid_data = self.df[[dependent_var, group_var]].dropna()
        
        if len(valid_data) < 3:
            return {"error": "Insufficient observations for Levene's test"}
        
        groups = valid_data[group_var].unique()
        k = len(groups)
        
        if k < 2:
            return {"error": "Need at least 2 groups for Levene's test"}
        
        warnings_list = []
        group_variances = {}
        
        # Prepare data for scipy
        group_samples = []
        for group in groups:
            group_data = valid_data[valid_data[group_var] == group][dependent_var].values
            group_samples.append(group_data)
            group_variances[str(group)] = float(np.var(group_data, ddof=1))
            
            if len(group_data) < 2:
                warnings_list.append(f"Group '{group}' has fewer than 2 observations")
        
        # Run Levene's test (using median-based method for robustness)
        w_statistic, p_value = stats.levene(*group_samples, center='median')
        
        # Compute variance ratio
        variances = list(group_variances.values())
        max_var = max(variances) if variances else 0
        min_var = min(variances) if variances else 0
        variance_ratio = max_var / min_var if min_var > 0 else float('inf')
        
        if variance_ratio > 4:
            warnings_list.append(f"High variance imbalance detected (ratio: {variance_ratio:.2f})")
        
        significance = "significant" if p_value < 0.05 else "not significant"
        interpretation = "Variances differ significantly across groups" if p_value < 0.05 else "Variances are approximately equal"
        
        self.operations_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": "levene_test",
            "details": {
                "dependent_var": dependent_var,
                "group_var": group_var,
                "W": float(w_statistic),
                "p": float(p_value),
                "variance_ratio": float(variance_ratio),
                "warnings": warnings_list
            }
        })
        
        return self._make_json_safe({
            "test": "levene",
            "dependent_var": dependent_var,
            "group_var": group_var,
            "n_groups": int(k),
            "n_total": int(len(valid_data)),
            "group_variances": group_variances,
            "variance_ratio": float(variance_ratio),
            "w_statistic": float(w_statistic),
            "p_value": float(p_value),
            "significance": significance,
            "interpretation": interpretation,
            "warnings": warnings_list
        })
    
    def run_shapiro(
        self,
        column: str
    ) -> Dict[str, Any]:
        """
        Perform Shapiro-Wilk normality test.
        
        Args:
            column: Numeric column to test for normality
            
        Returns:
            Dictionary containing:
            - w_statistic: W-statistic
            - p_value: p-value
            - skewness: Sample skewness
            - kurtosis: Sample kurtosis
            - normality: "normal" or "non-normal"
            - warnings: List of diagnostic warnings
        """
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"error": f"Column '{column}' must be numeric"}
        
        data = self.df[column].dropna()
        n = len(data)
        
        if n < 3:
            return {"error": "Need at least 3 observations for Shapiro-Wilk test"}
        
        if n > 5000:
            return {"error": "Shapiro-Wilk test not recommended for n > 5000 (too sensitive)"}
        
        warnings_list = []
        
        # Run Shapiro-Wilk test
        w_statistic, p_value = stats.shapiro(data)
        
        # Compute skewness and kurtosis
        skewness = float(data.skew())
        kurtosis = float(data.kurtosis())
        
        # Interpretation
        normality = "normal" if p_value >= 0.05 else "non-normal"
        
        if p_value < 0.05:
            warnings_list.append(f"Data is non-normal (p = {p_value:.4f})")
        
        if abs(skewness) > 2:
            warnings_list.append(f"Heavy skewness detected (skew = {skewness:.2f})")
        
        if abs(kurtosis) > 5:
            warnings_list.append(f"Heavy kurtosis detected (kurtosis = {kurtosis:.2f})")
        
        self.operations_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": "shapiro_wilk",
            "details": {
                "column": column,
                "W": float(w_statistic),
                "p": float(p_value),
                "normality": normality,
                "warnings": warnings_list
            }
        })
        
        return self._make_json_safe({
            "test": "shapiro_wilk",
            "column": column,
            "n": int(n),
            "w_statistic": float(w_statistic),
            "p_value": float(p_value),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "normality": normality,
            "interpretation": f"Data is {normality} (α = 0.05)",
            "warnings": warnings_list
        })
    
    def generate_analysis_report(self, weight_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive automated analysis report with auto-detection.
        
        Performs:
        - Auto-detection of numeric/categorical columns
        - Descriptive statistics with anomaly detection
        - Correlation analysis between numeric variables
        - Chi-square tests on categorical pairs
        - Simple regression if sufficient numeric columns
        - Subgroup analysis with risk assessment
        - Recommended actions based on findings
        
        Args:
            weight_column: Optional weight column for weighted analysis
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        report = {
            "summary": {},
            "descriptive_stats": {},
            "correlations": {},
            "chi_square_tests": [],
            "regression_summary": None,
            "subgroup_analysis": [],
            "anomalies": [],
            "recommendations": []
        }
        
        # Auto-detect column types
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Remove weight column from analysis columns
        if weight_column in numeric_cols:
            numeric_cols.remove(weight_column)
        if weight_column in categorical_cols:
            categorical_cols.remove(weight_column)
        
        report["summary"] = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "weighted_analysis": weight_column is not None
        }
        
        # 1. Descriptive Statistics with anomaly detection
        for col in numeric_cols[:10]:  # Limit to 10 numeric columns
            stats = self.descriptive_stats([col], weight_column)
            if col in stats:
                col_stats = stats[col]
                report["descriptive_stats"][col] = col_stats
                
                # Collect anomalies
                if col_stats.get("warnings"):
                    for warning in col_stats["warnings"]:
                        report["anomalies"].append({
                            "type": "descriptive",
                            "column": col,
                            "issue": warning
                        })
        
        for col in categorical_cols[:10]:  # Limit to 10 categorical columns
            stats = self.descriptive_stats([col], weight_column)
            if col in stats:
                report["descriptive_stats"][col] = stats[col]
        
        # 2. Correlation Analysis (top correlations > 0.5)
        if len(numeric_cols) >= 2:
            correlation_pairs = []
            for i, col1 in enumerate(numeric_cols[:10]):
                for col2 in numeric_cols[i+1:10]:
                    valid_data = self.df[[col1, col2]].dropna()
                    if len(valid_data) > 10:
                        corr = valid_data[col1].corr(valid_data[col2])
                        if abs(corr) > 0.5:
                            correlation_pairs.append({
                                "var1": col1,
                                "var2": col2,
                                "correlation": float(corr),
                                "strength": "strong" if abs(corr) > 0.7 else "moderate"
                            })
            
            # Sort by absolute correlation
            correlation_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            report["correlations"]["top_pairs"] = correlation_pairs[:10]
            
            if correlation_pairs:
                report["recommendations"].append({
                    "category": "correlation",
                    "message": f"Found {len(correlation_pairs)} strong correlations between numeric variables",
                    "action": "Review correlation pairs for potential multicollinearity in regression models"
                })
        
        # 3. Chi-square tests on categorical pairs (limit to avoid explosion)
        if len(categorical_cols) >= 2:
            chi_tests = []
            for i, col1 in enumerate(categorical_cols[:5]):
                for col2 in categorical_cols[i+1:5]:
                    crosstab_result = self.crosstab(col1, col2, weight_column)
                    if "chi_square_test" in crosstab_result:
                        chi_test = crosstab_result["chi_square_test"]
                        chi_tests.append({
                            "var1": col1,
                            "var2": col2,
                            "chi2": chi_test["chi2_statistic"],
                            "p_value": chi_test["p_value"],
                            "significant": chi_test["p_value"] < 0.05,
                            "warnings": chi_test.get("warnings", [])
                        })
                        
                        if chi_test.get("warnings"):
                            for warning in chi_test["warnings"]:
                                report["anomalies"].append({
                                    "type": "chi_square",
                                    "variables": f"{col1} x {col2}",
                                    "issue": warning
                                })
            
            report["chi_square_tests"] = chi_tests
            
            significant_tests = [t for t in chi_tests if t["significant"]]
            if significant_tests:
                report["recommendations"].append({
                    "category": "association",
                    "message": f"Found {len(significant_tests)} significant associations between categorical variables",
                    "action": "Consider these associations when designing models or stratifying analysis"
                })
        
        # 4. Simple Regression (if at least 3 numeric columns)
        if len(numeric_cols) >= 3:
            # Try OLS with first numeric as dependent, next 2 as predictors
            dependent = numeric_cols[0]
            independents = numeric_cols[1:min(3, len(numeric_cols))]
            
            regression_result = self.run_regression(dependent, independents, weight_column, "ols")
            
            if "error" not in regression_result:
                report["regression_summary"] = {
                    "model_type": regression_result.get("model_type", "ols"),
                    "dependent": dependent,
                    "independents": independents,
                    "r_squared": regression_result.get("r_squared"),
                    "adj_r_squared": regression_result.get("adj_r_squared"),
                    "significant_predictors": []
                }
                
                # Identify significant predictors (p < 0.05)
                for var, p_val in regression_result.get("p_values", {}).items():
                    if p_val < 0.05 and var != "intercept":
                        report["regression_summary"]["significant_predictors"].append(var)
                
                # Warnings for regression
                if regression_result.get("warnings"):
                    for warning in regression_result["warnings"]:
                        report["anomalies"].append({
                            "type": "regression",
                            "issue": warning
                        })
                
                if report["regression_summary"]["significant_predictors"]:
                    report["recommendations"].append({
                        "category": "regression",
                        "message": f"Found {len(report['regression_summary']['significant_predictors'])} significant predictors",
                        "action": "Review regression coefficients for effect sizes and practical significance"
                    })
        
        # 5. Subgroup Analysis (if categorical and numeric columns available)
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # Analyze first numeric by first categorical
            group_by = categorical_cols[0]
            target = numeric_cols[0]
            
            subgroup_result = self.subgroup_analysis(
                group_by, target, ["mean", "std", "count"], weight_column
            )
            
            if "error" not in subgroup_result:
                report["subgroup_analysis"].append({
                    "group_by": group_by,
                    "target": target,
                    "groups": subgroup_result.get("groups", {})
                })
                
                # Identify high-risk subgroups
                high_risk_groups = []
                for group_name, group_data in subgroup_result.get("groups", {}).items():
                    if group_data.get("risk_level") == "high_risk":
                        high_risk_groups.append(group_name)
                
                if high_risk_groups:
                    report["anomalies"].append({
                        "type": "subgroup",
                        "issue": f"High-risk subgroups detected: {', '.join(high_risk_groups)}",
                        "group_by": group_by,
                        "target": target
                    })
                    
                    report["recommendations"].append({
                        "category": "subgroup_risk",
                        "message": f"Found {len(high_risk_groups)} high-risk subgroups in {group_by}",
                        "action": "Review sample sizes and variability in high-risk groups before making inferences"
                    })
        
        # 6. ANOVA Signals (test for group differences)
        report["anova_tests"] = []
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # Run ANOVA on first numeric by first categorical
            anova_result = self.run_anova(numeric_cols[0], categorical_cols[0], weight_column)
            
            if "error" not in anova_result:
                report["anova_tests"].append(anova_result)
                
                if anova_result["significance"] == "significant":
                    eta_sq = anova_result["eta_squared"]
                    effect_size_label = "large" if eta_sq > 0.14 else ("medium" if eta_sq > 0.06 else "small")
                    
                    report["anomalies"].append({
                        "type": "anova",
                        "issue": f"Significant group differences detected (F={anova_result['f_statistic']:.2f}, p={anova_result['p_value']:.4f})",
                        "variables": f"{numeric_cols[0]} by {categorical_cols[0]}",
                        "effect_size": f"{effect_size_label} (η²={eta_sq:.4f})"
                    })
                    
                    report["recommendations"].append({
                        "category": "anova",
                        "message": f"Significant group differences with {effect_size_label} effect size",
                        "action": "Consider post-hoc tests or pairwise comparisons to identify which groups differ"
                    })
                
                # Check for variance imbalance warnings
                for warning in anova_result.get("warnings", []):
                    if "variance imbalance" in warning.lower():
                        report["anomalies"].append({
                            "type": "anova",
                            "issue": warning,
                            "variables": f"{numeric_cols[0]} by {categorical_cols[0]}"
                        })
        
        # 7. MANOVA Signals (multivariate group differences)
        report["manova_tests"] = []
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 3:
            # Run MANOVA on first 3 numeric by first categorical
            manova_vars = numeric_cols[:3]
            manova_result = self.run_manova(manova_vars, categorical_cols[0], weight_column)
            
            if "error" not in manova_result:
                report["manova_tests"].append(manova_result)
                
                if manova_result["overall_significance"] == "significant":
                    report["anomalies"].append({
                        "type": "manova",
                        "issue": f"Multivariate group differences detected (Wilks' λ={manova_result['wilks_lambda']:.4f}, p={manova_result['wilks_p_value']:.4f})",
                        "variables": f"{', '.join(manova_vars)} by {categorical_cols[0]}"
                    })
                    
                    report["recommendations"].append({
                        "category": "manova",
                        "message": "Significant multivariate group differences detected",
                        "action": f"Review which of {len(manova_vars)} variables contribute most to group separation"
                    })
        
        # 8. Normality Warnings (Shapiro-Wilk tests)
        report["normality_tests"] = []
        non_normal_cols = []
        for col in numeric_cols[:5]:  # Test first 5 numeric columns
            shapiro_result = self.run_shapiro(col)
            
            if "error" not in shapiro_result:
                report["normality_tests"].append(shapiro_result)
                
                if shapiro_result["normality"] == "non-normal":
                    non_normal_cols.append(col)
                    report["anomalies"].append({
                        "type": "normality",
                        "issue": f"Column '{col}' fails normality test (W={shapiro_result['w_statistic']:.4f}, p={shapiro_result['p_value']:.4f})",
                        "skewness": shapiro_result["skewness"],
                        "kurtosis": shapiro_result["kurtosis"]
                    })
        
        if non_normal_cols:
            report["recommendations"].append({
                "category": "normality",
                "message": f"{len(non_normal_cols)} columns fail normality assumption",
                "action": "Consider non-parametric tests (e.g., Kruskal-Wallis) or transformations for non-normal data"
            })
        
        # Final recommendations
        if len(report["anomalies"]) == 0:
            report["recommendations"].append({
                "category": "quality",
                "message": "No major data quality issues detected",
                "action": "Data appears suitable for analysis, but always validate assumptions"
            })
        elif len(report["anomalies"]) > 10:
            report["recommendations"].append({
                "category": "quality",
                "message": f"Detected {len(report['anomalies'])} data quality issues",
                "action": "Prioritize addressing data quality concerns before proceeding with analysis"
            })
        
        self.operations_log.append({
            "operation": "generate_analysis_report",
            "weighted": weight_column is not None
        })
        
        return self._make_json_safe(report)
    
    # ==========================================
    # STATISTICAL TEST METHODS
    # ==========================================
    
    def run_stat_test(
        self,
        test_type: str,
        var1: str,
        var2: Optional[str] = None,
        group: Optional[str] = None,
        weights: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a specific statistical test.
        
        Args:
            test_type: Type of test to run (e.g., 'one_sample_t', 'independent_t', etc.)
            var1: Primary variable name
            var2: Secondary variable name (for two-variable tests)
            group: Grouping variable name (for group comparison tests)
            weights: Weight column name (currently for documentation, not all tests support weights)
            
        Returns:
            Dict with test results in JSON-safe format
        """
        # Map test types to methods
        test_methods = {
            'one_sample_t': lambda: self.stats.one_sample_t(self.df, var1, popmean=0),
            'independent_t': lambda: self.stats.independent_t(self.df, group or var2, var1),
            'paired_t': lambda: self.stats.paired_t(self.df, var1, var2),
            'anova': lambda: self.stats.anova(self.df, group or var2, var1),
            'chi_square': lambda: self.stats.chi_square(self.df, var1, var2),
            'f_test': lambda: self.stats.f_test(self.df, var1, var2),
            'levene_test': lambda: self.stats.levene_test(self.df, var1, var2),
            'bartlett_test': lambda: self.stats.bartlett_test(self.df, var1, var2),
            'kruskal_test': lambda: self.stats.kruskal_test(self.df, group or var2, var1),
            'pearson_corr': lambda: self.stats.pearson_corr(self.df, var1, var2),
            'spearman_corr': lambda: self.stats.spearman_corr(self.df, var1, var2),
            'kendall_corr': lambda: self.stats.kendall_corr(self.df, var1, var2),
            'shapiro_test': lambda: self.stats.shapiro_test(self.df, var1),
            'ks_test': lambda: self.stats.ks_test(self.df, var1),
            'anderson_test': lambda: self.stats.anderson_test(self.df, var1),
            'jb_test': lambda: self.stats.jb_test(self.df, var1),
            'proportion_test': lambda: self.stats.proportion_test(self.df, var1),
            'two_proportion_test': lambda: self.stats.two_proportion_test(self.df, var1, group or var2),
        }
        
        if test_type not in test_methods:
            raise ValueError(f"Unknown test type: {test_type}. Available: {list(test_methods.keys())}")
        
        # Validate variables exist
        vars_to_check = [v for v in [var1, var2, group, weights] if v is not None]
        for v in vars_to_check:
            if v not in self.df.columns:
                raise ValueError(f"Variable '{v}' not found in data. Available: {list(self.df.columns)}")
        
        # Run the test
        result = test_methods[test_type]()
        
        # Add weights info if provided
        if weights:
            result['details']['weights_column'] = weights
            result['warnings'].append("Weights column specified but test may not support weighted analysis")
        
        # Log the operation
        self._log_operation("run_stat_test", {
            "test_type": test_type,
            "var1": var1,
            "var2": var2,
            "group": group,
            "weights": weights
        })
        
        return self._make_json_safe(result)
    
    def auto_test(
        self,
        var1: str,
        var2: Optional[str] = None,
        group: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Automatically select and run the appropriate statistical test.
        
        Args:
            var1: Primary variable name
            var2: Secondary variable name (optional)
            group: Grouping variable name (optional)
            
        Returns:
            Dict with test selection reasoning and test results
        """
        # Validate variables exist
        vars_to_check = [v for v in [var1, var2, group] if v is not None]
        for v in vars_to_check:
            if v not in self.df.columns:
                raise ValueError(f"Variable '{v}' not found in data. Available: {list(self.df.columns)}")
        
        # Get test recommendation
        selection = self.stats.auto_select_test(self.df, var1, var2, group)
        
        # Map recommended test to actual test type
        test_mapping = {
            'normality_tests': 'shapiro_test',
            'frequency_analysis': 'proportion_test',
            'correlation': 'pearson_corr',
            'chi_square': 'chi_square',
            'independent_t': 'independent_t',
            'paired_t': 'paired_t',
            'anova': 'anova',
            'two_proportion_test': 'two_proportion_test'
        }
        
        selected_test = selection.get('test', 'unknown')
        actual_test = test_mapping.get(selected_test, selected_test)
        
        # Get variable mapping if provided
        var_mapping = selection.get('variables', {})
        
        # Determine which variables to use
        test_var1 = var_mapping.get('value_col', var_mapping.get('col1', var1))
        test_var2 = var_mapping.get('col2', var2)
        test_group = var_mapping.get('group_col', group)
        
        # Run the selected test
        try:
            if actual_test in ['shapiro_test', 'ks_test', 'anderson_test', 'jb_test', 'proportion_test']:
                result = self.run_stat_test(actual_test, test_var1)
            elif actual_test in ['independent_t', 'anova', 'kruskal_test', 'two_proportion_test']:
                result = self.run_stat_test(actual_test, test_var1, group=test_group)
            elif actual_test in ['pearson_corr', 'spearman_corr', 'kendall_corr', 'chi_square', 'paired_t']:
                result = self.run_stat_test(actual_test, test_var1, test_var2)
            else:
                result = {"error": f"Could not determine how to run test: {actual_test}"}
        except Exception as e:
            result = {"error": str(e)}
        
        # Log the operation
        self._log_operation("auto_test", {
            "var1": var1,
            "var2": var2,
            "group": group,
            "selected_test": selected_test,
            "actual_test": actual_test
        })
        
        return self._make_json_safe({
            "selection": selection,
            "test_executed": actual_test,
            "result": result
        })
    
    # ==========================================
    # WELCH ANOVA, SHAPIRO-WILK, TUKEY HSD
    # ==========================================
    
    def run_welch_anova(
        self,
        group_col: str,
        value_col: str,
        weight_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform Welch's ANOVA (robust to unequal variances).
        
        Args:
            group_col: Name of the grouping column
            value_col: Name of the numeric value column
            weight_col: Optional weight column for weighted analysis
            
        Returns:
            Dict with Welch ANOVA results including F-statistic, p-value, effect size
        """
        timestamp = datetime.now().isoformat()
        warnings_list = []
        
        # Validate columns
        if group_col not in self.df.columns:
            return self._make_json_safe({
                "error": f"Group column '{group_col}' not found",
                "available_columns": list(self.df.columns)
            })
        
        if value_col not in self.df.columns:
            return self._make_json_safe({
                "error": f"Value column '{value_col}' not found",
                "available_columns": list(self.df.columns)
            })
        
        if weight_col and weight_col not in self.df.columns:
            return self._make_json_safe({
                "error": f"Weight column '{weight_col}' not found",
                "available_columns": list(self.df.columns)
            })
        
        # Check if value column is numeric
        if not pd.api.types.is_numeric_dtype(self.df[value_col]):
            return self._make_json_safe({
                "error": f"Value column '{value_col}' must be numeric"
            })
        
        # Get unique groups
        groups = self.df[group_col].dropna().unique()
        
        if len(groups) < 2:
            return self._make_json_safe({
                "error": f"Need at least 2 groups for ANOVA, found {len(groups)}",
                "groups_found": [str(g) for g in groups]
            })
        
        # Extract group data
        group_data = {}
        group_weights = {}
        group_stats = {}
        
        for g in groups:
            mask = self.df[group_col] == g
            values = self.df.loc[mask, value_col].dropna()
            
            if len(values) < 2:
                warnings_list.append(f"Group '{g}' has fewer than 2 observations ({len(values)})")
                continue
            
            if weight_col:
                weights = self.df.loc[mask, weight_col].reindex(values.index).fillna(0)
                valid_mask = weights > 0
                values = values[valid_mask]
                weights = weights[valid_mask]
                
                if len(values) < 2:
                    warnings_list.append(f"Group '{g}' has fewer than 2 valid weighted observations")
                    continue
                
                # Normalize weights within group
                weights = weights / weights.sum()
                
                # Weighted mean and variance
                mean = np.average(values, weights=weights)
                variance = np.average((values - mean) ** 2, weights=weights)
                # Effective sample size for weighted data
                n_eff = (weights.sum() ** 2) / (weights ** 2).sum()
                
                group_data[g] = values
                group_weights[g] = weights
                group_stats[g] = {
                    "n": len(values),
                    "n_effective": float(n_eff),
                    "mean": float(mean),
                    "variance": float(variance),
                    "std": float(np.sqrt(variance))
                }
            else:
                mean = values.mean()
                variance = values.var(ddof=1)
                
                group_data[g] = values
                group_stats[g] = {
                    "n": len(values),
                    "mean": float(mean),
                    "variance": float(variance),
                    "std": float(np.sqrt(variance)) if variance > 0 else 0
                }
        
        # Check if we have enough groups after filtering
        valid_groups = list(group_data.keys())
        if len(valid_groups) < 2:
            return self._make_json_safe({
                "error": "Need at least 2 groups with sufficient data for ANOVA",
                "warnings": warnings_list
            })
        
        # Check variance ratio
        variances = [group_stats[g]["variance"] for g in valid_groups if group_stats[g]["variance"] > 0]
        if len(variances) >= 2:
            var_ratio = max(variances) / min(variances) if min(variances) > 0 else float('inf')
            if var_ratio > 4:
                warnings_list.append(f"Variance ratio ({var_ratio:.2f}) > 4, suggesting heterogeneity of variances")
        
        # Check sample size equality
        sample_sizes = [group_stats[g]["n"] for g in valid_groups]
        if max(sample_sizes) / min(sample_sizes) > 1.5:
            warnings_list.append("Unequal sample sizes detected across groups")
        
        # Compute Welch's ANOVA
        k = len(valid_groups)
        
        if weight_col:
            # Weighted Welch's ANOVA
            n_eff = [group_stats[g]["n_effective"] for g in valid_groups]
            means = [group_stats[g]["mean"] for g in valid_groups]
            variances = [group_stats[g]["variance"] for g in valid_groups]
            
            # Weights for Welch: w_i = n_i / s_i^2
            w = []
            for i, g in enumerate(valid_groups):
                if variances[i] > 0:
                    w.append(n_eff[i] / variances[i])
                else:
                    w.append(0)
            w = np.array(w)
            
            if w.sum() == 0:
                return self._make_json_safe({
                    "error": "Cannot compute Welch ANOVA: all groups have zero variance"
                })
            
            # Weighted grand mean
            grand_mean = np.sum(w * means) / np.sum(w)
            
            # Welch F-statistic
            numerator = np.sum(w * (np.array(means) - grand_mean) ** 2) / (k - 1)
            
            # Lambda term for denominator adjustment
            lambda_term = 0
            for i, g in enumerate(valid_groups):
                if w[i] > 0 and n_eff[i] > 1:
                    lambda_term += ((1 - w[i] / np.sum(w)) ** 2) / (n_eff[i] - 1)
            lambda_term = 3 * lambda_term / (k ** 2 - 1)
            
            denominator = 1 + 2 * (k - 2) * lambda_term / 3
            F_stat = numerator / denominator if denominator > 0 else 0
            
            # Degrees of freedom
            df1 = k - 1
            df2_inv = lambda_term
            df2 = 1 / df2_inv if df2_inv > 0 else float('inf')
        else:
            # Unweighted Welch's ANOVA using scipy
            group_arrays = [group_data[g].values for g in valid_groups]
            
            # Use scipy's one-way ANOVA with Welch correction
            # First compute using standard formula
            n = np.array([len(arr) for arr in group_arrays])
            means = np.array([arr.mean() for arr in group_arrays])
            variances = np.array([arr.var(ddof=1) for arr in group_arrays])
            
            # Weights: w_i = n_i / s_i^2
            w = np.array([n[i] / variances[i] if variances[i] > 0 else 0 for i in range(k)])
            
            if w.sum() == 0:
                return self._make_json_safe({
                    "error": "Cannot compute Welch ANOVA: all groups have zero variance"
                })
            
            # Weighted grand mean
            grand_mean = np.sum(w * means) / np.sum(w)
            
            # Welch F-statistic
            numerator = np.sum(w * (means - grand_mean) ** 2) / (k - 1)
            
            # Lambda term
            lambda_term = 0
            for i in range(k):
                if w[i] > 0 and n[i] > 1:
                    lambda_term += ((1 - w[i] / np.sum(w)) ** 2) / (n[i] - 1)
            lambda_term = 3 * lambda_term / (k ** 2 - 1)
            
            denominator = 1 + 2 * (k - 2) * lambda_term / 3
            F_stat = numerator / denominator if denominator > 0 else 0
            
            # Degrees of freedom
            df1 = k - 1
            df2_inv = lambda_term
            df2 = 1 / df2_inv if df2_inv > 0 else float('inf')
        
        # P-value from F-distribution
        if df2 > 0 and not np.isinf(df2):
            p_value = 1 - stats.f.cdf(F_stat, df1, df2)
        else:
            p_value = None
            warnings_list.append("Could not compute p-value due to invalid degrees of freedom")
        
        # Effect size (eta-squared)
        all_values = pd.concat([group_data[g] for g in valid_groups])
        if weight_col:
            all_weights = pd.concat([group_weights[g] for g in valid_groups])
            all_weights = all_weights / all_weights.sum()
            overall_mean = np.average(all_values, weights=all_weights)
            ss_total = np.sum(all_weights * (all_values - overall_mean) ** 2)
            ss_between = sum(
                group_weights[g].sum() * (group_stats[g]["mean"] - overall_mean) ** 2
                for g in valid_groups
            )
        else:
            overall_mean = all_values.mean()
            ss_total = np.sum((all_values - overall_mean) ** 2)
            ss_between = sum(
                len(group_data[g]) * (group_stats[g]["mean"] - overall_mean) ** 2
                for g in valid_groups
            )
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        omega_squared = (ss_between - (k - 1) * (ss_total - ss_between) / (len(all_values) - k)) / (ss_total + (ss_total - ss_between) / (len(all_values) - k)) if ss_total > 0 else 0
        
        # Build result
        result = {
            "test": "welch_anova",
            "F_statistic": float(F_stat),
            "p_value": float(p_value) if p_value is not None else None,
            "df_between": int(df1),
            "df_within": float(df2) if not np.isinf(df2) else None,
            "effect_size": {
                "eta_squared": float(eta_squared),
                "omega_squared": float(omega_squared) if omega_squared > 0 else 0
            },
            "group_statistics": {str(g): group_stats[g] for g in valid_groups},
            "n_groups": len(valid_groups),
            "total_n": len(all_values),
            "weighted": weight_col is not None,
            "warnings": warnings_list,
            "interpretation": self._interpret_welch_anova(F_stat, p_value, eta_squared)
        }
        
        # Log operation
        self.operations_log.append({
            "timestamp": timestamp,
            "operation": "welch_anova",
            "details": {
                "group_col": group_col,
                "value_col": value_col,
                "weight_col": weight_col,
                "n_groups": len(valid_groups),
                "F_statistic": float(F_stat),
                "p_value": float(p_value) if p_value is not None else None
            }
        })
        
        return self._make_json_safe(result)
    
    def _interpret_welch_anova(self, F: float, p: Optional[float], eta_sq: float) -> str:
        """Generate interpretation for Welch ANOVA results."""
        if p is None:
            return "Could not determine significance due to computational issues"
        
        significance = "significant" if p < 0.05 else "not significant"
        
        if eta_sq < 0.01:
            effect = "negligible"
        elif eta_sq < 0.06:
            effect = "small"
        elif eta_sq < 0.14:
            effect = "medium"
        else:
            effect = "large"
        
        return f"The difference between groups is {significance} (p={p:.4f}) with a {effect} effect size (η²={eta_sq:.4f})"
    
    def run_shapiro_wilk(
        self,
        value_col: str
    ) -> Dict[str, Any]:
        """
        Perform Shapiro-Wilk test for normality with detailed diagnostics.
        
        Args:
            value_col: Name of the numeric column to test
            
        Returns:
            Dict with W statistic, p-value, interpretation, and recommendations
        """
        timestamp = datetime.now().isoformat()
        warnings_list = []
        
        # Validate column
        if value_col not in self.df.columns:
            return self._make_json_safe({
                "error": f"Column '{value_col}' not found",
                "available_columns": list(self.df.columns)
            })
        
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(self.df[value_col]):
            return self._make_json_safe({
                "error": f"Column '{value_col}' must be numeric for Shapiro-Wilk test"
            })
        
        # Get clean data
        data = self.df[value_col].dropna()
        
        # Handle edge cases
        if len(data) < 3:
            return self._make_json_safe({
                "error": f"Need at least 3 observations for Shapiro-Wilk test, found {len(data)}"
            })
        
        # Check for constant column
        if data.nunique() == 1:
            return self._make_json_safe({
                "error": "Column is constant (all values are identical)",
                "value": float(data.iloc[0])
            })
        
        # Limit sample size (scipy limitation)
        if len(data) > 5000:
            warnings_list.append(f"Sample size ({len(data)}) exceeds 5000, using random subsample")
            data = data.sample(n=5000, random_state=42)
        
        # Perform Shapiro-Wilk test
        try:
            w_stat, p_value = stats.shapiro(data)
        except Exception as e:
            return self._make_json_safe({
                "error": f"Shapiro-Wilk test failed: {str(e)}"
            })
        
        # Compute distribution characteristics
        skewness = float(stats.skew(data))
        kurtosis = float(stats.kurtosis(data))
        
        # Distribution flags
        distribution_flags = []
        recommended_transformations = []
        
        # Skewness interpretation
        if abs(skewness) > 2:
            distribution_flags.append("highly_skewed")
            if skewness > 0:
                distribution_flags.append("right_skewed")
                recommended_transformations.append("log")
                recommended_transformations.append("sqrt")
            else:
                distribution_flags.append("left_skewed")
                recommended_transformations.append("square")
                recommended_transformations.append("reflect_and_log")
        elif abs(skewness) > 1:
            distribution_flags.append("moderately_skewed")
            if skewness > 0:
                distribution_flags.append("right_skewed")
                recommended_transformations.append("sqrt")
            else:
                distribution_flags.append("left_skewed")
        
        # Kurtosis interpretation
        if kurtosis > 3:
            distribution_flags.append("heavy_tails")
            distribution_flags.append("leptokurtic")
        elif kurtosis < -1:
            distribution_flags.append("light_tails")
            distribution_flags.append("platykurtic")
        
        # Check for potential multimodality using Hartigan's dip test approximation
        # Simple check: compare histogram to normal distribution
        try:
            hist, bin_edges = np.histogram(data, bins='auto', density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            normal_pdf = stats.norm.pdf(bin_centers, data.mean(), data.std())
            
            # Count local maxima in histogram
            peaks = 0
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks += 1
            
            if peaks >= 2:
                distribution_flags.append("potentially_multimodal")
                warnings_list.append(f"Distribution may be multimodal ({peaks} peaks detected)")
        except:
            pass
        
        # Interpretation
        if p_value > 0.05:
            normality = "normal"
            interpretation = "Data appears to be normally distributed (fail to reject H0)"
        else:
            normality = "non-normal"
            interpretation = "Data does not appear to be normally distributed (reject H0)"
        
        # Add transformation recommendations based on Box-Cox
        if normality == "non-normal" and all(data > 0):
            recommended_transformations.append("box_cox")
        
        # Remove duplicates
        recommended_transformations = list(dict.fromkeys(recommended_transformations))
        
        # Build result
        result = {
            "test": "shapiro_wilk",
            "W_statistic": float(w_stat),
            "p_value": float(p_value),
            "n": len(data),
            "normality": normality,
            "interpretation": interpretation,
            "distribution_characteristics": {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std())
            },
            "distribution_flags": distribution_flags,
            "recommended_transformations": recommended_transformations if normality == "non-normal" else [],
            "warnings": warnings_list
        }
        
        # Log operation
        self.operations_log.append({
            "timestamp": timestamp,
            "operation": "shapiro_wilk",
            "details": {
                "value_col": value_col,
                "n": len(data),
                "W_statistic": float(w_stat),
                "p_value": float(p_value),
                "normality": normality
            }
        })
        
        return self._make_json_safe(result)
    
    def run_tukey_hsd(
        self,
        group_col: str,
        value_col: str,
        weight_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform Tukey's HSD (Honestly Significant Difference) post-hoc test.
        
        Args:
            group_col: Name of the grouping column
            value_col: Name of the numeric value column
            weight_col: Optional weight column for weighted analysis
            
        Returns:
            Dict with pairwise comparisons, mean differences, q-statistics, p-values, and CIs
        """
        timestamp = datetime.now().isoformat()
        warnings_list = []
        
        # Validate columns
        if group_col not in self.df.columns:
            return self._make_json_safe({
                "error": f"Group column '{group_col}' not found",
                "available_columns": list(self.df.columns)
            })
        
        if value_col not in self.df.columns:
            return self._make_json_safe({
                "error": f"Value column '{value_col}' not found",
                "available_columns": list(self.df.columns)
            })
        
        if weight_col and weight_col not in self.df.columns:
            return self._make_json_safe({
                "error": f"Weight column '{weight_col}' not found",
                "available_columns": list(self.df.columns)
            })
        
        # Check if value column is numeric
        if not pd.api.types.is_numeric_dtype(self.df[value_col]):
            return self._make_json_safe({
                "error": f"Value column '{value_col}' must be numeric"
            })
        
        # Get unique groups
        groups = sorted([str(g) for g in self.df[group_col].dropna().unique()])
        
        if len(groups) < 2:
            return self._make_json_safe({
                "error": f"Need at least 2 groups for Tukey HSD, found {len(groups)}"
            })
        
        # Extract group data
        group_data = {}
        group_weights = {}
        group_stats = {}
        
        for g in groups:
            mask = self.df[group_col].astype(str) == g
            values = self.df.loc[mask, value_col].dropna()
            
            if len(values) < 2:
                warnings_list.append(f"Group '{g}' has fewer than 2 observations ({len(values)})")
                continue
            
            if weight_col:
                weights = self.df.loc[mask, weight_col].reindex(values.index).fillna(0)
                valid_mask = weights > 0
                values = values[valid_mask]
                weights = weights[valid_mask]
                
                if len(values) < 2:
                    warnings_list.append(f"Group '{g}' has fewer than 2 valid weighted observations")
                    continue
                
                # Normalized weights
                norm_weights = weights / weights.sum()
                mean = np.average(values, weights=norm_weights)
                variance = np.average((values - mean) ** 2, weights=norm_weights)
                n_eff = (weights.sum() ** 2) / (weights ** 2).sum()
                
                group_data[g] = values
                group_weights[g] = weights
                group_stats[g] = {
                    "n": len(values),
                    "n_effective": float(n_eff),
                    "mean": float(mean),
                    "variance": float(variance),
                    "std": float(np.sqrt(variance))
                }
            else:
                group_data[g] = values
                group_stats[g] = {
                    "n": len(values),
                    "mean": float(values.mean()),
                    "variance": float(values.var(ddof=1)),
                    "std": float(values.std(ddof=1))
                }
        
        valid_groups = list(group_data.keys())
        
        if len(valid_groups) < 2:
            return self._make_json_safe({
                "error": "Need at least 2 groups with sufficient data for Tukey HSD",
                "warnings": warnings_list
            })
        
        k = len(valid_groups)
        
        # Compute pooled statistics
        if weight_col:
            # Weighted pooled variance
            total_weight = sum(group_weights[g].sum() for g in valid_groups)
            ss_within = sum(
                np.sum(group_weights[g] * (group_data[g] - group_stats[g]["mean"]) ** 2)
                for g in valid_groups
            )
            n_total = sum(group_stats[g]["n_effective"] for g in valid_groups)
            df_within = n_total - k
            ms_within = ss_within / df_within if df_within > 0 else 0
        else:
            # Unweighted pooled variance
            n_total = sum(len(group_data[g]) for g in valid_groups)
            df_within = n_total - k
            ss_within = sum(
                np.sum((group_data[g] - group_stats[g]["mean"]) ** 2)
                for g in valid_groups
            )
            ms_within = ss_within / df_within if df_within > 0 else 0
        
        pooled_std = np.sqrt(ms_within)
        
        # Pairwise comparisons
        comparisons = []
        
        for i in range(len(valid_groups)):
            for j in range(i + 1, len(valid_groups)):
                g1, g2 = valid_groups[i], valid_groups[j]
                
                mean1 = group_stats[g1]["mean"]
                mean2 = group_stats[g2]["mean"]
                mean_diff = mean1 - mean2
                
                if weight_col:
                    n1 = group_stats[g1]["n_effective"]
                    n2 = group_stats[g2]["n_effective"]
                else:
                    n1 = group_stats[g1]["n"]
                    n2 = group_stats[g2]["n"]
                
                # Standard error for pairwise comparison
                se = pooled_std * np.sqrt(0.5 * (1/n1 + 1/n2))
                
                # Q-statistic (studentized range)
                q_stat = abs(mean_diff) / se if se > 0 else 0
                
                # P-value from studentized range distribution
                # Using approximation via scipy
                try:
                    # Tukey's HSD uses the studentized range distribution
                    # p-value = P(q > |mean_diff|/SE)
                    from scipy.stats import studentized_range
                    p_value = 1 - studentized_range.cdf(q_stat, k, df_within)
                except (ImportError, AttributeError):
                    # Fallback: use t-distribution approximation
                    t_stat = q_stat / np.sqrt(2)
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_within))
                
                # Critical value for 95% CI
                try:
                    from scipy.stats import studentized_range
                    q_crit = studentized_range.ppf(0.95, k, df_within)
                except (ImportError, AttributeError):
                    # Approximation using t-distribution
                    q_crit = stats.t.ppf(0.975, df_within) * np.sqrt(2)
                
                # Confidence interval
                margin = q_crit * se
                ci_lower = mean_diff - margin
                ci_upper = mean_diff + margin
                
                # Significance
                significant = p_value < 0.05
                
                comparisons.append({
                    "group1": g1,
                    "group2": g2,
                    "mean1": float(mean1),
                    "mean2": float(mean2),
                    "mean_difference": float(mean_diff),
                    "std_error": float(se),
                    "q_statistic": float(q_stat),
                    "p_value": float(p_value),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "significant": significant,
                    "interpretation": f"{'Significant' if significant else 'No significant'} difference between {g1} and {g2}"
                })
        
        # Summary
        n_significant = sum(1 for c in comparisons if c["significant"])
        
        result = {
            "test": "tukey_hsd",
            "n_groups": k,
            "n_comparisons": len(comparisons),
            "n_significant": n_significant,
            "alpha": 0.05,
            "pooled_std": float(pooled_std),
            "df_within": float(df_within),
            "group_statistics": group_stats,
            "pairwise_comparisons": comparisons,
            "weighted": weight_col is not None,
            "warnings": warnings_list,
            "summary": f"{n_significant} of {len(comparisons)} pairwise comparisons are significant at α=0.05"
        }
        
        # Log operation
        self.operations_log.append({
            "timestamp": timestamp,
            "operation": "tukey_hsd",
            "details": {
                "group_col": group_col,
                "value_col": value_col,
                "weight_col": weight_col,
                "n_groups": k,
                "n_comparisons": len(comparisons),
                "n_significant": n_significant
            }
        })
        
        return self._make_json_safe(result)
    
    def get_operations_log(self) -> List[Dict[str, Any]]:
        """Return the operations log."""
        return self.operations_log
    
    def _make_json_safe(self, obj: Any) -> Any:
        """
        Recursively convert NaN, Inf, and -Inf to None for JSON serialization.
        
        Args:
            obj: Object to convert (dict, list, or scalar)
            
        Returns:
            JSON-safe object
        """
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return self._make_json_safe(obj.item())
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, np.ndarray):
            return self._make_json_safe(obj.tolist())
        else:
            return obj
