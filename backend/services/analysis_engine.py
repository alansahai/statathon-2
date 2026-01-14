"""
Analysis Engine - MoSPI-Compliant
Handles comprehensive statistical analysis including:
- Descriptive statistics (mean, median, std, skewness, kurtosis)
- Frequency distributions
- Crosstabulations
- Weighted statistics (if weight column exists)
- Distribution shape detection

All operations are deterministic and reproducible.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from scipy import stats

from services.file_manager import FileManager


class AnalysisEngine:
    """
    MoSPI-compliant statistical analysis engine for survey data processing.
    
    Provides comprehensive analysis including:
    - Descriptive statistics for numeric variables
    - Frequency distributions for categorical variables
    - Crosstabulations for categorical pairs
    - Weighted estimates (when weight column available)
    - Distribution shape detection (skewness, kurtosis)
    """
    
    def __init__(self):
        """Initialize AnalysisEngine."""
        pass
    
    @staticmethod
    def generate_statistics(filename: str) -> Dict[str, Any]:
        """
        Generate comprehensive statistical analysis for a dataset.
        
        Process Flow:
        1. Load best available file using FileManager
        2. Compute descriptive statistics for numeric columns
        3. Generate frequency distributions for categorical columns
        4. Create crosstabs for categorical pairs (limited to 3)
        5. Calculate weighted statistics if weight column exists
        6. Detect distribution shapes
        7. Return structured summary dictionary
        
        Args:
            filename: Name of the file to analyze (e.g., "survey.csv")
            
        Returns:
            Dictionary with structure:
            {
                "descriptive_stats": {...},
                "frequencies": {...},
                "crosstabs": {...},
                "weighted_stats": {...} (optional),
                "distribution_notes": {...}
            }
            
        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If file is empty or unreadable
            Exception: For other processing errors
        """
        try:
            # ============================================
            # STEP 1: Load Best Available File
            # ============================================
            file_path = FileManager.get_best_available_file(filename)
            
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except:
                try:
                    df = pd.read_excel(file_path)
                except Exception as e:
                    raise ValueError(f"Cannot read file format: {str(e)}")
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Initialize summary structure
            summary = {
                "descriptive_stats": {},
                "frequencies": {},
                "crosstabs": {},
                "weighted_stats": {},
                "distribution_notes": {}
            }
            
            # Identify column types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Detect weight column
            weight_col = AnalysisEngine._detect_weight_column(df)
            has_weights = weight_col is not None
            
            # ============================================
            # STEP 2: Descriptive Statistics (Numeric)
            # ============================================
            for col in numeric_cols:
                # Skip weight columns in descriptive stats
                if weight_col and col == weight_col:
                    continue
                
                try:
                    stats_dict = AnalysisEngine._compute_descriptive_stats(df, col)
                    summary["descriptive_stats"][col] = stats_dict
                except Exception as e:
                    summary["descriptive_stats"][col] = {"error": str(e)}
            
            # ============================================
            # STEP 3: Frequency Distributions (Categorical)
            # ============================================
            for col in categorical_cols:
                try:
                    freq_dict = AnalysisEngine._compute_frequencies(df, col)
                    summary["frequencies"][col] = freq_dict
                except Exception as e:
                    summary["frequencies"][col] = {"error": str(e)}
            
            # ============================================
            # STEP 4: Crosstabulations
            # ============================================
            if len(categorical_cols) >= 2:
                # Limit to first 3 categorical columns to avoid excessive computation
                cat_cols_limited = categorical_cols[:3]
                
                # Generate all pairs
                for i in range(len(cat_cols_limited)):
                    for j in range(i + 1, len(cat_cols_limited)):
                        col1 = cat_cols_limited[i]
                        col2 = cat_cols_limited[j]
                        crosstab_key = f"{col1}_vs_{col2}"
                        
                        try:
                            crosstab_dict = AnalysisEngine._compute_crosstab(df, col1, col2)
                            summary["crosstabs"][crosstab_key] = crosstab_dict
                        except Exception as e:
                            summary["crosstabs"][crosstab_key] = {"error": str(e)}
            
            # ============================================
            # STEP 5: Weighted Statistics (if weight exists)
            # ============================================
            if has_weights:
                for col in numeric_cols:
                    # Skip weight column itself
                    if col == weight_col:
                        continue
                    
                    try:
                        weighted_stats_dict = AnalysisEngine._compute_weighted_stats(
                            df, col, weight_col
                        )
                        summary["weighted_stats"][col] = weighted_stats_dict
                    except Exception as e:
                        summary["weighted_stats"][col] = {"error": str(e)}
                
                # Also compute weighted frequencies for categorical
                for col in categorical_cols:
                    try:
                        weighted_freq_dict = AnalysisEngine._compute_weighted_frequencies(
                            df, col, weight_col
                        )
                        
                        # Add to weighted_stats under special key
                        summary["weighted_stats"][f"{col}_frequencies"] = weighted_freq_dict
                    except Exception as e:
                        summary["weighted_stats"][f"{col}_frequencies"] = {"error": str(e)}
            
            # ============================================
            # STEP 6: Distribution Shape Detection
            # ============================================
            for col in numeric_cols:
                # Skip weight columns
                if weight_col and col == weight_col:
                    continue
                
                if col in summary["descriptive_stats"]:
                    stats_dict = summary["descriptive_stats"][col]
                    
                    # Check for skewness
                    skewness = stats_dict.get("skewness", 0)
                    kurtosis = stats_dict.get("kurtosis", 0)
                    
                    notes = []
                    
                    if abs(skewness) > 2:
                        if skewness > 0:
                            notes.append("highly right-skewed distribution")
                        else:
                            notes.append("highly left-skewed distribution")
                    elif abs(skewness) > 1:
                        if skewness > 0:
                            notes.append("moderately right-skewed")
                        else:
                            notes.append("moderately left-skewed")
                    
                    if kurtosis > 5:
                        notes.append("heavy-tailed distribution (high kurtosis)")
                    elif kurtosis < -1:
                        notes.append("light-tailed distribution (low kurtosis)")
                    
                    if notes:
                        summary["distribution_notes"][col] = ", ".join(notes)
                    else:
                        summary["distribution_notes"][col] = "approximately normal distribution"
            
            # ============================================
            # STEP 7: Return Structured Summary
            # ============================================
            return summary
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty or unreadable")
        
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise Exception(f"Analysis failed: {str(e)}")
    
    @staticmethod
    def _detect_weight_column(df: pd.DataFrame) -> Optional[str]:
        """
        Detect weight column in dataset.
        
        Searches for columns with names containing:
        - "weight"
        - "wt"
        - "w_"
        
        Args:
            df: Input DataFrame
            
        Returns:
            Weight column name or None if not found
        """
        # Check for common weight column names
        weight_keywords = ['weight', 'wt', 'w_', 'base_weight', 'poststrat_weight']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in weight_keywords):
                print(f"Detected weight column: '{col}'")
                return col
        
        return None
    
    @staticmethod
    def _compute_descriptive_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Compute comprehensive descriptive statistics for a numeric column.
        
        Statistics computed:
        - mean, median, std, min, max
        - q1 (25th percentile), q3 (75th percentile)
        - skewness, kurtosis
        - missing_count, missing_pct
        - non_missing_count
        
        Args:
            df: Input DataFrame
            column: Column name
            
        Returns:
            Dictionary with statistics
        """
        col_data = df[column]
        
        # Missing value info
        missing_count = col_data.isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        non_null = col_data.dropna()
        
        if len(non_null) == 0:
            return {
                "error": "All values are missing",
                "missing_count": int(missing_count),
                "missing_pct": float(missing_pct)
            }
        
        # Basic statistics
        stats_dict = {
            "count": int(len(non_null)),
            "missing_count": int(missing_count),
            "missing_pct": float(round(missing_pct, 2)),
            "mean": float(non_null.mean()),
            "median": float(non_null.median()),
            "std": float(non_null.std()),
            "min": float(non_null.min()),
            "max": float(non_null.max()),
            "q1": float(non_null.quantile(0.25)),
            "q3": float(non_null.quantile(0.75))
        }
        
        # Skewness and kurtosis (require at least 3 observations)
        if len(non_null) >= 3:
            try:
                stats_dict["skewness"] = float(stats.skew(non_null))
                stats_dict["kurtosis"] = float(stats.kurtosis(non_null))
            except:
                stats_dict["skewness"] = 0.0
                stats_dict["kurtosis"] = 0.0
        else:
            stats_dict["skewness"] = 0.0
            stats_dict["kurtosis"] = 0.0
        
        return stats_dict
    
    @staticmethod
    def _compute_frequencies(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Compute frequency distribution for a categorical column.
        
        Args:
            df: Input DataFrame
            column: Column name
            
        Returns:
            Dictionary with:
            - counts: Raw frequencies
            - proportions: Relative frequencies (%)
            - unique_count: Number of unique categories
            - missing_count: Number of missing values
        """
        col_data = df[column]
        
        # Missing values
        missing_count = col_data.isna().sum()
        
        # Value counts
        counts = col_data.value_counts(dropna=True)
        total = counts.sum()
        
        # Convert to proportions (percentages)
        proportions = (counts / total * 100).round(2)
        
        return {
            "counts": counts.to_dict(),
            "proportions": proportions.to_dict(),
            "unique_count": int(len(counts)),
            "missing_count": int(missing_count),
            "total_valid": int(total)
        }
    
    @staticmethod
    def _compute_crosstab(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Compute crosstabulation between two categorical variables.
        
        Args:
            df: Input DataFrame
            col1: First categorical column
            col2: Second categorical column
            
        Returns:
            Dictionary with:
            - crosstab: Raw frequency table
            - row_percentages: Row-wise percentages
            - col_percentages: Column-wise percentages
            - chi2_statistic: Chi-square test statistic (if applicable)
            - p_value: P-value from chi-square test
        """
        # Create crosstab
        ct = pd.crosstab(df[col1], df[col2], dropna=True)
        
        # Row percentages
        row_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        
        # Column percentages
        col_pct = ct.div(ct.sum(axis=0), axis=1) * 100
        
        # Chi-square test (if enough data)
        chi2_stat = None
        p_value = None
        
        if ct.shape[0] >= 2 and ct.shape[1] >= 2 and ct.min().min() >= 5:
            try:
                from scipy.stats import chi2_contingency
                chi2_stat, p_value, dof, expected = chi2_contingency(ct)
            except:
                pass
        
        result = {
            "crosstab": ct.to_dict(),
            "row_percentages": row_pct.round(2).to_dict(),
            "col_percentages": col_pct.round(2).to_dict()
        }
        
        if chi2_stat is not None:
            result["chi2_statistic"] = float(chi2_stat)
            result["p_value"] = float(p_value)
            result["significant"] = p_value < 0.05 if p_value is not None else False
        
        return result
    
    @staticmethod
    def _compute_weighted_stats(df: pd.DataFrame, column: str, weight_col: str) -> Dict[str, Any]:
        """
        Compute weighted statistics for a numeric column.
        
        Args:
            df: Input DataFrame
            column: Column name
            weight_col: Weight column name
            
        Returns:
            Dictionary with:
            - weighted_mean
            - weighted_total
            - unweighted_mean (for comparison)
            - design_effect
        """
        # Get valid observations
        valid_mask = df[column].notna() & df[weight_col].notna() & (df[weight_col] > 0)
        
        if not valid_mask.any():
            return {"error": "No valid observations"}
        
        values = df.loc[valid_mask, column]
        weights = df.loc[valid_mask, weight_col]
        
        # Weighted mean
        weighted_mean_val = AnalysisEngine.weighted_mean(values, weights)
        
        # Weighted total
        weighted_total_val = AnalysisEngine.weighted_total(values, weights)
        
        # Unweighted mean (for comparison)
        unweighted_mean_val = float(values.mean())
        
        # Design effect (DEFF) - ratio of weighted to unweighted variance
        # Simplified: uses effective sample size approximation
        effective_n = (weights.sum() ** 2) / (weights ** 2).sum()
        design_effect = len(weights) / effective_n
        
        return {
            "weighted_mean": float(weighted_mean_val),
            "weighted_total": float(weighted_total_val),
            "unweighted_mean": float(unweighted_mean_val),
            "design_effect": float(design_effect),
            "effective_sample_size": float(effective_n)
        }
    
    @staticmethod
    def _compute_weighted_frequencies(df: pd.DataFrame, column: str, weight_col: str) -> Dict[str, Any]:
        """
        Compute weighted frequency distribution for a categorical column.
        
        Args:
            df: Input DataFrame
            column: Column name
            weight_col: Weight column name
            
        Returns:
            Dictionary with weighted counts and proportions
        """
        # Get valid observations
        valid_mask = df[column].notna() & df[weight_col].notna() & (df[weight_col] > 0)
        
        if not valid_mask.any():
            return {"error": "No valid observations"}
        
        # Compute weighted counts
        weighted_counts = df.loc[valid_mask].groupby(column)[weight_col].sum()
        total_weight = weighted_counts.sum()
        
        # Compute proportions
        weighted_proportions = (weighted_counts / total_weight * 100).round(2)
        
        return {
            "weighted_counts": weighted_counts.to_dict(),
            "weighted_proportions": weighted_proportions.to_dict(),
            "total_weight": float(total_weight)
        }
    
    @staticmethod
    def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        """
        Calculate weighted mean.
        
        Formula: Σ(w_i * x_i) / Σ(w_i)
        
        Args:
            values: Series of values
            weights: Series of weights
            
        Returns:
            Weighted mean
        """
        if len(values) != len(weights):
            raise ValueError("Values and weights must have same length")
        
        weighted_sum = np.sum(weights * values)
        weight_sum = np.sum(weights)
        
        if weight_sum == 0:
            raise ValueError("Sum of weights is zero")
        
        return float(weighted_sum / weight_sum)
    
    @staticmethod
    def weighted_total(values: pd.Series, weights: pd.Series) -> float:
        """
        Calculate weighted total (population estimate).
        
        Formula: Σ(w_i * x_i)
        
        Args:
            values: Series of values
            weights: Series of weights
            
        Returns:
            Weighted total
        """
        if len(values) != len(weights):
            raise ValueError("Values and weights must have same length")
        
        return float(np.sum(weights * values))
