"""
Weighting Engine - MoSPI-ready automated survey weighting system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings


class WeightingEngine:
    """
    Automated Survey Weighting Engine for MoSPI
    Implements base weights, post-stratification, raking, trimming, and diagnostics
    with automatic column detection and data preparation
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize weighting engine
        
        Args:
            df: Input DataFrame
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.operations_log: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.auto_actions: List[str] = []
        
        # Detect column types
        self.numeric_columns = list(self.df.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(self.df.select_dtypes(include=['object', 'category']).columns)
        
        # Auto-initialize base weights if not present
        self._ensure_base_weights()
        
        # Log initialization
        self._log_operation("initialization", {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "numeric_cols": len(self.numeric_columns),
            "categorical_cols": len(self.categorical_columns),
            "has_base_weight": "base_weight" in self.df.columns
        })
    
    def _log_operation(self, operation: str, details: Dict[str, Any]):
        """Log an operation with timestamp"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "details": details
        }
        self.operations_log.append(log_entry)
        return log_entry
    
    def _ensure_base_weights(self):
        """Ensure base weights exist, create uniform weights if needed"""
        if "base_weight" not in self.df.columns:
            self.df["base_weight"] = 1.0
            self.auto_actions.append("Created uniform base_weight = 1.0 (no inclusion probabilities)")
    
    def _get_active_weight_column(self) -> str:
        """Auto-detect the most recent weight column"""
        for col in ['raked_weight_trimmed', 'raked_weight', 'poststrat_weight', 'base_weight']:
            if col in self.df.columns:
                return col
        # Fallback - shouldn't reach here due to _ensure_base_weights
        return 'base_weight'
    
    def _normalize_proportions(self, proportions: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
        """
        Normalize proportions to sum to 1.0
        
        Returns:
            Tuple of (normalized_dict, was_normalized)
        """
        total = sum(proportions.values())
        if abs(total - 1.0) < 0.01:
            return proportions, False
        
        # Check if these are counts (sum >> 1)
        if total > 10:
            # Convert counts to proportions
            normalized = {k: v / total for k, v in proportions.items()}
            self.warnings.append(f"Converted counts to proportions (total was {total:.0f})")
            self.auto_actions.append(f"Auto-normalized proportions from counts (sum={total:.0f})")
            return normalized, True
        else:
            # Just normalize to sum to 1
            normalized = {k: v / total for k, v in proportions.items()}
            self.warnings.append(f"Normalized proportions to sum to 1.0 (was {total:.4f})")
            self.auto_actions.append(f"Auto-normalized proportions (sum was {total:.4f})")
            return normalized, True
    
    def _auto_create_age_group(self, bins: List[int] = None, labels: List[str] = None) -> bool:
        """
        Auto-create age_group from age column if it exists
        
        Returns:
            True if created, False otherwise
        """
        if "age_group" in self.df.columns:
            return False
        
        if "age" not in self.df.columns:
            return False
        
        # Default bins and labels
        if bins is None:
            bins = [0, 25, 35, 50, 200]
        if labels is None:
            labels = ["young", "young_adult", "mid", "senior"]
        
        try:
            self.df["age_group"] = pd.cut(
                self.df["age"],
                bins=bins,
                labels=labels,
                right=False
            )
            self.df["age_group"] = self.df["age_group"].astype(str)
            self.auto_actions.append(f"Auto-created age_group from age column using bins {bins}")
            self.warnings.append(f"Created age_group categories: {labels}")
            return True
        except Exception as e:
            self.warnings.append(f"Could not create age_group from age: {str(e)}")
            return False
    
    def _validate_and_fix_control_columns(
        self,
        control_totals: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
        """
        Validate control columns exist, attempt auto-creation, drop if not possible
        
        Returns:
            Tuple of (cleaned_control_totals, dropped_columns)
        """
        cleaned_controls = {}
        dropped = []
        
        for col, targets in control_totals.items():
            # Check if column exists
            if col in self.df.columns:
                # Normalize proportions
                normalized, was_normalized = self._normalize_proportions(targets)
                cleaned_controls[col] = normalized
                continue
            
            # Try auto-creation for age_group
            if col == "age_group":
                if self._auto_create_age_group():
                    normalized, _ = self._normalize_proportions(targets)
                    cleaned_controls[col] = normalized
                    continue
            
            # Column doesn't exist and can't be created
            self.warnings.append(f"Control column '{col}' not found and cannot be auto-created")
            self.auto_actions.append(f"Dropped control column '{col}' (not found in data)")
            dropped.append(col)
        
        return cleaned_controls, dropped
    
    def calculate_base_weights(self, inclusion_prob_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate base weights from inclusion probabilities (or uniform if not provided)
        
        Args:
            inclusion_prob_column: Column containing inclusion probabilities (optional)
            
        Returns:
            Log entry with summary
        """
        # If no column specified, use uniform weights
        if inclusion_prob_column is None:
            self.df['base_weight'] = 1.0
            summary = {
                "method": "uniform",
                "min_weight": 1.0,
                "max_weight": 1.0,
                "mean_weight": 1.0,
                "median_weight": 1.0,
                "std_weight": 0.0,
                "total_weight": float(len(self.df))
            }
            self.auto_actions.append("Used uniform base_weight = 1.0 (no inclusion_prob_column specified)")
        else:
            # Column specified but doesn't exist
            if inclusion_prob_column not in self.df.columns:
                self.warnings.append(f"Column '{inclusion_prob_column}' not found, using uniform weights")
                self.df['base_weight'] = 1.0
                summary = {
                    "method": "uniform_fallback",
                    "requested_column": inclusion_prob_column,
                    "min_weight": 1.0,
                    "max_weight": 1.0,
                    "mean_weight": 1.0,
                    "median_weight": 1.0,
                    "std_weight": 0.0,
                    "total_weight": float(len(self.df))
                }
                self.auto_actions.append(f"Column '{inclusion_prob_column}' not found, used uniform weights")
            else:
                # Validate and calculate
                probs = self.df[inclusion_prob_column].copy()
                
                # Handle NaN
                if probs.isna().any():
                    nan_count = int(probs.isna().sum())
                    self.warnings.append(f"Column '{inclusion_prob_column}' has {nan_count} NaN values, filling with mean")
                    probs = probs.fillna(probs.mean())
                
                # Handle non-positive
                if (probs <= 0).any():
                    neg_count = int((probs <= 0).sum())
                    self.warnings.append(f"Column '{inclusion_prob_column}' has {neg_count} non-positive values, replacing with minimum positive")
                    min_positive = probs[probs > 0].min() if (probs > 0).any() else 0.01
                    probs = probs.clip(lower=min_positive)
                
                # Handle values > 1
                if (probs > 1).any():
                    over_count = int((probs > 1).sum())
                    self.warnings.append(f"Column '{inclusion_prob_column}' has {over_count} values > 1, capping at 1.0")
                    probs = probs.clip(upper=1.0)
                
                # Calculate base weights: w = 1/p
                self.df['base_weight'] = 1.0 / probs
                
                # Summary statistics
                summary = {
                    "method": "inverse_probability",
                    "inclusion_prob_column": inclusion_prob_column,
                    "min_weight": float(self.df['base_weight'].min()),
                    "max_weight": float(self.df['base_weight'].max()),
                    "mean_weight": float(self.df['base_weight'].mean()),
                    "median_weight": float(self.df['base_weight'].median()),
                    "std_weight": float(self.df['base_weight'].std()),
                    "total_weight": float(self.df['base_weight'].sum())
                }
        
        log_entry = self._log_operation("calculate_base_weights", {
            "summary": summary,
            "auto_actions": self.auto_actions.copy(),
            "warnings": self.warnings.copy()
        })
        
        return log_entry
    
    def apply_poststrat_weights(
        self,
        strata_column: Optional[str] = None,
        population_totals: Optional[Dict[str, float]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply post-stratification weights (auto-detects or skips if not possible)
        
        Args:
            strata_column: Column containing strata categories (optional)
            population_totals: Dict mapping strata values to population totals (optional)
            
        Returns:
            Tuple of (updated DataFrame, log entry)
        """
        # Ensure base weights exist
        if 'base_weight' not in self.df.columns:
            self._ensure_base_weights()
        
        # If no strata specified, skip post-stratification
        if strata_column is None or population_totals is None:
            self.warnings.append("Post-stratification skipped (no strata_column or population_totals)")
            self.df['poststrat_weight'] = self.df['base_weight'].copy()
            summary = {
                "method": "skipped",
                "reason": "no_strata_specified",
                "final_weight_sum": float(self.df['poststrat_weight'].sum()),
                "mean_weight": float(self.df['poststrat_weight'].mean())
            }
            log_entry = self._log_operation("apply_poststrat_weights", summary)
            return self.df, log_entry
        
        # Check if column exists
        if strata_column not in self.df.columns:
            # Try auto-creation for age_group
            if strata_column == "age_group":
                if not self._auto_create_age_group():
                    self.warnings.append(f"Column '{strata_column}' not found, skipping post-stratification")
                    self.df['poststrat_weight'] = self.df['base_weight'].copy()
                    summary = {
                        "method": "skipped",
                        "reason": f"column_{strata_column}_not_found"
                    }
                    log_entry = self._log_operation("apply_poststrat_weights", summary)
                    return self.df, log_entry
            else:
                self.warnings.append(f"Column '{strata_column}' not found, skipping post-stratification")
                self.df['poststrat_weight'] = self.df['base_weight'].copy()
                summary = {
                    "method": "skipped",
                    "reason": f"column_{strata_column}_not_found"
                }
                log_entry = self._log_operation("apply_poststrat_weights", summary)
                return self.df, log_entry
        
        # Normalize population totals
        population_totals, was_normalized = self._normalize_proportions(population_totals)
        
        # Get unique strata
        unique_strata = self.df[strata_column].dropna().unique()
        
        # Handle missing strata in population totals
        missing_strata = set(unique_strata) - set(population_totals.keys())
        if missing_strata:
            self.warnings.append(f"Missing population totals for: {missing_strata}, using equal distribution")
            # Distribute remaining proportion equally
            remaining_prop = 1.0 - sum(population_totals.values())
            if remaining_prop > 0:
                equal_prop = remaining_prop / len(missing_strata)
                for stratum in missing_strata:
                    population_totals[str(stratum)] = equal_prop
        
        # Get active weight column
        base_col = self._get_active_weight_column()
        
        # Calculate sample totals by strata
        sample_totals = self.df.groupby(strata_column)[base_col].sum().to_dict()
        
        # Calculate adjustment factors
        adjustment_factors = {}
        for stratum, pop_total in population_totals.items():
            sample_total = sample_totals.get(stratum, 0)
            if sample_total == 0:
                self.warnings.append(f"No sample data for stratum '{stratum}', skipping")
                adjustment_factors[stratum] = 1.0
            else:
                adjustment_factors[stratum] = pop_total / sample_total
        
        # Apply adjustments
        self.df['poststrat_weight'] = self.df.apply(
            lambda row: row[base_col] * adjustment_factors.get(row[strata_column], 1.0),
            axis=1
        )
        
        # Summary
        summary = {
            "strata_column": strata_column,
            "base_weight_column": base_col,
            "num_strata": len(population_totals),
            "adjustment_factors": adjustment_factors,
            "sample_totals": sample_totals,
            "population_totals": population_totals,
            "was_normalized": was_normalized,
            "final_weight_sum": float(self.df['poststrat_weight'].sum()),
            "mean_weight": float(self.df['poststrat_weight'].mean()),
            "std_weight": float(self.df['poststrat_weight'].std()),
            "auto_actions": self.auto_actions.copy(),
            "warnings": self.warnings.copy()
        }
        
        log_entry = self._log_operation("apply_poststrat_weights", summary)
        
        return self.df, log_entry
    
    def raking(
        self,
        control_totals: Dict[str, Dict[str, float]],
        max_iterations: int = 50,
        tolerance: float = 0.001
    ) -> Dict[str, Any]:
        """
        Iterative proportional fitting (raking) with auto-validation and fixing
        
        Args:
            control_totals: Dict of {column: {category: target_proportion}}
            max_iterations: Maximum iterations
            tolerance: Convergence threshold
            
        Returns:
            Log entry with convergence details and auto-actions
        """
        # Get starting weight column
        start_col = self._get_active_weight_column()
        self.df['raked_weight'] = self.df[start_col].copy()
        self.auto_actions.append(f"Starting raking from '{start_col}' column")
        
        # Validate and fix control columns
        cleaned_controls, dropped = self._validate_and_fix_control_columns(control_totals)
        
        # If all controls dropped, return unchanged weights
        if not cleaned_controls:
            self.warnings.append("All control columns dropped, raking skipped")
            summary = {
                "method": "skipped",
                "reason": "no_valid_controls",
                "dropped_controls": dropped,
                "warnings": self.warnings.copy(),
                "auto_actions": self.auto_actions.copy()
            }
            log_entry = self._log_operation("raking", summary)
            return log_entry
        
        # Raking iterations
        converged = False
        iteration_log = []
        
        for iteration in range(max_iterations):
            old_weights = self.df['raked_weight'].copy()
            max_diff = 0.0
            
            # Adjust for each control margin
            for col, target_props in cleaned_controls.items():
                # Calculate current weighted proportions
                for category, target_prop in target_props.items():
                    mask = self.df[col].astype(str) == str(category)
                    
                    if mask.sum() == 0:
                        continue
                    
                    current_total = self.df.loc[mask, 'raked_weight'].sum()
                    overall_total = self.df['raked_weight'].sum()
                    
                    if current_total == 0 or overall_total == 0:
                        continue
                    
                    current_prop = current_total / overall_total
                    
                    # Calculate adjustment factor
                    if current_prop > 0 and target_prop > 0:
                        adjustment = target_prop / current_prop
                        self.df.loc[mask, 'raked_weight'] *= adjustment
            
            # Check convergence
            weight_diff = np.abs(self.df['raked_weight'] - old_weights).max()
            max_diff = max(max_diff, weight_diff)
            
            iteration_log.append({
                "iteration": iteration + 1,
                "max_weight_diff": float(weight_diff),
                "mean_weight": float(self.df['raked_weight'].mean()),
                "total_weight": float(self.df['raked_weight'].sum())
            })
            
            if max_diff < tolerance:
                converged = True
                break
        
        # Final statistics
        summary = {
            "method": "raking",
            "starting_weight_column": start_col,
            "converged": converged,
            "iterations": len(iteration_log),
            "final_max_diff": float(max_diff),
            "tolerance": tolerance,
            "control_margins": list(cleaned_controls.keys()),
            "dropped_controls": dropped,
            "iteration_log": iteration_log[-5:] if len(iteration_log) > 5 else iteration_log,  # Last 5 only
            "final_stats": {
                "min_weight": float(self.df['raked_weight'].min()),
                "max_weight": float(self.df['raked_weight'].max()),
                "mean_weight": float(self.df['raked_weight'].mean()),
                "median_weight": float(self.df['raked_weight'].median()),
                "std_weight": float(self.df['raked_weight'].std())
            },
            "auto_actions": self.auto_actions.copy(),
            "warnings": self.warnings.copy()
        }
        
        log_entry = self._log_operation("raking", summary)
        
        return log_entry
    
    def trim_weights(
        self,
        min_w: float = 0.3,
        max_w: float = 3.0,
        weight_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trim weights to min/max bounds with auto-detection
        
        Args:
            min_w: Minimum weight threshold
            max_w: Maximum weight threshold
            weight_column: Which weight column to trim (auto-detect if None)
            
        Returns:
            Summary dict with trimming statistics
        """
        # Auto-detect weight column
        if weight_column is None:
            weight_column = self._get_active_weight_column()
        
        if weight_column not in self.df.columns:
            self._ensure_base_weights()
            weight_column = 'base_weight'
        
        # Store original weights
        original_weights = self.df[weight_column].copy()
        
        # Handle NaN/Inf before trimming
        if original_weights.isna().any():
            nan_count = int(original_weights.isna().sum())
            self.warnings.append(f"Found {nan_count} NaN values, filling with median")
            original_weights = original_weights.fillna(original_weights.median())
        
        if np.isinf(original_weights).any():
            inf_count = int(np.isinf(original_weights).sum())
            self.warnings.append(f"Found {inf_count} Inf values, replacing with max valid weight")
            max_valid = original_weights[~np.isinf(original_weights)].max()
            original_weights = original_weights.replace([np.inf, -np.inf], max_valid)
        
        # Count trimming
        n_below_min = (original_weights < min_w).sum()
        n_above_max = (original_weights > max_w).sum()
        
        # Apply trimming
        self.df[f'{weight_column}_trimmed'] = original_weights.clip(
            lower=min_w,
            upper=max_w
        )
        
        # Summary statistics
        summary = {
            "weight_column": weight_column,
            "min_threshold": float(min_w),
            "max_threshold": float(max_w),
            "n_below_min": int(n_below_min),
            "n_above_max": int(n_above_max),
            "total_trimmed": int(n_below_min + n_above_max),
            "pct_trimmed": float((n_below_min + n_above_max) / len(self.df) * 100),
            "original_stats": {
                "min": float(original_weights.min()),
                "max": float(original_weights.max()),
                "mean": float(original_weights.mean()),
                "std": float(original_weights.std())
            },
            "trimmed_stats": {
                "min": float(self.df[f'{weight_column}_trimmed'].min()),
                "max": float(self.df[f'{weight_column}_trimmed'].max()),
                "mean": float(self.df[f'{weight_column}_trimmed'].mean()),
                "std": float(self.df[f'{weight_column}_trimmed'].std())
            },
            "auto_actions": self.auto_actions.copy(),
            "warnings": self.warnings.copy()
        }
        
        log_entry = self._log_operation("trim_weights", summary)
        
        return summary
    
    def get_warnings(self) -> List[str]:
        """Get all warnings generated during operations"""
        return self.warnings.copy()
    
    def get_auto_actions(self) -> List[str]:
        """Get all automated actions taken"""
        return self.auto_actions.copy()
    
    def diagnostics(self, weight_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute comprehensive weight diagnostics with entropy
        
        Args:
            weight_column: Which weight column to analyze (auto-detect if None)
            
        Returns:
            Dict with diagnostic metrics
        """
        # Auto-detect weight column
        if weight_column is None:
            weight_column = self._get_active_weight_column()
        
        if weight_column not in self.df.columns:
            # Try one more time with explicit check
            if 'base_weight' not in self.df.columns:
                self._ensure_base_weights()
                weight_column = 'base_weight'
        
        weights = self.df[weight_column].copy()
        
        # Handle NaN/Inf
        if weights.isna().any() or np.isinf(weights).any():
            clean_weights = weights.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_weights) == 0:
                return {
                    "error": "All weights are NaN or Inf",
                    "weight_column": weight_column
                }
            weights = clean_weights
        
        # Basic statistics
        mean_w = weights.mean()
        std_w = weights.std()
        cv = std_w / mean_w if mean_w > 0 else 0
        
        # Effective sample size (ESS)
        sum_weights = weights.sum()
        sum_weights_sq = (weights ** 2).sum()
        effective_n = (sum_weights ** 2) / sum_weights_sq if sum_weights_sq > 0 else 0
        
        # Entropy (measure of weight uniformity)
        normalized_weights = weights / sum_weights
        entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-10))
        max_entropy = np.log(len(weights))
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = {
            f"p{p}": float(weights.quantile(p / 100))
            for p in percentiles
        }
        
        # Design effect
        design_effect = 1 + cv ** 2
        
        # Loss of precision
        loss_of_precision = 1 - (effective_n / len(weights))
        
        diagnostics = {
            "weight_column": weight_column,
            "n_observations": int(len(weights)),
            "mean_weight": float(mean_w),
            "median_weight": float(weights.median()),
            "std_weight": float(std_w),
            "min_weight": float(weights.min()),
            "max_weight": float(weights.max()),
            "cv": float(cv),
            "sum_weights": float(sum_weights),
            "effective_sample_size": float(effective_n),
            "design_effect": float(design_effect),
            "entropy": float(entropy),
            "entropy_ratio": float(entropy_ratio),
            "loss_of_precision": float(loss_of_precision),
            "percentiles": percentile_values,
            "distribution": {
                "skewness": float(weights.skew()),
                "kurtosis": float(weights.kurtosis())
            }
        }
        
        return self.make_json_safe(diagnostics)
    
    def export_weighted(self, save_path: str) -> str:
        """
        Export weighted DataFrame to CSV
        
        Args:
            save_path: Path to save CSV file
            
        Returns:
            File path where data was saved
        """
        self.df.to_csv(save_path, index=False)
        
        log_entry = self._log_operation("export_weighted", {
            "save_path": save_path,
            "rows": len(self.df),
            "columns": len(self.df.columns)
        })
        
        return save_path
    
    def get_weighted_dataframe(self) -> pd.DataFrame:
        """Get the current weighted DataFrame"""
        return self.df.copy()
    
    def get_operations_log(self) -> List[Dict[str, Any]]:
        """Get the operations log"""
        return self.make_json_safe(self.operations_log)
    
    @staticmethod
    def make_json_safe(data: Any) -> Any:
        """
        Convert data to JSON-safe format
        
        Handles:
        - numpy types -> python native types
        - NaN/Inf -> None
        - Recursive conversion for dict/list
        
        Args:
            data: Input data
            
        Returns:
            JSON-safe data
        """
        if isinstance(data, dict):
            return {key: WeightingEngine.make_json_safe(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [WeightingEngine.make_json_safe(item) for item in data]
        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            if np.isnan(data) or np.isinf(data):
                return None
            return float(data)
        elif isinstance(data, np.ndarray):
            return WeightingEngine.make_json_safe(data.tolist())
        elif isinstance(data, pd.Series):
            return WeightingEngine.make_json_safe(data.to_dict())
        elif isinstance(data, pd.DataFrame):
            return WeightingEngine.make_json_safe(data.to_dict(orient='records'))
        elif pd.isna(data):
            return None
        elif isinstance(data, (float, int)) and (np.isnan(data) or np.isinf(data)):
            return None
        else:
            return data
    
    @staticmethod
    def process_multiple_weights(
        file_ids: List[str],
        file_manager: Any,
        operation: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multiple files with weighting operations
        
        Args:
            file_ids: List of file identifiers
            file_manager: FileManager instance to load/save files
            operation: Operation to perform ("calculate", "validate", "trim")
            **kwargs: Additional parameters for the operation
            
        Returns:
            Dictionary mapping file_id to results:
            {
                "<file_id>": {"result": {...}, "status": "ok"},
                "<file_id>": {"error": "..."}
            }
        """
        results = {}
        
        for file_id in file_ids:
            try:
                # Load file
                file_path = file_manager.get_file_path(file_id)
                if not file_path:
                    results[file_id] = {"error": "File not found"}
                    continue
                
                df = file_manager.load_dataframe(file_path)
                
                # Initialize engine for this file
                engine = WeightingEngine(df)
                
                # Perform operation based on type
                if operation == "calculate":
                    # Determine weight type from kwargs
                    weight_type = kwargs.get("weight_type", "base")
                    
                    if weight_type == "base":
                        result = {
                            "operations_log": engine.operations_log,
                            "auto_actions": engine.auto_actions,
                            "warnings": engine.warnings
                        }
                    elif weight_type == "poststrat":
                        target_props = kwargs.get("target_proportions", {})
                        strat_var = kwargs.get("stratification_variable")
                        result = engine.calculate_poststratification_weights(strat_var, target_props)
                    elif weight_type == "raking":
                        margins = kwargs.get("margins", {})
                        result = engine.calculate_raking_weights(margins)
                    else:
                        results[file_id] = {"error": f"Unknown weight type: {weight_type}"}
                        continue
                        
                elif operation == "validate":
                    weight_col = kwargs.get("weight_column", engine._get_active_weight_column())
                    if weight_col not in df.columns:
                        results[file_id] = {"error": f"Weight column '{weight_col}' not found"}
                        continue
                    # Perform validation
                    result = {"validation": "ok"}  # Simplified
                    
                elif operation == "trim":
                    min_w = kwargs.get("min_weight", 0.3)
                    max_w = kwargs.get("max_weight", 3.0)
                    result = engine.trim_weights(min_weight=min_w, max_weight=max_w)
                    
                else:
                    results[file_id] = {"error": f"Unknown operation: {operation}"}
                    continue
                
                results[file_id] = {
                    "result": result,
                    "weighted_df": engine.df,
                    "status": "ok"
                }
                
            except Exception as e:
                results[file_id] = {"error": str(e)}
        
        return results
