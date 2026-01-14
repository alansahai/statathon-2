"""
Weighting Engine - MoSPI-Compliant
Handles survey weighting with:
- Base weights (inverse probability)
- Post-stratification calibration
- Weighted estimates (means, totals)
- Standard error estimation
- Margin of Error (MoE) calculation

All operations are deterministic and reproducible.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from services.file_manager import FileManager


class WeightingEngine:
    """
    MoSPI-compliant survey weighting engine for statistical data processing.
    
    Supports:
    - Base weight calculation (inverse probability of selection)
    - Post-stratification calibration to known population targets
    - Weighted mean and total computation
    - Standard error estimation using Taylor series linearization
    - Margin of error calculation (95% confidence level)
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize WeightingEngine with optional DataFrame.
        
        Args:
            df: Optional DataFrame to work with. If provided, enables instance methods.
        """
        self.df = df.copy() if df is not None else None
        self.operations_log = []
        self.auto_actions = []
        self.warnings = []
        self.weight_column = None
    
    # ==================== Instance Methods ====================
    
    def calculate_base_weights(self, inclusion_prob_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate base weights for the loaded DataFrame.
        
        Args:
            inclusion_prob_column: Optional column name containing inclusion probabilities
            
        Returns:
            Dictionary with operation log and summary
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Initialize with a DataFrame.")
        
        # Calculate base weights
        if inclusion_prob_column and inclusion_prob_column in self.df.columns:
            self.df['base_weight'] = 1.0 / self.df[inclusion_prob_column]
            self.auto_actions.append(f"Calculated base weights from '{inclusion_prob_column}' column")
        else:
            # Uniform weights (simple random sampling)
            self.df['base_weight'] = 1.0
            self.auto_actions.append("Created uniform base weights (SRS assumption)")
        
        self.weight_column = 'base_weight'
        
        # Calculate summary statistics
        summary = {
            'mean': float(self.df['base_weight'].mean()),
            'min': float(self.df['base_weight'].min()),
            'max': float(self.df['base_weight'].max()),
            'sum': float(self.df['base_weight'].sum()),
            'count': int(len(self.df))
        }
        
        log_entry = {
            'operation': 'calculate_base_weights',
            'status': 'success',
            'details': {
                'inclusion_prob_column': inclusion_prob_column,
                'summary': summary
            }
        }
        
        self.operations_log.append(log_entry)
        return log_entry
    
    def apply_poststrat_weights(self, strata_column: Optional[str] = None, 
                                 population_totals: Optional[Dict[str, float]] = None) -> tuple:
        """
        Apply post-stratification weights.
        
        Args:
            strata_column: Column to use for stratification
            population_totals: Known population totals for each stratum
            
        Returns:
            Tuple of (weighted_df, log_entry)
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Initialize with a DataFrame.")
        
        # Auto-detect strata column if not provided
        if not strata_column:
            # Look for common stratification columns
            candidate_cols = ['age_group', 'region', 'gender', 'stratum', 'strata']
            for col in candidate_cols:
                if col in self.df.columns:
                    strata_column = col
                    self.auto_actions.append(f"Auto-detected strata column: '{strata_column}'")
                    break
            
            if not strata_column:
                # Create age groups if age column exists
                age_cols = [col for col in self.df.columns if 'age' in col.lower()]
                if age_cols:
                    age_col = age_cols[0]
                    self.df['age_group'] = pd.cut(self.df[age_col], 
                                                   bins=[0, 18, 35, 50, 65, 120],
                                                   labels=['<18', '18-34', '35-49', '50-64', '65+'])
                    strata_column = 'age_group'
                    self.auto_actions.append(f"Created age_group from '{age_col}' column")
                else:
                    self.warnings.append("No suitable stratification column found")
                    return self.df, {'operation': 'apply_poststrat_weights', 'status': 'skipped', 'details': {}}
        
        # Ensure base weights exist
        if 'base_weight' not in self.df.columns:
            self.calculate_base_weights()
        
        # Apply post-stratification
        if population_totals:
            # Calculate adjustment factors
            sample_dist = self.df.groupby(strata_column).size() / len(self.df)
            
            adjustments = {}
            for stratum, pop_total in population_totals.items():
                if stratum in sample_dist.index:
                    pop_prop = pop_total / sum(population_totals.values())
                    sample_prop = sample_dist[stratum]
                    adjustments[stratum] = pop_prop / sample_prop if sample_prop > 0 else 1.0
            
            # Apply adjustments
            self.df['poststrat_weight'] = self.df.apply(
                lambda row: row['base_weight'] * adjustments.get(row[strata_column], 1.0),
                axis=1
            )
            self.weight_column = 'poststrat_weight'
            self.auto_actions.append(f"Applied post-stratification using '{strata_column}'")
        else:
            self.df['poststrat_weight'] = self.df['base_weight']
            self.warnings.append("No population totals provided - weights unchanged")
        
        summary = {
            'mean': float(self.df[self.weight_column].mean()),
            'min': float(self.df[self.weight_column].min()),
            'max': float(self.df[self.weight_column].max()),
            'sum': float(self.df[self.weight_column].sum())
        }
        
        log_entry = {
            'operation': 'apply_poststrat_weights',
            'status': 'success',
            'details': {
                'strata_column': strata_column,
                'population_totals': population_totals,
                'summary': summary
            }
        }
        
        self.operations_log.append(log_entry)
        return self.df, log_entry
    
    def raking(self, control_totals: Dict[str, Dict[str, float]], 
               max_iterations: int = 50, tolerance: float = 0.001) -> Dict[str, Any]:
        """
        Apply raking (iterative proportional fitting) to adjust weights.
        
        Args:
            control_totals: Dictionary of control variables and their target distributions
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary with operation log and summary
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Initialize with a DataFrame.")
        
        # Ensure base weights exist
        if 'base_weight' not in self.df.columns:
            self.calculate_base_weights()
        
        # Initialize raking weights
        self.df['raking_weight'] = self.df['base_weight'].copy()
        
        # Iterative proportional fitting
        for iteration in range(max_iterations):
            max_change = 0
            
            for control_var, target_dist in control_totals.items():
                if control_var not in self.df.columns:
                    self.warnings.append(f"Control variable '{control_var}' not found - skipping")
                    continue
                
                # Calculate current distribution
                current_totals = self.df.groupby(control_var)['raking_weight'].sum()
                target_total = sum(target_dist.values())
                
                # Calculate adjustment factors
                for category, target_count in target_dist.items():
                    if category in current_totals.index:
                        current_count = current_totals[category]
                        if current_count > 0:
                            adjustment = target_count / current_count
                            mask = self.df[control_var] == category
                            old_weights = self.df.loc[mask, 'raking_weight'].copy()
                            self.df.loc[mask, 'raking_weight'] *= adjustment
                            
                            # Track convergence
                            change = abs(adjustment - 1.0)
                            max_change = max(max_change, change)
            
            # Check convergence
            if max_change < tolerance:
                self.auto_actions.append(f"Raking converged after {iteration + 1} iterations")
                break
        else:
            self.warnings.append(f"Raking did not converge after {max_iterations} iterations")
        
        self.weight_column = 'raking_weight'
        
        summary = {
            'mean': float(self.df['raking_weight'].mean()),
            'min': float(self.df['raking_weight'].min()),
            'max': float(self.df['raking_weight'].max()),
            'sum': float(self.df['raking_weight'].sum()),
            'iterations': iteration + 1,
            'converged': max_change < tolerance
        }
        
        log_entry = {
            'operation': 'raking',
            'status': 'success',
            'details': {
                'control_totals': control_totals,
                'max_iterations': max_iterations,
                'tolerance': tolerance,
                'summary': summary
            }
        }
        
        self.operations_log.append(log_entry)
        return log_entry
    
    def export_weighted(self, file_path: str):
        """
        Export the weighted DataFrame to a CSV file.
        
        Args:
            file_path: Path where to save the weighted file
        """
        if self.df is None:
            raise ValueError("No DataFrame to export")
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        self.df.to_csv(file_path, index=False)
        self.auto_actions.append(f"Exported weighted data to {file_path}")
    
    def trim_weights(self, min_w: float = 0.3, max_w: float = 3.0, 
                     weight_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Trim extreme weights to improve stability.
        
        Caps weights at minimum and maximum thresholds. This reduces the influence
        of extreme weights while maintaining overall distribution characteristics.
        
        Args:
            min_w: Minimum weight threshold (default: 0.3)
            max_w: Maximum weight threshold (default: 3.0)
            weight_column: Optional weight column name. If not provided, auto-detects.
            
        Returns:
            Dictionary with trim summary
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Initialize with a DataFrame.")
        
        # Auto-detect weight column
        if not weight_column:
            weight_column = self._get_active_weight_column()
        
        if weight_column not in self.df.columns:
            raise ValueError(f"Weight column '{weight_column}' not found")
        
        # Get original weights (clean copy)
        original_weights = self.df[weight_column].copy()
        
        # Handle NaN and Inf values
        clean_mask = np.isfinite(original_weights)
        n_invalid = (~clean_mask).sum()
        
        if n_invalid > 0:
            self.warnings.append(f"Replaced {n_invalid} NaN/Inf values with median weight before trimming")
            median_weight = original_weights[clean_mask].median()
            original_weights = original_weights.fillna(median_weight).replace([np.inf, -np.inf], median_weight)
        
        # Count how many will be trimmed
        n_below = int((original_weights < min_w).sum())
        n_above = int((original_weights > max_w).sum())
        n_trimmed = n_below + n_above
        
        # Apply trimming
        trimmed_weights = original_weights.clip(lower=min_w, upper=max_w)
        
        # Store in DataFrame with descriptive column name
        trimmed_col_name = f"{weight_column}_trimmed"
        self.df[trimmed_col_name] = trimmed_weights
        self.weight_column = trimmed_col_name
        
        # Calculate summary statistics
        summary = {
            'original_column': weight_column,
            'trimmed_column': trimmed_col_name,
            'min_threshold': float(min_w),
            'max_threshold': float(max_w),
            'n_observations': int(len(original_weights)),
            'n_trimmed_below': n_below,
            'n_trimmed_above': n_above,
            'trimmed_count': n_trimmed,
            'trimmed_pct': float(100.0 * n_trimmed / len(original_weights)) if len(original_weights) > 0 else 0.0,
            'original_mean': float(original_weights.mean()),
            'original_min': float(original_weights.min()),
            'original_max': float(original_weights.max()),
            'new_mean': float(trimmed_weights.mean()),
            'new_min': float(trimmed_weights.min()),
            'new_max': float(trimmed_weights.max()),
            'new_cv': float(trimmed_weights.std() / trimmed_weights.mean()) if trimmed_weights.mean() > 0 else None
        }
        
        # Log action
        self.auto_actions.append(
            f"Trimmed {n_trimmed} weights ({summary['trimmed_pct']:.2f}%) to range [{min_w}, {max_w}]"
        )
        
        log_entry = {
            'operation': 'trim_weights',
            'status': 'success',
            'details': summary
        }
        
        self.operations_log.append(log_entry)
        return summary
    
    def get_auto_actions(self) -> list:
        """Return list of auto-actions performed."""
        return self.auto_actions
    
    def get_warnings(self) -> list:
        """Return list of warnings generated."""
        return self.warnings
    
    def get_operations_log(self) -> list:
        """Return operations log."""
        return self.operations_log
    
    def make_json_safe(self, obj: Any = None) -> Any:
        """
        Convert objects to JSON-serializable format.
        Can be called as instance method or class method.
        
        Args:
            obj: Object to convert. If None and called on instance, returns a safe version of instance data.
            
        Returns:
            JSON-serializable version of the object
        """
        # Handle None case for instance method pattern
        if obj is None:
            # Don't recursively call make_json_safe on self
            return {
                'operations_log': self._make_safe(self.operations_log),
                'auto_actions': self.auto_actions,
                'warnings': self.warnings
            }
        
        return self._make_safe(obj)
    
    def _make_safe(self, obj: Any) -> Any:
        """Internal helper to recursively make objects JSON-safe."""
        if isinstance(obj, dict):
            return {k: self._make_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_safe(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _get_active_weight_column(self) -> str:
        """
        Get the currently active weight column.
        
        Returns:
            Name of the weight column to use
        """
        if self.weight_column:
            return self.weight_column
        
        # Look for weight columns in order of preference
        weight_candidates = ['raking_weight', 'poststrat_weight', 'base_weight', 'weight', 'wt', 'w']
        
        for col in weight_candidates:
            if col in self.df.columns:
                return col
        
        return 'base_weight'  # Default
    
    def diagnostics(self, weight_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive weight diagnostics.
        
        Args:
            weight_column: Optional weight column name. If not provided, auto-detects.
            
        Returns:
            Dictionary with comprehensive diagnostics
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded")
        
        # Auto-detect weight column
        if not weight_column:
            weight_column = self._get_active_weight_column()
        
        if weight_column not in self.df.columns:
            raise ValueError(f"Weight column '{weight_column}' not found")
        
        weights = self.df[weight_column].copy()
        
        # Basic statistics
        statistics = {
            'count': int(len(weights)),
            'mean': float(weights.mean()),
            'median': float(weights.median()),
            'std': float(weights.std()),
            'min': float(weights.min()),
            'max': float(weights.max()),
            'cv': float(weights.std() / weights.mean()) if weights.mean() > 0 else None,
            'sum': float(weights.sum())
        }
        
        # Percentiles
        percentiles = {
            '1%': float(weights.quantile(0.01)),
            '5%': float(weights.quantile(0.05)),
            '10%': float(weights.quantile(0.10)),
            '25%': float(weights.quantile(0.25)),
            '50%': float(weights.quantile(0.50)),
            '75%': float(weights.quantile(0.75)),
            '90%': float(weights.quantile(0.90)),
            '95%': float(weights.quantile(0.95)),
            '99%': float(weights.quantile(0.99))
        }
        
        # Effective sample size
        n = len(weights)
        sum_w = weights.sum()
        sum_w2 = (weights ** 2).sum()
        effective_n = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else n
        
        # Design effect (DEFF)
        deff = n / effective_n if effective_n > 0 else 1.0
        
        # Quality checks
        quality_checks = {
            'no_zeros': bool((weights == 0).sum() == 0),
            'no_negatives': bool((weights < 0).sum() == 0),
            'no_missing': bool(weights.isna().sum() == 0),
            'no_infinite': bool(np.isinf(weights).sum() == 0),
            'reasonable_cv': bool(statistics['cv'] and statistics['cv'] < 1.0) if statistics['cv'] else False
        }
        
        return {
            'weight_column': weight_column,
            'statistics': statistics,
            'percentiles': percentiles,
            'effective_sample_size': float(effective_n),
            'design_effect': float(deff),
            'quality_checks': quality_checks
        }
    
    # ==================== Static Methods (Legacy) ====================
    
    @staticmethod
    def apply_weights(filename: str, method: str = "base", **kwargs) -> str:
        """
        Apply survey weights to a dataset.
        
        Process Flow:
        1. Load best available file using FileManager
        2. Calculate base weights (inverse probability)
        3. Apply post-stratification if requested
        4. Calculate weighted estimates
        5. Compute standard errors and margin of error
        6. Save weighted file
        
        Args:
            filename: Name of the file to weight (e.g., "survey.csv")
            method: Weighting method - "base" or "poststrat"
            **kwargs: Additional parameters:
                - strat_col: Column name for stratification (required for poststrat)
                - targets: Dictionary of target proportions (required for poststrat)
                
        Returns:
            Full path to the weighted file
            
        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If method is invalid or required parameters missing
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
            
            # ============================================
            # STEP 2: Calculate Base Weights
            # ============================================
            df = WeightingEngine._calculate_base_weights(df)
            
            # ============================================
            # STEP 3: Apply Post-Stratification (if requested)
            # ============================================
            if method == "poststrat":
                # Validate required parameters
                strat_col = kwargs.get('strat_col') or kwargs.get('stratification_column')
                targets = kwargs.get('targets') or kwargs.get('target_proportions')
                
                if not strat_col:
                    raise ValueError("Post-stratification requires 'strat_col' parameter")
                if not targets:
                    raise ValueError("Post-stratification requires 'targets' parameter (dictionary)")
                
                df = WeightingEngine.poststratify(df, strat_col, targets)
            
            elif method != "base":
                raise ValueError(f"Invalid method '{method}'. Valid options: 'base', 'poststrat'")
            
            # ============================================
            # STEP 4: Calculate Weighted Estimates
            # ============================================
            # Store weight column name
            weight_col = "poststrat_weight" if method == "poststrat" else "base_weight"
            
            # Calculate weighted means for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            weighted_means = {}
            standard_errors = {}
            margins_of_error = {}
            
            for col in numeric_cols:
                # Skip weight columns themselves
                if 'weight' in col.lower() or col == weight_col:
                    continue
                
                try:
                    # Calculate weighted mean
                    w_mean = WeightingEngine.weighted_mean(df[col], df[weight_col])
                    weighted_means[col] = float(w_mean)
                    
                    # Calculate standard error
                    se = WeightingEngine.compute_standard_error(df, col, weight_col)
                    standard_errors[col] = float(se)
                    
                    # Calculate margin of error (95% CI)
                    moe = 1.96 * se
                    margins_of_error[col] = float(moe)
                except:
                    pass
            
            # ============================================
            # STEP 5: Save Weighted File
            # ============================================
            # Extract base filename
            base_filename = Path(filename).stem
            if base_filename.endswith('_mapped'):
                base_filename = base_filename[:-7]
            if base_filename.endswith('_cleaned'):
                base_filename = base_filename[:-8]
            if base_filename.endswith('_weighted'):
                base_filename = base_filename[:-9]
            
            weighted_path = FileManager.get_weighted_path(f"{base_filename}.csv")
            
            # Ensure directory exists
            Path(weighted_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save CSV
            df.to_csv(weighted_path, index=False)
            
            # Print summary (optional - could be logged or returned)
            print(f"Weighting Summary:")
            print(f"  Method: {method}")
            print(f"  Weight Column: {weight_col}")
            print(f"  Weight Range: [{df[weight_col].min():.4f}, {df[weight_col].max():.4f}]")
            print(f"  Weighted Means: {len(weighted_means)} variables")
            print(f"  Output: {weighted_path}")
            
            return weighted_path
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty or unreadable")
        
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise Exception(f"Weighting failed: {str(e)}")
    
    @staticmethod
    def _calculate_base_weights(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate base weights using inverse probability of selection.
        
        Logic:
        - If "probability" column exists: base_weight = 1 / probability
        - Otherwise: assume simple random sampling (SRS) with weight = 1.0
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with base_weight column added
        """
        df = df.copy()
        
        # Check for probability column (various naming conventions)
        prob_cols = [col for col in df.columns if 'prob' in col.lower()]
        
        if prob_cols:
            prob_col = prob_cols[0]
            print(f"Found probability column: '{prob_col}'")
            
            # Calculate inverse probability weights
            try:
                # Handle zero or NaN probabilities
                df['base_weight'] = np.where(
                    (df[prob_col] > 0) & (df[prob_col].notna()),
                    1.0 / df[prob_col],
                    1.0  # Default to 1.0 for invalid probabilities
                )
                
                # Check for extreme weights
                extreme_weights = (df['base_weight'] > 10) | (df['base_weight'] < 0.1)
                if extreme_weights.any():
                    print(f"Warning: {extreme_weights.sum()} observations have extreme weights (>10 or <0.1)")
                
            except Exception as e:
                print(f"Error calculating inverse probability weights: {e}")
                df['base_weight'] = 1.0
        else:
            # No probability column - assume SRS
            print("No probability column found - assuming Simple Random Sampling (SRS)")
            df['base_weight'] = 1.0
        
        return df
    
    @staticmethod
    def poststratify(df: pd.DataFrame, strat_col: str, targets: Dict[str, float]) -> pd.DataFrame:
        """
        Apply post-stratification calibration to align sample with population targets.
        
        Post-stratification adjusts weights so that weighted sample proportions
        match known population proportions for a stratification variable.
        
        Algorithm:
        1. Calculate sample proportions using base weights
        2. Compute adjustment factors: target_prop / sample_prop
        3. Multiply base_weight by adjustment factor
        
        Args:
            df: DataFrame with base_weight column
            strat_col: Column name to stratify by (e.g., "age_group", "region")
            targets: Dictionary mapping categories to target proportions
                     Example: {"18-30": 0.25, "31-50": 0.45, "51+": 0.30}
                     Must sum to 1.0 (or will be normalized)
            
        Returns:
            DataFrame with poststrat_weight column added
            
        Raises:
            ValueError: If strat_col doesn't exist or targets invalid
        """
        df = df.copy()
        
        # Validate stratification column exists
        if strat_col not in df.columns:
            raise ValueError(f"Stratification column '{strat_col}' not found in dataset")
        
        # Ensure base_weight exists
        if 'base_weight' not in df.columns:
            df = WeightingEngine._calculate_base_weights(df)
        
        # Normalize targets to sum to 1.0
        target_sum = sum(targets.values())
        if abs(target_sum - 1.0) > 0.01:
            print(f"Warning: Target proportions sum to {target_sum:.4f}, normalizing to 1.0")
            targets = {k: v / target_sum for k, v in targets.items()}
        
        # Calculate sample proportions using base weights
        sample_counts = df.groupby(strat_col)['base_weight'].sum()
        total_weight = sample_counts.sum()
        sample_props = sample_counts / total_weight
        
        # Calculate adjustment factors for each stratum
        adjustment_factors = {}
        
        for category in df[strat_col].unique():
            if pd.isna(category):
                # Handle missing values
                adjustment_factors[category] = 1.0
                print(f"Warning: Missing values in '{strat_col}' - using adjustment factor 1.0")
                continue
            
            # Convert to string for consistent matching
            category_str = str(category)
            
            # Find matching target (case-insensitive)
            target_prop = None
            for target_key, target_val in targets.items():
                if str(target_key).lower() == category_str.lower():
                    target_prop = target_val
                    break
            
            if target_prop is None:
                print(f"Warning: Category '{category}' not in targets - using adjustment factor 1.0")
                adjustment_factors[category] = 1.0
                continue
            
            # Get sample proportion
            sample_prop = sample_props.get(category, 0)
            
            if sample_prop == 0:
                print(f"Warning: Category '{category}' has zero sample weight - using adjustment factor 1.0")
                adjustment_factors[category] = 1.0
            else:
                # Calculate adjustment factor
                adj_factor = target_prop / sample_prop
                adjustment_factors[category] = adj_factor
                print(f"  {category}: sample={sample_prop:.4f}, target={target_prop:.4f}, adjustment={adj_factor:.4f}")
        
        # Apply adjustment factors
        df['adjustment_factor'] = df[strat_col].map(adjustment_factors).fillna(1.0)
        df['poststrat_weight'] = df['base_weight'] * df['adjustment_factor']
        
        # Clean up temporary column
        df.drop('adjustment_factor', axis=1, inplace=True)
        
        # Verify calibration
        calibrated_props = df.groupby(strat_col)['poststrat_weight'].sum() / df['poststrat_weight'].sum()
        print(f"\nCalibration verification:")
        for category in calibrated_props.index:
            category_str = str(category)
            target_val = targets.get(category_str, targets.get(category, None))
            if target_val:
                print(f"  {category}: calibrated={calibrated_props[category]:.4f}, target={target_val:.4f}")
        
        return df
    
    @staticmethod
    def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        """
        Calculate weighted mean.
        
        Formula: Σ(w_i * x_i) / Σ(w_i)
        
        Args:
            values: Series of values
            weights: Series of weights (same length as values)
            
        Returns:
            Weighted mean
            
        Raises:
            ValueError: If weights sum to zero or series lengths don't match
        """
        # Remove NaN values
        valid_mask = values.notna() & weights.notna() & (weights > 0)
        
        if not valid_mask.any():
            raise ValueError("No valid data points (all NaN or zero weights)")
        
        values_valid = values[valid_mask]
        weights_valid = weights[valid_mask]
        
        # Check lengths match
        if len(values_valid) != len(weights_valid):
            raise ValueError("Values and weights must have same length")
        
        # Calculate weighted sum
        weighted_sum = np.sum(weights_valid * values_valid)
        weight_sum = np.sum(weights_valid)
        
        if weight_sum == 0:
            raise ValueError("Sum of weights is zero")
        
        return weighted_sum / weight_sum
    
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
        # Remove NaN values
        valid_mask = values.notna() & weights.notna() & (weights > 0)
        
        if not valid_mask.any():
            return 0.0
        
        values_valid = values[valid_mask]
        weights_valid = weights[valid_mask]
        
        return float(np.sum(weights_valid * values_valid))
    
    @staticmethod
    def compute_standard_error(df: pd.DataFrame, column: str, weight_col: str = "base_weight") -> float:
        """
        Compute standard error of weighted mean using Taylor series linearization.
        
        Simplified formula (approximation):
        SE(mean_w) ≈ sqrt(Σ w_i^2 * (x_i - mean_w)^2) / Σ w_i
        
        This is a conservative approximation suitable for simple random sampling
        with unequal weights or post-stratification.
        
        Args:
            df: DataFrame containing data
            column: Column name to compute SE for
            weight_col: Weight column name (default: "base_weight")
            
        Returns:
            Standard error estimate
            
        Raises:
            ValueError: If column or weight_col not found, or insufficient data
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        
        if weight_col not in df.columns:
            raise ValueError(f"Weight column '{weight_col}' not found in dataset")
        
        # Get valid observations
        valid_mask = df[column].notna() & df[weight_col].notna() & (df[weight_col] > 0)
        
        if valid_mask.sum() < 2:
            raise ValueError(f"Insufficient data for SE computation (need at least 2 valid observations)")
        
        values = df.loc[valid_mask, column]
        weights = df.loc[valid_mask, weight_col]
        
        # Calculate weighted mean
        weighted_mean_val = WeightingEngine.weighted_mean(values, weights)
        
        # Calculate deviations
        deviations = values - weighted_mean_val
        
        # Calculate variance components
        weight_sum = np.sum(weights)
        squared_weighted_deviations = np.sum(weights**2 * deviations**2)
        
        # Standard error (Taylor series approximation)
        se = np.sqrt(squared_weighted_deviations) / weight_sum
        
        return float(se)
    
    @staticmethod
    def calculate_margin_of_error(df: pd.DataFrame, column: str, weight_col: str = "base_weight", confidence_level: float = 0.95) -> float:
        """
        Calculate margin of error for weighted mean.
        
        Formula: MoE = z * SE
        Where z is the critical value from standard normal distribution
        (1.96 for 95% confidence level)
        
        Args:
            df: DataFrame containing data
            column: Column name to compute MoE for
            weight_col: Weight column name
            confidence_level: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Margin of error
        """
        # Get standard error
        se = WeightingEngine.compute_standard_error(df, column, weight_col)
        
        # Get critical value (z-score)
        if confidence_level == 0.95:
            z = 1.96
        elif confidence_level == 0.90:
            z = 1.645
        elif confidence_level == 0.99:
            z = 2.576
        else:
            # General case (using normal approximation)
            from scipy import stats
            z = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Calculate MoE
        moe = z * se
        
        return float(moe)
    
    @staticmethod
    def get_weight_distribution(df: pd.DataFrame, weight_col: str = "base_weight") -> Dict[str, float]:
        """
        Get summary statistics for weight distribution.
        
        Args:
            df: DataFrame with weights
            weight_col: Weight column name
            
        Returns:
            Dictionary with weight statistics
        """
        if weight_col not in df.columns:
            raise ValueError(f"Weight column '{weight_col}' not found")
        
        weights = df[weight_col]
        
        return {
            "min": float(weights.min()),
            "q1": float(weights.quantile(0.25)),
            "median": float(weights.median()),
            "q3": float(weights.quantile(0.75)),
            "max": float(weights.max()),
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "cv": float(weights.std() / weights.mean()) if weights.mean() > 0 else 0.0,
            "effective_sample_size": float((weights.sum()**2) / (weights**2).sum())
        }
