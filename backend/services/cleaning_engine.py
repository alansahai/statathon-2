"""
Data Cleaning Engine - MoSPI-Compliant
Handles automatic data cleaning with:
- Missing value imputation
- Outlier detection and treatment
- Logical validation rules
- Comprehensive cleaning summary

All operations are deterministic and reproducible.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from scipy import stats

from services.file_manager import FileManager


class CleaningEngine:
    """
    MoSPI-compliant data cleaning engine for statistical data processing.
    
    Provides deterministic cleaning operations including:
    - Missing value imputation (median/mode/forward fill)
    - Outlier detection and treatment (Z-score and IQR methods)
    - Logical validation rules (age, employment, skip patterns)
    - Comprehensive cleaning summary reporting
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize CleaningEngine.
        
        Args:
            df: Optional DataFrame to clean. If provided, instance methods can be used.
        """
        self.df = df.copy() if df is not None else None
        self.original_df = df.copy() if df is not None else None
        self.cleaning_summary = {}
    
    def auto_clean(self) -> Dict[str, Any]:
        """
        Perform complete automatic cleaning pipeline on the stored DataFrame.
        
        Returns:
            Dictionary with cleaning summary
            
        Raises:
            ValueError: If no DataFrame is loaded
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Initialize with a DataFrame or use static method.")
        
        # Store original state for summary
        missing_before = self.df.isnull().sum().sum()
        missing_by_column_before = {col: int(count) for col, count in self.df.isnull().sum().items() if count > 0}
        rows_before = len(self.df)
        
        # Apply cleaning operations
        self.df = self._impute_missing(self.df)
        missing_by_column_after = {col: int(count) for col, count in self.df.isnull().sum().items() if count > 0}
        
        self.df, outliers_dict = self._treat_outliers(self.df)
        print(f"[CleaningEngine.auto_clean] Outliers dict after treatment: {len(outliers_dict)} columns")
        print(f"[CleaningEngine.auto_clean] Outlier columns: {list(outliers_dict.keys())}")
        
        self.df, logical_dict = self._apply_logical_rules(self.df)
        print(f"[CleaningEngine.auto_clean] Logical dict after validation: {len(logical_dict)} issues")
        
        # Generate cleaning summary
        missing_after = self.df.isnull().sum().sum()
        
        # Count outliers and logical issues
        outliers_count = sum(v.get('count', 0) if isinstance(v, dict) else 0 for v in outliers_dict.values())
        logical_count = sum(v.get('count', 0) if isinstance(v, dict) else 0 for v in logical_dict.values())
        
        self.cleaning_summary = {
            "issues_detected": int(missing_before) + outliers_count + logical_count,
            "issues_fixed": int(missing_before - missing_after) + outliers_count,
            "rows_before": rows_before,
            "rows_after": len(self.df),
            "outliers_detected": outliers_count,
            "logical_issues": logical_count,
            "outliers_details": outliers_dict,
            "logical_details": logical_dict,
            "missing_values_before": missing_by_column_before,
            "missing_values_after": missing_by_column_after
        }
        
        print(f"[CleaningEngine.auto_clean] Final summary - outliers_details has {len(self.cleaning_summary['outliers_details'])} columns")
        print(f"[CleaningEngine.auto_clean] Final outlier columns: {list(self.cleaning_summary['outliers_details'].keys())}")
        
        return self.cleaning_summary
    
    @staticmethod
    def auto_clean_file(filename: str) -> str:
        """
        Perform complete automatic cleaning pipeline on a dataset.
        
        Process Flow:
        1. Resolve file path using FileManager
        2. Load dataset (CSV/XLSX)
        3. Apply missing value imputation
        4. Detect and treat outliers
        5. Apply logical validation rules
        6. Generate cleaning summary
        7. Save cleaned file
        
        Args:
            filename: Name of the file to clean (e.g., "survey.csv")
            
        Returns:
            Full path to the cleaned file
            
        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If file is empty or unreadable
            Exception: For other processing errors
        """
        try:
            # ============================================
            # STEP 1: Resolve File Path
            # ============================================
            print(f"[CleaningEngine] Looking for file: {filename}")
            file_path = FileManager.get_best_available_file(filename)
            print(f"[CleaningEngine] Found file at: {file_path}")
            
            # ============================================
            # STEP 2: Load Dataset
            # ============================================
            try:
                # Try CSV first
                df_original = pd.read_csv(file_path, low_memory=False)
            except:
                try:
                    # Fallback to Excel
                    df_original = pd.read_excel(file_path)
                except Exception as e:
                    raise ValueError(f"Cannot read file format: {str(e)}")
            
            if df_original.empty:
                raise ValueError("Dataset is empty")
            
            # Create working copy
            df = df_original.copy()
            
            # Store original state for summary
            missing_before = df.isnull().sum().to_dict()
            rows_before = len(df)
            
            # ============================================
            # STEP 3: Missing Value Imputation
            # ============================================
            df = CleaningEngine._impute_missing_static(df)
            
            # ============================================
            # STEP 4: Outlier Treatment
            # ============================================
            df, outliers_detected = CleaningEngine._treat_outliers_static(df)
            
            # ============================================
            # STEP 5: Logical Validation
            # ============================================
            df, logical_issues = CleaningEngine._apply_logical_rules_static(df)
            
            # ============================================
            # STEP 6: Generate Cleaning Summary
            # ============================================
            missing_after = df.isnull().sum().to_dict()
            
            clean_summary = {
                "missing_values_before": {k: int(v) for k, v in missing_before.items() if v > 0},
                "missing_values_after": {k: int(v) for k, v in missing_after.items() if v > 0},
                "outliers_detected": outliers_detected,
                "logical_issues_detected": logical_issues,
                "total_rows_cleaned": int(len(df)),
                "total_columns": int(len(df.columns)),
                "rows_dropped": int(rows_before - len(df))
            }
            
            # Store summary as metadata (could be saved separately if needed)
            # For now, we'll just log it
            print(f"Cleaning Summary: {clean_summary}")
            
            # ============================================
            # STEP 7: Save Cleaned File
            # ============================================
            # Extract base filename without extensions
            base_filename = Path(filename).stem
            if base_filename.endswith('_mapped'):
                base_filename = base_filename[:-7]
            if base_filename.endswith('_cleaned'):
                base_filename = base_filename[:-8]
            if base_filename.endswith('_weighted'):
                base_filename = base_filename[:-9]
            
            cleaned_path = FileManager.get_cleaned_path(f"{base_filename}.csv")
            
            # Ensure directory exists
            Path(cleaned_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save CSV
            df.to_csv(cleaned_path, index=False)
            
            return cleaned_path
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty or unreadable")
        
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise Exception(f"Cleaning failed: {str(e)}")
    
    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Instance method wrapper for impute_missing."""
        return self._impute_missing_static(df)
    
    def _treat_outliers(self, df: pd.DataFrame):
        """Instance method wrapper for treat_outliers."""
        return self._treat_outliers_static(df)
    
    def _apply_logical_rules(self, df: pd.DataFrame):
        """Instance method wrapper for apply_logical_rules."""
        return self._apply_logical_rules_static(df)
    
    @staticmethod
    def _impute_missing_static(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using deterministic methods.
        
        Imputation Strategy:
        - Numeric columns: Median imputation
        - Categorical columns: Mode imputation
        - Datetime columns: Forward fill, then backward fill
        
        Args:
            df: Input DataFrame with missing values
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        for col in df.columns:
            # Skip if no missing values
            if df[col].isnull().sum() == 0:
                continue
            
            try:
                # Numeric columns → median
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_value = df[col].median()
                    if pd.notna(median_value):
                        df[col].fillna(median_value, inplace=True)
                
                # Datetime columns → forward fill + backward fill
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(method='bfill', inplace=True)
                
                # Categorical/text columns → mode
                else:
                    mode_values = df[col].mode()
                    if len(mode_values) > 0:
                        mode_value = mode_values.iloc[0]
                        df[col].fillna(mode_value, inplace=True)
            
            except Exception as e:
                # If imputation fails for a column, skip it
                print(f"Warning: Failed to impute column '{col}': {str(e)}")
                continue
        
        return df
    
    @staticmethod
    def _treat_outliers_static(df: pd.DataFrame) -> tuple:
        """
        Detect and treat outliers using hybrid method.
        
        Detection Strategy:
        - Z-score method for normally distributed data (threshold: |z| > 3.0)
        - IQR method for skewed distributions (threshold: Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        
        Treatment:
        - Z-score outliers: Cap to 99th percentile (upper) and 1st percentile (lower)
        - IQR outliers: Winsorize to Q1 and Q3 boundaries
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned_df, outliers_dict)
            - cleaned_df: DataFrame with outliers treated
            - outliers_dict: Dictionary summarizing outliers detected per column
        """
        df = df.copy()
        outliers_detected = {}
        
        # Process only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\n[CleaningEngine._treat_outliers_static] Starting outlier detection")
        print(f"[CleaningEngine._treat_outliers_static] Total numeric columns to check: {len(numeric_cols)}")
        print(f"[CleaningEngine._treat_outliers_static] Numeric columns: {list(numeric_cols)}")
        
        for col in numeric_cols:
            # Skip if insufficient data
            non_null = df[col].dropna()
            if len(non_null) < 10:
                continue
            
            try:
                # Calculate skewness to choose method
                skewness = stats.skew(non_null)
                
                # ============================================
                # Method Selection
                # ============================================
                # If approximately normal (|skewness| < 1), use Z-score
                # Otherwise use IQR for robustness
                
                if abs(skewness) < 1.0:
                    # ============================================
                    # Z-SCORE METHOD (Normal-like distributions)
                    # ============================================
                    mean = non_null.mean()
                    std = non_null.std()
                    
                    if std == 0:
                        continue
                    
                    # Calculate z-scores on non-null values
                    z_scores = np.abs((non_null - mean) / std)
                    
                    # Detect outliers (|z| > 3.0)
                    outlier_mask = z_scores > 3.0
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        # Cap to 99th and 1st percentiles
                        lower_cap = non_null.quantile(0.01)
                        upper_cap = non_null.quantile(0.99)
                        
                        # Apply clipping to the entire column
                        df.loc[df[col].notna(), col] = df.loc[df[col].notna(), col].clip(lower=lower_cap, upper=upper_cap)
                        
                        outliers_detected[col] = {
                            "method": "z-score",
                            "count": int(outlier_count),
                            "percentage": float(round((outlier_count / len(non_null)) * 100, 2)),
                            "treatment": "capped_to_percentiles",
                            "lower_cap": float(lower_cap),
                            "upper_cap": float(upper_cap)
                        }
                        print(f"  ✓ {col}: Z-score detected {outlier_count} outliers")
                    else:
                        print(f"  - {col}: No Z-score outliers (skewness={skewness:.2f})")
                
                else:
                    # ============================================
                    # IQR METHOD (Skewed distributions)
                    # ============================================
                    Q1 = non_null.quantile(0.25)
                    Q3 = non_null.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR == 0:
                        continue
                    
                    # Calculate fences
                    lower_fence = Q1 - 1.5 * IQR
                    upper_fence = Q3 + 1.5 * IQR
                    
                    # Detect outliers on non-null values
                    outlier_mask = (non_null < lower_fence) | (non_null > upper_fence)
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        # Winsorize to Q1/Q3 boundaries on the entire column
                        df.loc[df[col].notna(), col] = df.loc[df[col].notna(), col].clip(lower=Q1, upper=Q3)
                        
                        outliers_detected[col] = {
                            "method": "iqr",
                            "count": int(outlier_count),
                            "percentage": float(round((outlier_count / len(non_null)) * 100, 2)),
                            "treatment": "winsorized_to_quartiles",
                            "q1": float(Q1),
                            "q3": float(Q3),
                            "iqr": float(IQR),
                            "lower_fence": float(lower_fence),
                            "upper_fence": float(upper_fence)
                        }
                        print(f"  ✓ {col}: IQR detected {outlier_count} outliers (skewness={abs(skewness):.2f})")
                    else:
                        print(f"  - {col}: No IQR outliers (skewness={abs(skewness):.2f})")
            
            except Exception as e:
                print(f"Warning: Failed to treat outliers in column '{col}': {str(e)}")
                continue
        
        print(f"[CleaningEngine] Total outlier columns detected: {len(outliers_detected)}")
        print(f"[CleaningEngine] Outlier columns: {list(outliers_detected.keys())}")
        
        return df, outliers_detected
    
    @staticmethod
    def _apply_logical_rules_static(df: pd.DataFrame) -> tuple:
        """
        Apply logical validation rules and auto-fix where possible.
        
        Validation Rules:
        1. Age validation: age < 0 or age > 120 → set to NaN
        2. Employment-income consistency: employed == 0 & income > 0 → income = 0
        3. Marital status consistency: married == 0 & spouse_age > 0 → spouse_age = NaN
        4. Skip pattern validation: if Q5 == "NO" then Q6 must be NaN
        5. Negative value checks: non-negative fields (income, hours, etc.)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned_df, issues_dict)
            - cleaned_df: DataFrame with logical corrections applied
            - issues_dict: Dictionary summarizing logical issues detected
        """
        df = df.copy()
        logical_issues = {}
        
        # ============================================
        # RULE 1: Age Validation
        # ============================================
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        for col in age_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    # Detect invalid ages
                    invalid_mask = (df[col] < 0) | (df[col] > 120)
                    invalid_count = invalid_mask.sum()
                    
                    if invalid_count > 0:
                        df.loc[invalid_mask, col] = np.nan
                        logical_issues[f"{col}_invalid_range"] = {
                            "rule": "age must be between 0 and 120",
                            "violations": int(invalid_count),
                            "action": "set to NaN"
                        }
                except:
                    pass
        
        # ============================================
        # RULE 2: Employment-Income Consistency
        # ============================================
        employment_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['employ', 'job', 'work'])]
        income_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['income', 'salary', 'wage', 'earnings'])]
        
        for emp_col in employment_cols:
            for inc_col in income_cols:
                if emp_col in df.columns and inc_col in df.columns:
                    try:
                        # If not employed (0 or False) but has income > 0
                        inconsistent_mask = (df[emp_col] == 0) & (df[inc_col] > 0)
                        inconsistent_count = inconsistent_mask.sum()
                        
                        if inconsistent_count > 0:
                            df.loc[inconsistent_mask, inc_col] = 0
                            logical_issues[f"{emp_col}_{inc_col}_inconsistency"] = {
                                "rule": "unemployed cannot have positive income",
                                "violations": int(inconsistent_count),
                                "action": "set income to 0"
                            }
                    except:
                        pass
        
        # ============================================
        # RULE 3: Marital Status-Spouse Consistency
        # ============================================
        marital_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['married', 'marital', 'spouse'])]
        spouse_cols = [col for col in df.columns if 'spouse' in col.lower() and col not in marital_cols]
        
        for mar_col in marital_cols:
            for sp_col in spouse_cols:
                if mar_col in df.columns and sp_col in df.columns:
                    try:
                        # If not married (0 or False) but has spouse data
                        inconsistent_mask = (df[mar_col] == 0) & (df[sp_col].notna()) & (df[sp_col] > 0)
                        inconsistent_count = inconsistent_mask.sum()
                        
                        if inconsistent_count > 0:
                            df.loc[inconsistent_mask, sp_col] = np.nan
                            logical_issues[f"{mar_col}_{sp_col}_inconsistency"] = {
                                "rule": "unmarried cannot have spouse data",
                                "violations": int(inconsistent_count),
                                "action": "set spouse data to NaN"
                            }
                    except:
                        pass
        
        # ============================================
        # RULE 4: Skip Pattern Validation
        # ============================================
        # Look for Q/question patterns
        q_cols = [col for col in df.columns if col.startswith('Q') or 'question' in col.lower()]
        
        # Common skip pattern: if Q5 = "NO", Q6 should be blank
        for i in range(len(q_cols) - 1):
            try:
                q_current = q_cols[i]
                q_next = q_cols[i + 1]
                
                if q_current in df.columns and q_next in df.columns:
                    # Check if current question has "NO" answers
                    no_mask = df[q_current].astype(str).str.upper().isin(['NO', 'N', '0', 'FALSE'])
                    
                    # Check if next question has data when current is NO
                    skip_violation_mask = no_mask & df[q_next].notna()
                    violation_count = skip_violation_mask.sum()
                    
                    if violation_count > 0:
                        df.loc[skip_violation_mask, q_next] = np.nan
                        logical_issues[f"skip_pattern_{q_current}_{q_next}"] = {
                            "rule": f"if {q_current} is NO, {q_next} must be blank",
                            "violations": int(violation_count),
                            "action": f"set {q_next} to NaN"
                        }
            except:
                pass
        
        # ============================================
        # RULE 5: Non-Negative Value Checks
        # ============================================
        non_negative_keywords = ['income', 'salary', 'hours', 'count', 'number', 'quantity', 'amount', 'price', 'cost']
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if column name suggests non-negative values
                if any(kw in col.lower() for kw in non_negative_keywords):
                    try:
                        negative_mask = df[col] < 0
                        negative_count = negative_mask.sum()
                        
                        if negative_count > 0:
                            df.loc[negative_mask, col] = 0
                            logical_issues[f"{col}_negative_values"] = {
                                "rule": f"{col} cannot be negative",
                                "violations": int(negative_count),
                                "action": "set to 0"
                            }
                    except:
                        pass
        
        # ============================================
        # RULE 6: Duplicate Row Detection
        # ============================================
        try:
            duplicate_mask = df.duplicated()
            duplicate_count = duplicate_mask.sum()
            
            if duplicate_count > 0:
                # Remove duplicates
                df = df[~duplicate_mask]
                logical_issues["duplicate_rows"] = {
                    "rule": "remove duplicate rows",
                    "violations": int(duplicate_count),
                    "action": "rows removed"
                }
        except:
            pass
        
        return df, logical_issues
    
    @staticmethod
    def generate_summary(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive cleaning summary comparing before/after states.
        
        Args:
            df_before: Original DataFrame before cleaning
            df_after: DataFrame after cleaning operations
            
        Returns:
            Dictionary containing:
            - missing_values_before: Column-wise missing counts before
            - missing_values_after: Column-wise missing counts after
            - outliers_detected: Summary of outliers found (would need to store this)
            - logical_issues_detected: Summary of logical issues (would need to store this)
            - total_rows_cleaned: Number of rows in cleaned dataset
            - total_columns: Number of columns
            - rows_dropped: Number of rows removed
        """
        missing_before = df_before.isnull().sum().to_dict()
        missing_after = df_after.isnull().sum().to_dict()
        
        summary = {
            "missing_values_before": {k: int(v) for k, v in missing_before.items() if v > 0},
            "missing_values_after": {k: int(v) for k, v in missing_after.items() if v > 0},
            "outliers_detected": {},  # Would need to be passed in from treat_outliers
            "logical_issues_detected": {},  # Would need to be passed in from apply_logical_rules
            "total_rows_cleaned": int(len(df_after)),
            "total_columns": int(len(df_after.columns)),
            "rows_dropped": int(len(df_before) - len(df_after))
        }
        
        return summary
