import pandas as pd
import numpy as np

class CleaningEngine:
    """
    Handles Missing Value Imputation, Outlier Treatment, and Rule-Based Validation.
    """

    @staticmethod
    def impute_data(df, method='mean', columns=None):
        df_clean = df.copy()
        log = []
        cols_to_process = columns if columns else df_clean.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols_to_process:
            if col not in df_clean.columns: continue
            
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                if method == 'mean':
                    val = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(val)
                elif method == 'median':
                    val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(val)
                elif method == 'mode':
                    if not df_clean[col].mode().empty:
                        val = df_clean[col].mode()[0]
                        df_clean[col] = df_clean[col].fillna(val)
                elif method == 'drop':
                    df_clean.dropna(subset=[col], inplace=True)
                
                log.append(f"Imputed {missing_count} missing values in '{col}' using {method}.")
        
        return df_clean, log

    @staticmethod
    def remove_outliers(df, columns=None, method='iqr'):
        df_clean = df.copy()
        log = []
        original_len = len(df_clean)
        cols_to_process = columns if columns else df_clean.select_dtypes(include=[np.number]).columns.tolist()

        mask = pd.Series([True] * len(df_clean), index=df_clean.index)

        for col in cols_to_process:
            if col not in df_clean.columns: continue
            
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                col_mask = (df_clean[col] >= lower) & (df_clean[col] <= upper)
                mask = mask & col_mask
        
        df_clean = df_clean[mask]
        removed = original_len - len(df_clean)
        if removed > 0:
            log.append(f"Removed {removed} rows containing outliers using {method} method.")
            
        return df_clean, log

    @staticmethod
    def apply_rules(df, rules=None):
        """
        NEW: Applies logical validation rules.
        Supported rules:
        - 'range': {'col': 'Age', 'min': 0, 'max': 120}
        - 'mandatory': ['Respondent_ID', 'Survey_Weight']
        """
        df_clean = df.copy()
        log = []
        initial_len = len(df_clean)

        if not rules:
            return df_clean, log

        # 1. Range Validation (e.g., Age cannot be negative)
        if 'range' in rules:
            for rule in rules['range']:
                col = rule.get('col')
                min_val = rule.get('min', -float('inf'))
                max_val = rule.get('max', float('inf'))
                
                if col in df_clean.columns:
                    mask = (df_clean[col] >= min_val) & (df_clean[col] <= max_val)
                    invalid_count = (~mask).sum()
                    
                    if invalid_count > 0:
                        df_clean = df_clean[mask]
                        log.append(f"Rule Validation: Removed {invalid_count} rows where '{col}' was outside range [{min_val}, {max_val}].")

        # 2. Mandatory Columns Check (Drop rows if critical ID/Weight is missing)
        if 'mandatory' in rules:
            cols = [c for c in rules['mandatory'] if c in df_clean.columns]
            if cols:
                before_n = len(df_clean)
                df_clean.dropna(subset=cols, inplace=True)
                dropped = before_n - len(df_clean)
                if dropped > 0:
                    log.append(f"Rule Validation: Dropped {dropped} rows missing mandatory values in {cols}.")

        return df_clean, log