import numpy as np
import pandas as pd
from scipy import stats

class StatisticsEngine:
    @staticmethod
    def calculate_effective_sample_size(weights):
        if len(weights) == 0: return 0
        weight_sum = np.sum(weights)
        weight_sq_sum = np.sum(weights ** 2)
        return (weight_sum ** 2) / weight_sq_sum

    @staticmethod
    def calculate_weighted_stats(df, target_col, weight_col, pop_control_total=None):
        clean_df = df[[target_col, weight_col]].dropna()
        if clean_df.empty: return None

        values = clean_df[target_col].values
        weights = clean_df[weight_col].values
        n = len(values)
        if n <= 1: return None

        sum_weights = np.sum(weights)
        weighted_mean = np.average(values, weights=weights)

        numerator = np.sum(weights * (values - weighted_mean) ** 2)
        weighted_var = (numerator / sum_weights) * (n / (n - 1))
        weighted_std = np.sqrt(weighted_var)

        n_eff = StatisticsEngine.calculate_effective_sample_size(weights)
        se = weighted_std / np.sqrt(n_eff)

        # Confidence Intervals & MOE
        confidence_level = 0.95
        degrees_of_freedom = n_eff - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        
        margin_of_error = t_critical * se  # <--- NEW: MOE Calculation
        
        ci_lower = weighted_mean - margin_of_error
        ci_upper = weighted_mean + margin_of_error

        pop_estimate = None
        if pop_control_total is not None and pop_control_total > 0:
            pop_estimate = weighted_mean * pop_control_total

        return {
            "column": target_col,
            "n_sample": int(n),
            "n_effective": round(n_eff, 2),
            "mean": round(weighted_mean, 4),
            "std_error": round(se, 4),
            "margin_of_error": round(margin_of_error, 4), # <--- NEW
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "population_estimate": round(pop_estimate, 2) if pop_estimate else "N/A"
        }

class SchemaValidator:
    """
    Handles schema mapping and type validation logic.
    """
    @staticmethod
    def map_columns(df, mapping_dict):
        """
        Renames columns based on user provided mapping.
        mapping_dict example: {"user_age": "Age", "user_income": "Income"}
        """
        # Invert mapping to check if rename is possible (ensure no collisions)
        return df.rename(columns=mapping_dict)

    @staticmethod
    def validate_schema(df, required_columns):
        """
        Checks if required columns exist in the dataframe.
        """
        missing = [col for col in required_columns if col not in df.columns]
        return missing