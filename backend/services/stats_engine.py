"""
Statistical Test Engine Module
Provides comprehensive statistical testing capabilities for survey data analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional, Union
import warnings


class StatisticalTestEngine:
    """
    Engine for performing various statistical tests on survey data.
    All methods return JSON-safe dictionaries with standardized structure.
    """

    def __init__(self):
        self.numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    def _validate_column(self, data: pd.DataFrame, column: str) -> None:
        """Validate that a column exists in the dataframe."""
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data. Available columns: {list(data.columns)}")

    def _validate_columns(self, data: pd.DataFrame, columns: List[str]) -> None:
        """Validate that multiple columns exist in the dataframe."""
        for col in columns:
            self._validate_column(data, col)

    def _is_numeric(self, data: pd.DataFrame, column: str) -> bool:
        """Check if a column is numeric."""
        return data[column].dtype.name in self.numeric_types

    def _is_categorical(self, data: pd.DataFrame, column: str) -> bool:
        """Check if a column is categorical."""
        return not self._is_numeric(data, column) or data[column].nunique() < 10

    def _clean_numeric(self, series: pd.Series) -> pd.Series:
        """Remove NaN values from numeric series."""
        return series.dropna()

    def _to_json_safe(self, value: Any) -> Any:
        """Convert numpy types to JSON-safe Python types."""
        if isinstance(value, (np.integer,)):
            return int(value)
        elif isinstance(value, (np.floating,)):
            if np.isnan(value):
                return None
            if np.isinf(value):
                return "inf" if value > 0 else "-inf"
            return float(value)
        elif isinstance(value, np.ndarray):
            return [self._to_json_safe(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._to_json_safe(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._to_json_safe(v) for v in value]
        return value

    def _build_result(
        self,
        test_name: str,
        statistic: float,
        p_value: float,
        df: Optional[Union[int, float, tuple]] = None,
        warnings_list: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a standardized result dictionary."""
        result = {
            "test": test_name,
            "statistic": self._to_json_safe(statistic),
            "p_value": self._to_json_safe(p_value),
            "warnings": warnings_list or [],
            "details": self._to_json_safe(details or {})
        }
        if df is not None:
            result["df"] = self._to_json_safe(df)
        return result

    # ==================== T-TESTS ====================

    def one_sample_t(self, data: pd.DataFrame, column: str, popmean: float) -> Dict[str, Any]:
        """
        Perform a one-sample t-test.
        
        Args:
            data: DataFrame containing the data
            column: Name of the numeric column to test
            popmean: Hypothesized population mean
            
        Returns:
            Dict with test results
        """
        self._validate_column(data, column)
        warnings_list = []

        if not self._is_numeric(data, column):
            raise ValueError(f"Column '{column}' must be numeric for one-sample t-test")

        sample = self._clean_numeric(data[column])
        
        if len(sample) < 2:
            raise ValueError("Sample size must be at least 2 for t-test")
        
        if len(sample) < 30:
            warnings_list.append("Sample size < 30, t-test assumes normality")

        result = stats.ttest_1samp(sample, popmean)
        
        return self._build_result(
            test_name="one_sample_t_test",
            statistic=result.statistic,
            p_value=result.pvalue,
            df=len(sample) - 1,
            warnings_list=warnings_list,
            details={
                "sample_mean": float(sample.mean()),
                "sample_std": float(sample.std()),
                "sample_size": len(sample),
                "hypothesized_mean": popmean,
                "confidence_interval_95": list(stats.t.interval(0.95, len(sample)-1, loc=sample.mean(), scale=stats.sem(sample)))
            }
        )

    def independent_t(self, data: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
        """
        Perform an independent samples t-test.
        
        Args:
            data: DataFrame containing the data
            group_col: Name of the grouping column (must have exactly 2 groups)
            value_col: Name of the numeric column to compare
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [group_col, value_col])
        warnings_list = []

        if not self._is_numeric(data, value_col):
            raise ValueError(f"Column '{value_col}' must be numeric for independent t-test")

        groups = data[group_col].dropna().unique()
        if len(groups) != 2:
            raise ValueError(f"Group column must have exactly 2 groups, found {len(groups)}: {list(groups)}")

        group1 = self._clean_numeric(data[data[group_col] == groups[0]][value_col])
        group2 = self._clean_numeric(data[data[group_col] == groups[1]][value_col])

        if len(group1) < 2 or len(group2) < 2:
            raise ValueError("Each group must have at least 2 observations")

        if len(group1) < 30 or len(group2) < 30:
            warnings_list.append("One or both groups have n < 30, t-test assumes normality")

        # Levene's test for equal variances
        levene_stat, levene_p = stats.levene(group1, group2)
        equal_var = levene_p > 0.05
        
        if not equal_var:
            warnings_list.append("Unequal variances detected (Levene's p < 0.05), using Welch's t-test")

        result = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        # Calculate degrees of freedom (Welch-Satterthwaite if unequal variances)
        if equal_var:
            df = len(group1) + len(group2) - 2
        else:
            # Welch-Satterthwaite approximation
            v1, v2 = group1.var(), group2.var()
            n1, n2 = len(group1), len(group2)
            df = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))

        return self._build_result(
            test_name="independent_t_test",
            statistic=result.statistic,
            p_value=result.pvalue,
            df=df,
            warnings_list=warnings_list,
            details={
                "group1_name": str(groups[0]),
                "group2_name": str(groups[1]),
                "group1_mean": float(group1.mean()),
                "group2_mean": float(group2.mean()),
                "group1_std": float(group1.std()),
                "group2_std": float(group2.std()),
                "group1_n": len(group1),
                "group2_n": len(group2),
                "mean_difference": float(group1.mean() - group2.mean()),
                "equal_variances_assumed": equal_var,
                "levene_statistic": float(levene_stat),
                "levene_p_value": float(levene_p),
                "effect_size_cohens_d": float((group1.mean() - group2.mean()) / np.sqrt(((len(group1)-1)*group1.var() + (len(group2)-1)*group2.var()) / (len(group1)+len(group2)-2)))
            }
        )

    def paired_t(self, data: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Perform a paired samples t-test.
        
        Args:
            data: DataFrame containing the data
            col1: Name of the first numeric column
            col2: Name of the second numeric column
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [col1, col2])
        warnings_list = []

        if not self._is_numeric(data, col1) or not self._is_numeric(data, col2):
            raise ValueError(f"Both columns must be numeric for paired t-test")

        # Remove rows with NaN in either column
        clean_data = data[[col1, col2]].dropna()
        sample1 = clean_data[col1]
        sample2 = clean_data[col2]

        if len(sample1) < 2:
            raise ValueError("Must have at least 2 paired observations")

        if len(sample1) < 30:
            warnings_list.append("Sample size < 30, paired t-test assumes normality of differences")

        result = stats.ttest_rel(sample1, sample2)
        differences = sample1 - sample2

        return self._build_result(
            test_name="paired_t_test",
            statistic=result.statistic,
            p_value=result.pvalue,
            df=len(sample1) - 1,
            warnings_list=warnings_list,
            details={
                "mean1": float(sample1.mean()),
                "mean2": float(sample2.mean()),
                "std1": float(sample1.std()),
                "std2": float(sample2.std()),
                "mean_difference": float(differences.mean()),
                "std_difference": float(differences.std()),
                "n_pairs": len(sample1),
                "correlation": float(sample1.corr(sample2)),
                "confidence_interval_95": list(stats.t.interval(0.95, len(differences)-1, loc=differences.mean(), scale=stats.sem(differences)))
            }
        )

    # ==================== ANOVA & NON-PARAMETRIC ====================

    def anova(self, data: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
        """
        Perform one-way ANOVA test.
        
        Args:
            data: DataFrame containing the data
            group_col: Name of the grouping column
            value_col: Name of the numeric column to compare
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [group_col, value_col])
        warnings_list = []

        if not self._is_numeric(data, value_col):
            raise ValueError(f"Column '{value_col}' must be numeric for ANOVA")

        groups = data[group_col].dropna().unique()
        if len(groups) < 2:
            raise ValueError(f"Need at least 2 groups for ANOVA, found {len(groups)}")

        group_data = [self._clean_numeric(data[data[group_col] == g][value_col]) for g in groups]
        
        # Check group sizes
        group_sizes = [len(g) for g in group_data]
        if any(s < 2 for s in group_sizes):
            raise ValueError("Each group must have at least 2 observations")

        if any(s < 30 for s in group_sizes):
            warnings_list.append("One or more groups have n < 30, ANOVA assumes normality")

        # Levene's test for homogeneity of variances
        levene_stat, levene_p = stats.levene(*group_data)
        if levene_p < 0.05:
            warnings_list.append("Unequal variances detected (Levene's p < 0.05), consider Welch ANOVA or Kruskal-Wallis")

        result = stats.f_oneway(*group_data)
        
        # Calculate effect size (eta-squared)
        grand_mean = data[value_col].dropna().mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_data)
        ss_total = sum((data[value_col].dropna() - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        df_between = len(groups) - 1
        df_within = sum(group_sizes) - len(groups)

        return self._build_result(
            test_name="one_way_anova",
            statistic=result.statistic,
            p_value=result.pvalue,
            df=(df_between, df_within),
            warnings_list=warnings_list,
            details={
                "n_groups": len(groups),
                "group_names": [str(g) for g in groups],
                "group_means": {str(g): float(group_data[i].mean()) for i, g in enumerate(groups)},
                "group_stds": {str(g): float(group_data[i].std()) for i, g in enumerate(groups)},
                "group_sizes": {str(g): group_sizes[i] for i, g in enumerate(groups)},
                "eta_squared": float(eta_squared),
                "levene_statistic": float(levene_stat),
                "levene_p_value": float(levene_p)
            }
        )

    def kruskal_test(self, data: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
        """
        Perform Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA).
        
        Args:
            data: DataFrame containing the data
            group_col: Name of the grouping column
            value_col: Name of the numeric column to compare
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [group_col, value_col])
        warnings_list = []

        if not self._is_numeric(data, value_col):
            raise ValueError(f"Column '{value_col}' must be numeric for Kruskal-Wallis test")

        groups = data[group_col].dropna().unique()
        if len(groups) < 2:
            raise ValueError(f"Need at least 2 groups for Kruskal-Wallis test, found {len(groups)}")

        group_data = [self._clean_numeric(data[data[group_col] == g][value_col]) for g in groups]
        
        group_sizes = [len(g) for g in group_data]
        if any(s < 2 for s in group_sizes):
            raise ValueError("Each group must have at least 2 observations")

        if any(s < 5 for s in group_sizes):
            warnings_list.append("One or more groups have n < 5, results may be unreliable")

        result = stats.kruskal(*group_data)

        return self._build_result(
            test_name="kruskal_wallis_test",
            statistic=result.statistic,
            p_value=result.pvalue,
            df=len(groups) - 1,
            warnings_list=warnings_list,
            details={
                "n_groups": len(groups),
                "group_names": [str(g) for g in groups],
                "group_medians": {str(g): float(group_data[i].median()) for i, g in enumerate(groups)},
                "group_sizes": {str(g): group_sizes[i] for i, g in enumerate(groups)}
            }
        )

    # ==================== CHI-SQUARE TEST ====================

    def chi_square(self, data: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Perform Chi-square test of independence.
        
        Args:
            data: DataFrame containing the data
            col1: Name of the first categorical column
            col2: Name of the second categorical column
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [col1, col2])
        warnings_list = []

        # Create contingency table
        contingency = pd.crosstab(data[col1], data[col2])
        
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            raise ValueError("Need at least 2 categories in each variable for chi-square test")

        # Check expected frequencies
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        if np.any(expected < 5):
            warnings_list.append("Some expected frequencies < 5, chi-square approximation may be unreliable")
        if np.any(expected < 1):
            warnings_list.append("Some expected frequencies < 1, consider Fisher's exact test")

        # Calculate Cramér's V (effect size)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0

        return self._build_result(
            test_name="chi_square_test",
            statistic=chi2,
            p_value=p_value,
            df=dof,
            warnings_list=warnings_list,
            details={
                "contingency_table": contingency.to_dict(),
                "expected_frequencies": pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).to_dict(),
                "cramers_v": float(cramers_v),
                "n_observations": int(n),
                "table_shape": list(contingency.shape)
            }
        )

    # ==================== VARIANCE TESTS ====================

    def f_test(self, data: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Perform F-test for equality of variances (two samples).
        
        Args:
            data: DataFrame containing the data
            col1: Name of the first numeric column
            col2: Name of the second numeric column
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [col1, col2])
        warnings_list = []

        if not self._is_numeric(data, col1) or not self._is_numeric(data, col2):
            raise ValueError("Both columns must be numeric for F-test")

        sample1 = self._clean_numeric(data[col1])
        sample2 = self._clean_numeric(data[col2])

        if len(sample1) < 2 or len(sample2) < 2:
            raise ValueError("Each sample must have at least 2 observations")

        warnings_list.append("F-test for variances is sensitive to non-normality; consider Levene's test as alternative")

        var1, var2 = sample1.var(), sample2.var()
        
        # F-statistic is ratio of larger to smaller variance
        if var1 >= var2:
            f_stat = var1 / var2 if var2 > 0 else np.inf
            df1, df2 = len(sample1) - 1, len(sample2) - 1
        else:
            f_stat = var2 / var1 if var1 > 0 else np.inf
            df1, df2 = len(sample2) - 1, len(sample1) - 1

        # Two-tailed p-value
        p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

        return self._build_result(
            test_name="f_test_variance",
            statistic=f_stat,
            p_value=p_value,
            df=(df1, df2),
            warnings_list=warnings_list,
            details={
                "variance1": float(var1),
                "variance2": float(var2),
                "variance_ratio": float(var1 / var2) if var2 > 0 else None,
                "n1": len(sample1),
                "n2": len(sample2)
            }
        )

    def levene_test(self, data: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Perform Levene's test for equality of variances.
        
        Args:
            data: DataFrame containing the data
            col1: Name of the first numeric column
            col2: Name of the second numeric column
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [col1, col2])
        warnings_list = []

        if not self._is_numeric(data, col1) or not self._is_numeric(data, col2):
            raise ValueError("Both columns must be numeric for Levene's test")

        sample1 = self._clean_numeric(data[col1])
        sample2 = self._clean_numeric(data[col2])

        if len(sample1) < 2 or len(sample2) < 2:
            raise ValueError("Each sample must have at least 2 observations")

        result = stats.levene(sample1, sample2, center='median')

        return self._build_result(
            test_name="levene_test",
            statistic=result.statistic,
            p_value=result.pvalue,
            df=(1, len(sample1) + len(sample2) - 2),
            warnings_list=warnings_list,
            details={
                "variance1": float(sample1.var()),
                "variance2": float(sample2.var()),
                "n1": len(sample1),
                "n2": len(sample2),
                "center": "median"
            }
        )

    def bartlett_test(self, data: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Perform Bartlett's test for equality of variances.
        
        Args:
            data: DataFrame containing the data
            col1: Name of the first numeric column
            col2: Name of the second numeric column
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [col1, col2])
        warnings_list = []

        if not self._is_numeric(data, col1) or not self._is_numeric(data, col2):
            raise ValueError("Both columns must be numeric for Bartlett's test")

        sample1 = self._clean_numeric(data[col1])
        sample2 = self._clean_numeric(data[col2])

        if len(sample1) < 2 or len(sample2) < 2:
            raise ValueError("Each sample must have at least 2 observations")

        warnings_list.append("Bartlett's test is sensitive to non-normality; consider Levene's test if normality is questionable")

        result = stats.bartlett(sample1, sample2)

        return self._build_result(
            test_name="bartlett_test",
            statistic=result.statistic,
            p_value=result.pvalue,
            df=1,  # k-1 where k=2 groups
            warnings_list=warnings_list,
            details={
                "variance1": float(sample1.var()),
                "variance2": float(sample2.var()),
                "n1": len(sample1),
                "n2": len(sample2)
            }
        )

    # ==================== CORRELATION TESTS ====================

    def pearson_corr(self, data: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Compute Pearson correlation coefficient.
        
        Args:
            data: DataFrame containing the data
            col1: Name of the first numeric column
            col2: Name of the second numeric column
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [col1, col2])
        warnings_list = []

        if not self._is_numeric(data, col1) or not self._is_numeric(data, col2):
            raise ValueError("Both columns must be numeric for Pearson correlation")

        clean_data = data[[col1, col2]].dropna()
        
        if len(clean_data) < 3:
            raise ValueError("Need at least 3 observations for correlation")

        if len(clean_data) < 30:
            warnings_list.append("Sample size < 30, Pearson correlation assumes bivariate normality")

        result = stats.pearsonr(clean_data[col1], clean_data[col2])

        return self._build_result(
            test_name="pearson_correlation",
            statistic=result.statistic,
            p_value=result.pvalue,
            df=len(clean_data) - 2,
            warnings_list=warnings_list,
            details={
                "correlation": float(result.statistic),
                "r_squared": float(result.statistic ** 2),
                "n": len(clean_data),
                "interpretation": self._interpret_correlation(result.statistic)
            }
        )

    def spearman_corr(self, data: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Compute Spearman rank correlation coefficient.
        
        Args:
            data: DataFrame containing the data
            col1: Name of the first column
            col2: Name of the second column
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [col1, col2])
        warnings_list = []

        clean_data = data[[col1, col2]].dropna()
        
        if len(clean_data) < 3:
            raise ValueError("Need at least 3 observations for correlation")

        result = stats.spearmanr(clean_data[col1], clean_data[col2])

        return self._build_result(
            test_name="spearman_correlation",
            statistic=result.statistic,
            p_value=result.pvalue,
            warnings_list=warnings_list,
            details={
                "correlation": float(result.statistic),
                "n": len(clean_data),
                "interpretation": self._interpret_correlation(result.statistic)
            }
        )

    def kendall_corr(self, data: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Compute Kendall's tau correlation coefficient.
        
        Args:
            data: DataFrame containing the data
            col1: Name of the first column
            col2: Name of the second column
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [col1, col2])
        warnings_list = []

        clean_data = data[[col1, col2]].dropna()
        
        if len(clean_data) < 3:
            raise ValueError("Need at least 3 observations for correlation")

        if len(clean_data) > 1000:
            warnings_list.append("Large sample size may make Kendall's tau computationally expensive")

        result = stats.kendalltau(clean_data[col1], clean_data[col2])

        return self._build_result(
            test_name="kendall_correlation",
            statistic=result.statistic,
            p_value=result.pvalue,
            warnings_list=warnings_list,
            details={
                "tau": float(result.statistic),
                "n": len(clean_data),
                "interpretation": self._interpret_correlation(result.statistic)
            }
        )

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient magnitude."""
        r_abs = abs(r)
        if r_abs < 0.1:
            strength = "negligible"
        elif r_abs < 0.3:
            strength = "weak"
        elif r_abs < 0.5:
            strength = "moderate"
        elif r_abs < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if r >= 0 else "negative"
        return f"{strength} {direction} correlation"

    # ==================== NORMALITY TESTS ====================

    def shapiro_test(self, data: pd.DataFrame, col: str) -> Dict[str, Any]:
        """
        Perform Shapiro-Wilk test for normality.
        
        Args:
            data: DataFrame containing the data
            col: Name of the numeric column to test
            
        Returns:
            Dict with test results
        """
        self._validate_column(data, col)
        warnings_list = []

        if not self._is_numeric(data, col):
            raise ValueError(f"Column '{col}' must be numeric for Shapiro-Wilk test")

        sample = self._clean_numeric(data[col])
        
        if len(sample) < 3:
            raise ValueError("Need at least 3 observations for Shapiro-Wilk test")

        if len(sample) > 5000:
            warnings_list.append("Sample size > 5000, using first 5000 observations (scipy limitation)")
            sample = sample.head(5000)

        result = stats.shapiro(sample)

        return self._build_result(
            test_name="shapiro_wilk_test",
            statistic=result.statistic,
            p_value=result.pvalue,
            warnings_list=warnings_list,
            details={
                "n": len(sample),
                "is_normal": result.pvalue > 0.05,
                "interpretation": "Data appears normally distributed" if result.pvalue > 0.05 else "Data does not appear normally distributed"
            }
        )

    def ks_test(self, data: pd.DataFrame, col: str) -> Dict[str, Any]:
        """
        Perform Kolmogorov-Smirnov test for normality.
        
        Args:
            data: DataFrame containing the data
            col: Name of the numeric column to test
            
        Returns:
            Dict with test results
        """
        self._validate_column(data, col)
        warnings_list = []

        if not self._is_numeric(data, col):
            raise ValueError(f"Column '{col}' must be numeric for Kolmogorov-Smirnov test")

        sample = self._clean_numeric(data[col])
        
        if len(sample) < 3:
            raise ValueError("Need at least 3 observations for K-S test")

        # Standardize data for comparison with standard normal
        standardized = (sample - sample.mean()) / sample.std()
        result = stats.kstest(standardized, 'norm')

        return self._build_result(
            test_name="kolmogorov_smirnov_test",
            statistic=result.statistic,
            p_value=result.pvalue,
            warnings_list=warnings_list,
            details={
                "n": len(sample),
                "is_normal": result.pvalue > 0.05,
                "interpretation": "Data appears normally distributed" if result.pvalue > 0.05 else "Data does not appear normally distributed",
                "sample_mean": float(sample.mean()),
                "sample_std": float(sample.std())
            }
        )

    def anderson_test(self, data: pd.DataFrame, col: str) -> Dict[str, Any]:
        """
        Perform Anderson-Darling test for normality.
        
        Args:
            data: DataFrame containing the data
            col: Name of the numeric column to test
            
        Returns:
            Dict with test results
        """
        self._validate_column(data, col)
        warnings_list = []

        if not self._is_numeric(data, col):
            raise ValueError(f"Column '{col}' must be numeric for Anderson-Darling test")

        sample = self._clean_numeric(data[col])
        
        if len(sample) < 3:
            raise ValueError("Need at least 3 observations for Anderson-Darling test")

        result = stats.anderson(sample, dist='norm')

        # Determine significance level
        significance_levels = {
            0: "15%",
            1: "10%",
            2: "5%",
            3: "2.5%",
            4: "1%"
        }
        
        rejected_at = []
        for i, (cv, sl) in enumerate(zip(result.critical_values, result.significance_level)):
            if result.statistic > cv:
                rejected_at.append(f"{sl}%")

        return self._build_result(
            test_name="anderson_darling_test",
            statistic=result.statistic,
            p_value=None,  # Anderson-Darling doesn't provide a p-value
            warnings_list=warnings_list,
            details={
                "n": len(sample),
                "critical_values": {f"{sl}%": float(cv) for cv, sl in zip(result.critical_values, result.significance_level)},
                "rejected_at_significance_levels": rejected_at,
                "is_normal_at_5pct": result.statistic <= result.critical_values[2],
                "interpretation": "Data appears normally distributed at 5% level" if result.statistic <= result.critical_values[2] else "Data does not appear normally distributed at 5% level"
            }
        )

    def jb_test(self, data: pd.DataFrame, col: str) -> Dict[str, Any]:
        """
        Perform Jarque-Bera test for normality.
        
        Args:
            data: DataFrame containing the data
            col: Name of the numeric column to test
            
        Returns:
            Dict with test results
        """
        self._validate_column(data, col)
        warnings_list = []

        if not self._is_numeric(data, col):
            raise ValueError(f"Column '{col}' must be numeric for Jarque-Bera test")

        sample = self._clean_numeric(data[col])
        
        if len(sample) < 3:
            raise ValueError("Need at least 3 observations for Jarque-Bera test")

        if len(sample) < 20:
            warnings_list.append("Sample size < 20, Jarque-Bera test may not be reliable")

        result = stats.jarque_bera(sample)
        skewness = stats.skew(sample)
        kurtosis = stats.kurtosis(sample)

        return self._build_result(
            test_name="jarque_bera_test",
            statistic=result.statistic,
            p_value=result.pvalue,
            df=2,  # JB test statistic has 2 degrees of freedom
            warnings_list=warnings_list,
            details={
                "n": len(sample),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "is_normal": result.pvalue > 0.05,
                "interpretation": "Data appears normally distributed" if result.pvalue > 0.05 else "Data does not appear normally distributed"
            }
        )

    # ==================== PROPORTION TESTS ====================

    def proportion_test(self, data: pd.DataFrame, col: str) -> Dict[str, Any]:
        """
        Perform one-sample proportion test (binomial test).
        Tests if proportion equals 0.5.
        
        Args:
            data: DataFrame containing the data
            col: Name of the binary/categorical column
            
        Returns:
            Dict with test results
        """
        self._validate_column(data, col)
        warnings_list = []

        clean_data = data[col].dropna()
        unique_vals = clean_data.unique()
        
        if len(unique_vals) != 2:
            warnings_list.append(f"Column has {len(unique_vals)} unique values, using most frequent two")
            value_counts = clean_data.value_counts()
            if len(value_counts) < 2:
                raise ValueError("Need at least 2 categories for proportion test")
            clean_data = clean_data[clean_data.isin(value_counts.head(2).index)]
            unique_vals = clean_data.unique()

        # Count successes (first category)
        n = len(clean_data)
        success_val = unique_vals[0]
        k = (clean_data == success_val).sum()
        
        # Binomial test against p=0.5
        result = stats.binom_test(k, n, p=0.5, alternative='two-sided')
        
        # Normal approximation for confidence interval
        p_hat = k / n
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        ci = (p_hat - 1.96 * se, p_hat + 1.96 * se)

        return self._build_result(
            test_name="one_sample_proportion_test",
            statistic=k,  # Number of successes
            p_value=result if isinstance(result, float) else result.pvalue,
            warnings_list=warnings_list,
            details={
                "n": n,
                "successes": int(k),
                "success_value": str(success_val),
                "sample_proportion": float(p_hat),
                "hypothesized_proportion": 0.5,
                "confidence_interval_95": [float(ci[0]), float(ci[1])]
            }
        )

    def two_proportion_test(self, data: pd.DataFrame, col: str, group_col: str) -> Dict[str, Any]:
        """
        Perform two-sample proportion test (z-test for proportions).
        
        Args:
            data: DataFrame containing the data
            col: Name of the binary/categorical column
            group_col: Name of the grouping column (must have exactly 2 groups)
            
        Returns:
            Dict with test results
        """
        self._validate_columns(data, [col, group_col])
        warnings_list = []

        groups = data[group_col].dropna().unique()
        if len(groups) != 2:
            raise ValueError(f"Group column must have exactly 2 groups, found {len(groups)}")

        # Get data for each group
        group1_data = data[data[group_col] == groups[0]][col].dropna()
        group2_data = data[data[group_col] == groups[1]][col].dropna()

        # Determine success value
        all_values = pd.concat([group1_data, group2_data])
        unique_vals = all_values.unique()
        
        if len(unique_vals) != 2:
            warnings_list.append(f"Column has {len(unique_vals)} unique values, using most frequent two")
            value_counts = all_values.value_counts()
            if len(value_counts) < 2:
                raise ValueError("Need at least 2 categories for proportion test")
            success_val = value_counts.index[0]
        else:
            success_val = unique_vals[0]

        # Calculate proportions
        n1, n2 = len(group1_data), len(group2_data)
        x1 = (group1_data == success_val).sum()
        x2 = (group2_data == success_val).sum()
        
        if n1 < 5 or n2 < 5:
            warnings_list.append("Small sample size, normal approximation may not be accurate")

        p1, p2 = x1 / n1, x2 / n2
        p_pooled = (x1 + x2) / (n1 + n2)
        
        # Z-test statistic
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z_stat = (p1 - p2) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return self._build_result(
            test_name="two_sample_proportion_test",
            statistic=z_stat,
            p_value=p_value,
            warnings_list=warnings_list,
            details={
                "group1_name": str(groups[0]),
                "group2_name": str(groups[1]),
                "group1_n": n1,
                "group2_n": n2,
                "group1_successes": int(x1),
                "group2_successes": int(x2),
                "group1_proportion": float(p1),
                "group2_proportion": float(p2),
                "pooled_proportion": float(p_pooled),
                "proportion_difference": float(p1 - p2),
                "success_value": str(success_val)
            }
        )

    # ==================== AUTO-SELECT TEST ====================

    def auto_select_test(
        self,
        data: pd.DataFrame,
        var1: str,
        var2: Optional[str] = None,
        group: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Automatically select the appropriate statistical test based on variable types.
        
        Args:
            data: DataFrame containing the data
            var1: Name of the primary variable
            var2: Name of the second variable (optional)
            group: Name of the grouping variable (optional)
            
        Returns:
            Dict with recommended test and reasoning
        """
        # Validate inputs
        if var1:
            self._validate_column(data, var1)
        if var2:
            self._validate_column(data, var2)
        if group:
            self._validate_column(data, group)

        var1_numeric = self._is_numeric(data, var1) if var1 else False
        var2_numeric = self._is_numeric(data, var2) if var2 else False
        
        # Determine number of groups if group variable provided
        n_groups = data[group].nunique() if group else None

        # Decision logic
        if var2 is None and group is None:
            # Single variable - normality tests
            if var1_numeric:
                return {
                    "test": "normality_tests",
                    "recommended_tests": ["shapiro_test", "ks_test", "anderson_test", "jb_test"],
                    "reason": "Single numeric variable - testing distribution normality"
                }
            else:
                return {
                    "test": "frequency_analysis",
                    "recommended_tests": ["proportion_test"],
                    "reason": "Single categorical variable - frequency/proportion analysis"
                }

        elif var2 is not None and group is None:
            # Two variables, no grouping
            if var1_numeric and var2_numeric:
                return {
                    "test": "correlation",
                    "recommended_tests": ["pearson_corr", "spearman_corr", "kendall_corr"],
                    "reason": "Two numeric variables - correlation analysis",
                    "additional_tests": ["levene_test", "f_test", "bartlett_test"]
                }
            elif not var1_numeric and not var2_numeric:
                return {
                    "test": "chi_square",
                    "recommended_tests": ["chi_square"],
                    "reason": "Two categorical variables - chi-square test of independence"
                }
            else:
                # One numeric, one categorical - treat categorical as group
                numeric_var = var1 if var1_numeric else var2
                cat_var = var2 if var1_numeric else var1
                n_cats = data[cat_var].nunique()
                
                if n_cats == 2:
                    return {
                        "test": "independent_t",
                        "recommended_tests": ["independent_t"],
                        "reason": f"One numeric and one binary categorical variable - independent t-test",
                        "variables": {"value_col": numeric_var, "group_col": cat_var}
                    }
                else:
                    return {
                        "test": "anova",
                        "recommended_tests": ["anova", "kruskal_test"],
                        "reason": f"One numeric and one categorical variable ({n_cats} groups) - ANOVA or Kruskal-Wallis",
                        "variables": {"value_col": numeric_var, "group_col": cat_var}
                    }

        elif group is not None:
            # Grouping variable provided
            if n_groups == 2:
                if var1_numeric:
                    if var2 is None:
                        return {
                            "test": "independent_t",
                            "recommended_tests": ["independent_t"],
                            "reason": "Numeric variable with 2 groups - independent t-test",
                            "variables": {"value_col": var1, "group_col": group}
                        }
                    elif var2_numeric:
                        return {
                            "test": "paired_t",
                            "recommended_tests": ["paired_t"],
                            "reason": "Two numeric variables with paired observations - paired t-test",
                            "variables": {"col1": var1, "col2": var2}
                        }
                else:
                    return {
                        "test": "two_proportion_test",
                        "recommended_tests": ["two_proportion_test"],
                        "reason": "Categorical variable with 2 groups - two-proportion z-test",
                        "variables": {"col": var1, "group_col": group}
                    }
            else:
                if var1_numeric:
                    return {
                        "test": "anova",
                        "recommended_tests": ["anova", "kruskal_test"],
                        "reason": f"Numeric variable with {n_groups} groups - ANOVA or Kruskal-Wallis test",
                        "variables": {"value_col": var1, "group_col": group}
                    }
                else:
                    return {
                        "test": "chi_square",
                        "recommended_tests": ["chi_square"],
                        "reason": f"Categorical variable with {n_groups} groups - chi-square test",
                        "variables": {"col1": var1, "col2": group}
                    }

        return {
            "test": "unknown",
            "reason": "Could not determine appropriate test based on provided variables"
        }


# Convenience function for module-level access
def get_engine() -> StatisticalTestEngine:
    """Get a StatisticalTestEngine instance."""
    return StatisticalTestEngine()
