"""
RecommendationEngine - Rule-based recommendation system for statistical analysis
Suggests methods, transformations, tests, and models based on dataset characteristics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class RecommendationEngine:
    """
    Intelligent recommendation system that analyzes dataset structure and suggests
    appropriate statistical methods, transformations, tests, and machine learning models.
    """
    
    def __init__(self, dataframe: pd.DataFrame, profiling: Optional[Dict[str, Any]] = None):
        """
        Initialize RecommendationEngine with dataset and optional profiling results.
        
        Args:
            dataframe: pandas DataFrame to analyze
            profiling: Optional dictionary with pre-computed statistics (risk indicators, etc.)
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        
        self.df = dataframe
        self.profiling = profiling or {}
        
        # Compute basic metadata
        self._analyze_columns()
    
    def _analyze_columns(self) -> None:
        """Analyze column types and characteristics for recommendations."""
        self.numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        self.categorical_cols = list(self.df.select_dtypes(exclude=[np.number]).columns)
        self.datetime_cols = []
        
        # Detect datetime columns
        for col in self.categorical_cols[:]:  # Iterate over copy
            try:
                if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    self.datetime_cols.append(col)
                    self.categorical_cols.remove(col)
                elif self.df[col].dtype == 'object':
                    # Try to parse as datetime
                    sample = self.df[col].dropna().head(100)
                    if len(sample) > 0:
                        try:
                            pd.to_datetime(sample)
                            self.datetime_cols.append(col)
                            self.categorical_cols.remove(col)
                        except Exception:
                            pass
            except Exception:
                pass
    
    def recommend_methods(self) -> List[str]:
        """
        Recommend analysis methods based on dataset structure.
        
        Rules:
        - >10 numeric columns → PCA + correlation analysis
        - >3 categorical columns → crosstabs + chi-square
        - Time column detected → forecasting
        
        Returns:
            List of recommended analysis method descriptions
        """
        recommendations = []
        
        # Dimensionality reduction for high-dimensional data
        if len(self.numeric_cols) > 10:
            recommendations.append(
                "Run PCA to reduce dimensionality and identify principal components"
            )
            recommendations.append(
                "Investigate correlation matrix to detect multicollinearity"
            )
        
        # Correlation analysis for numeric data
        if len(self.numeric_cols) >= 2:
            recommendations.append(
                "Perform correlation analysis to identify relationships between numeric variables"
            )
        
        # Categorical analysis
        if len(self.categorical_cols) >= 3:
            recommendations.append(
                "Use crosstab + chi-square tests for categorical variable interactions"
            )
        
        if len(self.categorical_cols) >= 1 and len(self.numeric_cols) >= 1:
            recommendations.append(
                "Conduct group-wise comparisons (ANOVA/t-tests) between categorical and numeric variables"
            )
        
        # Time series analysis
        if len(self.datetime_cols) > 0:
            recommendations.append(
                "Time series detected — apply ARIMA or Holt-Winters forecasting"
            )
            recommendations.append(
                "Perform seasonal decomposition to identify trends and patterns"
            )
        
        # Cluster analysis
        if len(self.numeric_cols) >= 3:
            recommendations.append(
                "Consider unsupervised clustering (K-Means) to discover natural groupings"
            )
        
        # Descriptive statistics
        if len(self.df) > 100:
            recommendations.append(
                "Generate comprehensive descriptive statistics and distribution plots"
            )
        
        return recommendations
    
    def recommend_transformations(self) -> List[Dict[str, str]]:
        """
        Recommend data transformations based on column characteristics.
        
        Rules:
        - Skewness > |1.5| → log/boxcox transform
        - Outliers > 10% → Winsorization or robust scaling
        - High cardinality categorical → grouping or hashing
        
        Returns:
            List of transformation recommendations with column, action, and reason
        """
        transformations = []
        
        # Analyze numeric columns
        for col in self.numeric_cols:
            try:
                data = self.df[col].dropna()
                
                if len(data) < 3:
                    continue
                
                # Check skewness
                skew = data.skew()
                if abs(skew) > 1.5:
                    transform_type = "log_transform" if skew > 0 else "power_transform"
                    transformations.append({
                        "column": col,
                        "action": transform_type,
                        "reason": f"High skew detected (skew={skew:.2f})"
                    })
                
                # Check outliers
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outlier_percent = (len(outliers) / len(data) * 100) if len(data) > 0 else 0
                
                if outlier_percent > 10:
                    transformations.append({
                        "column": col,
                        "action": "winsorize",
                        "reason": f"Many outliers detected ({outlier_percent:.1f}%)"
                    })
                elif outlier_percent > 5:
                    transformations.append({
                        "column": col,
                        "action": "robust_scale",
                        "reason": f"Moderate outliers detected ({outlier_percent:.1f}%)"
                    })
                
                # Check for zero or negative values if log transform suggested
                if any(t["action"] == "log_transform" and t["column"] == col for t in transformations):
                    if data.min() <= 0:
                        # Update recommendation
                        for t in transformations:
                            if t["column"] == col and t["action"] == "log_transform":
                                t["action"] = "yeo_johnson_transform"
                                t["reason"] += " (contains zero/negative values)"
                
            except Exception:
                # Skip problematic columns
                continue
        
        # Analyze categorical columns
        for col in self.categorical_cols:
            try:
                unique_count = self.df[col].nunique()
                total_count = len(self.df[col].dropna())
                
                # High cardinality
                if unique_count > 50:
                    transformations.append({
                        "column": col,
                        "action": "group_rare_categories",
                        "reason": f"High cardinality detected ({unique_count} unique values)"
                    })
                
                # Check for mixed types (numeric strings)
                if self.df[col].dtype == 'object':
                    sample = self.df[col].dropna().head(100)
                    if len(sample) > 0:
                        numeric_like = sum(str(val).replace('.', '').replace('-', '').isdigit() 
                                         for val in sample)
                        if numeric_like / len(sample) > 0.5:
                            transformations.append({
                                "column": col,
                                "action": "convert_to_numeric",
                                "reason": "Column contains numeric values stored as text"
                            })
            
            except Exception:
                continue
        
        return transformations
    
    def recommend_tests(self) -> List[Dict[str, Any]]:
        """
        Recommend statistical tests based on variable types and relationships.
        
        Rules:
        - Numeric vs Numeric → Correlation (Pearson/Spearman)
        - Numeric vs Categorical → ANOVA or t-test
        - Categorical vs Categorical → Chi-square test
        
        Returns:
            List of test recommendations with variables and suggested test
        """
        tests = []
        
        # Numeric vs Numeric - Correlation tests
        if len(self.numeric_cols) >= 2:
            for i, col1 in enumerate(self.numeric_cols[:5]):  # Limit to avoid too many
                for col2 in self.numeric_cols[i+1:6]:
                    tests.append({
                        "variables": [col1, col2],
                        "suggest": "Pearson correlation",
                        "reason": "Test linear relationship between numeric variables"
                    })
        
        # Numeric vs Categorical - ANOVA/t-test
        if len(self.numeric_cols) >= 1 and len(self.categorical_cols) >= 1:
            for num_col in self.numeric_cols[:3]:
                for cat_col in self.categorical_cols[:3]:
                    try:
                        num_categories = self.df[cat_col].nunique()
                        if num_categories == 2:
                            tests.append({
                                "variables": [num_col, cat_col],
                                "suggest": "Independent t-test",
                                "reason": "Compare numeric variable across 2 groups"
                            })
                        elif 2 < num_categories <= 10:
                            tests.append({
                                "variables": [num_col, cat_col],
                                "suggest": "One-way ANOVA",
                                "reason": f"Compare numeric variable across {num_categories} groups"
                            })
                    except Exception:
                        continue
        
        # Categorical vs Categorical - Chi-square test
        if len(self.categorical_cols) >= 2:
            for i, col1 in enumerate(self.categorical_cols[:3]):
                for col2 in self.categorical_cols[i+1:4]:
                    tests.append({
                        "variables": [col1, col2],
                        "suggest": "Chi-square test",
                        "reason": "Test independence between categorical variables"
                    })
        
        # Normality tests for numeric columns
        if len(self.numeric_cols) >= 1:
            for col in self.numeric_cols[:3]:
                tests.append({
                    "variables": [col],
                    "suggest": "Shapiro-Wilk normality test",
                    "reason": "Verify if data follows normal distribution"
                })
        
        return tests
    
    def recommend_models(self) -> List[Dict[str, str]]:
        """
        Recommend machine learning models based on dataset characteristics.
        
        Rules:
        - Binary target → Logistic regression or Random Forest classifier
        - Numeric target → Linear regression or Random Forest regressor
        - No clear target → Unsupervised clustering (K-Means)
        
        Returns:
            List of model recommendations with model type and reason
        """
        models = []
        
        # Try to identify potential target variables
        binary_targets = []
        numeric_targets = []
        
        for col in self.categorical_cols:
            try:
                unique_count = self.df[col].nunique()
                if unique_count == 2:
                    binary_targets.append(col)
            except Exception:
                continue
        
        for col in self.numeric_cols:
            try:
                # Check if column might be a target (e.g., outcome, result, target in name)
                if any(keyword in col.lower() for keyword in ['target', 'outcome', 'result', 'label', 'y']):
                    numeric_targets.append(col)
            except Exception:
                continue
        
        # Classification models
        if binary_targets:
            models.append({
                "model": "logistic_regression",
                "reason": f"Binary target detected ({binary_targets[0]}) - linear baseline model"
            })
            models.append({
                "model": "random_forest_classifier",
                "reason": "Binary classification with non-linear patterns likely"
            })
            models.append({
                "model": "gradient_boosting_classifier",
                "reason": "High-performance classification for complex relationships"
            })
        
        # Regression models
        if numeric_targets:
            models.append({
                "model": "linear_regression",
                "reason": f"Numeric target detected ({numeric_targets[0]}) - linear baseline"
            })
            models.append({
                "model": "random_forest_regressor",
                "reason": "Regression with non-linear patterns and feature importance"
            })
            models.append({
                "model": "gradient_boosting_regressor",
                "reason": "High-performance regression for complex relationships"
            })
        
        # Unsupervised learning
        if len(self.numeric_cols) >= 3:
            models.append({
                "model": "kmeans_clustering",
                "reason": "Discover natural groupings in data without labels"
            })
            models.append({
                "model": "hierarchical_clustering",
                "reason": "Explore nested cluster structures and dendrograms"
            })
        
        # Time series models
        if len(self.datetime_cols) > 0 and len(self.numeric_cols) >= 1:
            models.append({
                "model": "arima",
                "reason": "Time series forecasting with trend and seasonality"
            })
            models.append({
                "model": "prophet",
                "reason": "Robust time series forecasting with automatic seasonality detection"
            })
        
        # If no specific models recommended, suggest exploratory approaches
        if not models:
            if len(self.numeric_cols) >= 2:
                models.append({
                    "model": "pca",
                    "reason": "Dimensionality reduction and exploratory data analysis"
                })
            models.append({
                "model": "descriptive_statistics",
                "reason": "Start with comprehensive data exploration and profiling"
            })
        
        return models
    
    def build_summary(self) -> Dict[str, Any]:
        """
        Build comprehensive recommendation summary combining all analysis.
        
        Returns:
            Dictionary with all recommendations:
                - recommended_methods: Analysis method suggestions
                - recommended_transformations: Data transformation suggestions
                - recommended_tests: Statistical test recommendations
                - recommended_models: ML model suggestions
                - metadata: Dataset characteristics summary
        """
        return {
            "recommended_methods": self.recommend_methods(),
            "recommended_transformations": self.recommend_transformations(),
            "recommended_tests": self.recommend_tests(),
            "recommended_models": self.recommend_models(),
            "metadata": {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "numeric_columns": len(self.numeric_cols),
                "categorical_columns": len(self.categorical_cols),
                "datetime_columns": len(self.datetime_cols),
                "missing_percent": float(self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns)) * 100)
                    if len(self.df) > 0 else 0.0
            }
        }
    
    def generate_narrative(self, summary: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate human-readable narrative from recommendation summary.
        
        Converts structured recommendations into professional narrative text
        suitable for reports or dashboards.
        
        Args:
            summary: Optional pre-computed summary. If None, generates new summary.
            
        Returns:
            Human-readable narrative string with recommendations
        """
        if summary is None:
            summary = self.build_summary()
        
        narrative_parts = []
        
        # Introduction
        metadata = summary.get("metadata", {})
        narrative_parts.append(
            f"Based on the dataset containing {metadata.get('total_rows', 0):,} rows and "
            f"{metadata.get('total_columns', 0)} columns "
            f"({metadata.get('numeric_columns', 0)} numeric, "
            f"{metadata.get('categorical_columns', 0)} categorical), "
            f"the system recommends the following analysis strategy:"
        )
        narrative_parts.append("")
        
        # Analysis Methods
        methods = summary.get("recommended_methods", [])
        if methods:
            narrative_parts.append("**Analysis Methods:**")
            for i, method in enumerate(methods, 1):
                narrative_parts.append(f"  {i}. {method}")
            narrative_parts.append("")
        
        # Statistical Tests
        tests = summary.get("recommended_tests", [])
        if tests:
            narrative_parts.append("**Statistical Tests:**")
            for i, test in enumerate(tests[:5], 1):  # Limit to top 5
                variables = ", ".join(test.get("variables", []))
                suggestion = test.get("suggest", "")
                reason = test.get("reason", "")
                narrative_parts.append(f"  {i}. {suggestion} for {variables} — {reason}")
            if len(tests) > 5:
                narrative_parts.append(f"  ... and {len(tests) - 5} more tests")
            narrative_parts.append("")
        
        # Data Transformations
        transformations = summary.get("recommended_transformations", [])
        if transformations:
            narrative_parts.append("**Data Transformations:**")
            for i, transform in enumerate(transformations[:5], 1):  # Limit to top 5
                column = transform.get("column", "")
                action = transform.get("action", "")
                reason = transform.get("reason", "")
                narrative_parts.append(f"  {i}. Apply {action} to '{column}' — {reason}")
            if len(transformations) > 5:
                narrative_parts.append(f"  ... and {len(transformations) - 5} more transformations")
            narrative_parts.append("")
        
        # ML Models
        models = summary.get("recommended_models", [])
        if models:
            narrative_parts.append("**Suggested Machine Learning Models:**")
            for i, model in enumerate(models[:5], 1):  # Limit to top 5
                model_name = model.get("model", "")
                reason = model.get("reason", "")
                narrative_parts.append(f"  {i}. {model_name.replace('_', ' ').title()} — {reason}")
            if len(models) > 5:
                narrative_parts.append(f"  ... and {len(models) - 5} more models")
            narrative_parts.append("")
        
        # Conclusion
        if metadata.get("missing_percent", 0) > 5:
            narrative_parts.append(
                f"**Note:** The dataset contains {metadata['missing_percent']:.1f}% missing values. "
                "Consider imputation or removal strategies before analysis."
            )
        
        # Join all parts
        return "\n".join(narrative_parts)
