"""
DashboardEngine - Transform analysis results into chart-ready JSON structures
Prepares data for frontend charting libraries (Chart.js, D3.js, etc.)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime


class DashboardEngine:
    """
    Transforms pandas DataFrames into JSON-serializable chart data structures.
    Outputs are optimized for frontend charting libraries.
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize DashboardEngine with a pandas DataFrame.
        
        Args:
            dataframe: pandas DataFrame containing the data to transform
            
        Raises:
            TypeError: If dataframe is not a pandas DataFrame
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        
        self.dataframe = dataframe
    
    def _validate_column(self, column_name: str, require_numeric: bool = False) -> None:
        """
        Validate that a column exists and optionally check if it's numeric.
        
        Args:
            column_name: Name of the column to validate
            require_numeric: If True, raises TypeError if column is not numeric
            
        Raises:
            ValueError: If column doesn't exist in dataframe
            TypeError: If require_numeric=True and column is not numeric
            ValueError: If column contains only NaN values
        """
        if column_name not in self.dataframe.columns:
            raise ValueError(
                f"Column '{column_name}' not found in dataframe. "
                f"Available columns: {list(self.dataframe.columns)}"
            )
        
        # Check if column is empty (all NaN)
        if self.dataframe[column_name].isna().all():
            raise ValueError(f"Column '{column_name}' contains only NaN values")
        
        # Check numeric dtype if required
        if require_numeric:
            if not np.issubdtype(self.dataframe[column_name].dtype, np.number):
                raise TypeError(
                    f"Column '{column_name}' must be numeric. "
                    f"Current dtype: {self.dataframe[column_name].dtype}"
                )
    
    def to_chart_ready_bar(self, column: str) -> Dict[str, Any]:
        """
        Transform categorical column into bar chart data structure.
        
        Computes value counts and returns labels (categories) and values (counts).
        
        Args:
            column: Name of the categorical column
            
        Returns:
            Dictionary with:
                - labels: List of category names
                - values: List of counts for each category
                - metadata: Additional info (column name, type, total count)
                
        Raises:
            ValueError: If column doesn't exist or is empty
        """
        # Validate column exists
        self._validate_column(column, require_numeric=False)
        
        # Compute value counts
        value_counts = self.dataframe[column].value_counts().sort_index()
        
        if len(value_counts) == 0:
            raise ValueError(f"Column '{column}' has no valid values")
        
        # Convert to lists (JSON-serializable)
        labels = [str(label) for label in value_counts.index.tolist()]
        values = [int(val) for val in value_counts.values.tolist()]
        
        return {
            "labels": labels,
            "values": values,
            "metadata": {
                "column": column,
                "type": "bar",
                "total_count": int(value_counts.sum()),
                "num_categories": len(labels)
            }
        }
    
    def to_chart_ready_hist(self, column: str, bins: int = 10) -> Dict[str, Any]:
        """
        Transform numeric column into histogram data structure.
        
        Computes histogram bins and returns bin ranges as labels and counts as values.
        
        Args:
            column: Name of the numeric column
            bins: Number of histogram bins (default: 10)
            
        Returns:
            Dictionary with:
                - labels: List of bin range strings (e.g., "0.0-10.0")
                - values: List of counts for each bin
                - metadata: Additional info (column name, type, bin edges)
                
        Raises:
            ValueError: If column doesn't exist or is empty
            TypeError: If column is not numeric
        """
        # Validate column is numeric
        self._validate_column(column, require_numeric=True)
        
        # Extract non-null values
        data = self.dataframe[column].dropna()
        
        if len(data) == 0:
            raise ValueError(f"Column '{column}' has no valid numeric values")
        
        # Compute histogram
        counts, bin_edges = np.histogram(data, bins=bins)
        
        # Format bin ranges as labels
        labels = []
        for i in range(len(bin_edges) - 1):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            labels.append(f"{lower:.2f}-{upper:.2f}")
        
        # Convert to JSON-serializable types
        values = [int(count) for count in counts.tolist()]
        bin_edges_list = [float(edge) for edge in bin_edges.tolist()]
        
        return {
            "labels": labels,
            "values": values,
            "metadata": {
                "column": column,
                "type": "histogram",
                "bins": bins,
                "bin_edges": bin_edges_list,
                "total_count": int(sum(values)),
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "median": float(data.median())
            }
        }
    
    def to_chart_ready_pie(self, column: str, max_categories: int = 6) -> Dict[str, Any]:
        """
        Transform categorical column into pie chart data structure.
        
        Computes proportions for each category. Limits to max_categories by
        grouping smaller categories into "Other".
        
        Args:
            column: Name of the categorical column
            max_categories: Maximum number of slices (default: 6)
            
        Returns:
            Dictionary with:
                - labels: List of category names
                - values: List of proportions (0-1 range)
                - percentages: List of percentage strings
                - metadata: Additional info (column name, type, total count)
                
        Raises:
            ValueError: If column doesn't exist or is empty
        """
        # Validate column exists
        self._validate_column(column, require_numeric=False)
        
        # Compute value counts
        value_counts = self.dataframe[column].value_counts()
        
        if len(value_counts) == 0:
            raise ValueError(f"Column '{column}' has no valid values")
        
        total_count = value_counts.sum()
        
        # If too many categories, group smaller ones into "Other"
        if len(value_counts) > max_categories:
            # Keep top (max_categories - 1) and group rest as "Other"
            top_categories = value_counts.head(max_categories - 1)
            other_count = value_counts[max_categories - 1:].sum()
            
            # Combine
            labels = [str(label) for label in top_categories.index.tolist()] + ["Other"]
            counts = top_categories.tolist() + [other_count]
        else:
            labels = [str(label) for label in value_counts.index.tolist()]
            counts = value_counts.tolist()
        
        # Compute proportions
        proportions = [count / total_count for count in counts]
        
        # Format percentages
        percentages = [f"{prop * 100:.1f}%" for prop in proportions]
        
        # Convert to JSON-serializable types
        values = [float(prop) for prop in proportions]
        
        return {
            "labels": labels,
            "values": values,
            "percentages": percentages,
            "metadata": {
                "column": column,
                "type": "pie",
                "total_count": int(total_count),
                "num_categories": len(labels),
                "has_other": len(value_counts) > max_categories
            }
        }
    
    def to_chart_ready_timeseries(self, time_col: str, value_col: str) -> Dict[str, Any]:
        """
        Transform time series data into chart-ready structure.
        
        Converts time column to datetime, sorts by time, and returns ISO timestamp
        strings as labels and values.
        
        Args:
            time_col: Name of the column containing time/date values
            value_col: Name of the column containing numeric values
            
        Returns:
            Dictionary with:
                - labels: List of ISO timestamp strings
                - values: List of numeric values
                - metadata: Additional info (columns, type, date range, stats)
                
        Raises:
            ValueError: If columns don't exist or are empty
            TypeError: If value_col is not numeric
        """
        # Validate columns exist
        self._validate_column(time_col, require_numeric=False)
        self._validate_column(value_col, require_numeric=True)
        
        # Create a copy and convert time column to datetime
        data = self.dataframe[[time_col, value_col]].copy()
        
        # Try to convert time_col to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            try:
                data[time_col] = pd.to_datetime(data[time_col])
            except Exception as e:
                raise ValueError(f"Cannot convert column '{time_col}' to datetime: {e}")
        
        # Sort by time and drop NaN values
        data = data.sort_values(time_col).dropna()
        
        if len(data) == 0:
            raise ValueError(f"No valid data points for time series")
        
        # Convert to ISO format strings for labels
        labels = [ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts) 
                  for ts in data[time_col].tolist()]
        
        # Extract values
        values = [float(val) for val in data[value_col].tolist()]
        
        # Calculate statistics
        values_array = np.array(values)
        
        return {
            "labels": labels,
            "values": values,
            "metadata": {
                "time_column": time_col,
                "value_column": value_col,
                "type": "timeseries",
                "num_points": len(values),
                "start_date": labels[0],
                "end_date": labels[-1],
                "min_value": float(values_array.min()),
                "max_value": float(values_array.max()),
                "mean_value": float(values_array.mean()),
                "median_value": float(np.median(values_array))
            }
        }
    
    def to_chart_ready_scatter(self, x_col: str, y_col: str) -> Dict[str, Any]:
        """
        Transform two numeric columns into scatter plot data structure.
        
        Args:
            x_col: Name of the x-axis column
            y_col: Name of the y-axis column
            
        Returns:
            Dictionary with:
                - labels: Empty list (scatter plots don't have labels)
                - x_values: List of x coordinates
                - y_values: List of y coordinates
                - metadata: Additional info (columns, type, correlation)
                
        Raises:
            ValueError: If columns don't exist or are empty
            TypeError: If columns are not numeric
        """
        # Validate both columns are numeric
        self._validate_column(x_col, require_numeric=True)
        self._validate_column(y_col, require_numeric=True)
        
        # Extract data (drop rows with NaN in either column)
        data = self.dataframe[[x_col, y_col]].dropna()
        
        if len(data) == 0:
            raise ValueError(f"No valid data points for scatter plot")
        
        x_values = [float(val) for val in data[x_col].tolist()]
        y_values = [float(val) for val in data[y_col].tolist()]
        
        # Calculate correlation
        correlation = float(data[x_col].corr(data[y_col]))
        
        return {
            "labels": [],  # Scatter plots don't have labels
            "x_values": x_values,
            "y_values": y_values,
            "metadata": {
                "x_column": x_col,
                "y_column": y_col,
                "type": "scatter",
                "num_points": len(x_values),
                "correlation": correlation
            }
        }
    
    def to_chart_ready_boxplot(self, column: str) -> Dict[str, Any]:
        """
        Transform numeric column into boxplot data structure.
        
        Computes quartiles, median, min, max, and outliers.
        
        Args:
            column: Name of the numeric column
            
        Returns:
            Dictionary with:
                - labels: List with single column name
                - quartiles: Dict with q1, median, q3, min, max
                - outliers: List of outlier values
                - metadata: Additional info (column, type, stats)
                
        Raises:
            ValueError: If column doesn't exist or is empty
            TypeError: If column is not numeric
        """
        # Validate column is numeric
        self._validate_column(column, require_numeric=True)
        
        # Extract non-null values
        data = self.dataframe[column].dropna()
        
        if len(data) == 0:
            raise ValueError(f"Column '{column}' has no valid numeric values")
        
        # Calculate quartiles
        q1 = float(data.quantile(0.25))
        median = float(data.median())
        q3 = float(data.quantile(0.75))
        min_val = float(data.min())
        max_val = float(data.max())
        
        # Calculate IQR and outlier bounds
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Identify outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)].tolist()
        outliers = [float(val) for val in outliers]
        
        return {
            "labels": [column],
            "quartiles": {
                "min": min_val,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": max_val
            },
            "outliers": outliers,
            "metadata": {
                "column": column,
                "type": "boxplot",
                "num_values": len(data),
                "num_outliers": len(outliers),
                "iqr": float(iqr),
                "mean": float(data.mean()),
                "std": float(data.std())
            }
        }
    
    def get_kpi_summary(self) -> Dict[str, Any]:
        """
        Get high-level KPI metrics for the entire dataset.
        
        Computes overview statistics including row/column counts, missing data,
        and column type distribution.
        
        Returns:
            Dictionary with:
                - row_count: Number of rows in dataset
                - column_count: Number of columns in dataset
                - missing_cells: Total number of missing values
                - missing_percent: Percentage of missing values
                - numeric_columns: Count of numeric columns
                - categorical_columns: Count of categorical columns
                - total_cells: Total number of cells in dataset
        """
        # Basic counts
        row_count = len(self.dataframe)
        column_count = len(self.dataframe.columns)
        total_cells = row_count * column_count
        
        # Missing data
        missing_cells = int(self.dataframe.isna().sum().sum())
        missing_percent = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0
        
        # Column types
        numeric_columns = len(self.dataframe.select_dtypes(include=[np.number]).columns)
        categorical_columns = column_count - numeric_columns
        
        return {
            "row_count": row_count,
            "column_count": column_count,
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percent": round(float(missing_percent), 2),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns
        }
    
    def get_trend_summary(self, column: str) -> Dict[str, Any]:
        """
        Analyze trend direction for a numeric column.
        
        Computes percent change from first to last value and classifies
        trend as increasing, decreasing, or stable.
        
        Args:
            column: Name of the numeric column
            
        Returns:
            Dictionary with:
                - column: Column name
                - trend_direction: "increasing", "decreasing", or "stable"
                - percent_change: Percentage change from first to last value
                - first_value: First non-null value
                - last_value: Last non-null value
                - num_values: Number of non-null values
                
        Raises:
            ValueError: If column doesn't exist or is empty
            TypeError: If column is not numeric
        """
        # Validate column is numeric
        self._validate_column(column, require_numeric=True)
        
        # Extract non-null values
        data = self.dataframe[column].dropna()
        
        if len(data) < 2:
            raise ValueError(
                f"Column '{column}' must have at least 2 non-null values for trend analysis. "
                f"Found {len(data)} values."
            )
        
        # Get first and last values
        first_value = float(data.iloc[0])
        last_value = float(data.iloc[-1])
        
        # Calculate percent change
        if first_value != 0:
            percent_change = ((last_value - first_value) / abs(first_value)) * 100
        else:
            # Handle divide by zero - use last value as indicator
            if last_value > 0:
                percent_change = 100.0
            elif last_value < 0:
                percent_change = -100.0
            else:
                percent_change = 0.0
        
        # Determine trend direction
        if percent_change > 5.0:
            trend_direction = "increasing"
        elif percent_change < -5.0:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        return {
            "column": column,
            "trend_direction": trend_direction,
            "percent_change": round(float(percent_change), 2),
            "first_value": first_value,
            "last_value": last_value,
            "num_values": len(data)
        }
    
    def get_risk_indicators(self) -> Dict[str, Any]:
        """
        Identify data quality risk indicators across the dataset.
        
        Analyzes all columns for potential quality issues including high missing rates,
        outliers, skewed distributions, and high cardinality.
        
        Returns:
            Dictionary with:
                - high_missing_columns: Columns with >30% missing values
                - outlier_columns: Numeric columns with >10% outliers
                - skewed_columns: Numeric columns with |skew| > 1.0
                - high_cardinality_columns: Categorical columns with >50 unique values
                - summary: Counts of each risk type
        """
        high_missing_columns = []
        outlier_columns = []
        skewed_columns = []
        high_cardinality_columns = []
        
        total_rows = len(self.dataframe)
        
        for column in self.dataframe.columns:
            try:
                # Check missing data rate
                missing_count = self.dataframe[column].isna().sum()
                missing_percent = (missing_count / total_rows * 100) if total_rows > 0 else 0
                
                if missing_percent > 30.0:
                    high_missing_columns.append({
                        "column": column,
                        "missing_percent": round(float(missing_percent), 2)
                    })
                
                # Check if numeric
                is_numeric = np.issubdtype(self.dataframe[column].dtype, np.number)
                
                if is_numeric:
                    data = self.dataframe[column].dropna()
                    
                    if len(data) > 0:
                        # Check for outliers using IQR method
                        try:
                            q1 = data.quantile(0.25)
                            q3 = data.quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            
                            outliers = data[(data < lower_bound) | (data > upper_bound)]
                            outlier_percent = (len(outliers) / len(data) * 100) if len(data) > 0 else 0
                            
                            if outlier_percent > 10.0:
                                outlier_columns.append({
                                    "column": column,
                                    "outlier_percent": round(float(outlier_percent), 2),
                                    "num_outliers": len(outliers)
                                })
                        except Exception:
                            pass  # Skip if quartile calculation fails
                        
                        # Check for skewness
                        try:
                            skew_value = data.skew()
                            if abs(skew_value) > 1.0:
                                skewed_columns.append({
                                    "column": column,
                                    "skew": round(float(skew_value), 2)
                                })
                        except Exception:
                            pass  # Skip if skew calculation fails
                
                else:
                    # Check cardinality for categorical columns
                    unique_count = self.dataframe[column].nunique()
                    if unique_count > 50:
                        high_cardinality_columns.append({
                            "column": column,
                            "unique_values": int(unique_count)
                        })
            
            except Exception:
                # Skip columns that cause errors
                continue
        
        return {
            "high_missing_columns": high_missing_columns,
            "outlier_columns": outlier_columns,
            "skewed_columns": skewed_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "summary": {
                "total_columns": len(self.dataframe.columns),
                "high_missing_count": len(high_missing_columns),
                "outlier_count": len(outlier_columns),
                "skewed_count": len(skewed_columns),
                "high_cardinality_count": len(high_cardinality_columns)
            }
        }
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive dashboard summary combining all analytics.
        
        Provides a unified view including KPIs, risk indicators, top trends,
        recommended charts, and auto-generated highlights.
        
        Returns:
            Dictionary with:
                - kpis: Overview metrics from get_kpi_summary()
                - risks: Risk indicators from get_risk_indicators()
                - top_trends: Top 3 trends by absolute percent change
                - recommended_charts: Suggested visualizations by column
                - highlights: Auto-generated narrative insights
        """
        # TASK 1: Get KPIs
        kpis = self.get_kpi_summary()
        
        # TASK 2: Get Risk Indicators
        risks = self.get_risk_indicators()
        
        # TASK 3: Compute Top Trends
        top_trends = []
        numeric_columns = self.dataframe.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            try:
                # Only compute trend if column has at least 2 values
                if self.dataframe[col].dropna().shape[0] >= 2:
                    trend = self.get_trend_summary(col)
                    top_trends.append({
                        "column": trend["column"],
                        "direction": trend["trend_direction"],
                        "percent_change": trend["percent_change"]
                    })
            except Exception:
                # Skip columns that fail trend analysis
                continue
        
        # Sort by absolute percent change and take top 3
        top_trends = sorted(top_trends, key=lambda x: abs(x["percent_change"]), reverse=True)[:3]
        
        # TASK 4: Generate Recommended Charts
        recommended_charts = []
        
        for col in self.dataframe.columns:
            try:
                is_numeric = np.issubdtype(self.dataframe[col].dtype, np.number)
                is_datetime = pd.api.types.is_datetime64_any_dtype(self.dataframe[col])
                unique_count = self.dataframe[col].nunique()
                
                if is_numeric:
                    # Recommend histogram and boxplot for numeric columns
                    recommended_charts.append({
                        "column": col,
                        "chart_type": "histogram"
                    })
                    recommended_charts.append({
                        "column": col,
                        "chart_type": "boxplot"
                    })
                elif is_datetime:
                    # Recommend timeseries for datetime columns
                    recommended_charts.append({
                        "column": col,
                        "chart_type": "timeseries"
                    })
                else:
                    # Recommend bar chart for categorical columns
                    recommended_charts.append({
                        "column": col,
                        "chart_type": "bar"
                    })
                    
                    # Add pie chart if low cardinality
                    if unique_count <= 6:
                        recommended_charts.append({
                            "column": col,
                            "chart_type": "pie"
                        })
            except Exception:
                # Skip problematic columns
                continue
        
        # TASK 5: Generate Highlights
        highlights = []
        
        # Dataset size highlight
        highlights.append(
            f"Dataset contains {kpis['row_count']:,} rows and {kpis['column_count']} columns"
        )
        
        # Missing data highlight
        if kpis['missing_percent'] > 0:
            highlights.append(
                f"Missing data: {kpis['missing_percent']}% of cells are empty "
                f"({kpis['missing_cells']:,} missing values)"
            )
        
        # Top trend highlight
        if top_trends:
            top_trend = top_trends[0]
            direction_text = "increased" if top_trend['direction'] == "increasing" else \
                           "decreased" if top_trend['direction'] == "decreasing" else "remained stable"
            highlights.append(
                f"Top trend: {top_trend['column']} {direction_text} by "
                f"{abs(top_trend['percent_change']):.1f}%"
            )
        
        # Risk highlights
        risk_summary = risks['summary']
        total_risks = (
            risk_summary['high_missing_count'] +
            risk_summary['outlier_count'] +
            risk_summary['skewed_count'] +
            risk_summary['high_cardinality_count']
        )
        
        if total_risks > 0:
            risk_areas = []
            if risk_summary['high_missing_count'] > 0:
                risk_areas.append(f"{risk_summary['high_missing_count']} with high missing rates")
            if risk_summary['outlier_count'] > 0:
                risk_areas.append(f"{risk_summary['outlier_count']} with outliers")
            if risk_summary['skewed_count'] > 0:
                risk_areas.append(f"{risk_summary['skewed_count']} with skewed distributions")
            if risk_summary['high_cardinality_count'] > 0:
                risk_areas.append(f"{risk_summary['high_cardinality_count']} with high cardinality")
            
            highlights.append(
                f"Data quality issues detected: {', '.join(risk_areas)}"
            )
        else:
            highlights.append("No significant data quality issues detected")
        
        # Column type distribution
        highlights.append(
            f"Column types: {kpis['numeric_columns']} numeric, "
            f"{kpis['categorical_columns']} categorical"
        )
        
        # Recommended focus areas based on risks
        focus_areas = []
        if risk_summary['high_missing_count'] > 0:
            focus_areas.append("address missing data")
        if risk_summary['outlier_count'] > 0:
            focus_areas.append("investigate outliers")
        if risk_summary['skewed_count'] > 0:
            focus_areas.append("consider transformations for skewed distributions")
        
        if focus_areas:
            highlights.append(f"Recommended actions: {', '.join(focus_areas)}")
        
        return {
            "kpis": kpis,
            "risks": risks,
            "top_trends": top_trends,
            "recommended_charts": recommended_charts,
            "highlights": highlights
        }
    
    def get_augmented_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get enhanced dashboard summary with integrated AI recommendations.
        
        Combines standard dashboard analytics with intelligent recommendations
        for analysis methods, transformations, tests, and models.
        
        Returns:
            Dictionary with:
                - All standard dashboard summary fields (kpis, risks, trends, etc.)
                - recommendations: Intelligent analysis recommendations
                - narrative: Human-readable recommendation narrative
        """
        from services.recommendation_engine import RecommendationEngine
        
        # Get standard dashboard summary
        dashboard_summary = self.get_dashboard_summary()
        
        # Initialize recommendation engine
        try:
            rec_engine = RecommendationEngine(dataframe=self.dataframe)
            
            # Generate recommendations
            recommendations = rec_engine.build_summary()
            
            # Generate narrative
            narrative = rec_engine.generate_narrative(recommendations)
            
            # Augment dashboard summary
            dashboard_summary["recommendations"] = recommendations
            dashboard_summary["recommendations_narrative"] = narrative
            
        except Exception as e:
            # Fail gracefully - return dashboard without recommendations
            dashboard_summary["recommendations"] = {
                "error": f"Could not generate recommendations: {str(e)}"
            }
            dashboard_summary["recommendations_narrative"] = (
                "Recommendations are temporarily unavailable. "
                "Please try again or contact support."
            )
        
        return dashboard_summary
