"""
ChartEngine - Reusable chart generation service for StatFlow AI
Generates matplotlib-based visualizations with consistent styling and metadata output
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


class ChartEngine:
    """
    Generates statistical charts using matplotlib with consistent styling.
    Saves charts to temp_uploads/charts/<file_id>/ directory.
    """
    
    def __init__(self, dataframe: pd.DataFrame, file_id: str):
        """
        Initialize ChartEngine with a pandas DataFrame and file identifier.
        
        Args:
            dataframe: pandas DataFrame containing the data to visualize
            file_id: Unique identifier for the file/session (used in output filenames)
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if not isinstance(file_id, str) or not file_id.strip():
            raise ValueError("file_id must be a non-empty string")
        
        self.dataframe = dataframe
        self.file_id = file_id
        
        # Setup output directory: temp_uploads/charts/<file_id>/
        self.output_dir = Path("temp_uploads/charts") / file_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MoSPI-compliant color scheme
        self.primary_color = '#2c5aa0'
        self.secondary_color = '#1a5490'
    
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
    
    def generate_histogram(self, column_name: str) -> Dict[str, Any]:
        """
        Generate a histogram for a numeric column.
        
        Creates a frequency distribution histogram with automatic binning,
        using MoSPI-compliant styling. Saves as PNG with high resolution.
        
        Args:
            column_name: Name of the numeric column to plot
            
        Returns:
            Dictionary containing chart metadata:
                - type: "histogram"
                - column: column name
                - path: absolute path to saved PNG file
                - width: figure width in inches
                - height: figure height in inches
                - file_size: file size in bytes
                
        Raises:
            ValueError: If column doesn't exist or is empty
            TypeError: If column is not numeric
        """
        # Validate column
        self._validate_column(column_name, require_numeric=True)
        
        # Extract non-null values
        data = self.dataframe[column_name].dropna()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Generate histogram with automatic binning
        ax.hist(data, bins='auto', color=self.primary_color, edgecolor='white', alpha=0.8)
        
        # Set labels and title
        ax.set_title(f"Distribution of {column_name}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(column_name, fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        
        # Add grid for readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{self.file_id}_hist_{column_name}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return metadata
        return {
            'type': 'histogram',
            'column': column_name,
            'path': str(output_path.absolute()),
            'width': 8,
            'height': 5,
            'file_size': file_size
        }
    
    def generate_boxplot(self, column_name: str) -> Dict[str, Any]:
        """
        Generate a vertical boxplot for a numeric column.
        
        Creates a box-and-whisker plot showing quartiles, median, and outliers.
        Outliers are automatically highlighted with red circles (matplotlib default).
        
        Args:
            column_name: Name of the numeric column to plot
            
        Returns:
            Dictionary containing chart metadata:
                - type: "boxplot"
                - column: column name
                - path: absolute path to saved PNG file
                - width: figure width in inches
                - height: figure height in inches
                - file_size: file size in bytes
                
        Raises:
            ValueError: If column doesn't exist or is empty
            TypeError: If column is not numeric
        """
        # Validate column
        self._validate_column(column_name, require_numeric=True)
        
        # Extract non-null values
        data = self.dataframe[column_name].dropna()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 7))
        
        # Generate vertical boxplot
        bp = ax.boxplot(
            data,
            vert=True,
            patch_artist=True,  # Enable filling the box
            boxprops=dict(facecolor=self.primary_color, alpha=0.6, edgecolor=self.secondary_color, linewidth=1.5),
            whiskerprops=dict(color=self.secondary_color, linewidth=1.5),
            capprops=dict(color=self.secondary_color, linewidth=1.5),
            medianprops=dict(color='darkred', linewidth=2),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.5)  # Outliers
        )
        
        # Set labels and title
        ax.set_title(f"Boxplot of {column_name}", fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel(column_name, fontsize=11)
        
        # Remove x-axis ticks (single boxplot)
        ax.set_xticks([])
        
        # Add grid for readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{self.file_id}_box_{column_name}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return metadata
        return {
            'type': 'boxplot',
            'column': column_name,
            'path': str(output_path.absolute()),
            'width': 6,
            'height': 7,
            'file_size': file_size
        }
    
    def generate_bar_chart(self, column_name: str) -> Dict[str, Any]:
        """
        Generate a bar chart for a categorical column.
        
        Creates a vertical bar chart showing frequency counts for each category.
        Automatically rotates x-axis labels if there are more than 6 categories.
        
        Args:
            column_name: Name of the categorical column to plot
            
        Returns:
            Dictionary containing chart metadata:
                - type: "bar_chart"
                - column: column name
                - path: absolute path to saved PNG file
                - width: figure width in inches
                - height: figure height in inches
                - file_size: file size in bytes
                - num_categories: number of unique categories
                
        Raises:
            ValueError: If column doesn't exist or is empty
        """
        # Validate column exists
        self._validate_column(column_name, require_numeric=False)
        
        # Compute value counts
        value_counts = self.dataframe[column_name].value_counts().sort_index()
        
        if len(value_counts) == 0:
            raise ValueError(f"Column '{column_name}' has no valid values to plot")
        
        # Determine if labels need rotation
        rotate_labels = len(value_counts) > 6
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate bar chart
        bars = ax.bar(
            range(len(value_counts)),
            value_counts.values,
            color=self.primary_color,
            edgecolor=self.secondary_color,
            linewidth=1.2,
            alpha=0.8
        )
        
        # Set x-axis labels
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(
            value_counts.index,
            rotation=45 if rotate_labels else 0,
            ha='right' if rotate_labels else 'center',
            fontsize=9 if rotate_labels else 10
        )
        
        # Set labels and title
        ax.set_title(f"Bar Chart of {column_name}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(column_name, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # Add grid for readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{self.file_id}_bar_{column_name}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return metadata
        return {
            'type': 'bar_chart',
            'column': column_name,
            'path': str(output_path.absolute()),
            'width': 10,
            'height': 6,
            'file_size': file_size,
            'num_categories': len(value_counts)
        }
    
    def generate_pie_chart(self, column_name: str) -> Dict[str, Any]:
        """
        Generate a pie chart for a categorical column with <= 6 unique values.
        
        Creates a pie chart showing proportional distribution of categories
        with labels and percentage annotations.
        
        Args:
            column_name: Name of the categorical column to plot
            
        Returns:
            Dictionary containing chart metadata:
                - type: "pie_chart"
                - column: column name
                - path: absolute path to saved PNG file
                - width: figure width in inches
                - height: figure height in inches
                - file_size: file size in bytes
                - num_categories: number of unique categories
                
        Raises:
            ValueError: If column doesn't exist, is empty, or has > 6 unique values
        """
        # Validate column exists
        self._validate_column(column_name, require_numeric=False)
        
        # Compute value counts
        value_counts = self.dataframe[column_name].value_counts()
        
        if len(value_counts) == 0:
            raise ValueError(f"Column '{column_name}' has no valid values to plot")
        
        # Validate number of categories
        if len(value_counts) > 6:
            raise ValueError(
                f"Pie chart requires <= 6 unique values. "
                f"Column '{column_name}' has {len(value_counts)} unique values. "
                f"Consider using generate_bar_chart() instead."
            )
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Generate color palette (variations of MoSPI blue)
        colors = [
            self.primary_color,
            '#3d6fb0',
            '#4e84c0',
            self.secondary_color,
            '#0f3d70',
            '#5a94d0'
        ][:len(value_counts)]
        
        # Generate pie chart
        wedges, texts, autotexts = ax.pie(
            value_counts.values,
            labels=value_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10}
        )
        
        # Enhance percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        # Set title
        ax.set_title(f"Pie Chart of {column_name}", fontsize=14, fontweight='bold', pad=20)
        
        # Equal aspect ratio ensures circular pie
        ax.axis('equal')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{self.file_id}_pie_{column_name}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return metadata
        return {
            'type': 'pie_chart',
            'column': column_name,
            'path': str(output_path.absolute()),
            'width': 8,
            'height': 8,
            'file_size': file_size,
            'num_categories': len(value_counts)
        }
    
    def generate_frequency_plot(self, column_name: str) -> Dict[str, Any]:
        """
        Generate a frequency plot combining histogram with smooth line overlay.
        
        Creates a histogram with a smoothed frequency curve overlaid on top,
        using numpy interpolation (no seaborn/scipy KDE required).
        
        Args:
            column_name: Name of the numeric column to plot
            
        Returns:
            Dictionary containing chart metadata:
                - type: "frequency_plot"
                - column: column name
                - path: absolute path to saved PNG file
                - width: figure width in inches
                - height: figure height in inches
                - file_size: file size in bytes
                
        Raises:
            ValueError: If column doesn't exist or is empty
            TypeError: If column is not numeric
        """
        # Validate column
        self._validate_column(column_name, require_numeric=True)
        
        # Extract non-null values
        data = self.dataframe[column_name].dropna()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate histogram with automatic binning
        counts, bin_edges, patches = ax.hist(
            data,
            bins='auto',
            color=self.primary_color,
            edgecolor='white',
            alpha=0.6,
            label='Frequency'
        )
        
        # Calculate bin centers for smooth line
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create smooth curve using numpy interpolation
        # Generate more points for smoother curve
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 200)
        y_smooth = np.interp(x_smooth, bin_centers, counts)
        
        # Apply simple smoothing using moving average
        window_size = 5
        if len(y_smooth) >= window_size:
            y_smooth_avg = np.convolve(y_smooth, np.ones(window_size)/window_size, mode='same')
        else:
            y_smooth_avg = y_smooth
        
        # Plot smooth line
        ax.plot(
            x_smooth,
            y_smooth_avg,
            color='darkred',
            linewidth=2.5,
            label='Smoothed Frequency',
            alpha=0.9
        )
        
        # Set labels and title
        ax.set_title(
            f"Frequency Distribution of {column_name}",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel(column_name, fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Add grid for readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{self.file_id}_freq_{column_name}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return metadata
        return {
            'type': 'frequency_plot',
            'column': column_name,
            'path': str(output_path.absolute()),
            'width': 10,
            'height': 6,
            'file_size': file_size
        }
    
    def generate_correlation_heatmap(self) -> Dict[str, Any]:
        """
        Generate a correlation heatmap for all numeric columns in the dataframe.
        
        Computes the correlation matrix and visualizes it as a heatmap with
        color-coded cells and numeric correlation values displayed.
        
        Returns:
            Dictionary containing chart metadata:
                - type: "correlation_heatmap"
                - path: absolute path to saved PNG file
                - width: figure width in inches
                - height: figure height in inches
                - file_size: file size in bytes
                - num_variables: number of numeric columns
                
        Raises:
            ValueError: If dataframe has fewer than 2 numeric columns
        """
        # Select only numeric columns
        numeric_df = self.dataframe.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            raise ValueError(
                f"Correlation heatmap requires at least 2 numeric columns. "
                f"Found {numeric_df.shape[1]} numeric column(s)."
            )
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap using imshow
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', fontsize=11)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(corr_matrix.columns, fontsize=9)
        
        # Add correlation values as text in each cell
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                text = ax.text(
                    j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center',
                    color=text_color,
                    fontsize=8
                )
        
        # Set title
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{self.file_id}_corr_heatmap.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return metadata
        return {
            'type': 'correlation_heatmap',
            'path': str(output_path.absolute()),
            'width': 10,
            'height': 8,
            'file_size': file_size,
            'num_variables': len(corr_matrix.columns)
        }
    
    def generate_scatterplot(self, x_col: str, y_col: str) -> Dict[str, Any]:
        """
        Generate a scatter plot for two numeric columns.
        
        Creates a scatter plot showing the relationship between two variables.
        
        Args:
            x_col: Name of the column for x-axis (must be numeric)
            y_col: Name of the column for y-axis (must be numeric)
            
        Returns:
            Dictionary containing chart metadata:
                - type: "scatterplot"
                - x_column: x-axis column name
                - y_column: y-axis column name
                - path: absolute path to saved PNG file
                - width: figure width in inches
                - height: figure height in inches
                - file_size: file size in bytes
                
        Raises:
            ValueError: If columns don't exist or are empty
            TypeError: If columns are not numeric
        """
        # Validate both columns
        self._validate_column(x_col, require_numeric=True)
        self._validate_column(y_col, require_numeric=True)
        
        # Extract data (drop rows with NaN in either column)
        data = self.dataframe[[x_col, y_col]].dropna()
        
        if len(data) == 0:
            raise ValueError(f"No valid data points for scatter plot between '{x_col}' and '{y_col}'")
        
        x = data[x_col]
        y = data[y_col]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Generate scatter plot
        ax.scatter(
            x, y,
            color=self.primary_color,
            alpha=0.6,
            edgecolors=self.secondary_color,
            linewidth=0.5,
            s=50
        )
        
        # Set labels and title
        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        
        # Add grid for readability
        ax.grid(alpha=0.3, linestyle='--')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{self.file_id}_scatter_{x_col}_{y_col}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return metadata
        return {
            'type': 'scatterplot',
            'x_column': x_col,
            'y_column': y_col,
            'path': str(output_path.absolute()),
            'width': 9,
            'height': 7,
            'file_size': file_size
        }
    
    def generate_regression_fit_plot(self, x_col: str, y_col: str) -> Dict[str, Any]:
        """
        Generate a scatter plot with linear regression fit line.
        
        Creates a scatter plot with an overlaid linear regression line,
        showing the equation and fit.
        
        Args:
            x_col: Name of the column for x-axis (must be numeric)
            y_col: Name of the column for y-axis (must be numeric)
            
        Returns:
            Dictionary containing chart metadata:
                - type: "regression_fit"
                - x_column: x-axis column name
                - y_column: y-axis column name
                - path: absolute path to saved PNG file
                - width: figure width in inches
                - height: figure height in inches
                - file_size: file size in bytes
                - slope: regression slope
                - intercept: regression intercept
                - r_squared: R² value
                
        Raises:
            ValueError: If columns don't exist, are empty, or have insufficient data
            TypeError: If columns are not numeric
        """
        # Validate both columns
        self._validate_column(x_col, require_numeric=True)
        self._validate_column(y_col, require_numeric=True)
        
        # Extract data (drop rows with NaN in either column)
        data = self.dataframe[[x_col, y_col]].dropna()
        
        if len(data) < 2:
            raise ValueError(
                f"Regression requires at least 2 data points. "
                f"Found {len(data)} valid points for '{x_col}' and '{y_col}'"
            )
        
        x = data[x_col].values
        y = data[y_col].values
        
        # Compute linear regression using numpy polyfit
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        # Calculate R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Plot scatter points
        ax.scatter(
            x, y,
            color=self.primary_color,
            alpha=0.6,
            edgecolors=self.secondary_color,
            linewidth=0.5,
            s=50,
            label='Data Points'
        )
        
        # Plot regression line
        ax.plot(
            x, y_pred,
            color='darkred',
            linewidth=2.5,
            label='Regression Line'
        )
        
        # Add equation text
        equation_text = f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_squared:.3f}'
        ax.text(
            0.05, 0.95,
            equation_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        # Set labels and title
        ax.set_title(
            f"Regression Fit: {x_col} vs {y_col}",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        
        # Add legend
        ax.legend(loc='lower right', fontsize=10)
        
        # Add grid for readability
        ax.grid(alpha=0.3, linestyle='--')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{self.file_id}_regfit_{x_col}_{y_col}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return metadata
        return {
            'type': 'regression_fit',
            'x_column': x_col,
            'y_column': y_col,
            'path': str(output_path.absolute()),
            'width': 9,
            'height': 7,
            'file_size': file_size,
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared)
        }
    
    def generate_timeseries_plot(self, time_col: str, value_col: str) -> Dict[str, Any]:
        """
        Generate a time series line plot.
        
        Creates a line chart showing values over time, automatically converting
        and sorting by the time column.
        
        Args:
            time_col: Name of the column containing time/date values
            value_col: Name of the column containing numeric values to plot
            
        Returns:
            Dictionary containing chart metadata:
                - type: "timeseries"
                - time_column: time column name
                - value_column: value column name
                - path: absolute path to saved PNG file
                - width: figure width in inches
                - height: figure height in inches
                - file_size: file size in bytes
                
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
            raise ValueError(f"No valid data points for time series plot")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot time series
        ax.plot(
            data[time_col],
            data[value_col],
            color=self.primary_color,
            linewidth=2,
            marker='o',
            markersize=4,
            markerfacecolor=self.secondary_color,
            markeredgecolor='white',
            markeredgewidth=0.5
        )
        
        # Set labels and title
        ax.set_title(f"{value_col} Over Time", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(time_col, fontsize=11)
        ax.set_ylabel(value_col, fontsize=11)
        
        # Add grid for readability
        ax.grid(alpha=0.3, linestyle='--')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{self.file_id}_ts_{value_col}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return metadata
        return {
            'type': 'timeseries',
            'time_column': time_col,
            'value_column': value_col,
            'path': str(output_path.absolute()),
            'width': 12,
            'height': 6,
            'file_size': file_size
        }
    
    def generate_seasonal_decomposition_plots(
        self,
        time_col: str,
        value_col: str,
        decomposition: Dict[str, list]
    ) -> Dict[str, Any]:
        """
        Generate a 3-panel seasonal decomposition visualization.
        
        Creates a vertical 3-panel figure showing trend, seasonal, and residual
        components from time series decomposition.
        
        Args:
            time_col: Name of the column containing time/date values
            value_col: Name of the value column (for labeling purposes)
            decomposition: Dictionary containing 'trend', 'seasonal', and 'residual' lists
            
        Returns:
            Dictionary containing chart metadata:
                - type: "seasonal_decomposition"
                - value_column: value column name
                - path: absolute path to saved PNG file
                - width: figure width in inches
                - height: figure height in inches
                - file_size: file size in bytes
                
        Raises:
            ValueError: If decomposition is missing required components
        """
        # Validate decomposition structure
        required_keys = ['trend', 'seasonal', 'residual']
        for key in required_keys:
            if key not in decomposition:
                raise ValueError(f"Decomposition must contain '{key}' component")
        
        trend = decomposition['trend']
        seasonal = decomposition['seasonal']
        residual = decomposition['residual']
        
        # Validate all components have same length
        if not (len(trend) == len(seasonal) == len(residual)):
            raise ValueError("All decomposition components must have the same length")
        
        if len(trend) == 0:
            raise ValueError("Decomposition components are empty")
        
        # Create time index for x-axis
        time_index = np.arange(len(trend))
        
        # Create figure with 3 subplots (vertical)
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        # Panel 1: Trend
        axes[0].plot(time_index, trend, color=self.primary_color, linewidth=2)
        axes[0].set_title('Trend Component', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Trend', fontsize=10)
        axes[0].grid(alpha=0.3, linestyle='--')
        
        # Panel 2: Seasonal
        axes[1].plot(time_index, seasonal, color='darkgreen', linewidth=2)
        axes[1].set_title('Seasonal Component', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Seasonal', fontsize=10)
        axes[1].grid(alpha=0.3, linestyle='--')
        
        # Panel 3: Residual
        axes[2].plot(time_index, residual, color='darkred', linewidth=1.5, alpha=0.7)
        axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[2].set_title('Residual Component', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time Index', fontsize=10)
        axes[2].set_ylabel('Residual', fontsize=10)
        axes[2].grid(alpha=0.3, linestyle='--')
        
        # Overall title
        fig.suptitle(
            f'Seasonal Decomposition of {value_col}',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{self.file_id}_decomp_{value_col}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return metadata
        return {
            'type': 'seasonal_decomposition',
            'value_column': value_col,
            'path': str(output_path.absolute()),
            'width': 12,
            'height': 9,
            'file_size': file_size
        }
    
    def generate_residual_diagnostics(
        self,
        value_col: str,
        residuals: list
    ) -> Dict[str, Any]:
        """
        Generate residual diagnostics plots.
        
        Creates a 2-panel figure showing residuals over time and residual distribution.
        
        Args:
            value_col: Name of the value column (for labeling purposes)
            residuals: List of residual values from forecasting model
            
        Returns:
            Dictionary containing chart metadata:
                - type: "residual_diagnostics"
                - value_column: value column name
                - path: absolute path to saved PNG file
                - width: figure width in inches
                - height: figure height in inches
                - file_size: file size in bytes
                - mean_residual: mean of residuals
                - std_residual: standard deviation of residuals
                
        Raises:
            ValueError: If residuals list is empty
        """
        if not residuals or len(residuals) == 0:
            raise ValueError("Residuals list is empty")
        
        # Convert to numpy array
        residuals_array = np.array(residuals)
        
        # Calculate statistics
        mean_residual = np.mean(residuals_array)
        std_residual = np.std(residuals_array)
        
        # Create time index
        time_index = np.arange(len(residuals_array))
        
        # Create figure with 2 subplots (horizontal)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Residuals over time
        axes[0].plot(
            time_index,
            residuals_array,
            color=self.primary_color,
            linewidth=1.5,
            marker='o',
            markersize=3,
            alpha=0.7
        )
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[0].axhline(y=mean_residual, color='red', linestyle='--', linewidth=1, 
                       alpha=0.5, label=f'Mean: {mean_residual:.3f}')
        axes[0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time Index', fontsize=10)
        axes[0].set_ylabel('Residual', fontsize=10)
        axes[0].grid(alpha=0.3, linestyle='--')
        axes[0].legend(loc='best', fontsize=9)
        
        # Panel 2: Histogram of residuals
        axes[1].hist(
            residuals_array,
            bins='auto',
            color=self.primary_color,
            edgecolor='white',
            alpha=0.7,
            density=True
        )
        
        # Add mean line
        axes[1].axvline(
            x=mean_residual,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {mean_residual:.3f}'
        )
        
        # Add std lines
        axes[1].axvline(
            x=mean_residual + std_residual,
            color='orange',
            linestyle=':',
            linewidth=1.5,
            label=f'+1 Std: {std_residual:.3f}'
        )
        axes[1].axvline(
            x=mean_residual - std_residual,
            color='orange',
            linestyle=':',
            linewidth=1.5,
            label=f'-1 Std'
        )
        
        axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Residual Value', fontsize=10)
        axes[1].set_ylabel('Density', fontsize=10)
        axes[1].legend(loc='best', fontsize=9)
        axes[1].grid(alpha=0.3, linestyle='--', axis='y')
        
        # Overall title
        fig.suptitle(
            f'Residual Diagnostics for {value_col}',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_filename = f"{self.file_id}_residuals_{value_col}.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return metadata
        return {
            'type': 'residual_diagnostics',
            'value_column': value_col,
            'path': str(output_path.absolute()),
            'width': 14,
            'height': 5,
            'file_size': file_size,
            'mean_residual': float(mean_residual),
            'std_residual': float(std_residual)
        }
    
    def generate_chart(self, chart_type: str, **kwargs) -> Dict[str, Any]:
        """
        Unified chart dispatcher for dynamic chart generation.
        
        Routes to appropriate chart generation method based on chart_type.
        
        Args:
            chart_type: Type of chart to generate. Supported types:
                - "histogram": Frequency histogram (requires: column)
                - "boxplot": Box-and-whisker plot (requires: column)
                - "bar": Bar chart for categorical data (requires: column)
                - "pie": Pie chart for categorical data (requires: column)
                - "frequency": Histogram with smooth line (requires: column)
                - "heatmap": Correlation heatmap (no additional params)
                - "scatter": Scatter plot (requires: x_col, y_col)
                - "regression_fit": Scatter with regression line (requires: x_col, y_col)
                - "timeseries": Time series line plot (requires: time_col, value_col)
                - "decomposition": Seasonal decomposition (requires: time_col, value_col, decomposition)
                - "residuals": Residual diagnostics (requires: value_col, residuals)
            **kwargs: Additional parameters specific to each chart type
            
        Returns:
            Dictionary containing chart metadata with type and path
            
        Raises:
            ValueError: If chart_type is unsupported or required parameters are missing
        """
        # Normalize chart type
        chart_type = chart_type.lower().strip()
        
        # Route to appropriate method
        if chart_type == "histogram":
            if "column" not in kwargs:
                raise ValueError("Histogram requires 'column' parameter")
            return self.generate_histogram(kwargs["column"])
        
        elif chart_type == "boxplot":
            if "column" not in kwargs:
                raise ValueError("Boxplot requires 'column' parameter")
            return self.generate_boxplot(kwargs["column"])
        
        elif chart_type == "bar":
            if "column" not in kwargs:
                raise ValueError("Bar chart requires 'column' parameter")
            return self.generate_bar_chart(kwargs["column"])
        
        elif chart_type == "pie":
            if "column" not in kwargs:
                raise ValueError("Pie chart requires 'column' parameter")
            return self.generate_pie_chart(kwargs["column"])
        
        elif chart_type == "frequency":
            if "column" not in kwargs:
                raise ValueError("Frequency plot requires 'column' parameter")
            return self.generate_frequency_plot(kwargs["column"])
        
        elif chart_type == "heatmap" or chart_type == "correlation_heatmap":
            return self.generate_correlation_heatmap()
        
        elif chart_type == "scatter" or chart_type == "scatterplot":
            if "x_col" not in kwargs or "y_col" not in kwargs:
                raise ValueError("Scatter plot requires 'x_col' and 'y_col' parameters")
            return self.generate_scatterplot(kwargs["x_col"], kwargs["y_col"])
        
        elif chart_type == "regression_fit" or chart_type == "regression":
            if "x_col" not in kwargs or "y_col" not in kwargs:
                raise ValueError("Regression fit plot requires 'x_col' and 'y_col' parameters")
            return self.generate_regression_fit_plot(kwargs["x_col"], kwargs["y_col"])
        
        elif chart_type == "timeseries" or chart_type == "time_series":
            if "time_col" not in kwargs or "value_col" not in kwargs:
                raise ValueError("Time series plot requires 'time_col' and 'value_col' parameters")
            return self.generate_timeseries_plot(kwargs["time_col"], kwargs["value_col"])
        
        elif chart_type == "decomposition" or chart_type == "seasonal_decomposition":
            required = ["time_col", "value_col", "decomposition"]
            missing = [p for p in required if p not in kwargs]
            if missing:
                raise ValueError(f"Seasonal decomposition requires: {', '.join(missing)}")
            return self.generate_seasonal_decomposition_plots(
                kwargs["time_col"],
                kwargs["value_col"],
                kwargs["decomposition"]
            )
        
        elif chart_type == "residuals" or chart_type == "residual_diagnostics":
            if "value_col" not in kwargs or "residuals" not in kwargs:
                raise ValueError("Residual diagnostics requires 'value_col' and 'residuals' parameters")
            return self.generate_residual_diagnostics(kwargs["value_col"], kwargs["residuals"])
        
        else:
            supported_types = [
                "histogram", "boxplot", "bar", "pie", "frequency",
                "heatmap", "scatter", "regression_fit", "timeseries",
                "decomposition", "residuals"
            ]
            raise ValueError(
                f"Unsupported chart type: '{chart_type}'. "
                f"Supported types: {', '.join(supported_types)}"
            )
