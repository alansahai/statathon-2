"""
Schema Validator - Validates data schemas and structures
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class SchemaValidator:
    """
    Validates data schemas and ensures data quality
    """
    
    def __init__(self):
        """Initialize schema validator"""
        self.categorical_threshold = 20  # Max unique values for categorical
        self.numeric_types = [np.int8, np.int16, np.int32, np.int64, 
                             np.float16, np.float32, np.float64]
    
    def validate_column_types(
        self,
        df: pd.DataFrame,
        expected_schema: Dict[str, str]
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate column data types against expected schema
        
        Args:
            df: DataFrame to validate
            expected_schema: Expected column types (column_name -> type_name)
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        validation_details = {
            "valid_columns": [],
            "invalid_columns": [],
            "missing_columns": [],
            "extra_columns": []
        }
        
        actual_columns = set(df.columns)
        expected_columns = set(expected_schema.keys())
        
        # Find missing and extra columns
        validation_details["missing_columns"] = list(expected_columns - actual_columns)
        validation_details["extra_columns"] = list(actual_columns - expected_columns)
        
        # Validate types for common columns
        for col in expected_columns & actual_columns:
            expected_type = expected_schema[col].lower()
            actual_dtype = str(df[col].dtype).lower()
            
            type_match = (
                (expected_type in ["int", "integer"] and "int" in actual_dtype) or
                (expected_type in ["float", "numeric"] and ("float" in actual_dtype or "int" in actual_dtype)) or
                (expected_type in ["str", "string", "text", "object"] and "object" in actual_dtype) or
                (expected_type in ["datetime", "date"] and "datetime" in actual_dtype) or
                (expected_type in ["bool", "boolean"] and "bool" in actual_dtype)
            )
            
            if type_match:
                validation_details["valid_columns"].append(col)
            else:
                validation_details["invalid_columns"].append(f"{col}: expected {expected_type}, got {actual_dtype}")
        
        is_valid = (
            len(validation_details["missing_columns"]) == 0 and
            len(validation_details["invalid_columns"]) == 0
        )
        
        return (is_valid, validation_details)
    
    def validate_required_columns(
        self,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that required columns are present
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        actual_columns = set(df.columns)
        required_set = set(required_columns)
        
        missing_columns = list(required_set - actual_columns)
        is_valid = len(missing_columns) == 0
        
        return (is_valid, missing_columns)
    
    def validate_value_ranges(
        self,
        df: pd.DataFrame,
        range_constraints: Dict[str, Tuple[Any, Any]]
    ) -> Dict[str, List[int]]:
        """
        Validate that values fall within expected ranges
        
        TODO: Check numeric ranges
        TODO: Check categorical values
        TODO: Identify out-of-range values
        
        Args:
            df: DataFrame to validate
            range_constraints: Column range constraints
            
        Returns:
            Dictionary of column names to row indices with violations
        """
        violations = {}
        # TODO: Implement range validation
        return violations
    
    def validate_data_completeness(
        self,
        df: pd.DataFrame,
        max_missing_pct: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate data completeness
        
        Calculates missing value percentages and identifies problematic columns
        
        Args:
            df: DataFrame to validate
            max_missing_pct: Maximum allowed missing percentage (0.0 to 1.0)
            
        Returns:
            Completeness validation results
        """
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Calculate missing percentages per column
        missing_per_column = df.isnull().sum()
        missing_pct_per_column = (missing_per_column / total_rows * 100) if total_rows > 0 else pd.Series()
        
        # Find columns exceeding threshold
        threshold_pct = max_missing_pct * 100
        exceeding_threshold = missing_pct_per_column[missing_pct_per_column > threshold_pct].to_dict()
        
        # Find completely empty rows
        empty_rows = df.isnull().all(axis=1)
        empty_row_indices = df[empty_rows].index.tolist()
        
        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_missing_values": int(df.isnull().sum().sum()),
            "overall_missing_percentage": round(
                (df.isnull().sum().sum() / (total_rows * total_cols) * 100) if total_rows * total_cols > 0 else 0,
                2
            ),
            "columns_exceeding_threshold": exceeding_threshold,
            "empty_rows": empty_row_indices,
            "empty_row_count": len(empty_row_indices)
        }
    
    def validate_unique_constraints(
        self,
        df: pd.DataFrame,
        unique_columns: List[str]
    ) -> Dict[str, List[Any]]:
        """
        Validate uniqueness constraints
        
        Args:
            df: DataFrame to validate
            unique_columns: Columns that should have unique values
            
        Returns:
            Dictionary of column names to duplicate values
        """
        duplicates = {}
        
        for col in unique_columns:
            if col not in df.columns:
                continue
            
            # Find duplicate values
            dup_mask = df[col].duplicated(keep=False)
            dup_values = df[df[col].notna() & dup_mask][col].unique()
            
            if len(dup_values) > 0:
                duplicates[col] = dup_values.tolist()[:20]  # Limit to first 20
        
        return duplicates
    
    def infer_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Infer schema from DataFrame
        
        Detects column types, missing percentages, and data characteristics
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Inferred schema information
        """
        schema = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {},
            "missing_summary": {
                "total_missing": int(df.isnull().sum().sum()),
                "columns_with_missing": []
            }
        }
        
        for col in df.columns:
            col_info = self._analyze_column(df[col], col)
            schema["columns"][col] = col_info
            
            if col_info["missing_count"] > 0:
                schema["missing_summary"]["columns_with_missing"].append(col)
        
        return schema
    
    def _analyze_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """
        Analyze a single column to determine its characteristics
        
        Args:
            series: Pandas Series to analyze
            col_name: Name of the column
            
        Returns:
            Dictionary with column information
        """
        total_count = len(series)
        missing_count = int(series.isnull().sum())
        non_missing_count = total_count - missing_count
        missing_percentage = (missing_count / total_count * 100) if total_count > 0 else 0
        
        col_info = {
            "name": col_name,
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_percentage": round(missing_percentage, 2),
            "non_missing_count": non_missing_count,
            "unique_count": int(series.nunique()),
        }
        
        # Detect semantic type
        if non_missing_count == 0:
            col_info["semantic_type"] = "empty"
            col_info["category"] = "empty"
        elif pd.api.types.is_numeric_dtype(series):
            col_info.update(self._analyze_numeric_column(series))
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info["semantic_type"] = "datetime"
            col_info["category"] = "temporal"
            col_info.update(self._analyze_datetime_column(series))
        else:
            col_info.update(self._analyze_text_column(series))
        
        return col_info
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze numeric column characteristics
        
        Args:
            series: Numeric Series
            
        Returns:
            Dictionary with numeric column stats
        """
        non_null = series.dropna()
        unique_count = series.nunique()
        
        info = {
            "semantic_type": "numeric",
            "category": "continuous" if unique_count > self.categorical_threshold else "discrete"
        }
        
        if len(non_null) > 0:
            info.update({
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": float(non_null.mean()),
                "median": float(non_null.median()),
                "std": float(non_null.std()) if len(non_null) > 1 else 0.0
            })
            
            # Check if it's actually categorical (few unique values)
            if unique_count <= self.categorical_threshold:
                info["value_counts"] = non_null.value_counts().head(10).to_dict()
                info["is_categorical_like"] = True
            else:
                info["is_categorical_like"] = False
        
        return info
    
    def _analyze_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze text/categorical column characteristics
        
        Args:
            series: Text Series
            
        Returns:
            Dictionary with text column stats
        """
        non_null = series.dropna()
        unique_count = series.nunique()
        
        info = {
            "semantic_type": "text",
            "category": "categorical" if unique_count <= self.categorical_threshold else "text"
        }
        
        if len(non_null) > 0:
            # Get top values
            value_counts = non_null.value_counts().head(10)
            info["top_values"] = value_counts.to_dict()
            info["most_common"] = str(non_null.mode().iloc[0]) if len(non_null.mode()) > 0 else None
            
            # Check if it's high cardinality
            if unique_count == len(non_null):
                info["is_unique_identifier"] = True
            else:
                info["is_unique_identifier"] = False
        
        return info
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze datetime column characteristics
        
        Args:
            series: Datetime Series
            
        Returns:
            Dictionary with datetime column stats
        """
        non_null = series.dropna()
        
        info = {}
        
        if len(non_null) > 0:
            info.update({
                "min_date": str(non_null.min()),
                "max_date": str(non_null.max()),
            })
        
        return info
    
    def generate_validation_report(
        self,
        df: pd.DataFrame,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        
        Args:
            df: DataFrame to validate
            schema: Expected schema (optional)
            
        Returns:
            Comprehensive validation report
        """
        report = {
            "status": "completed",
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns)
            }
        }
        
        # Add completeness check
        report["completeness"] = self.validate_data_completeness(df)
        
        # Add inferred schema if no expected schema provided
        if schema is None:
            report["inferred_schema"] = self.infer_schema(df)
        else:
            # Validate against provided schema
            if "columns" in schema:
                expected_types = {col: info.get("dtype", "object") 
                                for col, info in schema["columns"].items()}
                is_valid, details = self.validate_column_types(df, expected_types)
                report["schema_validation"] = {
                    "is_valid": is_valid,
                    "details": details
                }
        
        return report
    
    def load_schema_for_files(
        self, 
        file_ids: List[str],
        file_manager: Any
    ) -> Dict[str, Any]:
        """
        Load and infer schemas for multiple files
        
        Args:
            file_ids: List of file identifiers
            file_manager: FileManager instance to load files
            
        Returns:
            Dictionary mapping file_id to schema info:
            {
                "<file_id>": {"schema": {...}, "status": "ok"},
                "<file_id>": {"error": "File not found"}
            }
        """
        results = {}
        
        for file_id in file_ids:
            try:
                # Get file path from file manager
                file_path = file_manager.get_file_path(file_id)
                if not file_path:
                    results[file_id] = {"error": "File not found"}
                    continue
                
                # Load dataframe
                df = file_manager.load_dataframe(file_path)
                
                # Infer schema
                schema = self.infer_schema(df)
                
                results[file_id] = {
                    "schema": schema,
                    "status": "ok"
                }
                
            except Exception as e:
                results[file_id] = {"error": str(e)}
        
        return results
    
    def apply_schema_for_files(
        self,
        file_ids: List[str],
        file_manager: Any,
        expected_schema: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Validate schemas for multiple files against expected schema
        
        Args:
            file_ids: List of file identifiers
            file_manager: FileManager instance to load files
            expected_schema: Expected column types (optional)
            
        Returns:
            Dictionary mapping file_id to validation results:
            {
                "<file_id>": {"validation": {...}, "status": "ok"},
                "<file_id>": {"error": "No mapping found"}
            }
        """
        results = {}
        
        for file_id in file_ids:
            try:
                # Get file path from file manager
                file_path = file_manager.get_file_path(file_id)
                if not file_path:
                    results[file_id] = {"error": "File not found"}
                    continue
                
                # Load dataframe
                df = file_manager.load_dataframe(file_path)
                
                # Generate validation report
                validation_report = self.generate_validation_report(
                    df, 
                    {"columns": {col: {"dtype": dtype} for col, dtype in expected_schema.items()}} 
                    if expected_schema else None
                )
                
                results[file_id] = {
                    "validation": validation_report,
                    "status": "ok"
                }
                
            except Exception as e:
                results[file_id] = {"error": str(e)}
        
        return results
