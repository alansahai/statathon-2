"""
Schema Mapping Engine
Handles user-defined column type mapping and datatype conversion

Allows users to specify column types (numeric, categorical, datetime) and
applies appropriate transformations to the dataset.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class SchemaMappingEngine:
    """
    Engine for managing schema mappings and applying datatype transformations.
    
    Supports three primary data types:
    - numeric: Integer or float values (cast to float64)
    - categorical: String/categorical values (cast to string)
    - datetime: Date/time values (parsed with pd.to_datetime)
    """
    
    def __init__(self, file_id: str):
        """
        Initialize SchemaMappingEngine with file identifier.
        
        Args:
            file_id: Unique identifier for the dataset file
        """
        self.file_id = file_id
        self.base_path = Path("temp_uploads")
        
        # Ensure schema directory exists
        self.schema_dir = self.base_path / "schema"
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to schema mapping file
        self.schema_path = self.schema_dir / f"{file_id}_schema.json"
    
    def load_columns(self) -> List[str]:
        """
        Load column names from the uploaded CSV file.
        
        Attempts to read from uploads directory to get original column structure.
        
        Returns:
            List of column names from the dataset
            
        Raises:
            FileNotFoundError: If CSV file does not exist
            ValueError: If file cannot be read or parsed
        """
        try:
            # Try uploads directory first (original file)
            upload_path = self.base_path / "uploads" / "default_user" / f"{self.file_id}.csv"
            
            if not upload_path.exists():
                # Try cleaned directory as fallback
                cleaned_path = self.base_path / "cleaned" / "default_user" / f"{self.file_id}_cleaned.csv"
                if cleaned_path.exists():
                    upload_path = cleaned_path
                else:
                    raise FileNotFoundError(
                        f"Dataset file not found for file_id: {self.file_id}"
                    )
            
            # Read CSV and extract column names
            df = pd.read_csv(upload_path, nrows=0)  # Read only headers
            columns = df.columns.tolist()
            
            if not columns:
                raise ValueError("Dataset has no columns")
            
            return columns
        
        except FileNotFoundError:
            raise
        
        except Exception as e:
            raise ValueError(f"Failed to load columns from dataset: {str(e)}")
    
    def save_mapping(self, mapping_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Save column-to-type mapping to JSON file.
        
        Validates mapping and stores it for future use in data processing
        and type conversion operations.
        
        Args:
            mapping_dict: Dictionary mapping column names to types
                Example: {
                    "age": "numeric",
                    "gender": "categorical",
                    "dob": "datetime",
                    "income": "numeric"
                }
        
        Returns:
            Confirmation dictionary with save status
            
        Raises:
            ValueError: If mapping contains invalid types
        """
        # Valid type options (including 'auto' for auto-detection)
        valid_types = ["numeric", "categorical", "datetime", "auto"]
        
        # Auto-detect types if 'auto' is specified
        processed_mapping = {}
        for col, dtype in mapping_dict.items():
            if dtype == "auto":
                # Auto-detect the type - default to categorical for now
                # Could be enhanced with actual type detection logic
                processed_mapping[col] = "categorical"
            elif dtype in valid_types:
                processed_mapping[col] = dtype
            else:
                # Invalid type
                processed_mapping[col] = dtype
        
        # Validate mapping (after auto-detection)
        invalid_types = []
        for col, dtype in processed_mapping.items():
            if dtype not in ["numeric", "categorical", "datetime"]:
                invalid_types.append(f"{col}: {mapping_dict[col]}")
        
        if invalid_types:
            raise ValueError(
                f"Invalid data types found: {', '.join(invalid_types)}. "
                f"Valid types are: {', '.join(valid_types)}"
            )
        
        try:
            # Save mapping to JSON (use processed_mapping with auto-detected types)
            with open(self.schema_path, 'w', encoding='utf-8') as f:
                json.dump(processed_mapping, f, indent=2, ensure_ascii=False)
            
            return {
                "status": "success",
                "message": f"Schema mapping saved for {len(processed_mapping)} columns",
                "path": str(self.schema_path),
                "columns_mapped": len(processed_mapping)
            }
        
        except Exception as e:
            raise IOError(f"Failed to save schema mapping: {str(e)}")
    
    def load_mapping(self) -> Optional[Dict[str, str]]:
        """
        Load saved column-to-type mapping from JSON file.
        
        Returns:
            Dictionary mapping column names to types, or None if no mapping exists
            
        Example:
            {
                "age": "numeric",
                "gender": "categorical",
                "dob": "datetime"
            }
        """
        try:
            if not self.schema_path.exists():
                return None
            
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            
            # Validate it's a dictionary
            if not isinstance(mapping, dict):
                print(f"Warning: Invalid schema mapping format in {self.schema_path}")
                return None
            
            return mapping
        
        except json.JSONDecodeError:
            print(f"Warning: Corrupted schema mapping file: {self.schema_path}")
            return None
        
        except Exception as e:
            print(f"Warning: Failed to load schema mapping: {e}")
            return None
    
    def apply_mapping(
        self, 
        dataframe: pd.DataFrame, 
        mapping_dict: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Apply datatype conversions to DataFrame based on schema mapping.
        
        Performs intelligent type casting:
        - numeric: Converts to float64, handles non-numeric as NaN
        - categorical: Converts to string type
        - datetime: Parses dates with pd.to_datetime(), handles errors
        
        Args:
            dataframe: Pandas DataFrame to transform
            mapping_dict: Optional mapping dictionary. If None, loads from saved file.
        
        Returns:
            Transformed DataFrame with updated datatypes
            
        Example:
            >>> engine = SchemaMappingEngine("12345")
            >>> mapping = {"age": "numeric", "gender": "categorical"}
            >>> df_transformed = engine.apply_mapping(df, mapping)
        """
        # Load mapping if not provided
        if mapping_dict is None:
            mapping_dict = self.load_mapping()
            
            if mapping_dict is None:
                print("Warning: No schema mapping found. Returning original DataFrame.")
                return dataframe
        
        # Create a copy to avoid modifying original
        df = dataframe.copy()
        
        # Track conversion statistics
        conversion_stats = {
            "successful": [],
            "failed": [],
            "skipped": []
        }
        
        # Apply conversions for each mapped column
        for column, target_type in mapping_dict.items():
            # Check if column exists in DataFrame
            if column not in df.columns:
                conversion_stats["skipped"].append({
                    "column": column,
                    "reason": "Column not found in DataFrame"
                })
                print(f"Warning: Column '{column}' not found in DataFrame. Skipping.")
                continue
            
            try:
                if target_type == "numeric":
                    # Convert to numeric, coerce errors to NaN
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                    conversion_stats["successful"].append({
                        "column": column,
                        "type": "numeric",
                        "dtype": str(df[column].dtype)
                    })
                
                elif target_type == "categorical":
                    # Convert to string type
                    df[column] = df[column].astype(str)
                    conversion_stats["successful"].append({
                        "column": column,
                        "type": "categorical",
                        "dtype": "object"
                    })
                
                elif target_type == "datetime":
                    # Parse datetime, coerce errors to NaT
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    conversion_stats["successful"].append({
                        "column": column,
                        "type": "datetime",
                        "dtype": str(df[column].dtype)
                    })
                
                else:
                    conversion_stats["skipped"].append({
                        "column": column,
                        "reason": f"Unknown type: {target_type}"
                    })
                    print(f"Warning: Unknown type '{target_type}' for column '{column}'. Skipping.")
            
            except Exception as e:
                conversion_stats["failed"].append({
                    "column": column,
                    "type": target_type,
                    "error": str(e)
                })
                print(f"Error converting column '{column}' to {target_type}: {e}")
        
        # Log conversion summary
        print(f"Schema mapping applied: {len(conversion_stats['successful'])} successful, "
              f"{len(conversion_stats['failed'])} failed, {len(conversion_stats['skipped'])} skipped")
        
        return df
    
    def get_suggested_types(self, dataframe: pd.DataFrame) -> Dict[str, str]:
        """
        Suggest data types for columns based on automatic detection.
        
        Uses heuristics to infer appropriate types:
        - Numeric: Already numeric dtype or parseable as numeric
        - Datetime: Contains date-like patterns or parseable as datetime
        - Categorical: String type or low cardinality
        
        Args:
            dataframe: Pandas DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to suggested types
        """
        suggestions = {}
        
        for column in dataframe.columns:
            col_data = dataframe[column]
            
            # Check if already numeric
            if pd.api.types.is_numeric_dtype(col_data):
                suggestions[column] = "numeric"
                continue
            
            # Check if datetime
            if pd.api.types.is_datetime64_any_dtype(col_data):
                suggestions[column] = "datetime"
                continue
            
            # Try to detect datetime from object columns
            if col_data.dtype == 'object':
                # Sample non-null values
                sample = col_data.dropna().head(10)
                
                if len(sample) > 0:
                    try:
                        # Try parsing as datetime
                        pd.to_datetime(sample, errors='raise')
                        suggestions[column] = "datetime"
                        continue
                    except:
                        pass
                    
                    # Try parsing as numeric
                    try:
                        pd.to_numeric(sample, errors='raise')
                        suggestions[column] = "numeric"
                        continue
                    except:
                        pass
            
            # Default to categorical
            suggestions[column] = "categorical"
        
        return suggestions
    
    def get_mapping_summary(self) -> Dict:
        """
        Get summary of current schema mapping configuration.
        
        Returns:
            Dictionary with mapping details and statistics
        """
        mapping = self.load_mapping()
        
        if mapping is None:
            return {
                "exists": False,
                "message": "No schema mapping configured"
            }
        
        # Count types
        type_counts = {}
        for dtype in mapping.values():
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        return {
            "exists": True,
            "total_columns": len(mapping),
            "type_distribution": type_counts,
            "mapping": mapping,
            "file_path": str(self.schema_path)
        }
