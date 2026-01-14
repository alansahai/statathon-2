"""
Schema Mapping Engine - MoSPI-Compliant
Handles automatic schema detection and column type mapping with validation.

Provides intelligent type detection using multiple heuristics:
- dtype inference
- regex patterns
- cardinality analysis
- value distribution
- string length analysis
- date parsing attempts
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings

from services.file_manager import FileManager

warnings.filterwarnings('ignore')


class SchemaMappingEngine:
    """
    MoSPI-compliant schema mapping engine for statistical data processing.
    
    Supports automatic detection and validation of:
    - numeric: Integer or float values
    - categorical: Low-cardinality discrete values
    - identifier: Unique identifiers (IDs, codes, keys)
    - boolean: Binary yes/no, true/false, 0/1 values
    - datetime: Date and time values
    - text_short: Short text fields (average length < 20)
    - text_long: Long text fields (free-form responses)
    """
    
    # Valid schema types
    VALID_TYPES = [
        "numeric",
        "categorical", 
        "identifier",
        "boolean",
        "datetime",
        "text_short",
        "text_long",
        "text"  # Generic text type for frontend compatibility
    ]
    
    # Identifier pattern keywords
    IDENTIFIER_KEYWORDS = ['id', '_id', 'uuid', 'key', 'code', 'number', 'ref']
    
    # Boolean patterns
    BOOLEAN_VALUES = [
        {0, 1},
        {'0', '1'},
        {'yes', 'no'},
        {'y', 'n'},
        {'true', 'false'},
        {'t', 'f'}
    ]
    
    # Date patterns (regex)
    DATE_PATTERNS = [
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD or YYYY/MM/DD
        r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # DD-MM-YYYY or DD/MM/YYYY
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2}',  # DD-MM-YY
    ]
    
    def __init__(self, file_id: Optional[str] = None):
        """
        Initialize SchemaMappingEngine.
        
        Args:
            file_id: Optional file identifier for compatibility with legacy code
        """
        self.file_id = file_id
    
    @staticmethod
    def _check_identifier(col_data: pd.Series, col_name: str) -> bool:
        """
        Check if column is likely an identifier.
        
        Identifiers have:
        - High uniqueness (>95%)
        - Low/no missing values
        - Name contains identifier keywords
        
        Args:
            col_data: Column data
            col_name: Column name
            
        Returns:
            True if likely identifier
        """
        # Check name pattern
        col_lower = col_name.lower()
        has_id_keyword = any(kw in col_lower for kw in SchemaMappingEngine.IDENTIFIER_KEYWORDS)
        
        # Check uniqueness
        non_null = col_data.dropna()
        if len(non_null) == 0:
            return False
        
        uniqueness = non_null.nunique() / len(non_null)
        
        # Identifier criteria: name pattern + high uniqueness
        return has_id_keyword and uniqueness > 0.95
    
    @staticmethod
    def _check_boolean(col_data: pd.Series) -> bool:
        """
        Check if column contains boolean values.
        
        Args:
            col_data: Column data
            
        Returns:
            True if boolean type
        """
        non_null = col_data.dropna()
        if len(non_null) == 0:
            return False
        
        # Get unique values as set (convert to lowercase strings for comparison)
        if col_data.dtype in ['int64', 'int32', 'float64', 'float32']:
            unique_vals = set(non_null.unique())
        else:
            unique_vals = set(str(v).lower().strip() for v in non_null.unique())
        
        # Check against boolean patterns
        return unique_vals in SchemaMappingEngine.BOOLEAN_VALUES
    
    @staticmethod
    def _check_datetime(col_data: pd.Series) -> bool:
        """
        Check if column contains datetime values.
        
        Args:
            col_data: Column data
            
        Returns:
            True if datetime type
        """
        non_null = col_data.dropna()
        if len(non_null) == 0:
            return False
        
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(col_data):
            return True
        
        # Sample first 100 values
        sample = non_null.head(100).astype(str)
        
        # Check for date patterns using regex
        matches = 0
        for pattern in SchemaMappingEngine.DATE_PATTERNS:
            pattern_matches = sample.str.contains(pattern, regex=True, na=False).sum()
            matches += pattern_matches
        
        # If >70% match date patterns, likely datetime
        if matches / len(sample) > 0.7:
            # Try to actually parse a sample
            try:
                pd.to_datetime(sample.head(10))
                return True
            except:
                pass
        
        return False
    
    @staticmethod
    def _check_text_type(col_data: pd.Series) -> str:
        """
        Determine if text column is short or long form.
        
        Args:
            col_data: Column data
            
        Returns:
            "text_short" or "text_long"
        """
        non_null = col_data.dropna()
        if len(non_null) == 0:
            return "text_short"
        
        # Calculate average string length
        try:
            avg_length = non_null.astype(str).str.len().mean()
            return "text_long" if avg_length > 20 else "text_short"
        except:
            return "text_short"
    
    def auto_detect_schema(self, file_path: str) -> Dict[str, Any]:
        """
        Automatically detect column types with comprehensive validation.
        
        Detection rules applied in order:
        1. Identifier: high uniqueness + identifier keywords
        2. Boolean: binary values (0/1, yes/no, true/false)
        3. Datetime: date patterns and parseable dates
        4. Numeric: numeric dtypes (int, float)
        5. Categorical: low cardinality (< 25 unique values)
        6. Text (short/long): based on average string length
        
        Args:
            file_path: Absolute path to the CSV file
            
        Returns:
            Dictionary with structure:
            {
                "columns": {
                    "col_name": {
                        "detected_type": str,
                        "unique": int,
                        "missing_count": int,
                        "missing_pct": float
                    },
                    ...
                },
                "warnings": [str, ...]
            }
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be read
        """
        try:
            # Load dataset
            df = pd.read_csv(file_path, low_memory=False)
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            columns_info = {}
            warnings_list = []
            
            # Check for duplicate column names
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicate_cols:
                warnings_list.append(f"Duplicate column names detected: {', '.join(duplicate_cols)}")
            
            # Check for duplicate rows
            duplicate_rows = df.duplicated().sum()
            if duplicate_rows > 0:
                dup_pct = (duplicate_rows / len(df)) * 100
                warnings_list.append(
                    f"{duplicate_rows} duplicate rows found ({dup_pct:.1f}% of dataset)"
                )
            
            # Analyze each column
            for col in df.columns:
                col_data = df[col]
                non_null = col_data.dropna()
                
                # Basic stats
                unique_count = col_data.nunique()
                missing_count = col_data.isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                
                # Initialize column info
                col_info = {
                    "detected_type": "categorical",  # default
                    "unique": int(unique_count),
                    "missing_count": int(missing_count),
                    "missing_pct": float(missing_pct)
                }
                
                # Detect type (in priority order)
                
                # 1. Check for identifier
                if self._check_identifier(col_data, col):
                    col_info["detected_type"] = "identifier"
                
                # 2. Check for boolean
                elif self._check_boolean(col_data):
                    col_info["detected_type"] = "boolean"
                
                # 3. Check for datetime
                elif self._check_datetime(col_data):
                    col_info["detected_type"] = "datetime"
                
                # 4. Check for numeric
                elif pd.api.types.is_numeric_dtype(col_data):
                    col_info["detected_type"] = "numeric"
                    
                    # Check for outliers (IQR method)
                    if len(non_null) >= 10:
                        q1 = non_null.quantile(0.25)
                        q3 = non_null.quantile(0.75)
                        iqr = q3 - q1
                        outliers = ((non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)).sum()
                        
                        if outliers > 0:
                            outlier_pct = (outliers / len(non_null)) * 100
                            if outlier_pct > 5:
                                warnings_list.append(
                                    f"Column '{col}' has {outliers} outliers ({outlier_pct:.1f}%)"
                                )
                
                # 5. Check for categorical (low cardinality)
                elif unique_count < 25:
                    col_info["detected_type"] = "categorical"
                    
                    # Check for inconsistent formatting
                    if col_data.dtype == 'object' and len(non_null) > 0:
                        # Check case inconsistency
                        sample = non_null.head(100).astype(str)
                        lower_sample = sample.str.lower()
                        if sample.nunique() != lower_sample.nunique():
                            warnings_list.append(
                                f"Column '{col}' contains inconsistent case (e.g., 'Yes' vs 'yes')"
                            )
                
                # 6. Text type (short vs long)
                else:
                    col_info["detected_type"] = self._check_text_type(col_data)
                
                # Add warning for high missing rate
                if missing_pct > 20:
                    warnings_list.append(
                        f"Column '{col}' has high missing rate: {missing_pct:.1f}%"
                    )
                
                # Add warning for constant columns
                if unique_count == 1:
                    warnings_list.append(
                        f"Column '{col}' has only one unique value (constant column)"
                    )
                
                columns_info[col] = col_info
            
            return {
                "columns": columns_info,
                "warnings": warnings_list
            }
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty or unreadable")
        
        except Exception as e:
            raise ValueError(f"Failed to detect schema: {str(e)}")
    
    def apply_schema(self, file_path: str, schema: Dict[str, str]) -> str:
        """
        Apply schema mapping to dataset and save to mapped directory.
        
        Type conversions:
        - numeric: pd.to_numeric(errors="coerce")
        - categorical: astype("category")
        - identifier: astype(str), no modification
        - boolean: normalize to 0/1 integer
        - datetime: pd.to_datetime(errors="coerce")
        - text_short: astype(str)
        - text_long: astype(str)
        
        Args:
            file_path: Absolute path to source CSV file
            schema: Dictionary mapping column names to types
                Example: {"age": "numeric", "gender": "categorical"}
                
        Returns:
            Full path to saved mapped file
            
        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If unsupported type or column doesn't exist
        """
        try:
            # Load dataset
            df = pd.read_csv(file_path, low_memory=False)
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Extract filename from path
            filename = Path(file_path).stem
            if filename.endswith('_mapped'):
                filename = filename[:-7]  # Remove '_mapped' suffix
            if filename.endswith('_cleaned'):
                filename = filename[:-8]  # Remove '_cleaned' suffix
            if filename.endswith('_weighted'):
                filename = filename[:-9]  # Remove '_weighted' suffix
            
            # Validate schema
            for col, dtype in schema.items():
                # Check column exists
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in dataset")
                
                # Check type is valid
                if dtype not in self.VALID_TYPES:
                    raise ValueError(
                        f"Unsupported type '{dtype}' for column '{col}'. "
                        f"Valid types: {', '.join(self.VALID_TYPES)}"
                    )
            
            # Apply conversions
            conversion_errors = []
            
            for col, target_type in schema.items():
                try:
                    if target_type == "numeric":
                        # Convert to numeric, coerce errors to NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    elif target_type == "categorical":
                        # Convert to category type
                        df[col] = df[col].astype('category')
                    
                    elif target_type == "identifier":
                        # Keep as string, no modification
                        df[col] = df[col].astype(str)
                    
                    elif target_type == "boolean":
                        # Normalize boolean values to 0/1
                        original = df[col].copy()
                        
                        # Try numeric conversion first
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].astype(int)
                        else:
                            # Map string values
                            df[col] = df[col].astype(str).str.lower().str.strip()
                            bool_map = {
                                'yes': 1, 'no': 0,
                                'y': 1, 'n': 0,
                                'true': 1, 'false': 0,
                                't': 1, 'f': 0,
                                '1': 1, '0': 0,
                                '1.0': 1, '0.0': 0
                            }
                            df[col] = df[col].map(bool_map)
                        
                        # Check if conversion was successful
                        if df[col].isna().sum() > original.isna().sum():
                            conversion_errors.append(
                                f"Warning: Some values in '{col}' could not be converted to boolean"
                            )
                    
                    elif target_type == "datetime":
                        # Parse datetime, coerce errors to NaT
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    elif target_type == "text_short" or target_type == "text_long":
                        # Convert to string
                        df[col] = df[col].astype(str)
                    
                    else:
                        raise ValueError(f"Unsupported type: {target_type}")
                
                except Exception as e:
                    conversion_errors.append(
                        f"Failed to convert column '{col}' to {target_type}: {str(e)}"
                    )
            
            # Raise error if any conversions failed
            if conversion_errors:
                raise ValueError("Conversion errors:\n" + "\n".join(conversion_errors))
            
            # Save to mapped directory using FileManager
            output_path = FileManager.get_mapped_path(f"{filename}.csv")
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save CSV
            df.to_csv(output_path, index=False)
            
            return output_path
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty or unreadable")
        
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise ValueError(f"Failed to apply schema: {str(e)}")
    
    # ==========================================
    # LEGACY COMPATIBILITY METHODS
    # ==========================================
    
    def load_columns(self) -> List[str]:
        """
        Legacy method: Load column names from dataset.
        Maintained for backward compatibility.
        
        Returns:
            List of column names
        """
        if not self.file_id:
            raise ValueError("file_id not set. Initialize with SchemaMappingEngine(file_id)")
        
        try:
            # Try to get best available file
            filename = f"{self.file_id}.csv"
            file_path = FileManager.get_best_available_file(filename)
            
            # Read only headers
            df = pd.read_csv(file_path, nrows=0)
            return df.columns.tolist()
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found for file_id: {self.file_id}")
        
        except Exception as e:
            raise ValueError(f"Failed to load columns: {str(e)}")
    
    @staticmethod
    def save_manual_schema(file_id: str, mapping: Dict[str, str]) -> str:
        """
        Legacy method: Save manual schema mapping.
        Maintained for backward compatibility.
        
        Args:
            file_id: File identifier
            mapping: Column to type mapping
            
        Returns:
            Path to saved schema file
        """
        # Normalize and validate types
        normalized_mapping = {}
        for col, dtype in mapping.items():
            # Normalize type names
            normalized_type = dtype.lower().strip()
            
            # Map generic "text" to "text_short" for internal consistency
            if normalized_type == "text":
                normalized_type = "text_short"
            
            if normalized_type not in SchemaMappingEngine.VALID_TYPES:
                raise ValueError(
                    f"Invalid type '{dtype}' for column '{col}'. "
                    f"Valid types: {', '.join(SchemaMappingEngine.VALID_TYPES)}"
                )
            
            normalized_mapping[col] = normalized_type
        
        # Get schema directory
        schema_path = FileManager.BASE_DIR / "schema" / "default_user"
        schema_path.mkdir(parents=True, exist_ok=True)
        
        # Save schema file
        schema_file = schema_path / f"{file_id}_schema.json"
        
        try:
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump(normalized_mapping, f, indent=2, ensure_ascii=False)
            
            return str(schema_file)
        
        except Exception as e:
            raise IOError(f"Failed to save schema: {str(e)}")
    
    @staticmethod
    def apply_schema_mapping(file_id: str) -> Dict[str, Any]:
        """
        Legacy method: Apply saved schema mapping to dataset.
        Maintained for backward compatibility.
        
        Args:
            file_id: File identifier
            
        Returns:
            Dictionary with mapping results
        """
        # Load schema file
        schema_path = FileManager.BASE_DIR / "schema" / "default_user" / f"{file_id}_schema.json"
        
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Schema mapping not found for file_id: {file_id}. "
                "Please save mapping first."
            )
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load schema: {str(e)}")
        
        # Get source file
        try:
            filename = f"{file_id}.csv"
            file_path = FileManager.get_best_available_file(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found for file_id: {file_id}")
        
        # Apply schema using new method
        engine = SchemaMappingEngine(file_id)
        mapped_path = engine.apply_schema(file_path, schema)
        
        # Return legacy format
        df = pd.read_csv(mapped_path)
        
        return {
            "file_id": file_id,
            "mapped_path": mapped_path,
            "columns_converted": len(schema),
            "rows_processed": len(df)
        }
