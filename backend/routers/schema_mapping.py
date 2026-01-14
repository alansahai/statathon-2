"""
Schema Mapping Router
API endpoints for managing column type mappings and datatype conversions
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

from services.schema_mapping_engine import SchemaMappingEngine


router = APIRouter(prefix="/schema", tags=["02 Schema Mapping"])


class MappingRequest(BaseModel):
    """Request model for saving schema mapping"""
    mapping: Dict[str, str]


class ColumnResponse(BaseModel):
    """Response model for column list"""
    status: str
    file_id: str
    columns: List[str]
    total_columns: int


class MappingSaveResponse(BaseModel):
    """Response model for save mapping operation"""
    status: str
    message: str
    path: str
    columns_mapped: int


class MappingGetResponse(BaseModel):
    """Response model for get mapping operation"""
    status: str
    exists: bool
    mapping: Optional[Dict[str, str]] = None
    total_columns: Optional[int] = None
    message: Optional[str] = None


class ApplyMappingResponse(BaseModel):
    """Response model for apply mapping operation"""
    status: str
    message: str
    output_path: str
    rows_processed: int
    columns_converted: int


class AutoMappingResponse(BaseModel):
    """Response model for auto-detection operation"""
    status: str
    data: Dict
    message: Optional[str] = None


@router.get("/columns/{file_id}", response_model=ColumnResponse)
async def get_columns(file_id: str):
    """
    Get list of column names from uploaded dataset.
    
    Returns column names to enable UI rendering of mapping dropdowns
    and type selection interfaces.
    
    **Path Parameters:**
    - file_id: Unique identifier for the dataset
    
    **Returns:**
    - Column name list with metadata
    
    **Error Codes:**
    - 404: Dataset file not found
    - 500: Error reading dataset
    
    **Example Response:**
    ```json
    {
        "status": "success",
        "file_id": "12345",
        "columns": ["age", "gender", "income", "dob"],
        "total_columns": 4
    }
    ```
    """
    try:
        # Initialize engine
        engine = SchemaMappingEngine(file_id)
        
        # Load columns
        columns = engine.load_columns()
        
        return ColumnResponse(
            status="success",
            file_id=file_id,
            columns=columns,
            total_columns=len(columns)
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {str(e)}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load columns: {str(e)}"
        )


@router.post("/save/{file_id}", response_model=MappingSaveResponse)
async def save_mapping(file_id: str, request: MappingRequest):
    """
    Save column-to-type mapping configuration.
    
    Stores user-defined schema mapping to JSON file for future use
    in data processing and type conversion operations.
    
    **Path Parameters:**
    - file_id: Unique identifier for the dataset
    
    **Request Body:**
    ```json
    {
        "mapping": {
            "age": "numeric",
            "gender": "categorical",
            "dob": "datetime",
            "income": "numeric",
            "user_id": "identifier",
            "bio": "text"
        }
    }
    ```
    
    **Valid Types:**
    - numeric: Integer or float values
    - categorical: String/categorical values
    - datetime: Date/time values
    - text: Long-form text content
    - identifier: Unique ID columns
    
    **Returns:**
    - Save confirmation with path and statistics
    
    **Error Codes:**
    - 400: Mapping is missing or empty
    - 404: Dataset file not found
    - 422: Invalid data types or columns not in dataset
    - 500: Failed to save mapping file
    """
    try:
        # Validate mapping not empty
        if not request.mapping:
            raise HTTPException(
                status_code=400,
                detail="Mapping cannot be empty"
            )
        
        # Initialize engine and save manual schema
        result = SchemaMappingEngine.save_manual_schema(file_id, request.mapping)
        
        return MappingSaveResponse(
            status="success",
            message="Manual schema saved",
            path=result,
            columns_mapped=len(request.mapping)
        )
    
    except FileNotFoundError as e:
        # Dataset file not found
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    
    except ValueError as e:
        # Invalid types or columns in mapping
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save mapping: {str(e)}"
        )


@router.get("/get/{file_id}", response_model=MappingGetResponse)
async def get_mapping(file_id: str):
    """
    Retrieve existing schema mapping configuration.
    
    Returns saved column-to-type mapping if it exists, otherwise
    indicates no mapping has been configured yet.
    
    **Path Parameters:**
    - file_id: Unique identifier for the dataset
    
    **Returns:**
    - Existing mapping or null if not configured
    
    **Example Response (mapping exists):**
    ```json
    {
        "status": "success",
        "exists": true,
        "mapping": {
            "age": "numeric",
            "gender": "categorical"
        },
        "total_columns": 2
    }
    ```
    
    **Example Response (no mapping):**
    ```json
    {
        "status": "success",
        "exists": false,
        "message": "No schema mapping configured"
    }
    ```
    """
    try:
        # Initialize engine
        engine = SchemaMappingEngine(file_id)
        
        # Load mapping
        mapping = engine.load_mapping()
        
        if mapping is None:
            return MappingGetResponse(
                status="success",
                exists=False,
                message="No schema mapping configured for this dataset"
            )
        
        return MappingGetResponse(
            status="success",
            exists=True,
            mapping=mapping,
            total_columns=len(mapping)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve mapping: {str(e)}"
        )


@router.post("/apply/{file_id}", response_model=ApplyMappingResponse)
async def apply_mapping(file_id: str):
    """
    Apply schema mapping to dataset and save processed version.
    
    Loads the dataset, applies saved schema mapping to convert column
    datatypes, and stores the transformed DataFrame to the mapped
    directory for downstream analysis.
    
    **Flow:**
    1. Check if mapping file exists
    2. Load original dataset
    3. Load saved schema mapping
    4. Apply type conversions (numeric, categorical, datetime, text, identifier)
    5. Save to temp_uploads/mapped/{file_id}_mapped.csv
    
    **Path Parameters:**
    - file_id: Unique identifier for the dataset
    
    **Type Conversions:**
    - numeric → pd.to_numeric(errors="coerce")
    - categorical → astype("category")
    - datetime → pd.to_datetime(errors="coerce")
    - text → astype(str)
    - identifier → keep as string (no changes)
    
    **Returns:**
    - Processing confirmation with output path and statistics
    
    **Error Codes:**
    - 404: Mapping file not found (must save mapping first)
    - 500: Processing/conversion failed
    
    **Example Response:**
    ```json
    {
        "status": "success",
        "file_id": "12345",
        "mapped_path": "temp_uploads/mapped/default_user/12345_mapped.csv",
        "columns_converted": 6
    }
    ```
    """
    try:
        # Apply schema mapping using static method
        result = SchemaMappingEngine.apply_schema_mapping(file_id)
        
        return ApplyMappingResponse(
            status="success",
            message=f"Schema mapping applied to {result['rows_processed']} rows",
            output_path=result['mapped_path'],
            rows_processed=result['rows_processed'],
            columns_converted=result['columns_converted']
        )
    
    except FileNotFoundError as e:
        # Mapping file or dataset not found
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    
    except Exception as e:
        # Pandas conversion or other errors
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply mapping: {str(e)}"
        )
    
    except HTTPException:
        raise
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply mapping: {str(e)}"
        )


@router.get("/suggestions/{file_id}")
async def get_type_suggestions(file_id: str):
    """
    Get AI-suggested data types for columns based on automatic detection.
    
    Analyzes the dataset and provides intelligent type suggestions
    to help users quickly configure schema mappings.
    
    **Path Parameters:**
    - file_id: Unique identifier for the dataset
    
    **Returns:**
    - Suggested type for each column with confidence indicators
    
    **Example Response:**
    ```json
    {
        "status": "success",
        "file_id": "12345",
        "suggestions": {
            "age": "numeric",
            "gender": "categorical",
            "dob": "datetime",
            "income": "numeric"
        }
    }
    ```
    """
    try:
        # Initialize engine
        engine = SchemaMappingEngine(file_id)
        
        # Load dataset
        base_path = Path("temp_uploads")
        upload_path = base_path / "uploads" / "default_user" / f"{file_id}.csv"
        
        if not upload_path.exists():
            cleaned_path = base_path / "cleaned" / "default_user" / f"{file_id}_cleaned.csv"
            if cleaned_path.exists():
                upload_path = cleaned_path
            else:
                raise FileNotFoundError(f"Dataset not found for file_id: {file_id}")
        
        # Read dataset (sample for efficiency)
        df = pd.read_csv(upload_path, nrows=1000)
        
        # Get suggestions
        suggestions = engine.get_suggested_types(df)
        
        return {
            "status": "success",
            "file_id": file_id,
            "suggestions": suggestions,
            "total_columns": len(suggestions)
        }
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate suggestions: {str(e)}"
        )


@router.get("/summary/{file_id}")
async def get_mapping_summary(file_id: str):
    """
    Get summary of current schema mapping configuration.
    
    Returns detailed information about the saved mapping including
    type distribution and file paths.
    
    **Path Parameters:**
    - file_id: Unique identifier for the dataset
    
    **Returns:**
    - Mapping summary with statistics and details
    """
    try:
        engine = SchemaMappingEngine(file_id)
        summary = engine.get_mapping_summary()
        
        return {
            "status": "success",
            "file_id": file_id,
            **summary
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve summary: {str(e)}"
        )


@router.post("/auto/{file_id}", response_model=AutoMappingResponse)
async def auto_detect_schema(file_id: str):
    """
    Automatically detect column types based on data patterns.
    
    Analyzes the dataset and intelligently suggests data types for each column
    without storing the mapping. User can review and modify before saving.
    
    **Detection Rules:**
    - int/float dtype → numeric
    - object with ≤20 unique values → categorical
    - object containing date patterns → datetime
    - object with avg text length >50 → text
    - unique id-like column names → identifier
    
    **Path Parameters:**
    - file_id: Unique identifier for the dataset
    
    **Returns:**
    - Auto-detected mapping for all columns
    
    **Valid Types:**
    - numeric: Integer or float values
    - categorical: Low-cardinality string/categorical values
    - datetime: Date/time values
    - text: Long-form text content
    - identifier: Unique ID columns
    
    **Error Codes:**
    - 404: Dataset file not found
    - 500: Failed to detect schema
    
    **Example Response:**
    ```json
    {
        "status": "success",
        "file_id": "12345",
        "auto_mapping": {
            "user_id": "identifier",
            "age": "numeric",
            "gender": "categorical",
            "registration_date": "datetime",
            "bio": "text"
        },
        "total_columns": 5
    }
    ```
    
    **Note:** This endpoint does NOT save the mapping automatically.
    Use POST /api/schema/save/{file_id} to persist the mapping.
    """
    try:
        # Initialize engine
        engine = SchemaMappingEngine(file_id)
        
        # Find the dataset file
        base_path = Path("temp_uploads")
        
        # Try uploads directory first
        upload_path = base_path / "uploads" / "default_user" / f"{file_id}.csv"
        
        if not upload_path.exists():
            # Try cleaned directory as fallback
            cleaned_path = base_path / "cleaned" / "default_user" / f"{file_id}_cleaned.csv"
            if cleaned_path.exists():
                upload_path = cleaned_path
            else:
                raise FileNotFoundError(f"Dataset not found for file_id: {file_id}")
        
        # Auto-detect schema
        schema_result = engine.auto_detect_schema(str(upload_path))
        
        # Extract columns info and warnings
        columns_info = schema_result.get("columns", {})
        warnings_list = schema_result.get("warnings", [])
        
        # Also get column list with types
        df = pd.read_csv(upload_path, nrows=5)  # Just read first few rows for column info
        columns = []
        auto_mapping = {}  # Simple mapping dict for backward compatibility
        
        for col in df.columns:
            col_info = columns_info.get(col, {})
            detected_type = col_info.get("detected_type", "categorical")
            
            # Normalize type names to match valid types
            # The engine may return "text_short", "text_long", "boolean" which we map to simpler types
            if detected_type in ["text_short", "text_long"]:
                detected_type = "text"
            elif detected_type == "boolean":
                detected_type = "categorical"  # Treat boolean as categorical for now
            
            columns.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "mapped_to": detected_type,
                "unique": col_info.get("unique", 0),
                "missing_count": col_info.get("missing_count", 0),
                "missing_pct": col_info.get("missing_pct", 0.0)
            })
            
            auto_mapping[col] = detected_type
        
        # Validate that all returned types are valid
        valid_types = ["numeric", "categorical", "datetime", "text", "identifier"]
        invalid_mappings = {
            col: dtype for col, dtype in auto_mapping.items()
            if dtype not in valid_types
        }
        
        if invalid_mappings:
            raise ValueError(
                f"Invalid types detected: {invalid_mappings}. "
                f"Valid types: {valid_types}"
            )
        
        # Calculate confidence score (simple heuristic: 100% if all columns mapped)
        confidence_score = len(auto_mapping) / len(columns) if columns else 0
        
        return AutoMappingResponse(
            status="success",
            data={
                "file_id": file_id,
                "mapping": auto_mapping,
                "auto_mapping": auto_mapping,  # Keep for backward compatibility
                "columns": columns,
                "total_columns": len(columns),
                "columns_mapped": len(auto_mapping),
                "confidence_score": confidence_score,
                "warnings": warnings_list  # Include warnings for frontend display
            },
            message=f"Successfully auto-mapped {len(auto_mapping)} columns"
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to auto-detect schema: {str(e)}"
        )
