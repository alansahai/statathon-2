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


router = APIRouter(prefix="/api/schema", tags=["Schema Mapping"])


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
            "income": "numeric"
        }
    }
    ```
    
    **Valid Types:**
    - numeric: Integer or float values
    - categorical: String/categorical values
    - datetime: Date/time values
    
    **Returns:**
    - Save confirmation with path and statistics
    
    **Error Codes:**
    - 400: Invalid mapping (unknown types, empty mapping)
    - 500: Failed to save mapping file
    """
    try:
        # Validate mapping not empty
        if not request.mapping:
            raise HTTPException(
                status_code=400,
                detail="Mapping cannot be empty"
            )
        
        # Initialize engine
        engine = SchemaMappingEngine(file_id)
        
        # Save mapping
        result = engine.save_mapping(request.mapping)
        
        return MappingSaveResponse(
            status=result["status"],
            message=result["message"],
            path=result["path"],
            columns_mapped=result["columns_mapped"]
        )
    
    except ValueError as e:
        # Invalid types in mapping
        raise HTTPException(
            status_code=400,
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
    datatypes, and stores the transformed DataFrame to the processed
    directory for downstream analysis.
    
    **Flow:**
    1. Load original dataset
    2. Load saved schema mapping
    3. Apply type conversions (numeric, categorical, datetime)
    4. Save to temp_uploads/processed/{file_id}.csv
    
    **Path Parameters:**
    - file_id: Unique identifier for the dataset
    
    **Returns:**
    - Processing confirmation with output path and statistics
    
    **Error Codes:**
    - 400: No mapping configured (must save mapping first)
    - 404: Dataset file not found
    - 500: Processing failed
    
    **Example Response:**
    ```json
    {
        "status": "success",
        "message": "Schema mapping applied successfully",
        "output_path": "temp_uploads/processed/12345.csv",
        "rows_processed": 1000,
        "columns_converted": 4
    }
    ```
    """
    try:
        # Initialize engine
        engine = SchemaMappingEngine(file_id)
        
        # Load mapping
        mapping = engine.load_mapping()
        
        if mapping is None:
            raise HTTPException(
                status_code=400,
                detail="No schema mapping configured. Please save mapping first using POST /api/schema/save/{file_id}"
            )
        
        # Load dataset
        base_path = Path("temp_uploads")
        
        # Try uploads first
        upload_path = base_path / "uploads" / "default_user" / f"{file_id}.csv"
        if not upload_path.exists():
            # Try cleaned
            cleaned_path = base_path / "cleaned" / "default_user" / f"{file_id}_cleaned.csv"
            if cleaned_path.exists():
                upload_path = cleaned_path
            else:
                raise FileNotFoundError(f"Dataset not found for file_id: {file_id}")
        
        # Read dataset
        df = pd.read_csv(upload_path)
        original_rows = len(df)
        
        # Apply mapping
        df_transformed = engine.apply_mapping(df, mapping)
        
        # Ensure processed directory exists
        processed_dir = base_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed dataset
        output_path = processed_dir / f"{file_id}.csv"
        df_transformed.to_csv(output_path, index=False)
        
        return ApplyMappingResponse(
            status="success",
            message="Schema mapping applied successfully",
            output_path=str(output_path),
            rows_processed=original_rows,
            columns_converted=len(mapping)
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
