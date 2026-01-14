"""
Common Pydantic models for multi-file support
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union


class FileIdentifierRequest(BaseModel):
    """
    Base model for requests that accept single or multiple file IDs.
    Provides backward compatibility with single file_id.
    """
    file_ids: List[str] = Field(
        ..., 
        description="List of file IDs to process (1-5 files)",
        min_items=1,
        max_items=5
    )
    
    @validator('file_ids')
    def validate_file_ids(cls, v):
        """Validate file IDs list"""
        if not v:
            raise ValueError("At least one file_id is required")
        
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate file_ids are not allowed")
        
        # Validate each file_id is non-empty
        for file_id in v:
            if not file_id or not file_id.strip():
                raise ValueError("Empty file_id is not allowed")
        
        return v
    
    @property
    def is_multi_file(self) -> bool:
        """Check if request is for multiple files"""
        return len(self.file_ids) > 1
    
    @property
    def single_file_id(self) -> str:
        """Get single file_id (for backward compatibility)"""
        return self.file_ids[0] if self.file_ids else None


class FileIdentifierRequestOptional(BaseModel):
    """
    Optional variant - allows providing either file_id or file_ids
    for maximum backward compatibility
    """
    file_id: Optional[str] = Field(None, description="Single file ID (deprecated, use file_ids)")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs (1-5 files)")
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id or file_ids into file_ids list"""
        file_id = values.get('file_id')
        
        # Case 1: Both provided
        if file_id and v:
            raise ValueError("Provide either file_id or file_ids, not both")
        
        # Case 2: Only file_id provided (legacy)
        if file_id and not v:
            return [file_id]
        
        # Case 3: Only file_ids provided
        if v and not file_id:
            if len(v) > 5:
                raise ValueError("Maximum 5 files allowed")
            if len(v) != len(set(v)):
                raise ValueError("Duplicate file_ids are not allowed")
            return v
        
        # Case 4: Neither provided
        raise ValueError("Either file_id or file_ids must be provided")
    
    @property
    def is_multi_file(self) -> bool:
        """Check if request is for multiple files"""
        return len(self.file_ids) > 1
    
    @property
    def single_file_id(self) -> str:
        """Get single file_id (for backward compatibility)"""
        return self.file_ids[0] if self.file_ids else None


class MultiFileResponse(BaseModel):
    """
    Standard response format for multi-file operations
    """
    status: str = Field(..., description="success, partial_success, or error")
    processed_count: int = Field(..., description="Number of files successfully processed")
    failed_count: int = Field(0, description="Number of files that failed")
    results_per_file: dict = Field(default_factory=dict, description="Results keyed by file_id")
    errors: Optional[List[dict]] = Field(None, description="List of errors if any")
    summary: Optional[dict] = Field(None, description="Aggregated summary across all files")


class MultiFileProcessingConfig(BaseModel):
    """
    Configuration for multi-file processing
    """
    fail_fast: bool = Field(
        False, 
        description="If True, stop processing on first error. If False, continue with remaining files"
    )
    aggregate_results: bool = Field(
        True, 
        description="If True, provide aggregated summary along with per-file results"
    )
    max_parallel: int = Field(
        1, 
        description="Maximum number of files to process in parallel (1 = sequential)",
        ge=1,
        le=5
    )
