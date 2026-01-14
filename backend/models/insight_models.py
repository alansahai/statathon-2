"""
StatFlow AI - Insight Models
Pydantic models for insight API endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List


class InsightOverviewRequest(BaseModel):
    """Request model for insight overview."""
    file_id: Optional[str] = Field(None, description="Single file identifier (legacy support)")
    file_ids: Optional[List[str]] = Field(None, description="List of file identifiers (1-5 files)")
    use_weighted: bool = Field(False, description="Use weighted data if available")
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id/file_ids to always use file_ids list."""
        file_id = values.get('file_id')
        if file_id and v:
            raise ValueError("Provide either file_id or file_ids, not both")
        if file_id and not v:
            return [file_id]
        if v and not file_id:
            if len(v) > 5:
                raise ValueError("Maximum 5 files allowed")
            if len(v) != len(set(v)):
                raise ValueError("Duplicate file_ids not allowed")
            return v
        raise ValueError("Either file_id or file_ids must be provided")


class FullInsightRequest(BaseModel):
    """Request model for full insights."""
    file_id: Optional[str] = Field(None, description="Single file identifier (legacy support)")
    file_ids: Optional[List[str]] = Field(None, description="List of file identifiers (1-5 files)")
    time_column: Optional[str] = Field(None, description="Time column for forecast insights")
    value_column: Optional[str] = Field(None, description="Value column for forecast insights")
    group_column: Optional[str] = Field(None, description="Group column for subgroup risk analysis")
    use_weighted: bool = Field(False, description="Use weighted data if available")
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id/file_ids to always use file_ids list."""
        file_id = values.get('file_id')
        if file_id and v:
            raise ValueError("Provide either file_id or file_ids, not both")
        if file_id and not v:
            return [file_id]
        if v and not file_id:
            if len(v) > 5:
                raise ValueError("Maximum 5 files allowed")
            if len(v) != len(set(v)):
                raise ValueError("Duplicate file_ids not allowed")
            return v
        raise ValueError("Either file_id or file_ids must be provided")
