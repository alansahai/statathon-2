"""
Pydantic models for statistical analysis endpoints.
Updated to support multi-file operations with backward compatibility.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union


class StatisticalTestRequest(BaseModel):
    """Request model for running a specific statistical test."""
    # Support both legacy single file_id and new multi-file file_ids
    file_id: Optional[str] = Field(None, description="Single file ID (legacy)")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs (1-5 files)")
    
    test_type: str
    var1: str
    var2: Optional[str] = None
    group: Optional[str] = None
    weights: Optional[str] = None
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id or file_ids into file_ids list"""
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


class AutoTestRequest(BaseModel):
    """Request model for auto-selecting and running a statistical test."""
    file_id: Optional[str] = Field(None, description="Single file ID (legacy)")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs (1-5 files)")
    
    var1: str
    var2: Optional[str] = None
    group: Optional[str] = None
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id or file_ids into file_ids list"""
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


class WelchANOVARequest(BaseModel):
    """Request model for Welch's ANOVA test."""
    file_id: Optional[str] = Field(None, description="Single file ID (legacy)")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs (1-5 files)")
    
    group_col: str
    value_col: str
    weight_column: Optional[str] = None
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id or file_ids into file_ids list"""
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


class ShapiroWilkRequest(BaseModel):
    """Request model for Shapiro-Wilk normality test."""
    file_id: Optional[str] = Field(None, description="Single file ID (legacy)")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs (1-5 files)")
    
    value_col: str
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id or file_ids into file_ids list"""
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


class TukeyHSDRequest(BaseModel):
    """Request model for Tukey HSD post-hoc test."""
    file_id: Optional[str] = Field(None, description="Single file ID (legacy)")
    file_ids: Optional[List[str]] = Field(None, description="List of file IDs (1-5 files)")
    
    group_col: str
    value_col: str
    weight_column: Optional[str] = None
    
    @validator('file_ids', always=True)
    def normalize_file_ids(cls, v, values):
        """Normalize file_id or file_ids into file_ids list"""
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

