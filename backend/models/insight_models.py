"""
StatFlow AI - Insight Models
Pydantic models for insight API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class InsightOverviewRequest(BaseModel):
    """Request model for insight overview."""
    file_id: str = Field(..., description="Unique file identifier")
    use_weighted: bool = Field(False, description="Use weighted data if available")


class FullInsightRequest(BaseModel):
    """Request model for full insights."""
    file_id: str = Field(..., description="Unique file identifier")
    time_column: Optional[str] = Field(None, description="Time column for forecast insights")
    value_column: Optional[str] = Field(None, description="Value column for forecast insights")
    group_column: Optional[str] = Field(None, description="Group column for subgroup risk analysis")
    use_weighted: bool = Field(False, description="Use weighted data if available")
