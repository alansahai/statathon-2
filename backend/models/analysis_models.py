"""
Pydantic models for statistical analysis endpoints.
"""

from pydantic import BaseModel
from typing import Optional


class StatisticalTestRequest(BaseModel):
    """Request model for running a specific statistical test."""
    file_id: str
    test_type: str
    var1: str
    var2: Optional[str] = None
    group: Optional[str] = None
    weights: Optional[str] = None


class AutoTestRequest(BaseModel):
    """Request model for auto-selecting and running a statistical test."""
    file_id: str
    var1: str
    var2: Optional[str] = None
    group: Optional[str] = None


class WelchANOVARequest(BaseModel):
    """Request model for Welch's ANOVA test."""
    file_id: str
    group_col: str
    value_col: str
    weight_column: Optional[str] = None


class ShapiroWilkRequest(BaseModel):
    """Request model for Shapiro-Wilk normality test."""
    file_id: str
    value_col: str


class TukeyHSDRequest(BaseModel):
    """Request model for Tukey HSD post-hoc test."""
    file_id: str
    group_col: str
    value_col: str
    weight_column: Optional[str] = None
