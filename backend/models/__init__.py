"""
StatFlow AI - Models Package
Pydantic models for API request/response validation.
"""

from models.analysis_models import (
    StatisticalTestRequest,
    AutoTestRequest,
    WelchANOVARequest,
    ShapiroWilkRequest,
    TukeyHSDRequest
)

from models.forecasting_models import (
    ForecastRequest,
    DecomposeRequest,
    TimeSeriesAnalysisRequest
)

from models.ml_models import (
    ClassificationRequest,
    RegressionRequest,
    ClusteringRequest,
    PCARequest,
    FeatureImportanceRequest
)

from models.insight_models import (
    InsightOverviewRequest,
    FullInsightRequest
)

__all__ = [
    # Analysis models
    "StatisticalTestRequest",
    "AutoTestRequest",
    "WelchANOVARequest",
    "ShapiroWilkRequest",
    "TukeyHSDRequest",
    # Forecasting models
    "ForecastRequest",
    "DecomposeRequest",
    "TimeSeriesAnalysisRequest",
    # ML models
    "ClassificationRequest",
    "RegressionRequest",
    "ClusteringRequest",
    "PCARequest",
    "FeatureImportanceRequest",
    # Insight models
    "InsightOverviewRequest",
    "FullInsightRequest",
]