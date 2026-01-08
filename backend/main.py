"""
StatFlow AI - Survey Analytics Application
Main FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import routers with explicit names
from routers.pipeline import router as pipeline_router
from routers.upload import router as upload_router
from routers.cleaning import router as cleaning_router
from routers.weighting import router as weighting_router
from routers.analysis import router as analysis_router
from routers.estimation import router as estimation_router
from routers.report import router as report_router
from routers.forecasting import router as forecasting_router
from routers.ml import router as ml_router
from routers.insight import router as insight_router
from routers.charts import router as charts_router
from routers.dashboard import router as dashboard_router
from routers.recommendation import router as recommendation_router
from routers.nlq import router as nlq_router
from routers.schema_mapping import router as schema_mapping_router

# Initialize FastAPI app
app = FastAPI(
    title="StatFlow AI",
    description="Survey Analytics and Data Processing API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (each exactly once)
app.include_router(pipeline_router)
app.include_router(upload_router, prefix="/api/upload")
app.include_router(schema_mapping_router)
app.include_router(cleaning_router)
app.include_router(weighting_router)
app.include_router(analysis_router)
app.include_router(forecasting_router)
app.include_router(ml_router)
app.include_router(insight_router)
app.include_router(nlq_router)
app.include_router(report_router, prefix="/api/report")
app.include_router(estimation_router, prefix="/api/estimation", tags=["Estimation"])
app.include_router(charts_router)
app.include_router(dashboard_router)
app.include_router(recommendation_router)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Welcome to StatFlow AI API",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
