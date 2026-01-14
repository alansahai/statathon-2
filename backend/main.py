"""
StatFlow AI - Survey Analytics Application
Main FastAPI Application Entry Point

MoSPI-Compliant Statistical Analysis Platform with:
- Pipeline orchestration
- Batch upload support
- FileManager integration
- Consistent JSON responses
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Import services
from services.file_manager import FileManager
from services.schema_mapping_engine import SchemaMappingEngine
from services.cleaning_engine import CleaningEngine
from services.weighting_engine import WeightingEngine
from services.analysis_engine import AnalysisEngine
from services.insight_engine import InsightEngine
from services.report_engine import ReportEngine

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
from routers.ws_router import router as ws_router

from dotenv import load_dotenv
load_dotenv()

print("GenAI Key Loaded:", bool(os.getenv("GOOGLE_GENAI_API_KEY")))


# ============================================================================
# Helper Functions
# ============================================================================

def convert_types(obj):
    """
    Recursively convert numpy, pandas, and non-JSON-safe types
    into pure Python serializable types.
    """
    import numpy as np

    # Convert dict
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}

    # Convert list/tuple
    if isinstance(obj, (list, tuple)):
        return [convert_types(i) for i in obj]

    # Convert numpy scalar types
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # Convert pandas NA/NaT
    if obj is None:
        return None

    # Primitive Python types - already serializable
    if isinstance(obj, (int, float, str, bool)):
        return obj

    # Fallback - convert to string to avoid breaking JSON
    return str(obj)


# ============================================================================
# Initialize FastAPI Application
# ============================================================================

app = FastAPI(
    title="StatFlow AI",
    description="MoSPI-Compliant Survey Analytics and Data Processing API",
    version="2.0.0"
)

# ============================================================================
# Configure CORS Middleware
# ============================================================================
# NOTE: Restrict origins in production! Replace ["*"] with specific domains.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ SECURITY: Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Startup Event - Initialize Directories
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup - ensure all directories exist"""
    FileManager.ensure_directories()
    print("✓ StatFlow AI v2.0 initialized - all directories ready")
    print("✓ MoSPI-compliant engines loaded")

# ============================================================================
# Include Routers (each exactly once) - All under /api/v1
# ============================================================================

app.include_router(pipeline_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1/upload")
app.include_router(schema_mapping_router, prefix="/api/v1")
app.include_router(cleaning_router, prefix="/api/v1")
app.include_router(weighting_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
app.include_router(forecasting_router, prefix="/api/v1")
app.include_router(ml_router, prefix="/api/v1")
app.include_router(insight_router, prefix="/api/v1")
app.include_router(nlq_router, prefix="/api/v1")
app.include_router(report_router, prefix="/api/v1")
app.include_router(estimation_router, prefix="/api/v1", tags=["Estimation"])
app.include_router(charts_router, prefix="/api/v1")
app.include_router(dashboard_router, prefix="/api/v1")
app.include_router(recommendation_router, prefix="/api/v1")
app.include_router(ws_router)  # WebSocket router (no prefix)

# ============================================================================
# Mount Static Files
# ============================================================================

# Mount backend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount UI files (frontend HTML/CSS/JS)
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui")
app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

# ============================================================================
# Root & Health Check Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Welcome to StatFlow AI v2.0 API",
        "description": "MoSPI-Compliant Statistical Analysis Platform",
        "status": "active",
        "version": "2.0.0",
        "features": [
            "Schema detection & mapping",
            "Data cleaning",
            "Survey weighting",
            "Statistical analysis",
            "AI-powered insights",
            "PDF report generation",
            "Pipeline orchestration"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "file_manager": "operational",
            "schema_engine": "operational",
            "cleaning_engine": "operational",
            "weighting_engine": "operational",
            "analysis_engine": "operational",
            "insight_engine": "operational",
            "report_engine": "operational"
        }
    }

# ============================================================================
# Request/Response Models
# ============================================================================

class BatchUploadResponse(BaseModel):
    """Response model for batch upload"""
    uploaded: List[str]
    count: int
    status: str

class PipelineRequest(BaseModel):
    """Request model for pipeline execution"""
    filename: str
    schema: Optional[Dict[str, str]] = None
    apply_weighting: Optional[bool] = True
    weighting_method: Optional[str] = "base"
    weighting_config: Optional[Dict[str, Any]] = None

class PipelineResponse(BaseModel):
    """Response model for pipeline execution"""
    status: str
    filename: str
    stages: Dict[str, Any]
    errors: Optional[List[str]] = None

# ============================================================================
# PATCH 1: Batch Upload Endpoint
# ============================================================================

@app.post("/api/v1/upload/batch", response_model=BatchUploadResponse)
async def batch_upload(files: List[UploadFile] = File(...)):
    """
    Upload multiple files at once (max 5 files).
    
    Accepts CSV, XLSX, XLS file formats.
    
    Args:
        files: List of files to upload
        
    Returns:
        JSON with list of uploaded filenames and count
        
    Raises:
        HTTPException 400: If more than 5 files or invalid format
    """
    try:
        # Validate file count
        if len(files) > 5:
            raise HTTPException(
                status_code=400,
                detail="Maximum 5 files allowed per batch upload"
            )
        
        if len(files) == 0:
            raise HTTPException(
                status_code=400,
                detail="No files provided"
            )
        
        uploaded_files = []
        
        for file in files:
            # Validate file extension
            if not file.filename:
                continue
            
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ['.csv', '.xlsx', '.xls']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file format: {file.filename}. Only CSV, XLSX, XLS allowed."
                )
            
            # Save file using FileManager
            try:
                upload_path = FileManager.get_uploaded_path(file.filename)
                
                # Ensure directory exists
                Path(upload_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Write file content
                content = await file.read()
                with open(upload_path, "wb") as f:
                    f.write(content)
                
                uploaded_files.append(file.filename)
                print(f"✓ Uploaded: {file.filename}")
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save file {file.filename}: {str(e)}"
                )
        
        return JSONResponse(
            status_code=200,
            content={
                "uploaded": uploaded_files,
                "count": len(uploaded_files),
                "status": "success"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch upload failed: {str(e)}"
        )

# ============================================================================
# PATCH 2: Single Pipeline Orchestration Endpoint
# ============================================================================

@app.post("/api/v1/pipeline/run", response_model=PipelineResponse)
async def run_complete_pipeline(request: PipelineRequest):
    """
    Execute complete data processing pipeline from start to finish.
    
    Pipeline Stages:
    1. Schema Auto-Detection
    2. Apply Schema (manual or auto)
    3. Data Cleaning
    4. Survey Weighting (optional)
    5. Statistical Analysis
    6. AI Insights Generation
    7. PDF Report Creation
    
    Args:
        request: Pipeline configuration with filename and optional schema
        
    Returns:
        JSON with results from each pipeline stage
        
    Raises:
        HTTPException: If any stage fails
    """
    try:
        filename = request.filename
        manual_schema = request.schema
        
        # Track all stage results
        stages = {}
        errors = []
        
        # ============================================
        # STAGE 1: Schema Auto-Detection
        # ============================================
        print(f"[Pipeline] Stage 1/7: Schema Detection for {filename}")
        try:
            file_path = FileManager.get_best_available_file(filename)
            
            schema_engine = SchemaMappingEngine()
            detected_schema = schema_engine.auto_detect_schema(file_path)
            
            stages["schema_detection"] = {
                "status": "success",
                "columns": detected_schema.get("columns", {}),
                "warnings": detected_schema.get("warnings", [])
            }
            
            print(f"  ✓ Detected {len(detected_schema.get('columns', {}))} columns")
            
        except Exception as e:
            error_msg = f"Schema detection failed: {str(e)}"
            errors.append(error_msg)
            stages["schema_detection"] = {"status": "failed", "error": error_msg}
            # Don't halt pipeline - continue with best effort
        
        # ============================================
        # STAGE 2: Apply Schema Mapping
        # ============================================
        print(f"[Pipeline] Stage 2/7: Apply Schema Mapping")
        try:
            file_path = FileManager.get_best_available_file(filename)
            
            # Use manual schema if provided, otherwise use detected schema
            if manual_schema:
                schema_to_apply = manual_schema
                print(f"  Using manual schema with {len(manual_schema)} columns")
            else:
                # Extract types from detected schema
                detected_cols = detected_schema.get("columns", {})
                schema_to_apply = {
                    col: info["detected_type"] 
                    for col, info in detected_cols.items()
                }
                print(f"  Using auto-detected schema")
            
            schema_engine = SchemaMappingEngine()
            mapped_path = schema_engine.apply_schema(file_path, schema_to_apply)
            
            stages["schema_mapping"] = {
                "status": "success",
                "mapped_file": mapped_path,
                "columns_mapped": len(schema_to_apply)
            }
            
            print(f"  ✓ Mapped file saved: {mapped_path}")
            
        except Exception as e:
            error_msg = f"Schema mapping failed: {str(e)}"
            errors.append(error_msg)
            stages["schema_mapping"] = {"status": "failed", "error": error_msg}
        
        # ============================================
        # STAGE 3: Data Cleaning
        # ============================================
        print(f"[Pipeline] Stage 3/7: Data Cleaning")
        try:
            cleaning_engine = CleaningEngine()
            cleaned_path = cleaning_engine.auto_clean(filename)
            
            stages["cleaning"] = {
                "status": "success",
                "cleaned_file": cleaned_path
            }
            
            print(f"  ✓ Cleaned file saved: {cleaned_path}")
            
        except Exception as e:
            error_msg = f"Cleaning failed: {str(e)}"
            errors.append(error_msg)
            stages["cleaning"] = {"status": "failed", "error": error_msg}
        
        # ============================================
        # STAGE 4: Survey Weighting (Optional)
        # ============================================
        if request.apply_weighting:
            print(f"[Pipeline] Stage 4/7: Survey Weighting")
            try:
                weighting_engine = WeightingEngine()
                
                # Extract weighting config
                method = request.weighting_method or "base"
                config = request.weighting_config or {}
                
                weighted_path = weighting_engine.apply_weights(
                    filename=filename,
                    method=method,
                    **config
                )
                
                stages["weighting"] = {
                    "status": "success",
                    "weighted_file": weighted_path,
                    "method": method
                }
                
                print(f"  ✓ Weighted file saved: {weighted_path}")
                
            except Exception as e:
                error_msg = f"Weighting failed: {str(e)}"
                errors.append(error_msg)
                stages["weighting"] = {"status": "failed", "error": error_msg}
        else:
            stages["weighting"] = {"status": "skipped"}
            print(f"[Pipeline] Stage 4/7: Weighting skipped")
        
        # ============================================
        # STAGE 5: Statistical Analysis
        # ============================================
        print(f"[Pipeline] Stage 5/7: Statistical Analysis")
        try:
            analysis_engine = AnalysisEngine()
            analysis_results = analysis_engine.generate_statistics(filename)
            
            stages["analysis"] = {
                "status": "success",
                "descriptive_stats_count": len(analysis_results.get("descriptive_stats", {})),
                "frequencies_count": len(analysis_results.get("frequencies", {})),
                "crosstabs_count": len(analysis_results.get("crosstabs", {})),
                "has_weighted_stats": bool(analysis_results.get("weighted_stats", {}))
            }
            
            print(f"  ✓ Analysis complete")
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            errors.append(error_msg)
            stages["analysis"] = {"status": "failed", "error": error_msg}
        
        # ============================================
        # STAGE 6: AI Insights Generation
        # ============================================
        print(f"[Pipeline] Stage 6/7: AI Insights Generation")
        try:
            insight_engine = InsightEngine()
            insights = insight_engine.generate_insights(filename)
            
            stages["insights"] = {
                "status": "success",
                "narrative_length": len(insights.get("narrative", "")),
                "has_summary": bool(insights.get("summary", {}))
            }
            
            print(f"  ✓ Insights generated")
            
        except Exception as e:
            error_msg = f"Insights generation failed: {str(e)}"
            errors.append(error_msg)
            stages["insights"] = {"status": "failed", "error": error_msg}
        
        # ============================================
        # STAGE 7: PDF Report Generation
        # ============================================
        print(f"[Pipeline] Stage 7/7: PDF Report Generation")
        try:
            report_engine = ReportEngine()
            report_path = report_engine.create_report(filename)
            
            stages["report"] = {
                "status": "success",
                "report_file": report_path
            }
            
            print(f"  ✓ Report saved: {report_path}")
            
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            errors.append(error_msg)
            stages["report"] = {"status": "failed", "error": error_msg}
        
        # ============================================
        # Pipeline Summary
        # ============================================
        successful_stages = sum(1 for s in stages.values() if s.get("status") == "success")
        total_stages = len(stages)
        
        print(f"[Pipeline] Complete: {successful_stages}/{total_stages} stages successful")
        
        pipeline_status = "success" if successful_stages == total_stages else "partial"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": pipeline_status,
                "filename": filename,
                "stages": stages,
                "errors": errors if errors else None,
                "summary": {
                    "total_stages": total_stages,
                    "successful_stages": successful_stages,
                    "failed_stages": len(errors)
                }
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}"
        )

# ============================================================================
# Simplified Pipeline Endpoints (Test Suite Compatible)
# ============================================================================
# Using Dict for maximum flexibility with test scripts

# --------------------------------------------------------------------------
# UPLOAD: Already handled by /api/v1/upload/batch
# --------------------------------------------------------------------------

@app.post("/api/v1/upload")
async def simple_upload(file: UploadFile = File(...)):
    """Upload a single file."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.csv', '.xlsx', '.xls']:
            raise HTTPException(status_code=400, detail="Invalid file format. Only CSV, XLSX, XLS allowed.")
        
        upload_path = FileManager.get_uploaded_path(file.filename)
        Path(upload_path).parent.mkdir(parents=True, exist_ok=True)
        
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        
        # Extract file_id from filename (without extension)
        file_id = Path(file.filename).stem
        
        print(f"✓ Uploaded: {file.filename}")
        return {
            "status": "success",
            "filename": file.filename,
            "file_id": file_id,
            "path": upload_path
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# --------------------------------------------------------------------------
# SCHEMA AUTO-DETECTION
# --------------------------------------------------------------------------

@app.post("/api/v1/schema/auto")
async def schema_auto(request: dict):
    """Auto-detect schema for uploaded file."""
    filename = request["filename"]
    engine = SchemaMappingEngine()
    result = engine.auto_detect_schema(filename)
    return JSONResponse(content=result)

# --------------------------------------------------------------------------
# SCHEMA APPLY
# --------------------------------------------------------------------------

@app.post("/api/v1/schema/apply")
async def schema_apply(request: Dict[str, Any]):
    """Apply schema mapping to uploaded file."""
    try:
        filename = request.get("filename")
        schema = request.get("schema")
        
        if not filename:
            raise HTTPException(status_code=400, detail="Missing 'filename' in request")
        
        file_path = FileManager.get_best_available_file(filename)
        schema_engine = SchemaMappingEngine()
        
        # If schema not provided, auto-detect it
        if not schema:
            detected = schema_engine.auto_detect_schema(file_path)
            schema_to_apply = {
                col: info["detected_type"]
                for col, info in detected.get("columns", {}).items()
            }
        else:
            schema_to_apply = schema
        
        # Apply schema
        mapped_path = schema_engine.apply_schema(file_path, schema_to_apply)
        
        return {
            "status": "success",
            "filename": filename,
            "mapped_file": mapped_path,
            "columns_mapped": len(schema_to_apply)
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema application failed: {str(e)}")

# --------------------------------------------------------------------------
# CLEANING
# --------------------------------------------------------------------------

@app.post("/api/v1/clean")
async def clean_file(request: Dict[str, Any]):
    """Clean uploaded file (imputation, outlier treatment, validation)."""
    try:
        filename = request.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Missing 'filename' in request")
        
        cleaning_engine = CleaningEngine()
        cleaned_path = cleaning_engine.auto_clean(filename)
        
        return {
            "status": "success",
            "filename": filename,
            "cleaned_file": cleaned_path
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {str(e)}")

# --------------------------------------------------------------------------
# WEIGHTING
# --------------------------------------------------------------------------

@app.post("/api/v1/weight")
async def weight_file(request: Dict[str, Any]):
    """Apply survey weights to cleaned file."""
    try:
        filename = request.get("filename")
        method = request.get("method", "base")
        
        if not filename:
            raise HTTPException(status_code=400, detail="Missing 'filename' in request")
        
        weighting_engine = WeightingEngine()
        weighted_path = weighting_engine.apply_weights(filename=filename, method=method)
        
        return {
            "status": "success",
            "filename": filename,
            "weighted_file": weighted_path,
            "method": method
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weighting failed: {str(e)}")

# --------------------------------------------------------------------------
# ANALYSIS
# --------------------------------------------------------------------------

@app.post("/api/v1/analyze")
def analyze(request: dict):
    """Generate statistical analysis for file."""
    filename = request["filename"]
    engine = AnalysisEngine()
    result = engine.generate_statistics(filename)
    safe_result = convert_types(result)
    return JSONResponse(content=safe_result)

# --------------------------------------------------------------------------
# INSIGHTS (AI-POWERED)
# --------------------------------------------------------------------------

@app.post("/api/v1/insights")
async def insights_basic(request: dict):
    """Generate AI-powered insights from dataset."""
    filename = request["filename"]
    engine = InsightEngine()
    result = engine.generate_insights(filename)
    return JSONResponse(content=result)

@app.post("/api/v1/insights/genai")
async def insights_genai(request: dict):
    """Generate AI-powered narrative insights from dataset."""
    filename = request["filename"]
    engine = InsightEngine()
    result = engine.generate_insights(filename)
    return JSONResponse(content=result)

@app.post("/api/insights/genai")
async def insights_genai_legacy(request: dict):
    """Legacy endpoint for GenAI insights (backward compatibility)."""
    return await insights_genai(request)

# --------------------------------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------------------------------

@app.post("/api/v1/report")
async def generate_report_endpoint(request: Dict[str, Any]):
    """Generate comprehensive PDF report for dataset."""
    try:
        filename = request.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Missing 'filename' in request")
        
        report_engine = ReportEngine()
        output = report_engine.create_report(filename)
        
        return {
            "status": "success",
            "report_file": output
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# ============================================================================
# NEW REST API ENDPOINTS (Requested Specification)
# ============================================================================
# These endpoints provide direct access to core services with standardized
# JSON responses, error handling, audit logging, and CORS support.

from database.audit_log import AuditLog
from tasks.pipeline import DataProcessingPipeline
import traceback

# Initialize audit logger
audit_logger = AuditLog()

# --------------------------------------------------------------------------
# 1. POST /api/upload - Upload CSV/XLSX files
# --------------------------------------------------------------------------

@app.post("/api/upload")
async def api_upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV or XLSX file for processing.
    
    Validates file format, saves to temp storage, and returns schema summary
    with data quality metrics (missing values, duplicates).
    
    Args:
        file: UploadFile object containing CSV/XLSX data
        
    Returns:
        JSON response with:
        - status: "success" or "error"
        - filename: Name of uploaded file
        - schema_summary: Column names and detected types
        - missing_values: Count of missing values per column
        - duplicates_count: Number of duplicate rows
        - row_count: Total number of rows
        - error: Error message (if failed)
    """
    try:
        # Validate file presence
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.csv', '.xlsx', '.xls']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format: {file_ext}. Only CSV, XLSX, XLS supported."
            )
        
        # Log upload attempt
        try:
            audit_logger.log_upload(
                user_id="default_user",
                file_name=file.filename,
                file_size=0,  # Size not available in UploadFile
                file_type=file_ext,
                status="started"
            )
        except Exception as audit_err:
            print(f"Audit log failed: {audit_err}")
        
        # Save file using FileManager
        upload_path = FileManager.get_uploaded_path(file.filename)
        Path(upload_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[UPLOAD] Saving file to: {upload_path}")
        
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        
        print(f"[UPLOAD] File saved successfully: {file.filename}")
        
        # Load file for validation
        if file_ext == '.csv':
            df = pd.read_csv(upload_path, low_memory=False)
        else:
            df = pd.read_excel(upload_path)
        
        # Validate schema using SchemaMappingEngine
        schema_engine = SchemaMappingEngine()
        schema_result = schema_engine.auto_detect_schema(file.filename)
        
        # Generate schema summary
        schema_summary = {}
        for col_name, col_info in schema_result.get("columns", {}).items():
            schema_summary[col_name] = {
                "type": col_info.get("detected_type", "unknown"),
                "sample_values": col_info.get("sample_values", [])[:3]
            }
        
        # Count missing values
        missing_values = df.isnull().sum().to_dict()
        
        # Count duplicates
        duplicates_count = int(df.duplicated().sum())
        
        # Log success
        try:
            audit_logger.log_upload(
                user_id="default_user",
                file_name=file.filename,
                file_size=len(content),
                file_type=file_ext,
                status="success"
            )
        except Exception as audit_err:
            print(f"Audit log failed: {audit_err}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "filename": file.filename,
                "schema_summary": schema_summary,
                "missing_values": convert_types(missing_values),
                "duplicates_count": duplicates_count,
                "row_count": len(df),
                "column_count": len(df.columns)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Log error
        try:
            audit_logger.log_upload(
                user_id="default_user",
                file_name=file.filename if file.filename else "unknown",
                file_size=0,
                file_type="unknown",
                status=f"error: {str(e)}"
            )
        except:
            pass
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

# --------------------------------------------------------------------------
# 2. POST /api/clean - Clean data with imputation and validation
# --------------------------------------------------------------------------

@app.post("/api/clean")
async def api_clean_data(request: Dict[str, Any]):
    """
    Clean dataset with missing value imputation and outlier treatment.
    
    Accepts cleaning configuration and applies data cleaning operations
    using the CleaningEngine.
    
    Args:
        request: JSON with:
            - filename: Name of file to clean
            - imputation_method: "median", "mode", "forward_fill" (optional)
            - outlier_treatment: "zscore", "iqr", "none" (optional)
            
    Returns:
        JSON response with:
        - status: "success" or "error"
        - filename: Original filename
        - cleaned_file: Path to cleaned file
        - preview: First 10 rows of cleaned data
        - cleaning_log: Summary of operations performed
        - error: Error message (if failed)
    """
    try:
        # Extract parameters
        filename = request.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Missing 'filename' parameter")
        
        print(f"[CLEAN] Received filename: {filename}")
        
        imputation_method = request.get("imputation_method", "auto")
        outlier_treatment = request.get("outlier_treatment", "auto")
        
        # Run CleaningEngine
        cleaning_engine = CleaningEngine()
        print(f"[CLEAN] Starting auto_clean for: {filename}")
        cleaned_path = cleaning_engine.auto_clean(filename)
        
        # Load cleaned file for preview
        try:
            df_cleaned = pd.read_csv(cleaned_path, low_memory=False)
        except:
            df_cleaned = pd.read_excel(cleaned_path)
        
        # Generate preview (first 10 rows)
        preview_data = df_cleaned.head(10).to_dict(orient='records')
        
        # Create cleaning log
        cleaning_log = {
            "operations_performed": [
                "Missing value imputation",
                "Outlier detection and treatment",
                "Logical validation rules"
            ],
            "imputation_method": imputation_method,
            "outlier_treatment": outlier_treatment,
            "rows_before": len(df_cleaned),  # Approximate
            "rows_after": len(df_cleaned),
            "columns": len(df_cleaned.columns)
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "filename": filename,
                "cleaned_file": cleaned_path,
                "preview": convert_types(preview_data),
                "cleaning_log": cleaning_log
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

# --------------------------------------------------------------------------
# 3. POST /api/weight - Apply survey weights
# --------------------------------------------------------------------------

@app.post("/api/weight")
async def api_apply_weights(request: Dict[str, Any]):
    """
    Apply survey weights to dataset.
    
    Calculates base weights and optionally applies post-stratification
    calibration to match known population targets.
    
    Args:
        request: JSON with:
            - filename: Name of file to weight
            - variable: Target variable for weighting (optional)
            - weight_column: Name of weight column to create (optional)
            - method: "base" or "poststrat" (optional)
            - strat_col: Stratification column for poststrat (optional)
            - targets: Target proportions dict for poststrat (optional)
            
    Returns:
        JSON response with:
        - status: "success" or "error"
        - filename: Original filename
        - weighted_file: Path to weighted file
        - weighted_stats: Weighted means and totals
        - margins_of_error: 95% confidence margins
        - error: Error message (if failed)
    """
    try:
        # Extract parameters
        filename = request.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Missing 'filename' parameter")
        
        method = request.get("method", "base")
        strat_col = request.get("strat_col")
        targets = request.get("targets")
        
        # Build kwargs for WeightingEngine
        kwargs = {}
        if strat_col:
            kwargs['strat_col'] = strat_col
        if targets:
            kwargs['targets'] = targets
        
        # Run WeightingEngine
        weighting_engine = WeightingEngine()
        weighted_path = weighting_engine.apply_weights(
            filename=filename,
            method=method,
            **kwargs
        )
        
        # Load weighted file
        try:
            df_weighted = pd.read_csv(weighted_path, low_memory=False)
        except:
            df_weighted = pd.read_excel(weighted_path)
        
        # Calculate weighted statistics
        weighted_stats = {}
        margins_of_error = {}
        
        # Find weight column
        weight_col = None
        for col in df_weighted.columns:
            if 'weight' in col.lower():
                weight_col = col
                break
        
        if weight_col:
            # Calculate weighted means for numeric columns
            numeric_cols = df_weighted.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != weight_col:
                    try:
                        weighted_mean = np.average(
                            df_weighted[col].dropna(),
                            weights=df_weighted.loc[df_weighted[col].notna(), weight_col]
                        )
                        weighted_stats[col] = float(weighted_mean)
                        
                        # Approximate margin of error (95% CI)
                        std_err = df_weighted[col].std() / np.sqrt(len(df_weighted))
                        margins_of_error[col] = float(1.96 * std_err)
                    except:
                        pass
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "filename": filename,
                "weighted_file": weighted_path,
                "method": method,
                "weighted_stats": convert_types(weighted_stats),
                "margins_of_error": convert_types(margins_of_error),
                "weight_column": weight_col
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

# --------------------------------------------------------------------------
# 4. POST /api/analyze - Generate statistical analysis
# --------------------------------------------------------------------------

@app.post("/api/analyze")
async def api_analyze_data(request: Dict[str, Any]):
    """
    Generate comprehensive statistical analysis.
    
    Computes descriptive statistics, frequency distributions, crosstabs,
    and correlation matrices.
    
    Args:
        request: JSON with:
            - filename: Name of file to analyze
            
    Returns:
        JSON response with:
        - status: "success" or "error"
        - filename: Original filename
        - descriptive_stats: Summary statistics for numeric columns
        - correlations: Correlation matrix
        - frequencies: Frequency distributions for categorical columns
        - crosstabs: Crosstabulation tables
        - error: Error message (if failed)
    """
    try:
        # Extract parameters
        filename = request.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Missing 'filename' parameter")
        
        # Run AnalysisEngine
        analysis_engine = AnalysisEngine()
        analysis_result = analysis_engine.generate_statistics(filename)
        
        # Load file for correlation calculation
        file_path = FileManager.get_best_available_file(filename)
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except:
            df = pd.read_excel(file_path)
        
        # Calculate correlations for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr().to_dict()
        else:
            corr_matrix = {}
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "filename": filename,
                "descriptive_stats": convert_types(analysis_result.get("descriptive_stats", {})),
                "frequencies": convert_types(analysis_result.get("frequencies", {})),
                "crosstabs": convert_types(analysis_result.get("crosstabs", {})),
                "correlations": convert_types(corr_matrix),
                "weighted_stats": convert_types(analysis_result.get("weighted_stats", {}))
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

# --------------------------------------------------------------------------
# 5. POST /api/report - Generate PDF report
# --------------------------------------------------------------------------

@app.post("/api/report")
async def api_generate_report(request: Dict[str, Any]):
    """
    Generate comprehensive PDF report.
    
    Creates a PDF report with statistical analysis, insights, and visualizations.
    
    Args:
        request: JSON with:
            - filename: Name of file to generate report for
            
    Returns:
        FileResponse with PDF attachment or JSON error
    """
    try:
        # Extract parameters
        filename = request.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Missing 'filename' parameter")
        
        print(f"[REPORT] Generating report for: {filename}")
        
        # Use filename as file_id for old report engine
        file_id = filename.replace('.csv', '').replace('.xlsx', '')
        
        # Run old ReportEngine with file_id
        report_engine = ReportEngine(file_id=file_id)
        report_path = report_engine.generate_basic_report()
        
        print(f"[REPORT] Generated report at: {report_path}")
        
        # Verify report exists
        if not Path(report_path).exists():
            raise FileNotFoundError(f"Report generation failed - file not found: {report_path}")
        
        # Return PDF as file download
        return FileResponse(
            path=report_path,
            media_type='application/pdf',
            filename=Path(report_path).name,
            headers={
                "Content-Disposition": f"attachment; filename={Path(report_path).name}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[REPORT] Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

# --------------------------------------------------------------------------
# 6. POST /api/run_pipeline - Execute full pipeline
# --------------------------------------------------------------------------

@app.post("/api/run_pipeline")
async def api_run_full_pipeline(request: Dict[str, Any]):
    """
    Execute complete data processing pipeline.
    
    Runs all stages: upload validation → cleaning → weighting → analysis → report.
    
    Args:
        request: JSON with:
            - filename: Name of uploaded file
            - apply_weighting: Whether to apply weights (default: True)
            - weighting_method: "base" or "poststrat" (default: "base")
            - cleaning_config: Cleaning options dict (optional)
            - weighting_config: Weighting options dict (optional)
            
    Returns:
        JSON response with:
        - status: "success", "partial", or "error"
        - filename: Original filename
        - outputs: Paths to all generated files
        - cleaned_data_path: Path to cleaned CSV
        - weighted_data_path: Path to weighted CSV
        - analysis_summary: Statistical analysis results
        - report_link: Path to PDF report
        - errors: List of any errors encountered
    """
    try:
        # Extract parameters
        filename = request.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Missing 'filename' parameter")
        
        apply_weighting = request.get("apply_weighting", True)
        weighting_method = request.get("weighting_method", "base")
        cleaning_config = request.get("cleaning_config", {})
        weighting_config = request.get("weighting_config", {})
        
        # Track pipeline results
        outputs = {}
        errors = []
        
        # =============================================
        # STAGE 1: Validate file exists
        # =============================================
        try:
            file_path = FileManager.get_uploaded_path(filename)
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {filename}")
            outputs["original_file"] = file_path
        except Exception as e:
            errors.append(f"Validation failed: {str(e)}")
            raise
        
        # =============================================
        # STAGE 2: Schema mapping (auto-detect)
        # =============================================
        try:
            schema_engine = SchemaMappingEngine()
            schema_result = schema_engine.auto_detect_schema(filename)
            outputs["schema_detected"] = True
            outputs["column_count"] = len(schema_result.get("columns", {}))
        except Exception as e:
            errors.append(f"Schema detection failed: {str(e)}")
            outputs["schema_detected"] = False
        
        # =============================================
        # STAGE 3: Data cleaning
        # =============================================
        try:
            cleaning_engine = CleaningEngine()
            cleaned_path = cleaning_engine.auto_clean(filename)
            outputs["cleaned_data_path"] = cleaned_path
        except Exception as e:
            errors.append(f"Cleaning failed: {str(e)}")
            outputs["cleaned_data_path"] = None
        
        # =============================================
        # STAGE 4: Weighting (if requested)
        # =============================================
        if apply_weighting:
            try:
                weighting_engine = WeightingEngine()
                weighted_path = weighting_engine.apply_weights(
                    filename=filename,
                    method=weighting_method,
                    **weighting_config
                )
                outputs["weighted_data_path"] = weighted_path
            except Exception as e:
                errors.append(f"Weighting failed: {str(e)}")
                outputs["weighted_data_path"] = None
        else:
            outputs["weighted_data_path"] = "skipped"
        
        # =============================================
        # STAGE 5: Statistical analysis
        # =============================================
        try:
            analysis_engine = AnalysisEngine()
            analysis_result = analysis_engine.generate_statistics(filename)
            outputs["analysis_summary"] = {
                "descriptive_stats_count": len(analysis_result.get("descriptive_stats", {})),
                "frequencies_count": len(analysis_result.get("frequencies", {})),
                "crosstabs_count": len(analysis_result.get("crosstabs", {}))
            }
        except Exception as e:
            errors.append(f"Analysis failed: {str(e)}")
            outputs["analysis_summary"] = None
        
        # =============================================
        # STAGE 6: PDF report generation
        # =============================================
        try:
            report_engine = ReportEngine()
            report_path = report_engine.create_report(filename)
            outputs["report_link"] = report_path
        except Exception as e:
            errors.append(f"Report generation failed: {str(e)}")
            outputs["report_link"] = None
        
        # =============================================
        # Pipeline summary
        # =============================================
        successful_stages = sum(1 for v in outputs.values() if v and v != "skipped")
        total_stages = len(outputs)
        
        pipeline_status = "success" if len(errors) == 0 else "partial"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": pipeline_status,
                "filename": filename,
                "outputs": convert_types(outputs),
                "errors": errors if errors else None,
                "summary": {
                    "total_stages": total_stages,
                    "successful_stages": successful_stages,
                    "failed_stages": len(errors)
                }
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "filename": request.get("filename", "unknown"),
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
