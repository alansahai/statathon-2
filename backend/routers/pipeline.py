"""
Pipeline Router - Orchestrates the complete data workflow
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from utils.file_manager import FileManager
from utils.schema_validator import SchemaValidator
from services.cleaning_engine import CleaningEngine
from services.weighting_engine import WeightingEngine
from services.analysis_engine import AnalysisEngine
from services.ml_engine import MLEngine
from services.forecasting_engine import ForecastingEngine
from services.insight_engine import InsightEngine
from services.report_engine import ReportEngine
from database.audit_log import AuditLog

router = APIRouter(prefix="/pipeline", tags=["00 Pipeline"])

# Initialize shared services
file_manager = FileManager(base_storage_path="./temp_uploads")
schema_validator = SchemaValidator()
audit_logger = AuditLog()

# In-memory pipeline status tracking
pipeline_status: Dict[str, Dict[str, Any]] = {}


class PipelineRequest(BaseModel):
    file_ids: List[str]
    include_forecast: bool = False
    include_ml: bool = False
    report_mode: str = "separate"  # or "combined"
    
    @validator('file_ids')
    def validate_file_ids(cls, v):
        """Validate file_ids list"""
        if not v or len(v) == 0:
            raise ValueError("file_ids cannot be empty")
        if len(v) > 5:
            raise ValueError("Maximum 5 files allowed per pipeline run")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate file_ids are not allowed")
        return v
    
    @validator('report_mode')
    def validate_report_mode(cls, v):
        """Validate report mode"""
        if v not in ["separate", "combined"]:
            raise ValueError("report_mode must be 'separate' or 'combined'")
        return v


@router.post("/run-full")
async def run_full_pipeline(request: PipelineRequest):
    """
    Execute the complete data processing pipeline
    
    Steps:
    1. Schema Mapping
    2. Cleaning
    3. Weighting
    4. Analysis
    5. Forecasting (optional)
    6. ML (optional)
    7. Insights
    8. Report generation
    
    Returns comprehensive results from all stages
    """
    file_ids = request.file_ids
    pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize pipeline status
    for fid in file_ids:
        pipeline_status[fid] = {
            "pipeline_id": pipeline_id,
            "status": "running",
            "current_stage": "schema",
            "started_at": datetime.now().isoformat(),
            "completed_stages": []
        }
    
    results = {
        "pipeline_id": pipeline_id,
        "status": "running",
        "file_ids": file_ids,
        "steps": {}
    }
    
    try:
        # ===================================
        # STEP 1: Schema Validation
        # ===================================
        audit_logger.log_action("pipeline", "schema_validation_start", {"file_ids": file_ids})
        
        try:
            schema_results = schema_validator.load_schema_for_files(file_ids, file_manager)
            results["steps"]["schema"] = schema_results
            
            # Update status
            for fid in file_ids:
                if fid in pipeline_status:
                    pipeline_status[fid]["completed_stages"].append("schema")
                    pipeline_status[fid]["current_stage"] = "cleaning"
            
            audit_logger.log_action("pipeline", "schema_validation_complete", {
                "success_count": len([r for r in schema_results.values() if "schema" in r])
            })
        except Exception as e:
            results["steps"]["schema"] = {"error": str(e)}
            audit_logger.log_action("pipeline", "schema_validation_failed", {"error": str(e)})
        
        # ===================================
        # STEP 2: Data Cleaning
        # ===================================
        audit_logger.log_action("pipeline", "cleaning_start", {"file_ids": file_ids})
        
        try:
            cleaning_results = CleaningEngine.process_multiple(
                file_ids=file_ids,
                file_manager=file_manager,
                operation="auto_clean"
            )
            results["steps"]["cleaning"] = cleaning_results
            
            # Save cleaned files
            for fid, result in cleaning_results.items():
                if "result" in result and result["status"] == "ok":
                    # Get cleaned dataframe from engine
                    file_path = file_manager.get_file_path(fid)
                    if file_path:
                        df = file_manager.load_dataframe(file_path)
                        engine = CleaningEngine(df)
                        engine.auto_clean()
                        
                        # Save cleaned file
                        cleaned_dir = Path("temp_uploads/cleaned/default_user")
                        cleaned_dir.mkdir(parents=True, exist_ok=True)
                        cleaned_path = cleaned_dir / f"{fid}_cleaned.csv"
                        engine.df.to_csv(cleaned_path, index=False)
            
            # Update status
            for fid in file_ids:
                if fid in pipeline_status:
                    pipeline_status[fid]["completed_stages"].append("cleaning")
                    pipeline_status[fid]["current_stage"] = "weighting"
            
            audit_logger.log_action("pipeline", "cleaning_complete", {
                "success_count": len([r for r in cleaning_results.values() if "result" in r])
            })
        except Exception as e:
            results["steps"]["cleaning"] = {"error": str(e)}
            audit_logger.log_action("pipeline", "cleaning_failed", {"error": str(e)})
        
        # ===================================
        # STEP 3: Weighting
        # ===================================
        audit_logger.log_action("pipeline", "weighting_start", {"file_ids": file_ids})
        
        try:
            # Use cleaned files for weighting
            weighting_results = {}
            for fid in file_ids:
                try:
                    # Load cleaned file
                    cleaned_path = Path(f"temp_uploads/cleaned/default_user/{fid}_cleaned.csv")
                    if cleaned_path.exists():
                        df = file_manager.load_dataframe(str(cleaned_path))
                    else:
                        # Fallback to original file
                        file_path = file_manager.get_file_path(fid)
                        df = file_manager.load_dataframe(file_path)
                    
                    # Initialize weighting engine
                    engine = WeightingEngine(df)
                    
                    # Calculate base weights (uniform)
                    result = {
                        "method": "base",
                        "operations_log": engine.operations_log,
                        "auto_actions": engine.auto_actions,
                        "warnings": engine.warnings
                    }
                    
                    # Save weighted file
                    weighted_dir = Path("temp_uploads/weighted/default_user")
                    weighted_dir.mkdir(parents=True, exist_ok=True)
                    weighted_path = weighted_dir / f"{fid}_weighted.csv"
                    engine.df.to_csv(weighted_path, index=False)
                    
                    weighting_results[fid] = {
                        "result": result,
                        "weighted_file_path": str(weighted_path),
                        "status": "ok"
                    }
                    
                except Exception as e:
                    weighting_results[fid] = {"error": str(e)}
            
            results["steps"]["weighting"] = weighting_results
            
            # Update status
            for fid in file_ids:
                if fid in pipeline_status:
                    pipeline_status[fid]["completed_stages"].append("weighting")
                    pipeline_status[fid]["current_stage"] = "analysis"
            
            audit_logger.log_action("pipeline", "weighting_complete", {
                "success_count": len([r for r in weighting_results.values() if "result" in r])
            })
        except Exception as e:
            results["steps"]["weighting"] = {"error": str(e)}
            audit_logger.log_action("pipeline", "weighting_failed", {"error": str(e)})
        
        # ===================================
        # STEP 4: Analysis
        # ===================================
        audit_logger.log_action("pipeline", "analysis_start", {"file_ids": file_ids})
        
        try:
            # Run descriptive analysis on weighted files
            analysis_results = {}
            for fid in file_ids:
                try:
                    # Load weighted file
                    weighted_path = Path(f"temp_uploads/weighted/default_user/{fid}_weighted.csv")
                    if weighted_path.exists():
                        df = file_manager.load_dataframe(str(weighted_path))
                    else:
                        # Fallback to cleaned file
                        cleaned_path = Path(f"temp_uploads/cleaned/default_user/{fid}_cleaned.csv")
                        if cleaned_path.exists():
                            df = file_manager.load_dataframe(str(cleaned_path))
                        else:
                            file_path = file_manager.get_file_path(fid)
                            df = file_manager.load_dataframe(file_path)
                    
                    # Run descriptive analysis
                    engine = AnalysisEngine(df)
                    numeric_cols = list(df.select_dtypes(include=['number']).columns)
                    
                    if numeric_cols:
                        analysis_result = engine.descriptive_stats(
                            columns=numeric_cols[:10],  # Limit to first 10 numeric columns
                            weight_column="base_weight" if "base_weight" in df.columns else None
                        )
                        
                        analysis_results[fid] = {
                            "result": analysis_result,
                            "operations_log": engine.operations_log,
                            "status": "ok"
                        }
                    else:
                        analysis_results[fid] = {"error": "No numeric columns found"}
                        
                except Exception as e:
                    analysis_results[fid] = {"error": str(e)}
            
            results["steps"]["analysis"] = analysis_results
            
            # Update status
            for fid in file_ids:
                if fid in pipeline_status:
                    pipeline_status[fid]["completed_stages"].append("analysis")
                    pipeline_status[fid]["current_stage"] = "forecasting" if request.include_forecast else "ml" if request.include_ml else "insights"
            
            audit_logger.log_action("pipeline", "analysis_complete", {
                "success_count": len([r for r in analysis_results.values() if "result" in r])
            })
        except Exception as e:
            results["steps"]["analysis"] = {"error": str(e)}
            audit_logger.log_action("pipeline", "analysis_failed", {"error": str(e)})
        
        # ===================================
        # STEP 5: Forecasting (Optional)
        # ===================================
        if request.include_forecast:
            audit_logger.log_action("pipeline", "forecasting_start", {"file_ids": file_ids})
            
            try:
                forecasting_results = ForecastingEngine.forecast_multiple(
                    file_ids=file_ids,
                    file_manager=file_manager,
                    params={
                        "method": "auto",
                        "periods": 12
                    }
                )
                results["steps"]["forecasting"] = forecasting_results
                
                # Update status
                for fid in file_ids:
                    if fid in pipeline_status:
                        pipeline_status[fid]["completed_stages"].append("forecasting")
                        pipeline_status[fid]["current_stage"] = "ml" if request.include_ml else "insights"
                
                audit_logger.log_action("pipeline", "forecasting_complete", {
                    "success_count": len([r for r in forecasting_results.values() if "result" in r])
                })
            except Exception as e:
                results["steps"]["forecasting"] = {"error": str(e)}
                audit_logger.log_action("pipeline", "forecasting_failed", {"error": str(e)})
        else:
            results["steps"]["forecasting"] = None
        
        # ===================================
        # STEP 6: ML (Optional)
        # ===================================
        if request.include_ml:
            audit_logger.log_action("pipeline", "ml_start", {"file_ids": file_ids})
            
            try:
                # Run PCA on weighted files
                ml_results = {}
                for fid in file_ids:
                    try:
                        weighted_path = Path(f"temp_uploads/weighted/default_user/{fid}_weighted.csv")
                        if weighted_path.exists():
                            df = file_manager.load_dataframe(str(weighted_path))
                        else:
                            file_path = file_manager.get_file_path(fid)
                            df = file_manager.load_dataframe(file_path)
                        
                        engine = MLEngine(df)
                        numeric_cols = list(df.select_dtypes(include=['number']).columns)
                        
                        if len(numeric_cols) >= 2:
                            ml_result = engine.pca(
                                feature_columns=numeric_cols[:5],  # Limit to first 5
                                n_components=min(2, len(numeric_cols))
                            )
                            ml_results[fid] = {
                                "result": ml_result,
                                "operations_log": engine.operations_log,
                                "status": "ok"
                            }
                        else:
                            ml_results[fid] = {"error": "Insufficient numeric columns for ML"}
                            
                    except Exception as e:
                        ml_results[fid] = {"error": str(e)}
                
                results["steps"]["ml"] = ml_results
                
                # Update status
                for fid in file_ids:
                    if fid in pipeline_status:
                        pipeline_status[fid]["completed_stages"].append("ml")
                        pipeline_status[fid]["current_stage"] = "insights"
                
                audit_logger.log_action("pipeline", "ml_complete", {
                    "success_count": len([r for r in ml_results.values() if "result" in r])
                })
            except Exception as e:
                results["steps"]["ml"] = {"error": str(e)}
                audit_logger.log_action("pipeline", "ml_failed", {"error": str(e)})
        else:
            results["steps"]["ml"] = None
        
        # ===================================
        # STEP 7: Insights
        # ===================================
        audit_logger.log_action("pipeline", "insights_start", {"file_ids": file_ids})
        
        try:
            insights_results = InsightEngine.insights_multiple(
                file_ids=file_ids,
                file_manager=file_manager,
                params={}
            )
            results["steps"]["insights"] = insights_results
            
            # Update status
            for fid in file_ids:
                if fid in pipeline_status:
                    pipeline_status[fid]["completed_stages"].append("insights")
                    pipeline_status[fid]["current_stage"] = "report"
            
            audit_logger.log_action("pipeline", "insights_complete", {
                "success_count": len([r for r in insights_results.values() if "insights" in r])
            })
        except Exception as e:
            results["steps"]["insights"] = {"error": str(e)}
            audit_logger.log_action("pipeline", "insights_failed", {"error": str(e)})
        
        # ===================================
        # STEP 8: Report Generation
        # ===================================
        audit_logger.log_action("pipeline", "report_start", {
            "file_ids": file_ids,
            "mode": request.report_mode
        })
        
        try:
            report_results = ReportEngine.batch_generate_reports(
                file_ids=file_ids,
                file_manager=file_manager,
                report_config={
                    "metadata": {
                        "pipeline_id": pipeline_id,
                        "generated_at": datetime.now().isoformat()
                    }
                }
            )
            
            # Handle combined mode - create ZIP
            if request.report_mode == "combined" and len(report_results) > 0:
                import zipfile
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = f"{timestamp}_combined_reports.zip"
                reports_dir = Path("temp_uploads/reports/default_user")
                reports_dir.mkdir(parents=True, exist_ok=True)
                zip_path = reports_dir / zip_filename
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for fid, result in report_results.items():
                        if "report_path" in result:
                            report_path = Path(result["report_path"])
                            if report_path.exists():
                                zipf.write(report_path, f"{fid}_report.pdf")
                
                results["steps"]["report"] = {
                    "mode": "combined",
                    "zip_path": str(zip_path),
                    "individual_reports": report_results
                }
            else:
                results["steps"]["report"] = {
                    "mode": "separate",
                    "reports": report_results
                }
            
            # Update status
            for fid in file_ids:
                if fid in pipeline_status:
                    pipeline_status[fid]["completed_stages"].append("report")
                    pipeline_status[fid]["current_stage"] = "completed"
                    pipeline_status[fid]["status"] = "completed"
                    pipeline_status[fid]["completed_at"] = datetime.now().isoformat()
            
            audit_logger.log_action("pipeline", "report_complete", {
                "success_count": len([r for r in report_results.values() if "report_path" in r]),
                "mode": request.report_mode
            })
        except Exception as e:
            results["steps"]["report"] = {"error": str(e)}
            audit_logger.log_action("pipeline", "report_failed", {"error": str(e)})
        
        # ===================================
        # Final Status
        # ===================================
        results["status"] = "pipeline_completed"
        results["completed_at"] = datetime.now().isoformat()
        
        audit_logger.log_action("pipeline", "full_pipeline_complete", {
            "pipeline_id": pipeline_id,
            "file_count": len(file_ids)
        })
        
        return results
        
    except Exception as e:
        # Mark all files as failed
        for fid in file_ids:
            if fid in pipeline_status:
                pipeline_status[fid]["status"] = "failed"
                pipeline_status[fid]["error"] = str(e)
        
        audit_logger.log_action("pipeline", "full_pipeline_failed", {
            "pipeline_id": pipeline_id,
            "error": str(e)
        })
        
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@router.post("/run-minimal")
async def run_minimal_pipeline(request: PipelineRequest):
    """
    Execute minimal pipeline: schema → cleaning → analysis → insights
    
    Skips weighting, forecasting, ML, and report generation
    """
    file_ids = request.file_ids
    pipeline_id = f"pipeline_minimal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize pipeline status
    for fid in file_ids:
        pipeline_status[fid] = {
            "pipeline_id": pipeline_id,
            "status": "running",
            "current_stage": "schema",
            "started_at": datetime.now().isoformat(),
            "completed_stages": []
        }
    
    results = {
        "pipeline_id": pipeline_id,
        "status": "running",
        "file_ids": file_ids,
        "steps": {}
    }
    
    try:
        # STEP 1: Schema Validation
        audit_logger.log_action("pipeline", "minimal_schema_start", {"file_ids": file_ids})
        schema_results = schema_validator.load_schema_for_files(file_ids, file_manager)
        results["steps"]["schema"] = schema_results
        
        for fid in file_ids:
            if fid in pipeline_status:
                pipeline_status[fid]["completed_stages"].append("schema")
                pipeline_status[fid]["current_stage"] = "cleaning"
        
        # STEP 2: Cleaning
        audit_logger.log_action("pipeline", "minimal_cleaning_start", {"file_ids": file_ids})
        cleaning_results = CleaningEngine.process_multiple(
            file_ids=file_ids,
            file_manager=file_manager,
            operation="auto_clean"
        )
        results["steps"]["cleaning"] = cleaning_results
        
        for fid in file_ids:
            if fid in pipeline_status:
                pipeline_status[fid]["completed_stages"].append("cleaning")
                pipeline_status[fid]["current_stage"] = "analysis"
        
        # STEP 3: Analysis
        audit_logger.log_action("pipeline", "minimal_analysis_start", {"file_ids": file_ids})
        analysis_results = {}
        for fid in file_ids:
            try:
                file_path = file_manager.get_file_path(fid)
                df = file_manager.load_dataframe(file_path)
                engine = AnalysisEngine(df)
                numeric_cols = list(df.select_dtypes(include=['number']).columns)
                
                if numeric_cols:
                    analysis_result = engine.descriptive_stats(columns=numeric_cols[:10])
                    analysis_results[fid] = {"result": analysis_result, "status": "ok"}
                else:
                    analysis_results[fid] = {"error": "No numeric columns found"}
            except Exception as e:
                analysis_results[fid] = {"error": str(e)}
        
        results["steps"]["analysis"] = analysis_results
        
        for fid in file_ids:
            if fid in pipeline_status:
                pipeline_status[fid]["completed_stages"].append("analysis")
                pipeline_status[fid]["current_stage"] = "insights"
        
        # STEP 4: Insights
        audit_logger.log_action("pipeline", "minimal_insights_start", {"file_ids": file_ids})
        insights_results = InsightEngine.insights_multiple(
            file_ids=file_ids,
            file_manager=file_manager,
            params={}
        )
        results["steps"]["insights"] = insights_results
        
        for fid in file_ids:
            if fid in pipeline_status:
                pipeline_status[fid]["completed_stages"].append("insights")
                pipeline_status[fid]["current_stage"] = "completed"
                pipeline_status[fid]["status"] = "completed"
                pipeline_status[fid]["completed_at"] = datetime.now().isoformat()
        
        results["status"] = "pipeline_completed"
        results["completed_at"] = datetime.now().isoformat()
        
        audit_logger.log_action("pipeline", "minimal_pipeline_complete", {
            "pipeline_id": pipeline_id,
            "file_count": len(file_ids)
        })
        
        return results
        
    except Exception as e:
        for fid in file_ids:
            if fid in pipeline_status:
                pipeline_status[fid]["status"] = "failed"
                pipeline_status[fid]["error"] = str(e)
        
        audit_logger.log_action("pipeline", "minimal_pipeline_failed", {
            "pipeline_id": pipeline_id,
            "error": str(e)
        })
        
        raise HTTPException(status_code=500, detail=f"Minimal pipeline execution failed: {str(e)}")


@router.get("/status/{file_id}")
async def get_pipeline_status(file_id: str):
    """
    Get current pipeline status for a specific file_id
    
    Returns:
        - pipeline_id: Unique pipeline identifier
        - status: running, completed, failed
        - current_stage: Current processing stage
        - completed_stages: List of completed stages
        - started_at: Start timestamp
        - completed_at: Completion timestamp (if completed)
        - error: Error message (if failed)
    """
    if file_id not in pipeline_status:
        raise HTTPException(status_code=404, detail=f"No pipeline status found for file_id: {file_id}")
    
    return pipeline_status[file_id]


@router.get("/status")
async def get_all_pipeline_status():
    """
    Get pipeline status for all files
    
    Returns dictionary of file_id -> status
    """
    if not pipeline_status:
        return {
            "message": "No active pipelines",
            "pipelines": {}
        }
    
    return {
        "active_pipelines": len([s for s in pipeline_status.values() if s["status"] == "running"]),
        "completed_pipelines": len([s for s in pipeline_status.values() if s["status"] == "completed"]),
        "failed_pipelines": len([s for s in pipeline_status.values() if s["status"] == "failed"]),
        "pipelines": pipeline_status
    }
