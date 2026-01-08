# Multi-File Engine Implementation Summary

## Overview
All core engines have been updated to support multi-file processing with isolated execution per file_id. Each engine now has a static wrapper function that processes multiple files independently without cross-file data merging or shared state.

## Implementation Date
January 8, 2026

---

## 1. Schema Validator (`utils/schema_validator.py`)

### Added Functions

#### `load_schema_for_files(file_ids, file_manager)`
- **Purpose**: Load and infer schemas for multiple files
- **Returns**: `{file_id: {"schema": {...}, "status": "ok"}}` or `{file_id: {"error": "..."}}`
- **Features**:
  - Loads each file independently
  - Infers schema using existing `infer_schema()` method
  - Returns isolated schema per file_id
  - Error handling per file

#### `apply_schema_for_files(file_ids, file_manager, expected_schema)`
- **Purpose**: Validate schemas for multiple files
- **Returns**: `{file_id: {"validation": {...}, "status": "ok"}}` or `{file_id: {"error": "..."}}`
- **Features**:
  - Validates each file against expected schema
  - Generates validation reports per file
  - No cross-file schema conflicts

---

## 2. Cleaning Engine (`services/cleaning_engine.py`)

### Added Function

#### `process_multiple(file_ids, file_manager, operation, **kwargs)` (static)
- **Purpose**: Process multiple files with cleaning operations
- **Operations Supported**:
  - `detect_issues` - Detect data quality issues
  - `auto_clean` - Automatic data cleaning
  - `manual_clean` - Rule-based cleaning
- **Returns**: `{file_id: {"result": {...}, "status": "ok"}}` or `{file_id: {"error": "..."}}`
- **Features**:
  - Loads each file independently
  - Creates new CleaningEngine instance per file
  - No shared state between files
  - Per-file error handling

---

## 3. Weighting Engine (`services/weighting_engine.py`)

### Added Function

#### `process_multiple_weights(file_ids, file_manager, operation, **kwargs)` (static)
- **Purpose**: Process multiple files with weighting operations
- **Operations Supported**:
  - `calculate` - Calculate base/poststrat/raking weights
  - `validate` - Validate weight quality
  - `trim` - Trim extreme weights
- **Returns**: `{file_id: {"result": {...}, "weighted_df": df, "status": "ok"}}` or `{file_id: {"error": "..."}}`
- **Features**:
  - Isolated WeightingEngine per file_id
  - Ensures file-specific storage paths: `temp_uploads/weighted/<user>/<file_id>_weighted.csv`
  - Returns weighted dataframe per file
  - No cross-file weight calculations

---

## 4. Analysis Engine (`services/analysis_engine.py`)

### Added Function

#### `analyze_multiple(file_ids, file_manager, analysis_type, params)` (static)
- **Purpose**: Perform analysis on multiple files
- **Analysis Types Supported**:
  - `descriptive` - Descriptive statistics
  - `crosstab` - Cross-tabulation analysis
  - `regression` - OLS regression
  - `subgroup` - Subgroup analysis
- **Returns**: `{file_id: {"result": {...}, "operations_log": [...], "status": "ok"}}` or `{file_id: {"error": "..."}}`
- **Features**:
  - Creates isolated AnalysisEngine per file
  - No combined dataframe operations
  - Returns operations_log per file
  - Independent statistical calculations

---

## 5. ML Engine (`services/ml_engine.py`)

### Added Function

#### `ml_multiple(file_ids, file_manager, ml_type, params)` (static)
- **Purpose**: Run ML operations on multiple files independently
- **ML Types Supported**:
  - `classify` - Logistic regression
  - `regress` - Linear regression
  - `cluster` - K-means clustering
  - `pca` - Principal Component Analysis
  - `feature_importance` - Random forest with feature importance
- **Returns**: `{file_id: {"result": {...}, "operations_log": [...], "status": "ok"}}` or `{file_id: {"error": "..."}}`
- **Features**:
  - Isolated preprocessing per file_id
  - NO shared encoders or scalers
  - Independent model training per file
  - Per-file feature engineering

---

## 6. Forecasting Engine (`services/forecasting_engine.py`)

### Added Function

#### `forecast_multiple(file_ids, file_manager, params)` (static)
- **Purpose**: Run forecasting operations on multiple files with error handling
- **Methods Supported**:
  - `auto` - Automatic method selection
  - `ma` - Moving average
  - `es` - Exponential smoothing
  - `holt_winters` - Holt-Winters seasonal
  - `arima` - ARIMA forecasting
- **Returns**: `{file_id: {"result": {...}, "operations_log": [...], "status": "ok"}}` or `{file_id: {"error": "..."}}`
- **Features**:
  - Detects if time column exists per file
  - Returns error if time column missing: `{"error": "No time column in dataset"}`
  - Validates required columns before processing
  - Independent time series analysis per file

---

## 7. Insight Engine (`services/insight_engine.py`)

### Added Function

#### `insights_multiple(file_ids, file_manager, params)` (static)
- **Purpose**: Generate insights for multiple files
- **Parameters**:
  - `time_column` (optional)
  - `value_column` (optional)
  - `group_column` (optional)
- **Returns**: `{file_id: {"insights": {...}, "operations_log": [...], "status": "ok"}}` or `{file_id: {"error": "..."}}`
- **Features**:
  - Generates full insights per file
  - Returns structured aggregated insights
  - Combines descriptive, forecast, and ML findings per file
  - Independent insight generation

---

## 8. Report Engine (`services/report_engine.py`)

### Added Function

#### `batch_generate_reports(file_ids, file_manager, report_config)` (static)
- **Purpose**: Generate reports for multiple files with unique filenames
- **Configuration Options**:
  - `metadata` - Report metadata
  - `cleaning_results` - Per-file cleaning results
  - `weighting_results` - Per-file weighting results
  - `analysis_results` - Per-file analysis results
- **Returns**: `{file_id: {"report_path": "...", "status": "ok"}}` or `{file_id: {"error": "..."}}`
- **Features**:
  - Creates unique ReportEngine per file_id
  - Ensures unique filenames: `temp_uploads/reports/<user>/<file_id>_report.pdf`
  - No overwriting between files
  - Per-file chart cache

---

## Architectural Principles

### 1. **Isolated Execution**
- Each file_id is processed independently
- No global state or shared variables
- New engine instance per file

### 2. **No Data Merging**
- DataFrames are never combined across files
- Results are aggregated at router level
- Engines only process single files

### 3. **Consistent Error Handling**
```python
results = {}
for file_id in file_ids:
    try:
        # Load and process file
        results[file_id] = {"result": {...}, "status": "ok"}
    except Exception as e:
        results[file_id] = {"error": str(e)}
return results
```

### 4. **Standard Return Format**
```python
{
    "<file_id>": {
        "result": {...},           # Operation-specific results
        "operations_log": [...],   # Optional: operation logs
        "status": "ok"
    },
    "<file_id>": {
        "error": "Error message"
    }
}
```

### 5. **File Storage Isolation**
- Cleaned files: `temp_uploads/cleaned/<user>/<file_id>_cleaned.csv`
- Weighted files: `temp_uploads/weighted/<user>/<file_id>_weighted.csv`
- Reports: `temp_uploads/reports/<user>/<file_id>_report.pdf`
- No file path conflicts

---

## Testing Scenarios

### CASE 1: Single file_id (Legacy Mode)
```python
results = CleaningEngine.process_multiple(
    file_ids=["file1"],
    file_manager=fm,
    operation="auto_clean"
)
# Expected: {"file1": {"result": {...}, "status": "ok"}}
```

### CASE 2: Multiple file_ids
```python
results = AnalysisEngine.analyze_multiple(
    file_ids=["file1", "file2", "file3"],
    file_manager=fm,
    analysis_type="descriptive",
    params={"columns": ["age", "income"]}
)
# Expected: {
#   "file1": {"result": {...}, "status": "ok"},
#   "file2": {"result": {...}, "status": "ok"},
#   "file3": {"result": {...}, "status": "ok"}
# }
```

### CASE 3: Partial Success
```python
results = ForecastingEngine.forecast_multiple(
    file_ids=["file1", "file2", "file3"],
    file_manager=fm,
    params={"time_column": "date", "value_column": "sales"}
)
# Expected: {
#   "file1": {"result": {...}, "status": "ok"},
#   "file2": {"error": "No time column in dataset"},
#   "file3": {"result": {...}, "status": "ok"}
# }
```

---

## Router Integration

Routers can now call engine multi-file functions directly:

```python
# Example: Analysis router
@router.post("/descriptive")
async def descriptive_analysis(request: DescriptiveStatsRequest):
    # Normalize file_ids
    file_ids = request.file_ids or [request.file_id]
    
    # Call engine multi-file function
    results = AnalysisEngine.analyze_multiple(
        file_ids=file_ids,
        file_manager=file_manager,
        analysis_type="descriptive",
        params={"columns": request.columns, "weight_column": request.weight_column}
    )
    
    # Extract results and errors
    results_per_file = {fid: r["result"] for fid, r in results.items() if "result" in r}
    errors = {fid: r["error"] for fid, r in results.items() if "error" in r}
    
    # Determine status
    status = "success" if len(results_per_file) == len(file_ids) else "partial_success"
    
    return {
        "status": status,
        "file_ids": file_ids,
        "results": results_per_file,
        "errors": errors
    }
```

---

## Key Benefits

1. **Scalability**: Process 1-5 files with same code path
2. **Isolation**: No cross-file interference or data leakage
3. **Error Resilience**: One file failure doesn't stop others
4. **Maintainability**: Core algorithms unchanged
5. **Backward Compatibility**: Single file_id still works
6. **Type Safety**: Static methods with clear signatures
7. **Testing**: Easy to unit test per file

---

## Next Steps (Part 5)

After engine updates complete, proceed with:
1. UI integration layer updates
2. Frontend multi-file upload component
3. Progress indicators for multi-file operations
4. Batch result visualization
5. Combined report ZIP download functionality

---

## Files Modified

1. ✅ `backend/utils/schema_validator.py` - Added multi-file schema functions
2. ✅ `backend/services/cleaning_engine.py` - Added `process_multiple()`
3. ✅ `backend/services/weighting_engine.py` - Added `process_multiple_weights()`
4. ✅ `backend/services/analysis_engine.py` - Added `analyze_multiple()`
5. ✅ `backend/services/ml_engine.py` - Added `ml_multiple()`
6. ✅ `backend/services/forecasting_engine.py` - Added `forecast_multiple()`
7. ✅ `backend/services/insight_engine.py` - Added `insights_multiple()`
8. ✅ `backend/services/report_engine.py` - Added `batch_generate_reports()`

---

## Implementation Complete ✅

All 8 engines now support multi-file processing with:
- ✅ Isolated execution per file_id
- ✅ No cross-file data merging
- ✅ Consistent error handling
- ✅ Standard return format
- ✅ File storage isolation
- ✅ Backward compatibility

**Status**: Ready for Part 5 (UI Integration)
