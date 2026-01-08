# Test file for StatFlow AI Backend

## Test the server

The server is running at: http://127.0.0.1:8000

## Available Endpoints

### Upload Endpoints
- `POST /api/upload/csv` - Upload CSV file
- `POST /api/upload/excel` - Upload Excel file
- `POST /api/upload/file` - Upload any supported file (auto-detect)
- `GET /api/upload/status/{file_id}` - Get upload status
- `DELETE /api/upload/file/{file_id}` - Delete uploaded file

### Test with curl or Postman

**Upload CSV:**
```bash
curl -X POST "http://127.0.0.1:8000/api/upload/file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_data.csv"
```

**Interactive API Docs:**
Visit http://127.0.0.1:8000/docs for Swagger UI

## Features Implemented

### 1. Upload Router (routers/upload.py)
✅ Accept CSV or Excel upload
✅ Save uploaded file to /temp_uploads
✅ Extract preview rows (first 10)
✅ Validate file size (max 50MB)
✅ Validate file extension (.csv, .xlsx, .xls)
✅ Return JSON with schema, preview, and filepath
✅ Error handling for empty files, wrong extensions, broken formats

### 2. Schema Validator (utils/schema_validator.py)
✅ Detect column types (numeric, text, datetime)
✅ Detect missing percentages per column
✅ Identify numeric vs categorical columns
✅ Calculate statistics (min, max, mean, median, std)
✅ Identify unique identifiers
✅ Detect semantic types (continuous, discrete, categorical, text)
✅ Return structured schema dictionary

### 3. File Manager (utils/file_manager.py)
✅ Generate unique filenames using UUID
✅ Save file to disk with proper directory structure
✅ Load as Pandas DataFrame safely
✅ Handle multiple encoding types (UTF-8, Latin-1, CP1252)
✅ Support CSV and Excel formats
✅ Delete temporary files when done
✅ Calculate file hash (SHA-256)
✅ Track file metadata in registry

## Response Format

### Successful Upload Response:
```json
{
  "status": "success",
  "file_info": {
    "file_id": "uuid-here",
    "filename": "data.csv",
    "filepath": "./temp_uploads/uploads/default_user/uuid.csv",
    "file_size": 12345,
    "upload_timestamp": "2026-01-07T...",
    "row_count": 100,
    "column_count": 5
  },
  "schema": {
    "row_count": 100,
    "column_count": 5,
    "columns": {
      "age": {
        "name": "age",
        "dtype": "int64",
        "missing_count": 2,
        "missing_percentage": 2.0,
        "non_missing_count": 98,
        "unique_count": 45,
        "semantic_type": "numeric",
        "category": "continuous",
        "min": 18.0,
        "max": 75.0,
        "mean": 42.5,
        "median": 41.0,
        "std": 12.3
      }
    },
    "missing_summary": {
      "total_missing": 10,
      "columns_with_missing": ["age", "income"]
    }
  },
  "columns": ["age", "income", "education", ...],
  "preview": [
    {"age": 25, "income": 50000, ...},
    ...
  ],
  "preview_row_count": 10
}
```

### Error Responses:
- 400: Invalid file extension, empty file, parsing error
- 404: File not found
- 500: Internal server error

## Next Steps

To continue development:
1. Implement cleaning endpoints in routers/cleaning.py
2. Implement weighting endpoints in routers/weighting.py
3. Implement estimation endpoints in routers/estimation.py
4. Implement report generation in routers/report.py
5. Add authentication and user management
6. Connect to a real database (PostgreSQL/MongoDB)
7. Add data visualization endpoints
8. Implement pipeline orchestration
