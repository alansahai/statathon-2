# File Upload Webpage - Frontend Developer Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [Backend API Specifications](#backend-api-specifications)
3. [File Upload Requirements](#file-upload-requirements)
4. [UI/UX Guidelines](#uiux-guidelines)
5. [Implementation Steps](#implementation-steps)
6. [API Integration Details](#api-integration-details)
7. [Error Handling](#error-handling)
8. [Testing Checklist](#testing-checklist)
9. [Example Code](#example-code)

---

## 🎯 Overview

This document provides comprehensive specifications for developing a file upload webpage for the StatFlow AI survey analytics platform. The webpage will allow users to upload CSV or Excel files for data analysis.

### Project Context
- **Application**: StatFlow AI - Survey Analytics Platform
- **Backend Framework**: FastAPI (Python)
- **Backend Base URL**: `http://localhost:8000` (development)
- **API Prefix**: `/api/upload`
- **CORS**: Enabled for all origins in development

### Key Features Required
- Drag-and-drop file upload
- File type validation (CSV, XLSX, XLS)
- File size validation (max 50MB)
- Real-time upload progress indicator
- Data preview after successful upload
- Schema/column information display
- Error handling and user feedback
- File deletion capability

---

## 🔌 Backend API Specifications

### Base Configuration
```
Backend Server: http://localhost:8000
API Base Path: /api/upload
Content-Type: multipart/form-data
```

### 1. Upload File (General)
**Endpoint**: `POST /api/upload/file`

**Description**: Upload any supported file type (CSV or Excel). The backend automatically detects the file type.

**Request**:
```
Method: POST
Content-Type: multipart/form-data
Body: 
  - file: File (required)
```

**Success Response** (Status: 200):
```json
{
  "status": "success",
  "file_info": {
    "file_id": "5f888f32-d9f0-4a8d-8dea-75af47f01ec3",
    "filename": "survey_data.csv",
    "filepath": "temp_uploads/uploads/default_user/5f888f32-d9f0-4a8d-8dea-75af47f01ec3.csv",
    "file_size": 245678,
    "upload_timestamp": "2025-01-08T10:30:45.123456",
    "row_count": 1500,
    "column_count": 25
  },
  "schema": {
    "respondent_id": {
      "type": "int64",
      "nullable": false,
      "unique": true
    },
    "age": {
      "type": "int64",
      "nullable": false,
      "unique": false
    },
    "income": {
      "type": "float64",
      "nullable": true,
      "unique": false
    }
  },
  "columns": ["respondent_id", "age", "income", "gender", "satisfaction"],
  "preview": [
    {
      "respondent_id": 1,
      "age": 25,
      "income": 45000.50,
      "gender": "Male",
      "satisfaction": 4
    },
    {
      "respondent_id": 2,
      "age": 34,
      "income": 62000.00,
      "gender": "Female",
      "satisfaction": 5
    }
  ],
  "preview_row_count": 10
}
```

**Error Responses**:
- **400 Bad Request**:
  ```json
  {
    "detail": "No filename provided"
  }
  ```
  ```json
  {
    "detail": "Invalid extension .txt"
  }
  ```
  ```json
  {
    "detail": "Uploaded file is empty"
  }
  ```
  ```json
  {
    "detail": "File size exceeds limit"
  }
  ```
  ```json
  {
    "detail": "Failed to parse data: [specific error]"
  }
  ```
  ```json
  {
    "detail": "Parsed file contains no data"
  }
  ```

- **500 Internal Server Error**:
  ```json
  {
    "detail": "Internal server error: [error details]"
  }
  ```

### 2. Upload CSV Specifically
**Endpoint**: `POST /api/upload/csv`

**Description**: Upload CSV files only. Returns error if non-CSV file is uploaded.

**Request/Response**: Same as `/file` endpoint, but enforces CSV file type validation.

### 3. Upload Excel Specifically
**Endpoint**: `POST /api/upload/excel`

**Description**: Upload Excel files (.xlsx, .xls) only. Returns error if non-Excel file is uploaded.

**Request/Response**: Same as `/file` endpoint, but enforces Excel file type validation.

### 4. Check Upload Status
**Endpoint**: `GET /api/upload/status/{file_id}`

**Description**: Check if an uploaded file still exists and get its metadata.

**Request**:
```
Method: GET
Path Parameter: file_id (string, UUID format)
```

**Success Response** (Status: 200):
```json
{
  "status": "success",
  "file_id": "5f888f32-d9f0-4a8d-8dea-75af47f01ec3",
  "filepath": "temp_uploads/uploads/default_user/5f888f32-d9f0-4a8d-8dea-75af47f01ec3.csv",
  "file_size": 245678,
  "exists": true
}
```

**Error Response** (Status: 404):
```json
{
  "detail": "File not found"
}
```

### 5. Delete Uploaded File
**Endpoint**: `DELETE /api/upload/file/{file_id}`

**Description**: Delete a previously uploaded file from the server.

**Request**:
```
Method: DELETE
Path Parameter: file_id (string, UUID format)
```

**Success Response** (Status: 200):
```json
{
  "status": "success",
  "message": "File deleted successfully",
  "file_id": "5f888f32-d9f0-4a8d-8dea-75af47f01ec3"
}
```

**Error Response** (Status: 404):
```json
{
  "detail": "File not found"
}
```

---

## 📁 File Upload Requirements

### Supported File Types
| Extension | MIME Type | Description |
|-----------|-----------|-------------|
| `.csv` | `text/csv` | Comma-separated values |
| `.xlsx` | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` | Excel 2007+ |
| `.xls` | `application/vnd.ms-excel` | Excel 97-2003 |

### File Constraints
- **Maximum File Size**: 50 MB (52,428,800 bytes)
- **Minimum Requirements**: File must contain at least one row of data
- **Encoding**: UTF-8 (for CSV files)
- **Preview Rows**: First 10 rows are returned in the preview

### User Session Management
- **User ID**: Currently defaults to `"default_user"` (hardcoded in backend)
- **Storage Path**: Files are saved to `temp_uploads/uploads/{user_id}/{file_id}{extension}`
- **File Naming**: Backend generates unique UUID for each file to prevent conflicts

---

## 🎨 UI/UX Guidelines

### Page Layout Structure
```
┌─────────────────────────────────────────────┐
│            Header / Navigation              │
├─────────────────────────────────────────────┤
│                                             │
│              Hero Section                   │
│         "Upload Your Dataset"               │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│          Information Panel                  │
│   • Supported formats: CSV, Excel           │
│   • Maximum size: 50 MB                     │
│   • Data preview available                  │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│          Upload Section                     │
│   ┌───────────────────────────────────┐    │
│   │    Drag & Drop Area               │    │
│   │    or Click to Browse             │    │
│   │                                   │    │
│   │    📁 [Icon]                      │    │
│   └───────────────────────────────────┘    │
│                                             │
│   [Choose File Button] [Upload Button]     │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│     Upload Progress (when uploading)        │
│   ████████████░░░░░░░░░░░ 60%              │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│      Success Message & File Info            │
│   ✓ Upload successful!                      │
│   File: survey_data.csv (240 KB)            │
│   Rows: 1,500 | Columns: 25                 │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│         Data Preview Section                │
│   ┌───────────────────────────────────┐    │
│   │  Column1 | Column2 | Column3 ...  │    │
│   │  ──────────────────────────────   │    │
│   │  Value1  | Value2  | Value3  ...  │    │
│   │  Value1  | Value2  | Value3  ...  │    │
│   └───────────────────────────────────┘    │
│                                             │
│   [Continue to Data Cleaning] [Delete File] │
│                                             │
└─────────────────────────────────────────────┘
```

### Visual States

#### 1. Initial State (No File Selected)
- Display drag-and-drop area with dashed border
- Show upload icon and instructional text
- Display file requirements and constraints
- "Choose File" and "Upload" buttons (Upload disabled)

#### 2. File Selected State
- Change border color to indicate file is ready
- Display selected filename and size
- Enable "Upload" button
- Show "Remove" button to clear selection

#### 3. Uploading State
- Show progress bar with percentage
- Display spinner/loading animation
- Disable all input controls
- Show "Uploading..." status text

#### 4. Success State
- Display success checkmark and message
- Show file information card:
  - File ID (for reference)
  - Original filename
  - File size (human-readable format)
  - Row count
  - Column count
  - Upload timestamp
- Display data preview table (10 rows)
- Show column names and data types
- Enable "Continue" and "Delete" actions

#### 5. Error State
- Display error icon and message
- Show specific error description
- Highlight the error with red styling
- Enable "Try Again" button
- Keep previous form state for retry

### Color Scheme (Based on Existing UI)
```css
--primary-color: #3498db;
--success-color: #27ae60;
--error-color: #e74c3c;
--warning-color: #f39c12;
--border-color: #dfe6e9;
--bg-light: #f8f9fa;
--text-dark: #2c3e50;
--text-muted: #7f8c8d;
```

### Responsive Design
- **Desktop (> 768px)**: Full-width upload area, side-by-side buttons
- **Tablet (768px - 1024px)**: Slightly narrower container
- **Mobile (< 768px)**: Stack elements vertically, full-width buttons

---

## 🛠 Implementation Steps

### Step 1: HTML Structure
Create the basic HTML structure with:
- Hero section with title and description
- Information panel with upload requirements
- File input section (drag-and-drop area)
- Hidden file input element
- Upload progress indicator (initially hidden)
- Success message container (initially hidden)
- File information display area (initially hidden)
- Data preview table container (initially hidden)
- Error message container (initially hidden)
- Action buttons (Continue, Delete, Retry)

### Step 2: CSS Styling
Implement styles for:
- Upload area with hover and drag-over states
- File input wrapper and custom file button
- Progress bar animation
- Success/error message cards
- Data preview table (scrollable, styled)
- Responsive breakpoints
- Loading spinners and animations

### Step 3: JavaScript - File Selection
Implement:
- Click-to-browse functionality
- Drag-and-drop event handlers:
  - `dragover` - Prevent default and show visual feedback
  - `dragleave` - Remove visual feedback
  - `drop` - Handle file drop
- File validation (type and size)
- Display selected file information
- Enable/disable upload button based on validation

### Step 4: JavaScript - File Upload
Implement:
- FormData creation with file
- Fetch API or XMLHttpRequest for upload
- Progress tracking (using `XMLHttpRequest.upload.onprogress`)
- Upload function that:
  - Disables form during upload
  - Shows progress bar
  - Handles response
  - Updates UI based on result

### Step 5: JavaScript - Response Handling
Implement:
- Success handler:
  - Store file_id in memory/localStorage
  - Display file information
  - Render data preview table
  - Show action buttons
  - Hide upload section
- Error handler:
  - Parse error message
  - Display user-friendly error
  - Enable retry functionality

### Step 6: JavaScript - Additional Features
Implement:
- Delete file function
- "Continue to Next Step" navigation
- "Upload Another File" reset functionality
- File status checking (optional)
- Format file size display (bytes to KB/MB)
- Format timestamps to readable dates

---

## 🔗 API Integration Details

### JavaScript Fetch Example - File Upload
```javascript
async function uploadFile(file) {
  // Create FormData
  const formData = new FormData();
  formData.append('file', file);

  try {
    // Make API request
    const response = await fetch('http://localhost:8000/api/upload/file', {
      method: 'POST',
      body: formData
      // Note: Don't set Content-Type header, browser will set it with boundary
    });

    // Parse response
    const data = await response.json();

    if (response.ok) {
      // Success - handle response
      handleUploadSuccess(data);
    } else {
      // Error from backend
      handleUploadError(data.detail || 'Upload failed');
    }
  } catch (error) {
    // Network or other error
    handleUploadError('Network error: ' + error.message);
  }
}
```

### JavaScript XMLHttpRequest with Progress
```javascript
function uploadFileWithProgress(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);

    // Progress tracking
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const percentComplete = (e.loaded / e.total) * 100;
        updateProgressBar(percentComplete);
      }
    });

    // Load event (completion)
    xhr.addEventListener('load', () => {
      if (xhr.status === 200) {
        const data = JSON.parse(xhr.responseText);
        resolve(data);
      } else {
        const error = JSON.parse(xhr.responseText);
        reject(new Error(error.detail || 'Upload failed'));
      }
    });

    // Error event
    xhr.addEventListener('error', () => {
      reject(new Error('Network error occurred'));
    });

    // Send request
    xhr.open('POST', 'http://localhost:8000/api/upload/file');
    xhr.send(formData);
  });
}
```

### File Validation Before Upload
```javascript
function validateFile(file) {
  const maxSize = 50 * 1024 * 1024; // 50 MB
  const allowedTypes = ['.csv', '.xlsx', '.xls'];
  const allowedMimes = [
    'text/csv',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel'
  ];

  // Check if file exists
  if (!file) {
    return { valid: false, error: 'No file selected' };
  }

  // Check file size
  if (file.size > maxSize) {
    return { 
      valid: false, 
      error: `File size exceeds 50 MB limit (${formatFileSize(file.size)} provided)` 
    };
  }

  // Check file size is not zero
  if (file.size === 0) {
    return { valid: false, error: 'File is empty' };
  }

  // Check file extension
  const fileName = file.name.toLowerCase();
  const hasValidExtension = allowedTypes.some(ext => fileName.endsWith(ext));
  
  if (!hasValidExtension) {
    return { 
      valid: false, 
      error: `Invalid file type. Allowed types: ${allowedTypes.join(', ')}` 
    };
  }

  // Check MIME type (additional validation)
  if (!allowedMimes.includes(file.type) && file.type !== '') {
    console.warn('MIME type mismatch:', file.type);
    // Don't reject, as some systems may not set MIME correctly
  }

  return { valid: true };
}
```

---

## ⚠️ Error Handling

### Client-Side Validation Errors
Display these before making API call:

| Error | Condition | Message |
|-------|-----------|---------|
| No file | User clicks upload without selecting file | "Please select a file to upload" |
| Invalid type | File extension not in allowed list | "Invalid file type. Please upload CSV or Excel files only" |
| File too large | File size > 50 MB | "File size exceeds 50 MB limit" |
| Empty file | File size = 0 bytes | "Cannot upload an empty file" |

### Server-Side Error Handling
Handle these from API responses:

| HTTP Status | Error Detail | User-Friendly Message | Action |
|-------------|--------------|------------------------|--------|
| 400 | "No filename provided" | "Invalid file upload. Please try again" | Enable retry |
| 400 | "Invalid extension" | "This file type is not supported. Please upload CSV or Excel files" | Enable retry |
| 400 | "Expected CSV file" | "Please upload a CSV file for this section" | Enable retry |
| 400 | "Expected Excel file" | "Please upload an Excel file for this section" | Enable retry |
| 400 | "Uploaded file is empty" | "The uploaded file contains no data" | Enable retry |
| 400 | "File size exceeds limit" | "File size exceeds the 50 MB limit" | Enable retry |
| 400 | "Failed to parse data" | "Unable to read file. Please ensure the file is not corrupted" | Enable retry |
| 400 | "Parsed file contains no data" | "The file contains no data rows" | Enable retry |
| 404 | "File not found" | "File not found on server" | Return to upload |
| 500 | Any | "An unexpected error occurred. Please try again" | Enable retry |
| Network error | - | "Network error. Please check your connection and try again" | Enable retry |

### Error Display Pattern
```javascript
function displayError(message) {
  // Hide other sections
  hideProgressBar();
  hideSuccessSection();
  
  // Show error section
  const errorContainer = document.getElementById('error-container');
  errorContainer.innerHTML = `
    <div class="error-message">
      <div class="error-icon">⚠️</div>
      <h3>Upload Failed</h3>
      <p>${escapeHtml(message)}</p>
      <button onclick="resetForm()" class="btn btn-primary">Try Again</button>
    </div>
  `;
  errorContainer.style.display = 'block';
  
  // Re-enable upload section
  enableUploadSection();
}
```

---

## ✅ Testing Checklist

### Functional Testing

#### File Selection
- [ ] Click "Choose File" button opens file browser
- [ ] Drag and drop file onto upload area works
- [ ] Selected file name displays correctly
- [ ] File size displays in human-readable format (KB, MB)
- [ ] "Remove" button clears file selection

#### File Validation
- [ ] CSV file (.csv) passes validation
- [ ] Excel file (.xlsx) passes validation
- [ ] Excel file (.xls) passes validation
- [ ] Text file (.txt) fails validation with correct error
- [ ] PDF file (.pdf) fails validation with correct error
- [ ] File larger than 50 MB fails validation
- [ ] Empty file (0 bytes) fails validation

#### Upload Process
- [ ] Upload button disabled when no file selected
- [ ] Upload button enabled when valid file selected
- [ ] Progress bar appears during upload
- [ ] Progress bar updates correctly (0% to 100%)
- [ ] Form controls disabled during upload
- [ ] Upload completes successfully for valid file
- [ ] Success message appears after upload

#### Response Handling
- [ ] File information displays correctly (name, size, rows, columns)
- [ ] Data preview table renders with correct columns
- [ ] Data preview shows up to 10 rows
- [ ] Column data types display correctly
- [ ] Timestamp formats correctly
- [ ] File ID stores for future operations

#### Error Handling
- [ ] Network error shows appropriate message
- [ ] Server error (500) shows user-friendly message
- [ ] Invalid file type error displays correctly
- [ ] File too large error displays correctly
- [ ] Empty file error displays correctly
- [ ] Parsing error displays correctly
- [ ] "Try Again" button resets form correctly

#### Additional Features
- [ ] Delete file button makes correct API call
- [ ] Delete success shows confirmation message
- [ ] Delete error shows appropriate message
- [ ] "Continue" button navigates to next page
- [ ] "Upload Another File" resets entire form
- [ ] File status check works correctly (if implemented)

### UI/UX Testing

#### Visual States
- [ ] Initial state displays correctly
- [ ] Hover effect on upload area works
- [ ] Drag-over effect shows visual feedback
- [ ] File selected state has visual indicator
- [ ] Progress bar animates smoothly
- [ ] Success state has checkmark and green styling
- [ ] Error state has warning icon and red styling

#### Responsive Design
- [ ] Layout works on desktop (> 1024px)
- [ ] Layout works on tablet (768px - 1024px)
- [ ] Layout works on mobile (< 768px)
- [ ] Touch interactions work on mobile devices
- [ ] Buttons are properly sized for touch on mobile

#### Accessibility
- [ ] All interactive elements are keyboard accessible
- [ ] Tab order is logical
- [ ] Error messages are announced by screen readers
- [ ] File input has proper labels
- [ ] ARIA attributes used where appropriate
- [ ] Color contrast meets WCAG standards

### Performance Testing
- [ ] Small file (< 1 MB) uploads quickly
- [ ] Medium file (5-10 MB) uploads with progress tracking
- [ ] Large file (40-50 MB) uploads successfully
- [ ] Multiple rapid uploads handled correctly
- [ ] Memory leaks checked (no issues after multiple uploads)

### Browser Compatibility
- [ ] Chrome (latest version)
- [ ] Firefox (latest version)
- [ ] Safari (latest version)
- [ ] Edge (latest version)

---

## 💻 Example Code

### Complete HTML Structure
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Dataset - StatFlow AI</title>
    <link rel="stylesheet" href="assets/styles.css">
    <link rel="stylesheet" href="assets/upload.css">
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar">
        <!-- Your existing navbar -->
    </nav>

    <div class="upload-container">
        <!-- Hero Section -->
        <div class="hero-upload">
            <h1>📊 Upload Your Dataset</h1>
            <p>Upload CSV or Excel files to begin analysis</p>
        </div>

        <!-- Information Panel -->
        <div class="info-box">
            <h3>📌 Before You Upload</h3>
            <ul>
                <li>Supported formats: CSV (.csv), Excel (.xlsx, .xls)</li>
                <li>Maximum file size: 50 MB</li>
                <li>Files must contain at least one row of data</li>
                <li>First 10 rows will be shown as preview</li>
            </ul>
        </div>

        <!-- Upload Section -->
        <div class="upload-section" id="upload-section">
            <h2>Select File</h2>
            
            <!-- Drag and Drop Area -->
            <div class="file-input-wrapper">
                <label for="file-input" class="file-input-label" id="file-label">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">
                        <p><strong>Drag and drop your file here</strong></p>
                        <p>or click to browse</p>
                    </div>
                </label>
                <input type="file" id="file-input" accept=".csv,.xlsx,.xls">
            </div>

            <!-- Selected File Display -->
            <div id="selected-file-info" style="display: none;">
                <p><strong>Selected File:</strong> <span id="selected-filename"></span></p>
                <p><strong>Size:</strong> <span id="selected-filesize"></span></p>
                <button type="button" id="remove-file-btn" class="btn btn-secondary">Remove</button>
            </div>

            <!-- Action Buttons -->
            <div class="button-group">
                <button type="button" id="upload-btn" class="btn btn-primary" disabled>
                    Upload File
                </button>
            </div>
        </div>

        <!-- Progress Section (Hidden Initially) -->
        <div class="progress-section" id="progress-section" style="display: none;">
            <h3>Uploading...</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
            <p class="progress-text"><span id="progress-percent">0</span>%</p>
        </div>

        <!-- Success Section (Hidden Initially) -->
        <div class="success-section" id="success-section" style="display: none;">
            <div class="success-message">
                <div class="success-icon">✓</div>
                <h3>Upload Successful!</h3>
            </div>

            <!-- File Information -->
            <div class="file-info-card">
                <h4>File Information</h4>
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">Filename:</span>
                        <span class="info-value" id="info-filename"></span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">File Size:</span>
                        <span class="info-value" id="info-filesize"></span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Rows:</span>
                        <span class="info-value" id="info-rows"></span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Columns:</span>
                        <span class="info-value" id="info-columns"></span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">File ID:</span>
                        <span class="info-value" id="info-fileid"></span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Uploaded:</span>
                        <span class="info-value" id="info-timestamp"></span>
                    </div>
                </div>
            </div>

            <!-- Data Preview -->
            <div class="preview-section">
                <h4>Data Preview (First 10 Rows)</h4>
                <div class="table-wrapper">
                    <table id="preview-table" class="data-table">
                        <!-- Table will be populated dynamically -->
                    </table>
                </div>
            </div>

            <!-- Action Buttons -->
            <div class="button-group">
                <button type="button" id="continue-btn" class="btn btn-primary">
                    Continue to Data Cleaning →
                </button>
                <button type="button" id="delete-btn" class="btn btn-danger">
                    Delete File
                </button>
                <button type="button" id="upload-another-btn" class="btn btn-secondary">
                    Upload Another File
                </button>
            </div>
        </div>

        <!-- Error Section (Hidden Initially) -->
        <div class="error-section" id="error-section" style="display: none;">
            <div class="error-message">
                <div class="error-icon">⚠️</div>
                <h3>Upload Failed</h3>
                <p id="error-text"></p>
                <button type="button" id="retry-btn" class="btn btn-primary">
                    Try Again
                </button>
            </div>
        </div>
    </div>

    <script src="assets/upload.js"></script>
</body>
</html>
```

### Complete JavaScript Implementation
```javascript
// upload.js - File Upload Handler

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB
const ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls'];

// State
let selectedFile = null;
let uploadedFileId = null;

// DOM Elements
const fileInput = document.getElementById('file-input');
const fileLabel = document.getElementById('file-label');
const selectedFileInfo = document.getElementById('selected-file-info');
const uploadBtn = document.getElementById('upload-btn');
const removeFileBtn = document.getElementById('remove-file-btn');

// Sections
const uploadSection = document.getElementById('upload-section');
const progressSection = document.getElementById('progress-section');
const successSection = document.getElementById('success-section');
const errorSection = document.getElementById('error-section');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
});

function initializeEventListeners() {
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Remove file button
    removeFileBtn.addEventListener('click', clearFileSelection);
    
    // Upload button
    uploadBtn.addEventListener('click', handleUpload);
    
    // Drag and drop events
    fileLabel.addEventListener('dragover', handleDragOver);
    fileLabel.addEventListener('dragleave', handleDragLeave);
    fileLabel.addEventListener('drop', handleDrop);
    
    // Success section buttons
    document.getElementById('continue-btn')?.addEventListener('click', handleContinue);
    document.getElementById('delete-btn')?.addEventListener('click', handleDelete);
    document.getElementById('upload-another-btn')?.addEventListener('click', resetForm);
    document.getElementById('retry-btn')?.addEventListener('click', resetForm);
}

// File Selection Handlers
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFileSelection(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    fileLabel.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    fileLabel.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    fileLabel.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file) {
        // Set the file to the input element
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        
        processFileSelection(file);
    }
}

function processFileSelection(file) {
    // Validate file
    const validation = validateFile(file);
    
    if (!validation.valid) {
        showError(validation.error);
        return;
    }
    
    // Store selected file
    selectedFile = file;
    
    // Update UI
    document.getElementById('selected-filename').textContent = file.name;
    document.getElementById('selected-filesize').textContent = formatFileSize(file.size);
    
    selectedFileInfo.style.display = 'block';
    fileLabel.classList.add('has-file');
    uploadBtn.disabled = false;
    
    // Hide error if any
    errorSection.style.display = 'none';
}

function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';
    selectedFileInfo.style.display = 'none';
    fileLabel.classList.remove('has-file');
    uploadBtn.disabled = true;
}

// File Validation
function validateFile(file) {
    if (!file) {
        return { valid: false, error: 'No file selected' };
    }
    
    if (file.size === 0) {
        return { valid: false, error: 'Cannot upload an empty file' };
    }
    
    if (file.size > MAX_FILE_SIZE) {
        return { 
            valid: false, 
            error: `File size exceeds 50 MB limit. Your file is ${formatFileSize(file.size)}` 
        };
    }
    
    const fileName = file.name.toLowerCase();
    const hasValidExtension = ALLOWED_EXTENSIONS.some(ext => fileName.endsWith(ext));
    
    if (!hasValidExtension) {
        return { 
            valid: false, 
            error: `Invalid file type. Please upload CSV or Excel files only` 
        };
    }
    
    return { valid: true };
}

// Upload Handler
async function handleUpload() {
    if (!selectedFile) {
        showError('Please select a file to upload');
        return;
    }
    
    // Validate again before upload
    const validation = validateFile(selectedFile);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }
    
    // Hide upload section, show progress
    uploadSection.style.display = 'none';
    progressSection.style.display = 'block';
    errorSection.style.display = 'none';
    
    try {
        const response = await uploadFileWithProgress(selectedFile);
        handleUploadSuccess(response);
    } catch (error) {
        handleUploadError(error.message);
    }
}

function uploadFileWithProgress(file) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append('file', file);
        
        // Progress tracking
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                updateProgressBar(percentComplete);
            }
        });
        
        // Load event
        xhr.addEventListener('load', () => {
            if (xhr.status === 200) {
                try {
                    const data = JSON.parse(xhr.responseText);
                    resolve(data);
                } catch (e) {
                    reject(new Error('Invalid response from server'));
                }
            } else {
                try {
                    const error = JSON.parse(xhr.responseText);
                    reject(new Error(error.detail || 'Upload failed'));
                } catch (e) {
                    reject(new Error(`Upload failed with status ${xhr.status}`));
                }
            }
        });
        
        // Error event
        xhr.addEventListener('error', () => {
            reject(new Error('Network error occurred. Please check your connection'));
        });
        
        // Abort event
        xhr.addEventListener('abort', () => {
            reject(new Error('Upload was cancelled'));
        });
        
        // Send request
        xhr.open('POST', `${API_BASE_URL}/api/upload/file`);
        xhr.send(formData);
    });
}

function updateProgressBar(percent) {
    const progressFill = document.getElementById('progress-fill');
    const progressPercent = document.getElementById('progress-percent');
    
    progressFill.style.width = percent + '%';
    progressPercent.textContent = percent;
}

// Success Handler
function handleUploadSuccess(response) {
    // Store file ID
    uploadedFileId = response.file_info.file_id;
    
    // Store in localStorage for other pages
    localStorage.setItem('current_file_id', uploadedFileId);
    localStorage.setItem('current_filename', response.file_info.filename);
    
    // Hide progress, show success
    progressSection.style.display = 'none';
    successSection.style.display = 'block';
    
    // Populate file information
    document.getElementById('info-filename').textContent = response.file_info.filename;
    document.getElementById('info-filesize').textContent = formatFileSize(response.file_info.file_size);
    document.getElementById('info-rows').textContent = response.file_info.row_count.toLocaleString();
    document.getElementById('info-columns').textContent = response.file_info.column_count;
    document.getElementById('info-fileid').textContent = response.file_info.file_id;
    document.getElementById('info-timestamp').textContent = formatTimestamp(response.file_info.upload_timestamp);
    
    // Populate data preview table
    populatePreviewTable(response.columns, response.preview);
}

function populatePreviewTable(columns, preview) {
    const table = document.getElementById('preview-table');
    table.innerHTML = '';
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    columns.forEach(column => {
        const th = document.createElement('th');
        th.textContent = column;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    
    preview.forEach(row => {
        const tr = document.createElement('tr');
        
        columns.forEach(column => {
            const td = document.createElement('td');
            const value = row[column];
            td.textContent = value !== null && value !== undefined ? value : 'null';
            tr.appendChild(td);
        });
        
        tbody.appendChild(tr);
    });
    
    table.appendChild(tbody);
}

// Error Handler
function handleUploadError(errorMessage) {
    progressSection.style.display = 'none';
    uploadSection.style.display = 'block';
    showError(errorMessage);
}

function showError(message) {
    errorSection.style.display = 'block';
    document.getElementById('error-text').textContent = message;
}

// Action Handlers
function handleContinue() {
    // Navigate to data cleaning page
    window.location.href = 'cleaning.html';
}

async function handleDelete() {
    if (!uploadedFileId) {
        showError('No file to delete');
        return;
    }
    
    if (!confirm('Are you sure you want to delete this file?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/upload/file/${uploadedFileId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            alert('File deleted successfully');
            resetForm();
        } else {
            const error = await response.json();
            showError(error.detail || 'Failed to delete file');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    }
}

function resetForm() {
    // Clear state
    selectedFile = null;
    uploadedFileId = null;
    
    // Clear localStorage
    localStorage.removeItem('current_file_id');
    localStorage.removeItem('current_filename');
    
    // Reset file input
    fileInput.value = '';
    
    // Hide all sections except upload
    uploadSection.style.display = 'block';
    progressSection.style.display = 'none';
    successSection.style.display = 'none';
    errorSection.style.display = 'none';
    selectedFileInfo.style.display = 'none';
    
    // Reset UI state
    fileLabel.classList.remove('has-file', 'drag-over');
    uploadBtn.disabled = true;
    
    // Reset progress bar
    updateProgressBar(0);
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function formatTimestamp(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
```

### CSS Styling Example
```css
/* upload.css - File Upload Page Styles */

:root {
    --primary-color: #3498db;
    --success-color: #27ae60;
    --error-color: #e74c3c;
    --warning-color: #f39c12;
    --border-color: #dfe6e9;
    --bg-light: #f8f9fa;
    --text-dark: #2c3e50;
    --text-muted: #7f8c8d;
    --shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.upload-container {
    max-width: 800px;
    margin: 2rem auto;
    padding: 0 1rem;
}

/* Hero Section */
.hero-upload {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, var(--primary-color), #2c6ba8);
    color: white;
    border-radius: 8px;
    margin-bottom: 2rem;
}

.hero-upload h1 {
    margin: 0 0 0.5rem 0;
    font-size: 2rem;
}

.hero-upload p {
    margin: 0;
    opacity: 0.9;
}

/* Info Box */
.info-box {
    background: #e8f4fd;
    border-left: 4px solid var(--primary-color);
    padding: 1.5rem;
    border-radius: 5px;
    margin-bottom: 2rem;
}

.info-box h3 {
    margin-top: 0;
    margin-bottom: 1rem;
    color: var(--primary-color);
}

.info-box ul {
    list-style: none;
    padding-left: 0;
    margin: 0;
}

.info-box li {
    padding: 0.5rem 0;
    padding-left: 1.5rem;
    position: relative;
}

.info-box li:before {
    content: "✓";
    position: absolute;
    left: 0;
    color: var(--success-color);
    font-weight: bold;
}

/* Upload Section */
.upload-section {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: var(--shadow);
    margin-bottom: 2rem;
}

.file-input-wrapper {
    margin: 1.5rem 0;
}

.file-input-label {
    display: block;
    padding: 3rem 2rem;
    border: 3px dashed var(--border-color);
    border-radius: 8px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    background: var(--bg-light);
}

.file-input-label:hover {
    border-color: var(--primary-color);
    background: #e8f4fd;
}

.file-input-label.drag-over {
    border-color: var(--primary-color);
    background: #e8f4fd;
    transform: scale(1.02);
}

.file-input-label.has-file {
    border-color: var(--success-color);
    background: #d4edda;
}

.upload-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.upload-text p {
    margin: 0.5rem 0;
}

.upload-text strong {
    color: var(--text-dark);
    font-size: 1.1rem;
}

#selected-file-info {
    background: var(--bg-light);
    padding: 1rem;
    border-radius: 5px;
    margin-top: 1rem;
}

#selected-file-info p {
    margin: 0.5rem 0;
}

/* Progress Section */
.progress-section {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: var(--shadow);
    text-align: center;
}

.progress-bar {
    width: 100%;
    height: 30px;
    background: var(--bg-light);
    border-radius: 15px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary-color), #2c6ba8);
    transition: width 0.3s ease;
    width: 0%;
}

.progress-text {
    font-size: 1.2rem;
    font-weight: bold;
    color: var(--primary-color);
}

/* Success Section */
.success-section {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: var(--shadow);
}

.success-message {
    text-align: center;
    margin-bottom: 2rem;
}

.success-icon {
    display: inline-block;
    width: 60px;
    height: 60px;
    background: var(--success-color);
    color: white;
    border-radius: 50%;
    line-height: 60px;
    font-size: 2rem;
    margin-bottom: 1rem;
}

.file-info-card {
    background: var(--bg-light);
    padding: 1.5rem;
    border-radius: 5px;
    margin-bottom: 2rem;
}

.file-info-card h4 {
    margin-top: 0;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
}

.info-item {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border-color);
}

.info-label {
    font-weight: bold;
    color: var(--text-muted);
}

.info-value {
    color: var(--text-dark);
}

/* Preview Section */
.preview-section {
    margin: 2rem 0;
}

.table-wrapper {
    overflow-x: auto;
    border: 1px solid var(--border-color);
    border-radius: 5px;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}

.data-table th {
    background: var(--primary-color);
    color: white;
    padding: 0.75rem;
    text-align: left;
    font-weight: 600;
    position: sticky;
    top: 0;
}

.data-table td {
    padding: 0.75rem;
    border-bottom: 1px solid var(--border-color);
}

.data-table tbody tr:hover {
    background: var(--bg-light);
}

/* Error Section */
.error-section {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: var(--shadow);
    text-align: center;
}

.error-message {
    color: var(--error-color);
}

.error-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

/* Buttons */
.button-group {
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}

.btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.btn-primary {
    background: var(--primary-color);
    color: white;
}

.btn-primary:hover:not(:disabled) {
    background: #2980b9;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.btn-secondary {
    background: var(--text-muted);
    color: white;
}

.btn-secondary:hover {
    background: #6c7a89;
}

.btn-danger {
    background: var(--error-color);
    color: white;
}

.btn-danger:hover {
    background: #c0392b;
}

/* Responsive Design */
@media (max-width: 768px) {
    .upload-container {
        margin: 1rem auto;
    }
    
    .hero-upload h1 {
        font-size: 1.5rem;
    }
    
    .upload-section,
    .success-section,
    .error-section {
        padding: 1.5rem;
    }
    
    .file-input-label {
        padding: 2rem 1rem;
    }
    
    .button-group {
        flex-direction: column;
    }
    
    .btn {
        width: 100%;
    }
    
    .info-grid {
        grid-template-columns: 1fr;
    }
}

/* Animations */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.success-section,
.error-section {
    animation: fadeIn 0.3s ease;
}
```

---

## 📝 Additional Notes

### Storage Locations
- **Uploaded Files**: `temp_uploads/uploads/default_user/{file_id}.csv`
- **Processed Files**: `temp_uploads/processed/{file_id}_analysis.json`
- **Reports**: `temp_uploads/reports/`
- **Cleaned Data**: `temp_uploads/cleaned/default_user/{file_id}_cleaned.csv`

### Session Management
Currently, the backend uses a hardcoded user ID (`default_user`). In future versions:
- Implement proper authentication
- Pass user tokens in headers
- Store files per authenticated user

### Next Steps After Upload
After successful upload, users typically proceed to:
1. **Data Cleaning** (`cleaning.html`) - Clean and prepare data
2. **Schema Mapping** (`schema.html`) - Map columns to expected schema
3. **Analysis** (`analysis.html`) - Perform statistical analysis
4. **Dashboard** (`dashboard.html`) - View visualizations

### File ID Storage
Store the `file_id` returned from the upload API:
- In `localStorage` for session persistence
- Pass as query parameter to next page: `cleaning.html?file_id=xxx`
- Use in subsequent API calls for data operations

### Performance Considerations
- For large files (>10MB), ensure progress tracking is smooth
- Consider implementing chunk upload for files near the 50MB limit
- Add file compression option for large datasets
- Implement client-side CSV parsing preview before upload (optional)

---

## 🆘 Support & Resources

### Backend Documentation
- FastAPI Docs: `http://localhost:8000/docs` (Interactive API documentation)
- ReDoc: `http://localhost:8000/redoc` (Alternative documentation view)

### Useful Libraries
- **File Upload**: Native FormData API, or use libraries like:
  - `axios` - For easier promise-based HTTP requests
  - `dropzone.js` - Advanced drag-and-drop functionality
  
- **Data Tables**: Consider using:
  - Native HTML tables (as shown)
  - `DataTables.js` - For sortable, searchable tables
  - `ag-Grid` - For advanced data grid features

### Testing the Backend
To start the backend server:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then access:
- Application: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

---

## ✨ Best Practices

1. **Always validate files client-side** before upload to save bandwidth
2. **Provide clear error messages** that help users fix issues
3. **Show upload progress** for better user experience
4. **Store file_id** securely for subsequent operations
5. **Handle network errors gracefully** with retry options
6. **Make the interface responsive** for mobile users
7. **Add loading states** for all async operations
8. **Sanitize data** before displaying (XSS protection)
9. **Test with various file sizes** and formats
10. **Implement proper error logging** for debugging

---

**Document Version**: 1.0  
**Last Updated**: January 8, 2026  
**Author**: Backend Development Team  
**Contact**: [Your contact information]
