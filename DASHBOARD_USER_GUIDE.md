# 📊 Dashboard Quick Start Guide

## Accessing the Dashboard

1. **From Homepage:** Click the "📊 View Dashboard" button on [index.html](index.html)
2. **From Navbar:** Click "Dashboard" in the navigation menu
3. **Direct Link:** Navigate to [dashboard.html](dashboard.html)

---

## Dashboard Sections

### 1. 📈 Overview Statistics
Four key metrics displayed in animated cards:
- **Files Processed:** Total number of uploaded CSV files
- **Steps Completed:** Pipeline progress (0-9 steps)
- **Analyses Run:** Number of statistical analyses executed
- **Reports Generated:** Total PDF/HTML reports created

### 2. 🔄 Pipeline Progress
Visual progress bar showing:
- **Percentage:** 0-100% based on completed steps
- **Status Label:** "Not started", "Step X of 9", or "Pipeline complete! 🎉"
- **Animated Fill:** Gradient progress indicator

### 3. 📁 Recent Files
Lists the 5 most recently uploaded files with:
- **File Name:** Truncated display name
- **File ID:** Full UUID for reference
- **Timestamp:** When the file was uploaded
- **Status Badge:** Current processing stage (8 stages)

#### Status Badge Colors:
| Badge | Color | Stage |
|-------|-------|-------|
| Uploaded | Gray | Initial upload |
| Schema | Blue | Schema configured |
| Cleaned | Yellow | Data cleaning complete |
| Weighted | Purple | Weighting applied |
| Analysis | Green | Statistical analysis done |
| Forecast | Teal | Forecasting complete |
| ML | Pink | Machine learning complete |
| Report | Gold | Final report generated |

### 4. 💡 Quick Insights
Mini analytics panels showing:
- **Column Name:** Variable being analyzed
- **Mean Value:** Average (large display)
- **Median:** Middle value
- **Data Source:** Cached analysis results

### 5. ⚡ Quick Actions
Six action buttons for workflow navigation:

| Button | Icon | Action |
|--------|------|--------|
| Upload New File | 📤 | Go to upload page |
| Configure Schema | 🗂️ | Set column types |
| View Analysis | 📈 | See statistical results |
| Generate Report | 📄 | Create PDF report |
| Restart Pipeline | 🔄 | Clear all progress |
| Download All | 💾 | Get ZIP file |

---

## User Workflows

### First-Time User (No Files)
1. Dashboard shows empty states
2. Click "📤 Upload New File"
3. Upload CSV file(s)
4. Return to dashboard to see progress

### Continuing User (Files Uploaded)
1. Dashboard shows overview stats
2. Check pipeline progress bar
3. View recent files and their status
4. Click quick action to continue workflow

### Power User (Multiple Files)
1. Monitor all files in Recent Files list
2. Check Quick Insights for data summaries
3. Use status badges to track each file's stage
4. Download all outputs when complete

---

## Features

### ✅ Real-Time Updates
- Stats refresh on page load
- Progress bar updates automatically
- Status badges reflect latest completion flags

### 🎨 Dark Mode Support
- All components themed for dark mode
- Status badges have dark variants
- Smooth theme transitions

### 📱 Responsive Design
- **Desktop:** Full 3-4 column layout
- **Tablet (900px):** 2 column layout
- **Mobile (600px):** Single column stacked

### 💾 Data Persistence
- Uses localStorage for all tracking
- File IDs stored persistently
- Analysis results cached for speed
- Step completion flags maintained

### 🔄 Pipeline Management
**Restart Pipeline:**
- Clears all step completion flags
- Keeps file IDs intact
- Resets analysis/report counters
- Shows confirmation dialog

**Download All:**
- Fetches ZIP file from backend
- Includes all processed files
- Auto-downloads to browser
- Shows toast notifications

---

## Technical Details

### LocalStorage Keys Used:
```javascript
'statflow_file_ids'          // Array of file UUIDs
'currentFileId'              // Active file being processed
'step1_complete' - 'step9_complete'  // Boolean flags
'analysis_count'             // Number of analyses run
'report_count'               // Number of reports generated
'analysis_{fileId}'          // Cached analysis results
'file_{fileId}_timestamp'    // Upload timestamp
```

### API Endpoints Called:
```
GET /api/dashboard/kpi/{file_id}
GET /api/dashboard/risks/{file_id}
GET /api/analysis/descriptive/{file_id}
GET /api/report/download/{file_id}
```

### JavaScript Service:
- **DashboardService** (dashboard.js)
  - `init()` - Initialize all sections
  - `loadOverview()` - Load stat cards
  - `loadRecentFiles()` - Display file list
  - `loadQuickCharts()` - Show mini analytics
  - `updatePipelineProgress()` - Update progress bar
  - `restartPipeline()` - Clear progress
  - `downloadAllOutputs()` - Download ZIP

---

## Troubleshooting

### Dashboard Shows Empty States
**Problem:** No files appear in Recent Files  
**Solution:** 
1. Go to index.html
2. Upload at least one CSV file
3. Return to dashboard

### Stats Show Zero
**Problem:** All counters at 0  
**Solution:** Complete at least one step in the pipeline

### Quick Insights Empty
**Problem:** No mini charts displayed  
**Solution:** 
1. Go to analysis.html
2. Run statistical analysis
3. Return to dashboard

### Progress Bar at 0%
**Problem:** No pipeline progress  
**Solution:** Follow the workflow from schema → cleaning → analysis

### Download Fails
**Problem:** "Download failed" error  
**Solution:** 
1. Ensure backend is running
2. Check file_id exists
3. Verify report has been generated

---

## Best Practices

1. **Check Dashboard First:** Always start your session by viewing the dashboard
2. **Monitor Progress:** Use the progress bar to track your workflow
3. **Complete Steps in Order:** Follow the pipeline: Upload → Schema → Cleaning → Weighting → Analysis
4. **Cache Awareness:** Dashboard caches analysis results for speed
5. **Regular Downloads:** Download outputs after completing major stages

---

## Keyboard Shortcuts (Future Enhancement)
- `Ctrl+D` - Go to Dashboard (planned)
- `Ctrl+U` - Upload New File (planned)
- `Ctrl+R` - Restart Pipeline (planned)

---

## Mobile Optimization

### Mobile Layout Changes:
- Single column layout
- Larger touch targets
- Stacked file items
- Full-width action buttons
- Compact stat cards

### Mobile Gestures:
- Swipe left/right on file items (future)
- Pull-to-refresh (future)

---

## Privacy & Data

### Data Storage:
- All data stored in browser localStorage
- No data sent to external servers
- Files processed locally by backend
- Session data persists across page refreshes

### Data Clearing:
- Click "Restart Pipeline" to clear progress flags
- Clear browser data to reset completely
- File uploads remain until manually deleted

---

## Support

For issues or questions:
1. Check [PART_7_SUMMARY.md](PART_7_SUMMARY.md) for technical details
2. Review console logs for errors
3. Ensure backend is running on port 8000
4. Verify localStorage has data

---

**Dashboard Version:** 1.0  
**Last Updated:** January 8, 2026  
**Compatible With:** StatFlow AI v6 MVP
