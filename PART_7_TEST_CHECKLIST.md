# ✅ Part 7 Testing Checklist

## Pre-Testing Setup

- [ ] Backend server running on `http://127.0.0.1:8000`
- [ ] Browser console open for debugging
- [ ] Network tab open to monitor API calls
- [ ] At least one CSV file ready for testing

---

## Test Suite 1: Dashboard Access

### Test 1.1: Direct Navigation
- [ ] Navigate to `dashboard.html` directly
- [ ] Page loads without errors
- [ ] Navbar and steps components load
- [ ] All sections render (stats, progress, files, insights, actions)

### Test 1.2: From Index Page
- [ ] Go to `index.html`
- [ ] Click "📊 View Dashboard" button
- [ ] Dashboard page loads correctly
- [ ] Button styling (gradient) displays correctly

### Test 1.3: From Navbar
- [ ] Click "Dashboard" in navbar from any page
- [ ] Dashboard loads
- [ ] Active state shows on navbar

**Expected Results:** All navigation methods work, page loads in < 2 seconds

---

## Test Suite 2: Empty State Handling

### Test 2.1: No Files Uploaded
- [ ] Clear localStorage: `localStorage.clear()`
- [ ] Refresh dashboard
- [ ] Overview stats show: 0 Files, 0 Steps, 0 Analyses, 0 Reports
- [ ] Recent Files shows empty state with "Upload Your First File" button
- [ ] Quick Insights shows "No analytics available yet"
- [ ] Progress bar at 0% with "Not started" label

### Test 2.2: Empty State Actions
- [ ] Click "Upload Your First File" button in Recent Files
- [ ] Redirects to `index.html`
- [ ] Click "Go to Analysis" in Quick Insights empty state
- [ ] Redirects to `analysis.html`

**Expected Results:** Empty states provide clear guidance and working CTAs

---

## Test Suite 3: File Upload & Display

### Test 3.1: Single File Upload
- [ ] Upload 1 CSV file via `index.html`
- [ ] Navigate to dashboard
- [ ] Recent Files shows 1 file
- [ ] File has correct ID (UUID format)
- [ ] Status badge shows "Uploaded" (gray)
- [ ] Timestamp displays correctly

### Test 3.2: Multiple File Uploads
- [ ] Upload 5 CSV files
- [ ] Dashboard shows all 5 files
- [ ] Files listed in reverse chronological order (newest first)
- [ ] Each file has unique ID
- [ ] All files show "Uploaded" badge

### Test 3.3: More Than 5 Files
- [ ] Upload 7 files total
- [ ] Dashboard shows only last 5 files
- [ ] Oldest files not displayed (as expected)

**Expected Results:** File list updates correctly, shows last 5 files

---

## Test Suite 4: Status Badge Progression

### Test 4.1: Schema Step
- [ ] Complete schema configuration for a file
- [ ] Mark `step2_complete` as true in localStorage
- [ ] Refresh dashboard
- [ ] File badge changes to "Schema" (blue)

### Test 4.2: Cleaning Step
- [ ] Complete data cleaning
- [ ] Mark `step3_complete` as true
- [ ] Badge changes to "Cleaned" (yellow)

### Test 4.3: Weighting Step
- [ ] Complete weighting
- [ ] Mark `step4_complete` as true
- [ ] Badge changes to "Weighted" (purple)

### Test 4.4: Analysis Step
- [ ] Complete analysis
- [ ] Mark `step5_complete` as true
- [ ] Badge changes to "Analysis" (green)

### Test 4.5: Full Pipeline
- [ ] Complete all 9 steps
- [ ] Badge shows highest completed step
- [ ] Badge changes to "Report" (gold) when step 9 complete

**Expected Results:** Badges update correctly as steps complete

---

## Test Suite 5: Pipeline Progress Bar

### Test 5.1: Initial State
- [ ] No steps completed
- [ ] Progress bar at 0%
- [ ] Label shows "Not started"
- [ ] Bar has no fill

### Test 5.2: Partial Progress
- [ ] Complete 3 steps
- [ ] Progress bar at 33%
- [ ] Label shows "Step 3 of 9"
- [ ] Bar fills ~1/3 width with gradient

### Test 5.3: Complete Pipeline
- [ ] Complete all 9 steps
- [ ] Progress bar at 100%
- [ ] Label shows "Pipeline complete! 🎉"
- [ ] Bar fully filled with gradient

### Test 5.4: Animation
- [ ] Refresh page with steps complete
- [ ] Progress bar animates from 0 to correct percentage
- [ ] Animation smooth (0.8s cubic-bezier)

**Expected Results:** Progress bar accurately reflects completion percentage

---

## Test Suite 6: Overview Statistics

### Test 6.1: File Count
- [ ] Upload 3 files
- [ ] Dashboard shows "3" in Files Processed stat
- [ ] Upload 2 more files
- [ ] Count updates to "5"

### Test 6.2: Steps Completed
- [ ] Complete 4 steps
- [ ] Stats show "4" in Steps Completed
- [ ] Increment by completing another step
- [ ] Count updates to "5"

### Test 6.3: Analysis Count
- [ ] Set `analysis_count` to "3" in localStorage
- [ ] Dashboard shows "3" in Analyses Run
- [ ] Increment counter
- [ ] Stat updates

### Test 6.4: Report Count
- [ ] Set `report_count` to "2" in localStorage
- [ ] Dashboard shows "2" in Reports Generated
- [ ] Stat displays correctly

**Expected Results:** All stats display accurate counts from localStorage

---

## Test Suite 7: Quick Insights

### Test 7.1: No Analysis Data
- [ ] No cached analysis results
- [ ] Quick Insights shows "Run analysis to see insights"
- [ ] "Go to Analysis" button present
- [ ] Button works and redirects

### Test 7.2: With Cached Data
- [ ] Run analysis on a file
- [ ] Cache results in localStorage
- [ ] Dashboard shows mini charts
- [ ] Up to 6 column summaries displayed

### Test 7.3: Mini Chart Content
- [ ] Each mini chart shows column name
- [ ] Large mean value displayed
- [ ] Subtitle shows "Mean: X | Median: Y"
- [ ] Numbers formatted to 2 decimal places

### Test 7.4: Hover Effects
- [ ] Hover over mini chart
- [ ] Card scales up (1.02)
- [ ] Shadow increases
- [ ] Transition smooth

**Expected Results:** Insights load from cache or show empty state

---

## Test Suite 8: Quick Actions

### Test 8.1: Upload New File
- [ ] Click "📤 Upload New File"
- [ ] Redirects to `index.html`
- [ ] Page loads correctly

### Test 8.2: Configure Schema
- [ ] Click "🗂️ Configure Schema"
- [ ] Redirects to `schema.html`
- [ ] Works correctly

### Test 8.3: View Analysis
- [ ] Click "📈 View Analysis"
- [ ] Redirects to `analysis.html`
- [ ] Loads correctly

### Test 8.4: Generate Report
- [ ] Click "📄 Generate Report"
- [ ] Redirects to `report.html`
- [ ] Works correctly

### Test 8.5: Restart Pipeline
- [ ] Click "🔄 Restart Pipeline"
- [ ] Confirmation dialog appears
- [ ] Click "OK"
- [ ] Step flags cleared from localStorage
- [ ] Page reloads
- [ ] Progress bar reset to 0%

### Test 8.6: Download All
- [ ] Click "💾 Download All"
- [ ] Toast shows "Preparing download..."
- [ ] API call to `/api/report/download/{file_id}`
- [ ] ZIP file downloads
- [ ] Toast shows "Download started"
- [ ] File named correctly (statflow_outputs_{id}.zip)

**Expected Results:** All action buttons work and navigate/execute correctly

---

## Test Suite 9: Restart Pipeline

### Test 9.1: Confirmation Dialog
- [ ] Click "Restart Pipeline"
- [ ] Dialog shows with warning message
- [ ] "Are you sure?" text present
- [ ] Cancel button works (no changes)

### Test 9.2: Successful Restart
- [ ] Complete several steps
- [ ] Click "Restart Pipeline" → OK
- [ ] All `step{N}_complete` flags removed from localStorage
- [ ] `analysis_count` reset to 0
- [ ] `report_count` reset to 0
- [ ] File IDs remain (not cleared)
- [ ] Page reloads automatically

### Test 9.3: After Restart
- [ ] Dashboard shows 0 steps completed
- [ ] Progress bar at 0%
- [ ] Files still present in Recent Files
- [ ] Status badges back to "Uploaded"

**Expected Results:** Restart clears progress but keeps files

---

## Test Suite 10: Download Functionality

### Test 10.1: Download with Files
- [ ] Have at least 1 file uploaded
- [ ] Click "Download All"
- [ ] API endpoint called: `/api/report/download/{file_id}`
- [ ] Response is blob
- [ ] Download triggered
- [ ] File saves to default download folder

### Test 10.2: Download Filename
- [ ] Check downloaded file name
- [ ] Format: `statflow_outputs_{first8chars}.zip`
- [ ] Extension is `.zip`

### Test 10.3: Download Error Handling
- [ ] Backend offline
- [ ] Click "Download All"
- [ ] Error toast shows: "Download failed. Please try again."
- [ ] Console logs error
- [ ] No crash or freeze

### Test 10.4: No Files Downloaded
- [ ] No files uploaded
- [ ] Click "Download All"
- [ ] Toast shows: "No files to download"
- [ ] No API call made

**Expected Results:** Download works correctly with proper error handling

---

## Test Suite 11: Dark Mode

### Test 11.1: Toggle Dark Mode
- [ ] Dashboard in light mode initially
- [ ] Click theme toggle button (bottom-right)
- [ ] Page transitions to dark mode
- [ ] All elements themed correctly

### Test 11.2: Dashboard Components in Dark Mode
- [ ] Background changes to dark gray
- [ ] Text changes to light color
- [ ] Cards have dark background
- [ ] Borders visible with good contrast
- [ ] Status badges have dark variants
- [ ] Progress bar visible on dark background

### Test 11.3: Hover States in Dark Mode
- [ ] Hover over file items
- [ ] Background changes to darker shade
- [ ] Hover over action buttons
- [ ] Proper dark mode hover effect

### Test 11.4: Theme Persistence
- [ ] Switch to dark mode
- [ ] Refresh page
- [ ] Dark mode persists
- [ ] localStorage has 'theme': 'dark'

**Expected Results:** Full dark mode support with all components themed

---

## Test Suite 12: Responsive Design

### Test 12.1: Desktop (1400px+)
- [ ] View dashboard at 1920px width
- [ ] Stats grid shows 4 columns
- [ ] Quick actions show 3 columns
- [ ] Mini charts show 3 columns
- [ ] All spacing appropriate

### Test 12.2: Tablet (900px)
- [ ] Resize to 900px width
- [ ] Stats grid shows 2 columns
- [ ] Quick actions show 2 columns
- [ ] Mini charts show 2 columns
- [ ] Cards adjust padding

### Test 12.3: Mobile (600px)
- [ ] Resize to 600px width
- [ ] Stats grid shows 1 column (stacked)
- [ ] Quick actions show 1 column
- [ ] Mini charts show 1 column
- [ ] File items stack properly

### Test 12.4: Small Mobile (450px)
- [ ] Resize to 450px width
- [ ] Base font size reduces to 14px
- [ ] Cards have minimal padding
- [ ] All content readable
- [ ] No horizontal scroll

**Expected Results:** Dashboard responsive at all breakpoints

---

## Test Suite 13: Animations

### Test 13.1: Page Load Animation
- [ ] Refresh dashboard
- [ ] Body fades in (pageEnter animation)
- [ ] Duration ~0.4s
- [ ] Smooth transition

### Test 13.2: Stat Card Animation
- [ ] Stat cards fade up on load
- [ ] Each card has `fadeInUp` animation
- [ ] Staggered effect (if implemented)

### Test 13.3: Progress Bar Animation
- [ ] Progress bar fills smoothly
- [ ] Gradient animates
- [ ] Duration ~0.8s
- [ ] Cubic-bezier easing

### Test 13.4: Hover Animations
- [ ] Hover over stat card → lifts up 4px
- [ ] Hover over file item → slides right 4px
- [ ] Hover over action button → lifts up 2px
- [ ] All transitions smooth (0.3s)

**Expected Results:** All animations smooth and performant

---

## Test Suite 14: Error Handling

### Test 14.1: API Errors
- [ ] Backend offline
- [ ] Refresh dashboard
- [ ] Quick Insights shows error state
- [ ] "Unable to load analytics" message
- [ ] Retry button present

### Test 14.2: Retry Functionality
- [ ] Error state showing
- [ ] Click "Retry" button
- [ ] Function re-executes
- [ ] Loads data if backend online

### Test 14.3: Malformed Data
- [ ] Corrupt localStorage data
- [ ] Dashboard handles gracefully
- [ ] No console errors (or handled errors)
- [ ] Shows empty states

### Test 14.4: Network Timeout
- [ ] Slow network (throttle to 3G)
- [ ] Dashboard loads
- [ ] Loading states show
- [ ] Eventually resolves or errors

**Expected Results:** Graceful error handling, no crashes

---

## Test Suite 15: Performance

### Test 15.1: Load Time
- [ ] Measure page load time
- [ ] Dashboard loads in < 2 seconds
- [ ] DOMContentLoaded < 1 second
- [ ] All assets load

### Test 15.2: Caching
- [ ] First load: API call made for insights
- [ ] Results cached in localStorage
- [ ] Second load: No API call (uses cache)
- [ ] Load time faster on subsequent visits

### Test 15.3: Memory Usage
- [ ] Open browser task manager
- [ ] Monitor memory usage
- [ ] Dashboard uses < 100MB
- [ ] No memory leaks on reload

### Test 15.4: Interaction Speed
- [ ] Click actions respond instantly
- [ ] Hover effects have no lag
- [ ] Animations smooth (60fps)
- [ ] No janky scrolling

**Expected Results:** Dashboard performs well, no lag or memory issues

---

## Test Suite 16: Browser Compatibility

### Test 16.1: Chrome/Edge
- [ ] Test in Chrome
- [ ] All features work
- [ ] Animations smooth
- [ ] No console errors

### Test 16.2: Firefox
- [ ] Test in Firefox
- [ ] Dashboard displays correctly
- [ ] Interactions work
- [ ] CSS Grid supported

### Test 16.3: Safari
- [ ] Test in Safari
- [ ] Gradient styles display
- [ ] LocalStorage works
- [ ] Animations render

### Test 16.4: Mobile Browsers
- [ ] Test on iOS Safari
- [ ] Test on Chrome Mobile
- [ ] Touch interactions work
- [ ] Responsive layout correct

**Expected Results:** Works on all modern browsers

---

## Test Suite 17: LocalStorage Management

### Test 17.1: Data Persistence
- [ ] Complete workflow with 1 file
- [ ] Close browser completely
- [ ] Reopen and go to dashboard
- [ ] All data still present
- [ ] Progress maintained

### Test 17.2: Multiple Sessions
- [ ] Open dashboard in 2 tabs
- [ ] Upload file in Tab 1
- [ ] Refresh Tab 2
- [ ] Tab 2 shows updated data

### Test 17.3: Storage Limits
- [ ] Upload many files (10+)
- [ ] Dashboard handles large data
- [ ] No performance degradation
- [ ] LocalStorage not exceeded

**Expected Results:** LocalStorage reliable and performant

---

## Test Suite 18: Integration with Other Pages

### Test 18.1: From Index to Dashboard
- [ ] Upload file on index.html
- [ ] Go to dashboard
- [ ] New file appears in Recent Files
- [ ] Stats update correctly

### Test 18.2: From Analysis to Dashboard
- [ ] Run analysis on analysis.html
- [ ] Go to dashboard
- [ ] Quick Insights shows analysis results
- [ ] Analysis count incremented

### Test 18.3: Workflow Continuity
- [ ] Start on index.html
- [ ] Upload → Schema → Cleaning → Dashboard
- [ ] Dashboard reflects all completed steps
- [ ] Progress bar updated

**Expected Results:** Seamless integration across pages

---

## Regression Testing

### After Each Code Change:
- [ ] Dashboard loads without errors
- [ ] All 6 quick actions work
- [ ] Status badges display correctly
- [ ] Progress bar updates
- [ ] Dark mode still works
- [ ] Responsive design intact
- [ ] No console errors

---

## Final Acceptance Criteria

### Must Pass:
- [ ] All Test Suites 1-18 pass
- [ ] No critical bugs
- [ ] Dashboard loads < 2 seconds
- [ ] Works on Chrome, Firefox, Safari
- [ ] Mobile responsive
- [ ] Dark mode functional
- [ ] All 6 quick actions work
- [ ] LocalStorage persists correctly
- [ ] Error handling graceful
- [ ] Animations smooth

### Nice to Have (Optional):
- [ ] Sub-second load time
- [ ] Advanced caching strategies
- [ ] Keyboard shortcuts
- [ ] Accessibility (ARIA labels)
- [ ] Print styles

---

## Bug Reporting Template

**Bug ID:** #  
**Severity:** Critical | High | Medium | Low  
**Test Suite:** Suite #  
**Test Case:** Test #  

**Steps to Reproduce:**
1. 
2. 
3. 

**Expected Result:**

**Actual Result:**

**Screenshots/Console Logs:**

**Browser/OS:**

**Additional Notes:**

---

## Testing Sign-Off

**Tester Name:** ___________________  
**Date:** ___________________  
**Test Suites Passed:** ___/18  
**Critical Bugs Found:** ___  
**Status:** ✅ Pass | ❌ Fail | ⚠️ Partial  

**Notes:**

---

**Testing Version:** 1.0  
**Last Updated:** January 8, 2026  
**Total Test Cases:** 100+
