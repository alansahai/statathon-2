// =======================================================================
// cleaning.js – Auto and Manual Data Cleaning
// =======================================================================

/**
 * Get file_id from URL or sessionStorage
 */
function getFileId() {
    const urlParams = new URLSearchParams(window.location.search);
    const fileIdFromUrl = urlParams.get('file');
    const fileIdFromSession = sessionStorage.getItem('uploadedFileId');

    return fileIdFromUrl || fileIdFromSession;
}

let cleaningMode = 'auto'; // 'auto' or 'manual'
let detectedIssues = null;

window.addEventListener("DOMContentLoaded", () => initCleaning());

function initCleaning() {
    const fileId = getFileId();
    const fileName = sessionStorage.getItem("uploadedFileName");

    console.log("[cleaning.js] Initializing with file_id:", fileId);

    if (!fileId) {
        showError("No file uploaded. Please upload a file first.");
        setTimeout(() => {
            window.location.href = "/ui/index.html";
        }, 2000);
        return;
    }

    // Show file info
    if (fileName) {
        const fileInfoEl = document.getElementById("fileInfo");
        if (fileInfoEl) {
            fileInfoEl.innerHTML = `
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid var(--primary-color);">
                    <strong>📄 File:</strong> ${fileName} | <strong>🆔 ID:</strong> <code>${fileId}</code>
                </div>
            `;
        }
    }

    // Setup mode toggle
    setupModeToggle();

    // Add event listeners
    const autoCleanBtn = document.getElementById("autoCleanBtn");
    const manualCleanBtn = document.getElementById("manualCleanBtn");
    const detectIssuesBtn = document.getElementById("detectIssuesBtn");

    if (autoCleanBtn) {
        autoCleanBtn.addEventListener("click", () => runAutoCleaning(fileId));
    }

    if (manualCleanBtn) {
        manualCleanBtn.addEventListener("click", () => runManualCleaning(fileId));
    }

    if (detectIssuesBtn) {
        detectIssuesBtn.addEventListener("click", () => detectIssues(fileId));
    }

    // Automatically run auto cleaning on page load
    console.log("[cleaning.js] Auto-running cleaning...");
    runAutoCleaning(fileId);
}

function setupModeToggle() {
    const autoModeBtn = document.getElementById("autoModeBtn");
    const manualModeBtn = document.getElementById("manualModeBtn");
    const autoSection = document.getElementById("autoCleaningSection");
    const manualSection = document.getElementById("manualCleaningSection");

    if (autoModeBtn && manualModeBtn) {
        autoModeBtn.addEventListener("click", () => {
            cleaningMode = 'auto';
            autoModeBtn.classList.add("active");
            manualModeBtn.classList.remove("active");
            if (autoSection) autoSection.style.display = "block";
            if (manualSection) manualSection.style.display = "none";
        });

        manualModeBtn.addEventListener("click", () => {
            cleaningMode = 'manual';
            manualModeBtn.classList.add("active");
            autoModeBtn.classList.remove("active");
            if (manualSection) manualSection.style.display = "block";
            if (autoSection) autoSection.style.display = "none";
        });
    }
}

async function runAutoCleaning(fileId) {
    console.log("[cleaning.js] Running auto cleaning for file_id:", fileId);

    const url = `${window.API_BASE}/cleaning/auto-clean`;
    const payload = { file_id: fileId };

    console.log("[cleaning.js] Sending request to:", url);
    console.log("[cleaning.js] Payload:", payload);

    const statusEl = document.getElementById("cleaningStatus");
    const resultsEl = document.getElementById("cleaningResults");

    showLoading(statusEl, "🧹 Cleaning your data...");
    if (resultsEl) resultsEl.innerHTML = "";

    try {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errorData = await res.json();
            throw new Error(errorData.detail || `Server error: ${res.status}`);
        }

        const data = await res.json();
        console.log("[cleaning.js] Response:", data);

        if (data.status !== "success" && data.status !== "partial_success") {
            throw new Error(data.message || data.error || "Cleaning failed");
        }

        // Show success message
        if (statusEl) {
            statusEl.innerHTML = `
                <div style="text-align: center; padding: 1.5rem; background: #d4edda; border-radius: 8px; border: 1px solid #c3e6cb;">
                    <h3 style="color: #155724; margin: 0;">✅ Data Cleaned Successfully!</h3>
                </div>
            `;
        }

        // Display cleaning results
        displayCleaningResults(resultsEl, data, fileId);

        console.log('✅ Data cleaning completed successfully');

    } catch (err) {
        console.error("[cleaning.js] Error:", err);
        showError(err.message, statusEl, resultsEl, fileId);
    }
}

async function detectIssues(fileId) {
    console.log("[cleaning.js] Detecting issues for file_id:", fileId);

    const url = `${window.API_BASE}/cleaning/detect-issues`;
    const payload = { file_id: fileId };

    const statusEl = document.getElementById("detectionStatus");
    const resultsEl = document.getElementById("detectionResults");

    showLoading(statusEl, "🔍 Detecting data quality issues...");
    if (resultsEl) resultsEl.innerHTML = "";

    try {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errorData = await res.json();
            throw new Error(errorData.detail || `Server error: ${res.status}`);
        }

        const data = await res.json();
        console.log("[cleaning.js] Detection response:", data);

        if (data.status !== "success") {
            throw new Error(data.error || "Issue detection failed");
        }

        detectedIssues = data;

        // Display detected issues
        displayDetectedIssues(resultsEl, data, fileId);

    } catch (err) {
        console.error("[cleaning.js] Detection error:", err);
        if (statusEl) {
            statusEl.innerHTML = `
                <div style="padding: 1rem; background: #f8d7da; border-radius: 8px; color: #721c24;">
                    ❌ ${err.message}
                </div>
            `;
        }
    }
}

async function runManualCleaning(fileId) {
    if (!detectedIssues) {
        alert("Please detect issues first before running manual cleaning.");
        return;
    }

    console.log("[cleaning.js] Running manual cleaning with detected issues");

    const url = `${window.API_BASE}/cleaning/manual-clean`;
    const payload = {
        file_id: fileId,
        options: {
            handle_missing: true,
            handle_duplicates: true,
            handle_outliers: false
        }
    };

    const statusEl = document.getElementById("detectionStatus");
    const resultsEl = document.getElementById("detectionResults");

    showLoading(statusEl, "🧹 Applying manual cleaning rules...");

    try {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errorData = await res.json();
            throw new Error(errorData.detail || `Server error: ${res.status}`);
        }

        const data = await res.json();
        console.log("[cleaning.js] Manual cleaning response:", data);

        if (data.status !== "success") {
            throw new Error(data.error || "Manual cleaning failed");
        }

        displayCleaningResults(resultsEl, data, fileId);

    } catch (err) {
        console.error("[cleaning.js] Manual cleaning error:", err);
        showError(err.message, statusEl, resultsEl, fileId);
    }
}

function showLoading(element, message) {
    if (!element) return;
    element.innerHTML = `
        <div style="text-align: center; padding: 2rem;">
            <div class="spinner"></div>
            <p style="margin-top: 1rem; color: var(--text-muted); font-size: 1.1rem;">${message}</p>
            <p style="color: var(--text-muted); font-size: 0.9rem;">This may take a few moments</p>
        </div>
    `;
}

function displayCleaningResults(resultsEl, data, fileId) {
    if (!resultsEl || !data.results) return;

    const fileResults = data.results[fileId] || {};
    const summary = fileResults.summary || {};
    const issuesFixed = summary.issues_fixed || 0;
    const issuesDetected = summary.issues_detected || 0;
    const rowsAfter = summary.rows_after || 'N/A';
    const outliersDetails = summary.outliers_details || {};
    const logicalDetails = summary.logical_details || {};

    console.log('[cleaning.js] Full summary:', summary);
    console.log('[cleaning.js] Outliers details:', outliersDetails);
    console.log('[cleaning.js] Outliers count:', Object.keys(outliersDetails).length);
    console.log('[cleaning.js] Logical details:', logicalDetails);
    console.log('[cleaning.js] Missing before:', summary.missing_values_before);
    console.log('[cleaning.js] Missing after:', summary.missing_values_after);

    // Generate detailed cleaning report
    let detailedReport = '';

    // Outliers section
    const outlierColumns = Object.keys(outliersDetails);
    console.log('[cleaning.js] Processing outlier columns:', outlierColumns);

    if (outlierColumns.length > 0) {
        detailedReport += `
            <div style="margin-top: 2rem; padding: 1.5rem; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 8px;">
                <h4 style="color: #856404; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">📊</span>
                    Outlier Detection & Treatment (${outlierColumns.length} columns)
                </h4>
                <div style="display: grid; gap: 1rem;">
                    ${outlierColumns.map(column => {
            const details = outliersDetails[column];
            console.log(`[cleaning.js] Processing outlier column: ${column}`, details);
            return `
                        <div style="background: white; padding: 1rem; border-radius: 6px; border: 1px solid #ffc107;">
                            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                                <strong style="color: #856404; font-size: 1rem;">📌 ${column}</strong>
                                <span style="background: #ffc107; color: #000; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">
                                    ${details.method ? details.method.toUpperCase() : 'DETECTED'}
                                </span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin-top: 0.75rem;">
                                ${details.count ? `
                                    <div style="font-size: 0.9rem; color: #666;">
                                        <strong>Count:</strong> ${details.count} outliers
                                    </div>
                                ` : ''}
                                ${details.percentage !== undefined ? `
                                    <div style="font-size: 0.9rem; color: #666;">
                                        <strong>Percentage:</strong> ${details.percentage}%
                                    </div>
                                ` : ''}
                                ${details.treatment ? `
                                    <div style="font-size: 0.9rem; color: #666;">
                                        <strong>Treatment:</strong> ${details.treatment.replace(/_/g, ' ')}
                                    </div>
                                ` : ''}
                                ${details.lower_cap !== undefined ? `
                                    <div style="font-size: 0.9rem; color: #666;">
                                        <strong>Lower Cap:</strong> ${details.lower_cap.toFixed(2)}
                                    </div>
                                ` : ''}
                                ${details.upper_cap !== undefined ? `
                                    <div style="font-size: 0.9rem; color: #666;">
                                        <strong>Upper Cap:</strong> ${details.upper_cap.toFixed(2)}
                                    </div>
                                ` : ''}
                                ${details.q1 !== undefined ? `
                                    <div style="font-size: 0.9rem; color: #666;">
                                        <strong>Q1:</strong> ${details.q1.toFixed(2)}
                                    </div>
                                ` : ''}
                                ${details.q3 !== undefined ? `
                                    <div style="font-size: 0.9rem; color: #666;">
                                        <strong>Q3:</strong> ${details.q3.toFixed(2)}
                                    </div>
                                ` : ''}
                                ${details.iqr !== undefined ? `
                                    <div style="font-size: 0.9rem; color: #666;">
                                        <strong>IQR:</strong> ${details.iqr.toFixed(2)}
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    `}).join('')}
                </div>
            </div>
        `;
    }

    // Logical validation section
    const logicalColumns = Object.keys(logicalDetails);
    console.log('[cleaning.js] Processing logical columns:', logicalColumns);

    if (logicalColumns.length > 0) {
        detailedReport += `
            <div style="margin-top: 2rem; padding: 1.5rem; background: #d1ecf1; border-left: 4px solid #17a2b8; border-radius: 8px;">
                <h4 style="color: #0c5460; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">🔍</span>
                    Logical Validation Issues (${logicalColumns.length} issues)
                </h4>
                <div style="display: grid; gap: 1rem;">
                    ${logicalColumns.map(key => {
            const details = logicalDetails[key];
            console.log(`[cleaning.js] Processing logical issue: ${key}`, details);
            const violationCount = details.violations || details.count || 0;
            const ruleText = details.rule || details.description || '';
            const actionText = details.action || '';

            return `
                        <div style="background: white; padding: 1rem; border-radius: 6px; border: 1px solid #17a2b8;">
                            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                                <strong style="color: #0c5460; font-size: 1rem;">⚠️ ${key.replace(/_/g, ' ').toUpperCase()}</strong>
                                ${violationCount > 0 ? `
                                    <span style="background: #17a2b8; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">
                                        ${violationCount} violations
                                    </span>
                                ` : ''}
                            </div>
                            ${ruleText ? `
                                <p style="font-size: 0.9rem; color: #666; margin: 0.5rem 0;">
                                    <strong>Rule:</strong> ${ruleText}
                                </p>
                            ` : ''}
                            ${actionText ? `
                                <p style="font-size: 0.9rem; color: #0c5460; margin: 0.5rem 0 0 0;">
                                    <strong>Action:</strong> ${actionText}
                                </p>
                            ` : ''}
                        </div>
                    `}).join('')}
                </div>
            </div>
        `;
    }

    // Missing values section (if available)
    const missingBefore = summary.missing_values_before || {};
    const missingAfter = summary.missing_values_after || {};

    const columnsWithMissing = new Set([...Object.keys(missingBefore), ...Object.keys(missingAfter)]);
    console.log('[cleaning.js] Columns with missing values:', Array.from(columnsWithMissing));

    if (columnsWithMissing.size > 0) {
        const missingColumns = Array.from(columnsWithMissing).filter(column => {
            const before = missingBefore[column] || 0;
            return before > 0;
        });

        console.log('[cleaning.js] Processing missing value columns:', missingColumns);

        if (missingColumns.length > 0) {
            detailedReport += `
                <div style="margin-top: 2rem; padding: 1.5rem; background: #e7f3ff; border-left: 4px solid #007bff; border-radius: 8px;">
                    <h4 style="color: #004085; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.5rem;">🔧</span>
                        Missing Value Imputation (${missingColumns.length} columns)
                    </h4>
                    <div style="display: grid; gap: 1rem;">
                        ${missingColumns.map(column => {
                const before = missingBefore[column] || 0;
                const after = missingAfter[column] || 0;
                const fixed = before - after;

                console.log(`[cleaning.js] Missing values for ${column}: before=${before}, after=${after}, fixed=${fixed}`);

                return `
                                <div style="background: white; padding: 1rem; border-radius: 6px; border: 1px solid #007bff;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <strong style="color: #004085; font-size: 1rem;">📋 ${column}</strong>
                                        <div style="display: flex; gap: 1rem; align-items: center;">
                                            <span style="font-size: 0.9rem; color: #dc3545;">
                                                <strong>Before:</strong> ${before} missing
                                            </span>
                                            <span style="font-size: 1.2rem;">→</span>
                                            <span style="font-size: 0.9rem; color: #28a745;">
                                                <strong>After:</strong> ${after} missing
                                            </span>
                                            ${fixed > 0 ? `
                                                <span style="background: #28a745; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">
                                                    ${fixed} fixed
                                                </span>
                                            ` : ''}
                                        </div>
                                    </div>
                                </div>
                            `;
            }).join('')}
                    </div>
                </div>
            `;
        }
    }

    resultsEl.innerHTML = `
        <div style="background: white; padding: 2rem; border-radius: 8px; border: 1px solid #dee2e6; margin-top: 1.5rem;">
            <h3 style="color: #28a745; margin-bottom: 1.5rem; text-align: center;">✨ Cleaning Summary</h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
                <div style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: #007bff; margin-bottom: 0.5rem;">${issuesDetected}</div>
                    <div style="color: #6c757d; font-size: 0.9rem; font-weight: 600;">Issues Detected</div>
                </div>
                
                <div style="text-align: center; padding: 1.5rem; background: #d4edda; border-radius: 8px;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: #28a745; margin-bottom: 0.5rem;">${issuesFixed}</div>
                    <div style="color: #155724; font-size: 0.9rem; font-weight: 600;">Issues Fixed</div>
                </div>
                
                <div style="text-align: center; padding: 1.5rem; background: #e8f4fd; border-radius: 8px;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: #0056b3; margin-bottom: 0.5rem;">${rowsAfter}</div>
                    <div style="color: #004085; font-size: 0.9rem; font-weight: 600;">Rows After Cleaning</div>
                </div>
            </div>

            ${detailedReport}

            <div style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 8px; margin-top: 2rem;">
                <p style="margin: 0 0 1.5rem 0; color: var(--text-muted); font-size: 0.95rem;">
                    ℹ️ Data cleaning completed. Ready to proceed to the next step.
                </p>
                <button 
                    id="continueToWeightingBtn"
                    class="btn btn-primary"
                    style="
                        padding: 1rem 2rem;
                        font-size: 1.1rem;
                        background: #007bff;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3);
                    "
                    onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(0, 123, 255, 0.4)'"
                    onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0, 123, 255, 0.3)'"
                >
                    ✓ Continue to Weighting →
                </button>
            </div>
        </div>
    `;

    // Attach click handler
    const continueBtn = document.getElementById('continueToWeightingBtn');
    if (continueBtn) {
        continueBtn.addEventListener('click', () => {
            window.location.href = `/ui/weighting.html?file=${fileId}`;
        });
    }
}

function displayDetectedIssues(resultsEl, data, fileId) {
    if (!resultsEl || !data.results) return;

    const fileResults = data.results[fileId] || {};
    const issues = fileResults.issues || {};

    resultsEl.innerHTML = `
        <div style="background: white; padding: 2rem; border-radius: 8px; border: 1px solid #dee2e6; margin-top: 1.5rem;">
            <h3 style="color: #007bff; margin-bottom: 1.5rem;">🔍 Detected Issues</h3>
            
            <div style="margin-bottom: 2rem;">
                ${Object.keys(issues).length === 0 ?
            '<p style="color: #28a745;">✅ No issues detected! Your data looks clean.</p>' :
            Object.entries(issues).map(([key, value]) => `
                        <div style="padding: 1rem; background: #fff3cd; border-left: 4px solid #ffc107; margin-bottom: 1rem; border-radius: 4px;">
                            <strong style="color: #856404;">${key}:</strong>
                            <span style="color: #856404;">${JSON.stringify(value)}</span>
                        </div>
                    `).join('')
        }
            </div>

            <div style="text-align: center;">
                <button 
                    id="proceedManualClean"
                    class="btn btn-success"
                    style="
                        padding: 1rem 2rem;
                        font-size: 1.1rem;
                        background: #28a745;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
                    "
                    onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(40, 167, 69, 0.4)'"
                    onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(40, 167, 69, 0.3)'"
                >
                    🧹 Apply Manual Cleaning
                </button>
            </div>
        </div>
    `;

    // Attach click handler
    const cleanBtn = document.getElementById('proceedManualClean');
    if (cleanBtn) {
        cleanBtn.addEventListener('click', () => {
            runManualCleaning(fileId);
        });
    }
}

function showError(message, statusEl, resultsEl, fileId) {
    if (statusEl) {
        statusEl.innerHTML = `
            <div style="text-align: center; padding: 1.5rem; background: #f8d7da; border-radius: 8px; border: 1px solid #f5c6cb;">
                <h3 style="color: #721c24; margin: 0 0 0.5rem 0;">❌ Cleaning Failed</h3>
                <p style="color: #721c24; margin: 0;">${message}</p>
            </div>
        `;
    }

    if (resultsEl && fileId) {
        resultsEl.innerHTML = `
            <div style="background: white; padding: 2rem; border-radius: 8px; border: 1px solid #f5c6cb; margin-top: 1rem; text-align: center;">
                <p style="color: #721c24; margin-bottom: 1.5rem;">Unable to complete data cleaning. Please try again or go back to schema mapping.</p>
                <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                    <button 
                        onclick="window.location.href='/ui/schema.html?file=${fileId}'"
                        class="btn btn-secondary"
                        style="padding: 0.75rem 1.5rem; border-radius: 6px; cursor: pointer; background: #6c757d; color: white; border: none;"
                    >
                        ← Back to Schema
                    </button>
                    <button 
                        onclick="location.reload()"
                        class="btn btn-primary"
                        style="padding: 0.75rem 1.5rem; border-radius: 6px; cursor: pointer; background: #007bff; color: white; border: none;"
                    >
                        🔄 Try Again
                    </button>
                </div>
            </div>
        `;
    }
}
