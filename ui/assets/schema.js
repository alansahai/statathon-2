/**
 * schema.js - Schema Mapping Page Module
 * Handles column detection and schema mapping
 */

if (!window.API_BASE) console.error("API_BASE not ready");

/**
 * Get file_id from URL or sessionStorage
 */
function getFileId() {
    const urlParams = new URLSearchParams(window.location.search);
    const fileIdFromUrl = urlParams.get('file');
    const fileIdFromSession = sessionStorage.getItem('uploadedFileId');

    return fileIdFromUrl || fileIdFromSession;
}

/**
 * Initialize schema page
 */
async function initSchemaPage() {
    const fileId = getFileId();
    const fileName = sessionStorage.getItem("uploadedFileName");

    console.log("[schema.js] Initializing with file_id:", fileId);

    const loadingArea = document.getElementById('loadingArea');
    const columnMapping = document.getElementById('columnMapping');

    if (!fileId) {
        console.warn("[schema.js] No file_id found, redirecting to upload page");
        if (loadingArea) {
            loadingArea.innerHTML = `
                <div style="text-align: center; padding: 2rem;">
                    <p style="color: #dc3545; font-size: 1.1rem;">❌ No file found. Please upload a file first.</p>
                    <button onclick="window.location.href='index.html'" class="btn btn-primary" style="margin-top: 1rem;">
                        ← Back to Upload
                    </button>
                </div>
            `;
        }
        return;
    }

    // Show file info
    if (fileName) {
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            fileInfo.innerHTML = `
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid var(--primary-color);">
                    <strong>📄 File:</strong> ${fileName} | <strong>🆔 ID:</strong> <code>${fileId}</code>
                </div>
            `;
        }
    }

    // Auto-load schema and apply mapping
    await loadSchemaColumns(fileId);

    console.log('✅ Schema page initialized');
}

/**
 * Load schema columns and auto-apply mapping
 */
async function loadSchemaColumns(fileId) {
    const loadingArea = document.getElementById('loadingArea');
    const columnMapping = document.getElementById('columnMapping');

    // Show loading state
    if (loadingArea) {
        loadingArea.style.display = 'block';
        loadingArea.innerHTML = `
            <div class="spinner"></div>
            <p style="color: var(--text-muted); font-size: 1.1rem;">🔍 Analyzing your data...</p>
            <p style="color: var(--text-muted); font-size: 0.9rem;">Detecting columns and auto-mapping schema...</p>
        `;
    }

    try {
        console.log('[schema.js] Calling auto-mapping endpoint:', `${window.API_BASE}/schema/auto/${fileId}`);

        // Call auto-mapping endpoint (it will detect columns and map automatically)
        const response = await fetch(`${window.API_BASE}/schema/auto/${fileId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();
        console.log('[schema.js] Auto-mapping response:', data);

        if (data.status === 'success' && data.data) {
            // Get columns from the response
            const mappingData = data.data;
            const columns = mappingData.columns || [];
            const mapping = mappingData.mapping || {};

            if (columns.length === 0) {
                throw new Error('No columns found in the dataset');
            }

            // Render the results
            renderColumnsTable(columns, mappingData);

            console.log('✅ Schema mapping completed successfully');

        } else {
            throw new Error(data.message || 'Auto-mapping failed');
        }

    } catch (error) {
        console.error('Schema error:', error);
        if (loadingArea) {
            loadingArea.style.display = 'block';
            loadingArea.innerHTML = `
                <div style="text-align: center; padding: 2rem; background: #f8d7da; border-radius: 8px; border: 1px solid #f5c6cb;">
                    <p style="color: #721c24; font-size: 1.1rem; margin-bottom: 1rem;">❌ Schema mapping failed</p>
                    <p style="color: #721c24; margin-bottom: 1.5rem;">${error.message}</p>
                    <button onclick="window.location.href='index.html'" class="btn btn-primary">
                        ← Back to Upload
                    </button>
                    <button onclick="location.reload()" class="btn btn-secondary" style="margin-left: 0.5rem;">
                        🔄 Try Again
                    </button>
                </div>
            `;
        }
    }
}

/**
 * Render columns table with mapping (editable)
 */
function renderColumnsTable(columns, mappingData) {
    const loadingArea = document.getElementById('loadingArea');
    const columnMapping = document.getElementById('columnMapping');

    const mapping = mappingData.mapping || {};
    const confidence = mappingData.confidence_score || 0;
    const warnings = mappingData.warnings || [];

    // Available mapping types
    const validTypes = ['numeric', 'categorical', 'datetime', 'text', 'identifier'];

    let tableHTML = `
        <div style="margin-bottom: 2rem;">
            <div style="
                background: ${confidence > 0.8 ? '#d4edda' : '#fff3cd'};
                border: 1px solid ${confidence > 0.8 ? '#c3e6cb' : '#ffeaa7'};
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
            ">
                <h3 style="margin: 0 0 0.5rem 0; color: ${confidence > 0.8 ? '#155724' : '#856404'};">
                    ${confidence > 0.8 ? '✅' : '⚠️'} Auto-Mapping ${confidence > 0.8 ? 'Complete' : 'Applied'}
                </h3>
                <p style="margin: 0; color: ${confidence > 0.8 ? '#155724' : '#856404'};">
                    Confidence Score: <strong>${(confidence * 100).toFixed(0)}%</strong> | 
                    ${columns.length} columns detected
                </p>
            </div>`;

    // Show warnings if any
    if (warnings && warnings.length > 0) {
        tableHTML += `
            <div style="
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1.5rem;
            ">
                <h4 style="margin: 0 0 0.5rem 0; color: #856404;">⚠️ Data Quality Warnings:</h4>
                <ul style="margin: 0.5rem 0 0 1.5rem; color: #856404;">
                    ${warnings.map(w => `<li>${w}</li>`).join('')}
                </ul>
            </div>`;
    }

    tableHTML += `
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <thead>
                        <tr style="background: var(--primary-color, #007bff); color: white;">
                            <th style="padding: 1rem; text-align: left; border: 1px solid #dee2e6; width: 30%;">Column Name</th>
                            <th style="padding: 1rem; text-align: left; border: 1px solid #dee2e6; width: 15%;">Data Type</th>
                            <th style="padding: 1rem; text-align: left; border: 1px solid #dee2e6; width: 25%;">Map To</th>
                            <th style="padding: 1rem; text-align: left; border: 1px solid #dee2e6; width: 30%;">Info</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    columns.forEach((col, index) => {
        const mappedField = mapping[col.name] || 'categorical';
        const unique = col.unique || 0;
        const missingPct = col.missing_pct || 0;

        // Generate dropdown options
        const typeOptions = validTypes.map(type =>
            `<option value="${type}" ${type === mappedField ? 'selected' : ''}>${type}</option>`
        ).join('');

        tableHTML += `
            <tr style="border-bottom: 1px solid #dee2e6; ${index % 2 === 0 ? 'background: #f8f9fa;' : 'background: white;'}">
                <td style="padding: 0.75rem; border: 1px solid #dee2e6;">
                    <code style="background: #e9ecef; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.9rem;">${col.name}</code>
                </td>
                <td style="padding: 0.75rem; border: 1px solid #dee2e6;">
                    <span style="background: #e8f4fd; color: #007bff; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                        ${col.dtype || 'unknown'}
                    </span>
                </td>
                <td style="padding: 0.75rem; border: 1px solid #dee2e6;">
                    <select 
                        class="mapping-select" 
                        data-column="${col.name}"
                        style="
                            width: 100%;
                            padding: 0.5rem;
                            border: 2px solid #007bff;
                            border-radius: 6px;
                            background: white;
                            font-size: 0.9rem;
                            font-weight: 600;
                            color: #007bff;
                            cursor: pointer;
                            transition: all 0.2s;
                        "
                        onchange="this.style.borderColor='#28a745'; this.style.color='#28a745';"
                    >
                        ${typeOptions}
                    </select>
                </td>
                <td style="padding: 0.75rem; border: 1px solid #dee2e6; font-size: 0.85rem; color: #6c757d;">
                    ${unique} unique | ${missingPct > 0 ? `${missingPct.toFixed(1)}% missing` : 'No missing'}
                </td>
            </tr>
        `;
    });

    tableHTML += `
                    </tbody>
                </table>
            </div>

            <div style="margin-top: 2rem; display: flex; gap: 1rem; justify-content: center; align-items: center; flex-wrap: wrap; padding: 1.5rem; background: #f8f9fa; border-radius: 8px;">
                <div style="flex: 1; min-width: 250px;">
                    <p style="margin: 0; color: var(--text-muted); font-size: 0.95rem;">
                        ℹ️ Review and adjust the column mappings above, then verify to proceed.
                    </p>
                </div>
                <div style="display: flex; gap: 0.75rem;">
                    <button 
                        id="verifyBtn" 
                        class="btn btn-success" 
                        style="
                            padding: 1rem 1.5rem;
                            font-size: 1rem;
                            background: #28a745;
                            color: white;
                            border: none;
                            border-radius: 8px;
                            cursor: pointer;
                            transition: all 0.3s ease;
                            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
                            font-weight: 600;
                        "
                        onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(40, 167, 69, 0.4)'"
                        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(40, 167, 69, 0.3)'"
                    >
                        ✓ Verify & Save Mapping
                    </button>
                    <button 
                        id="continueBtn" 
                        class="btn btn-primary" 
                        disabled
                        style="
                            padding: 1rem 1.5rem;
                            font-size: 1rem;
                            background: #6c757d;
                            color: white;
                            border: none;
                            border-radius: 8px;
                            cursor: not-allowed;
                            transition: all 0.3s ease;
                            opacity: 0.6;
                        "
                    >
                        Continue to Cleaning →
                    </button>
                </div>
            </div>
    `;

    // Hide loading, show results
    if (loadingArea) loadingArea.style.display = 'none';
    if (columnMapping) {
        columnMapping.style.display = 'block';
        columnMapping.innerHTML = tableHTML;
    }

    // Attach verify button handler
    const verifyBtn = document.getElementById('verifyBtn');
    const continueBtn = document.getElementById('continueBtn');

    if (verifyBtn) {
        verifyBtn.addEventListener('click', async () => {
            console.log('[schema.js] Verify button clicked');
            await verifyAndSaveMapping();
        });
    }
}

/**
 * Verify and save the mapping configuration
 */
async function verifyAndSaveMapping() {
    const fileId = getFileId();

    // Re-query buttons from DOM to ensure we have fresh references
    const verifyBtn = document.getElementById('verifyBtn');
    const continueBtn = document.getElementById('continueBtn');

    if (!verifyBtn || !continueBtn) {
        console.error('[schema.js] Buttons not found in DOM');
        return;
    }

    // Collect all mappings from dropdowns
    const mappingSelects = document.querySelectorAll('.mapping-select');
    const updatedMapping = {};

    mappingSelects.forEach(select => {
        const columnName = select.getAttribute('data-column');
        const mappedType = select.value;
        updatedMapping[columnName] = mappedType;
    });

    console.log('[schema.js] Saving mapping:', updatedMapping);

    // Update button state
    verifyBtn.disabled = true;
    verifyBtn.textContent = '⏳ Saving...';
    verifyBtn.style.background = '#6c757d';

    try {
        // Save mapping to backend
        const response = await fetch(`${window.API_BASE}/schema/save/${fileId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mapping: updatedMapping })
        });

        const data = await response.json();

        if (data.status === 'success') {
            console.log('[schema.js] Mapping saved successfully, enabling continue button');
            // Success - enable continue button
            verifyBtn.textContent = '✅ Verified!';
            verifyBtn.style.background = '#28a745';
            verifyBtn.disabled = true;  // Verify button should be disabled after success

            console.log('[schema.js] continueBtn element:', continueBtn);
            console.log('[schema.js] continueBtn disabled before:', continueBtn.disabled);
            console.log('[schema.js] continueBtn hasAttribute("disabled"):', continueBtn.hasAttribute('disabled'));

            // IMPORTANT: Remove the disabled attribute, not just set property to false
            continueBtn.removeAttribute('disabled');
            continueBtn.style.background = '#007bff';
            continueBtn.style.cursor = 'pointer';  // Change cursor from not-allowed to pointer
            continueBtn.style.opacity = '1';  // Change opacity from 0.6 to 1
            continueBtn.style.boxShadow = '0 4px 12px rgba(0, 123, 255, 0.3)';

            console.log('[schema.js] continueBtn disabled after:', continueBtn.disabled);
            console.log('[schema.js] continueBtn hasAttribute("disabled") after:', continueBtn.hasAttribute('disabled'));

            // Create a click handler function
            const handleNavigate = (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('[schema.js] Continue button clicked! Navigating to cleaning page');
                window.location.href = `/ui/cleaning.html?file=${fileId}`;
            };

            // Remove all existing listeners by cloning and replacing
            const newContinueBtn = continueBtn.cloneNode(true);
            continueBtn.parentNode.replaceChild(newContinueBtn, continueBtn);

            // Re-apply styles to the new button
            newContinueBtn.removeAttribute('disabled');
            newContinueBtn.style.background = '#007bff';
            newContinueBtn.style.cursor = 'pointer';
            newContinueBtn.style.opacity = '1';
            newContinueBtn.style.boxShadow = '0 4px 12px rgba(0, 123, 255, 0.3)';

            // Add click listener to the new button
            newContinueBtn.addEventListener('click', handleNavigate, false);

            console.log('[schema.js] Click event listener attached to continue button');

            // Also set hover handlers
            newContinueBtn.onmouseover = function () {
                this.style.transform = 'translateY(-2px)';
                this.style.boxShadow = '0 6px 16px rgba(0, 123, 255, 0.4)';
            };
            newContinueBtn.onmouseout = function () {
                this.style.transform = 'translateY(0)';
                this.style.boxShadow = '0 4px 12px rgba(0, 123, 255, 0.3)';
            };

            console.log('✅ Schema mapping saved successfully and continue button ready');

            // Auto-redirect after 3 seconds
            let countdown = 3;
            verifyBtn.textContent = `✅ Redirecting in ${countdown}s...`;

            const countdownInterval = setInterval(() => {
                countdown--;
                if (countdown > 0) {
                    verifyBtn.textContent = `✅ Redirecting in ${countdown}s...`;
                } else {
                    clearInterval(countdownInterval);
                    verifyBtn.textContent = '✅ Redirecting...';
                    console.log('[schema.js] Auto-redirecting to cleaning page');
                    window.location.href = `/ui/cleaning.html?file=${fileId}`;
                }
            }, 1000);

        } else {
            throw new Error(data.message || 'Failed to save mapping');
        }

    } catch (error) {
        console.error('Error saving mapping:', error);
        verifyBtn.textContent = '❌ Save Failed';
        verifyBtn.style.background = '#dc3545';
        verifyBtn.disabled = false;

        alert(`Failed to save mapping: ${error.message}\n\nPlease try again.`);

        // Reset button after 2 seconds
        setTimeout(() => {
            verifyBtn.textContent = '✓ Verify & Save Mapping';
            verifyBtn.style.background = '#28a745';
        }, 2000);
    }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', function () {
    console.log('[schema.js] Page loaded, initializing...');
    initSchemaPage();
});

// Export functions
window.SchemaModule = {
    initSchemaPage,
    loadSchemaColumns,
    renderColumnsTable,
    verifyAndSaveMapping,
    getFileId
};
