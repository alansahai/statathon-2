/**
 * StatFlow AI - Workflow Navigation Module
 * Auto-progression pipeline helper functions
 * 
 * Manages file_id propagation and page navigation across the analysis workflow:
 * Upload → Schema → Cleaning → Weighting → Analysis → Insights → Report
 */

if (!window.API_BASE) console.error("API_BASE not ready");

// Workflow order definition
const WORKFLOW_ORDER = [
    "index.html",
    "schema.html",
    "cleaning.html",
    "weighting.html",
    "analysis.html",
    "insight.html",
    "report.html"
];

// Core workflow functions
function getFileId() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('file');
}

function validateFileId() {
    const fileId = getFileId();
    if (!fileId || fileId === "" || fileId === "undefined") {
        window.location.href = "index.html";
        return null;
    }
    return fileId;
}

function buildUrl(page, fileId) {
    return `${page}?file=${encodeURIComponent(fileId)}`;
}

function getCurrentPage() {
    const path = window.location.pathname;
    return path.substring(path.lastIndexOf("/") + 1).toLowerCase();
}

function getNextStep(currentPage) {
    const index = WORKFLOW_ORDER.indexOf(currentPage);
    if (index === -1 || index === WORKFLOW_ORDER.length - 1) return null;
    return WORKFLOW_ORDER[index + 1];
}

function getPreviousStep(currentPage) {
    const index = WORKFLOW_ORDER.indexOf(currentPage);
    if (index === -1 || index === 0) return null;
    return WORKFLOW_ORDER[index - 1];
}

function getWorkflowProgress(page) {
    const index = WORKFLOW_ORDER.indexOf(page);
    if (index === -1) return 0;
    return Math.round((index / (WORKFLOW_ORDER.length - 1)) * 100);
}

function autoProgressToNext(currentPage, delayMs = 1500) {
    const fileId = validateFileId();
    const nextPage = getNextStep(currentPage);
    if (!fileId || !nextPage) return;
    setTimeout(() => window.location.href = buildUrl(nextPage, fileId), delayMs);
}

// Navigation helper functions
function goToUpload() {
    window.location.href = "index.html";
}

function goToSchema(fileId = null) {
    const id = fileId || getFileId();
    if (!id) {
        goToUpload();
        return;
    }
    window.location.href = buildUrl("schema.html", id);
}

function goToCleaning(fileId = null) {
    const id = fileId || getFileId();
    if (!id) {
        goToUpload();
        return;
    }
    window.location.href = buildUrl("cleaning.html", id);
}

function goToWeighting(fileId = null) {
    const id = fileId || getFileId();
    if (!id) {
        goToUpload();
        return;
    }
    window.location.href = buildUrl("weighting.html", id);
}

function goToAnalysis(fileId = null) {
    const id = fileId || getFileId();
    if (!id) {
        goToUpload();
        return;
    }
    window.location.href = buildUrl("analysis.html", id);
}

function goToInsights(fileId = null) {
    const id = fileId || getFileId();
    if (!id) {
        goToUpload();
        return;
    }
    window.location.href = buildUrl("insight.html", id);
}

function goToReport(fileId = null) {
    const id = fileId || getFileId();
    if (!id) {
        goToUpload();
        return;
    }
    window.location.href = buildUrl("report.html", id);
}

function goToDashboard(fileId = null) {
    const id = fileId || getFileId();
    if (!id) {
        goToUpload();
        return;
    }
    window.location.href = buildUrl("dashboard.html", id);
}

function goToPrevious(currentPage) {
    const previousPage = getPreviousStep(currentPage);
    if (previousPage) {
        if (previousPage === "index.html") {
            goToUpload();
        } else {
            const fileId = getFileId();
            if (fileId) {
                window.location.href = buildUrl(previousPage, fileId);
            } else {
                goToUpload();
            }
        }
    }
}

// Display functions
function displayFileInfo() {
    const fileId = getFileId();
    const el = document.getElementById("fileInfo");
    if (el && fileId) {
        el.innerText = `File ID: ${fileId}`;
    }

    // Also display in .file-info container if exists
    const container = document.querySelector('.file-info');
    if (container && fileId) {
        const currentPage = getCurrentPage();
        const progress = getWorkflowProgressDetails(currentPage);

        container.innerHTML = `
            <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid var(--primary-color);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: var(--primary-color);">📁 File ID:</strong> 
                        <code style="background: white; padding: 0.25rem 0.5rem; border-radius: 4px;">${fileId}</code>
                    </div>
                    <div>
                        <strong style="color: var(--primary-color);">Step:</strong> 
                        <span style="background: var(--primary-color); color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem;">
                            ${progress.currentStep?.icon || ''} ${progress.currentStep?.name || 'Unknown'} (${progress.currentIndex + 1}/${progress.totalSteps})
                        </span>
                    </div>
                </div>
            </div>
        `;
    }
}

function showProgress() {
    const page = getCurrentPage();
    const pct = getWorkflowProgress(page);
    const el = document.getElementById("progress");
    if (el) el.innerText = `Progress: ${pct}%`;
}

function getWorkflowProgressDetails(currentPage) {
    const steps = [
        { name: 'Upload', page: 'index.html', icon: '📤' },
        { name: 'Schema', page: 'schema.html', icon: '🔍' },
        { name: 'Cleaning', page: 'cleaning.html', icon: '🧹' },
        { name: 'Weighting', page: 'weighting.html', icon: '⚖️' },
        { name: 'Analysis', page: 'analysis.html', icon: '📊' },
        { name: 'Insights', page: 'insight.html', icon: '🧠' },
        { name: 'Report', page: 'report.html', icon: '📄' }
    ];

    const currentIndex = steps.findIndex(step => step.page === currentPage);

    return {
        steps,
        currentStep: currentIndex >= 0 ? steps[currentIndex] : null,
        currentIndex,
        totalSteps: steps.length,
        progress: currentIndex >= 0 ? ((currentIndex + 1) / steps.length * 100).toFixed(0) : 0,
        isFirstStep: currentIndex === 0,
        isLastStep: currentIndex === steps.length - 1,
        hasNext: currentIndex < steps.length - 1,
        hasPrevious: currentIndex > 0
    };
}

function logWorkflowProgress(currentPage) {
    const progress = getWorkflowProgressDetails(currentPage);

    console.log('=== Workflow Progress ===');
    console.log(`Current Step: ${progress.currentStep?.name} (${progress.currentIndex + 1}/${progress.totalSteps})`);
    console.log(`Progress: ${progress.progress}%`);
    console.log(`File ID: ${getFileId()}`);
    console.log('========================');
}

// Make WorkflowService globally available
window.WorkflowService = {
    getFileId,
    validateFileId,
    buildUrl,
    getCurrentPage,
    getNextStep,
    getPreviousStep,
    getWorkflowProgress,
    autoProgressToNext,
    goToUpload,
    goToSchema,
    goToCleaning,
    goToWeighting,
    goToAnalysis,
    goToInsights,
    goToReport,
    goToDashboard,
    goToPrevious,
    displayFileInfo,
    showProgress,
    logWorkflowProgress,
    WORKFLOW_ORDER
};

// Auto-display file info and progress on page load
window.addEventListener("DOMContentLoaded", () => {
    displayFileInfo();
    showProgress();
});

// Legacy compatibility - keep old object structure
const LegacyWorkflowService = {
    /**
     * Get file_id from URL query parameter
     * @returns {string|null} The file_id or null if not found
     */
    getFileId() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('file');
    },

    /**
     * Validate that file_id exists in URL
     * Redirects to upload page if not found
     * @param {boolean} showAlert - Whether to show alert before redirect (default: true)
     * @returns {string|null} The file_id or null if validation fails
     */
    validateFileId(showAlert = true) {
        const fileId = this.getFileId();

        if (!fileId) {
            if (showAlert) {
                alert('No file ID provided. Redirecting to upload page...');
            }
            this.goToUpload();
            return null;
        }

        return fileId;
    },

    /**
     * Build URL with file_id parameter
     * @param {string} page - The page name (e.g., 'schema.html')
     * @param {string} fileId - The file_id to append
     * @returns {string} Complete URL with file parameter
     */
    buildUrl(page, fileId) {
        return `${page}?file=${encodeURIComponent(fileId)}`;
    },

    /**
     * Navigate to a specific page with current file_id
     * @param {string} page - The page to navigate to
     * @param {string} fileId - Optional file_id (uses current if not provided)
     */
    navigateTo(page, fileId = null) {
        const id = fileId || this.getFileId();

        if (!id) {
            console.error('No file_id available for navigation');
            this.goToUpload();
            return;
        }

        window.location.href = this.buildUrl(page, id);
    },

    // ============================================
    // NAVIGATION HELPER FUNCTIONS
    // ============================================

    /**
     * Navigate to Upload page (index.html)
     * This is the starting point - no file_id needed
     */
    goToUpload() {
        window.location.href = 'index.html';
    },

    /**
     * Navigate to Schema Detection page
     * Step 1: Auto-detect column types
     * @param {string} fileId - Optional file_id
     */
    goToSchema(fileId = null) {
        this.navigateTo('schema.html', fileId);
    },

    /**
     * Navigate to Data Cleaning page
     * Step 2: Auto-clean missing values and outliers
     * @param {string} fileId - Optional file_id
     */
    goToCleaning(fileId = null) {
        this.navigateTo('cleaning.html', fileId);
    },

    /**
     * Navigate to Survey Weighting page
     * Step 3: Calculate and apply survey weights
     * @param {string} fileId - Optional file_id
     */
    goToWeighting(fileId = null) {
        this.navigateTo('weighting.html', fileId);
    },

    /**
     * Navigate to Statistical Analysis page
     * Step 4: Run descriptive statistics
     * @param {string} fileId - Optional file_id
     */
    goToAnalysis(fileId = null) {
        this.navigateTo('analysis.html', fileId);
    },

    /**
     * Navigate to AI Insights page
     * Step 5: Generate AI-powered insights
     * @param {string} fileId - Optional file_id
     */
    goToInsights(fileId = null) {
        this.navigateTo('insight.html', fileId);
    },

    /**
     * Navigate to Report Generation page
     * Step 6: Generate and download PDF report
     * @param {string} fileId - Optional file_id
     */
    goToReport(fileId = null) {
        this.navigateTo('report.html', fileId);
    },

    /**
     * Navigate to Dashboard
     * @param {string} fileId - Optional file_id
     */
    goToDashboard(fileId = null) {
        this.navigateTo('dashboard.html', fileId);
    },

    // ============================================
    // AUTO-PROGRESSION WORKFLOW
    // ============================================

    /**
     * Get the next step in the workflow
     * @param {string} currentPage - Current page name (e.g., 'schema.html')
     * @returns {string|null} Next page name or null if at end
     */
    getNextStep(currentPage) {
        const workflow = {
            'index.html': 'schema.html',
            'schema.html': 'cleaning.html',
            'cleaning.html': 'weighting.html',
            'weighting.html': 'analysis.html',
            'analysis.html': 'insight.html',
            'insight.html': 'report.html',
            'report.html': null // Final step
        };

        return workflow[currentPage] || null;
    },

    /**
     * Get the previous step in the workflow
     * @param {string} currentPage - Current page name
     * @returns {string|null} Previous page name or null if at start
     */
    getPreviousStep(currentPage) {
        const workflow = {
            'schema.html': 'index.html',
            'cleaning.html': 'schema.html',
            'weighting.html': 'cleaning.html',
            'analysis.html': 'weighting.html',
            'insight.html': 'analysis.html',
            'report.html': 'insight.html'
        };

        return workflow[currentPage] || null;
    },

    /**
     * Auto-redirect to next step after delay
     * @param {string} currentPage - Current page name
     * @param {number} delayMs - Delay in milliseconds (default: 2000)
     */
    autoProgressToNext(currentPage, delayMs = 2000) {
        const nextPage = this.getNextStep(currentPage);

        if (nextPage) {
            setTimeout(() => {
                this.navigateTo(nextPage);
            }, delayMs);
        }
    },

    /**
     * Navigate to previous step
     * @param {string} currentPage - Current page name
     */
    goToPrevious(currentPage) {
        const previousPage = this.getPreviousStep(currentPage);

        if (previousPage) {
            if (previousPage === 'index.html') {
                this.goToUpload();
            } else {
                this.navigateTo(previousPage);
            }
        }
    },

    // ============================================
    // WORKFLOW STATUS TRACKING
    // ============================================

    /**
     * Get workflow progress information
     * @param {string} currentPage - Current page name
     * @returns {object} Progress information
     */
    getWorkflowProgress(currentPage) {
        const steps = [
            { name: 'Upload', page: 'index.html', icon: '📤' },
            { name: 'Schema', page: 'schema.html', icon: '🔍' },
            { name: 'Cleaning', page: 'cleaning.html', icon: '🧹' },
            { name: 'Weighting', page: 'weighting.html', icon: '⚖️' },
            { name: 'Analysis', page: 'analysis.html', icon: '📊' },
            { name: 'Insights', page: 'insight.html', icon: '🧠' },
            { name: 'Report', page: 'report.html', icon: '📄' }
        ];

        const currentIndex = steps.findIndex(step => step.page === currentPage);

        return {
            steps,
            currentStep: currentIndex >= 0 ? steps[currentIndex] : null,
            currentIndex,
            totalSteps: steps.length,
            progress: currentIndex >= 0 ? ((currentIndex + 1) / steps.length * 100).toFixed(0) : 0,
            isFirstStep: currentIndex === 0,
            isLastStep: currentIndex === steps.length - 1,
            hasNext: currentIndex < steps.length - 1,
            hasPrevious: currentIndex > 0
        };
    },

    /**
     * Display workflow progress in console (for debugging)
     * @param {string} currentPage - Current page name
     */
    logWorkflowProgress(currentPage) {
        const progress = this.getWorkflowProgress(currentPage);

        console.log('=== Workflow Progress ===');
        console.log(`Current Step: ${progress.currentStep?.name} (${progress.currentIndex + 1}/${progress.totalSteps})`);
        console.log(`Progress: ${progress.progress}%`);
        console.log(`File ID: ${this.getFileId()}`);
        console.log('========================');
    },

    // ============================================
    // UTILITY FUNCTIONS
    // ============================================

    /**
     * Extract current page name from window.location
     * @returns {string} Current page name (e.g., 'schema.html')
     */
    getCurrentPage() {
        const path = window.location.pathname;
        return path.split('/').pop() || 'index.html';
    },

    /**
     * Check if currently on a specific page
     * @param {string} pageName - Page name to check
     * @returns {boolean} True if on the specified page
     */
    isCurrentPage(pageName) {
        return this.getCurrentPage() === pageName;
    },

    /**
     * Display file information in designated container
     * Shows file_id and current workflow step
     */
    displayFileInfo() {
        const fileId = this.getFileId();
        const container = document.querySelector('.file-info');

        if (container && fileId) {
            const currentPage = this.getCurrentPage();
            const progress = this.getWorkflowProgress(currentPage);

            container.innerHTML = `
                <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid var(--primary-color);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: var(--primary-color);">📁 File ID:</strong> 
                            <code style="background: white; padding: 0.25rem 0.5rem; border-radius: 4px;">${fileId}</code>
                        </div>
                        <div>
                            <strong style="color: var(--primary-color);">Step:</strong> 
                            <span style="background: var(--primary-color); color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem;">
                                ${progress.currentStep?.icon || ''} ${progress.currentStep?.name || 'Unknown'} (${progress.currentIndex + 1}/${progress.totalSteps})
                            </span>
                        </div>
                    </div>
                </div>
            `;
        }
    },

    /**
     * Get column suggestions for AI assistant
     * @param {string} fileId - File ID
     * @returns {Promise<Array>} Array of column names
     */
    async getColumnSuggestions(fileId) {
        try {
            const url = `${window.API_BASE}/upload/columns/${fileId}`;
            console.log("Calling:", url);
            const response = await fetch(url);
            const data = await response.json();
            return data.columns || [];
        } catch (error) {
            console.error('Failed to load columns for suggestions:', error);
            return [];
        }
    },

    /**
     * Preload assistant data for current file
     * @param {string} fileId - File ID
     * @returns {Promise<Object>} Object with columns and metadata
     */
    async preloadAssistantData(fileId) {
        try {
            const columns = await this.getColumnSuggestions(fileId);
            return {
                columns,
                fileId,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('Failed to preload assistant data:', error);
            return { columns: [], fileId, timestamp: new Date().toISOString() };
        }
    },

    /**
     * Start full pipeline execution
     * @param {string} fileId - File ID to process
     * @returns {Promise<Object>} Pipeline execution result
     */
    async startFullPipeline(fileId) {
        try {
            const url = `${window.API_BASE}/pipeline/execute`;
            console.log("Calling:", url);
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_id: fileId })
            });

            if (!response.ok) {
                throw new Error(`Pipeline execution failed: ${response.statusText}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Failed to start full pipeline:', error);
            throw error;
        }
    },

    /**
     * Get pipeline status
     * @param {string} fileId - File ID
     * @returns {Promise<Object>} Pipeline status
     */
    async getPipelineStatus(fileId) {
        try {
            const url = `${window.API_BASE}/pipeline/status/${fileId}`;
            console.log("Calling:", url);
            const response = await fetch(url);

            if (!response.ok) {
                throw new Error(`Failed to get pipeline status: ${response.statusText}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Failed to get pipeline status:', error);
            return { status: 'unknown', error: error.message };
        }
    },

    /**
     * Watch pipeline status with polling
     * @param {string} fileId - File ID
     * @param {number} intervalMs - Polling interval in milliseconds (default: 2000)
     * @param {Function} onUpdate - Callback function for status updates
     * @returns {Function} Stop function to cancel polling
     */
    watchPipelineStatus(fileId, intervalMs = 2000, onUpdate) {
        let isRunning = true;

        const poll = async () => {
            if (!isRunning) return;

            try {
                const status = await this.getPipelineStatus(fileId);

                if (onUpdate && typeof onUpdate === 'function') {
                    onUpdate(status);
                }

                // Continue polling if pipeline is still running
                if (status.status === 'running' && isRunning) {
                    setTimeout(poll, intervalMs);
                }
            } catch (error) {
                console.error('Pipeline status polling error:', error);
                if (isRunning) {
                    setTimeout(poll, intervalMs);
                }
            }
        };

        // Start polling
        poll();

        // Return stop function
        return () => {
            isRunning = false;
        };
    },

    /**
     * Connect to pipeline WebSocket
     * @param {string} fileId - File ID
     * @returns {WebSocket} WebSocket connection
     */
    connectPipelineWS(fileId) {
        if (typeof PipelineWebSocket === 'undefined') {
            console.error('PipelineWebSocket not loaded');
            return null;
        }

        const ws = new PipelineWebSocket();
        ws.connect(fileId);
        return ws;
    },

    /**
     * Disconnect pipeline WebSocket
     * @param {WebSocket} ws - WebSocket connection to disconnect
     */
    disconnectPipelineWS(ws) {
        if (ws && typeof ws.disconnect === 'function') {
            ws.disconnect();
        }
    },

    /**
     * Check if WebSocket is connected
     * @param {WebSocket} ws - WebSocket connection
     * @returns {boolean} Connection status
     */
    isWSConnected(ws) {
        return ws && typeof ws.isConnected === 'function' && ws.isConnected();
    },

    /**
     * Preload data for assistant context
     * @param {string} fileId - The file ID to load data for
     * @returns {Promise<Object>} File data including columns
     */
    async preloadAssistantData(fileId) {
        try {
            // Try to get columns from backend
            const response = await fetch(`${window.API_BASE}/upload/columns/${fileId}`);
            if (response.ok) {
                const data = await response.json();
                return { columns: data.columns || [] };
            }
        } catch (error) {
            console.warn('Could not load columns for assistant:', error);
        }
        return { columns: [] };
    }
};

// Make WorkflowService globally available
window.WorkflowService = WorkflowService;
