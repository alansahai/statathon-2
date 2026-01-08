/**
 * StatFlow AI - Pipeline Orchestration
 * 
 * Sequential workflow execution with UI progress tracking
 * 
 * Workflow Steps:
 * 1. Upload → 2. Schema → 3. Cleaning → 4. Weighting
 * → 5. Analysis → 6. Forecasting (opt) → 7. ML (opt)
 * → 8. Insights → 10. Report
 */

const PipelineService = {
    /**
     * =====================================================
     * FULL PIPELINE EXECUTION
     * =====================================================
     */

    async runFullPipeline(options = {}) {
        const {
            includeForecast = false,
            includeML = false,
            reportMode = 'separate'
        } = options;

        try {
            this.updateProgress('Starting pipeline...', 0);

            // Use backend pipeline orchestration
            const result = await ApiService.runFullPipeline(null, {
                includeForecast,
                includeML,
                reportMode
            });

            this.updateProgress('Pipeline completed!', 100);
            return result;

        } catch (error) {
            this.updateProgress('Pipeline failed', 0);
            ApiService.showError(`Pipeline error: ${error.message}`);
            throw error;
        }
    },

    /**
     * =====================================================
     * MANUAL STEP-BY-STEP PIPELINE
     * =====================================================
     */

    async runManualPipeline(options = {}) {
        const {
            includeForecast = false,
            includeML = false,
            reportMode = 'separate',
            schemaMapping = null
        } = options;

        const steps = [];

        try {
            // STEP 1: Schema Validation (if mapping provided)
            if (schemaMapping) {
                this.updateProgress('Step 1/8: Validating schema...', 10);
                const fileIds = ApiService.getFileIds();

                for (const fileId of fileIds) {
                    await ApiService.saveSchemaMapping(fileId, schemaMapping);
                    await ApiService.applySchemaMapping(fileId);
                }

                steps.push({ step: 'schema', status: 'completed' });
            }

            // STEP 2: Data Cleaning
            this.updateProgress('Step 2/8: Cleaning data...', 25);
            const cleaningResult = await ApiService.autoClean();
            steps.push({ step: 'cleaning', status: 'completed', result: cleaningResult });

            // STEP 3: Weighting
            this.updateProgress('Step 3/8: Calculating weights...', 40);
            const weightingResult = await ApiService.calculateWeights();
            steps.push({ step: 'weighting', status: 'completed', result: weightingResult });

            // STEP 4: Analysis
            this.updateProgress('Step 4/8: Running analysis...', 55);

            // Get columns for analysis
            const fileIds = ApiService.getFileIds();
            const firstFileId = fileIds[0];
            const columnsData = await ApiService.getColumns(firstFileId);

            // Run descriptive stats on numeric columns
            const analysisResult = await ApiService.runDescriptiveStats(
                null,
                columnsData.columns.slice(0, 10) // First 10 columns
            );
            steps.push({ step: 'analysis', status: 'completed', result: analysisResult });

            // STEP 5: Forecasting (optional)
            if (includeForecast) {
                this.updateProgress('Step 5/8: Generating forecasts...', 65);
                const forecastResult = await ApiService.runForecast(null, {
                    method: 'auto',
                    periods: 12
                });
                steps.push({ step: 'forecasting', status: 'completed', result: forecastResult });
            } else {
                steps.push({ step: 'forecasting', status: 'skipped' });
            }

            // STEP 6: Machine Learning (optional)
            if (includeML) {
                this.updateProgress('Step 6/8: Running ML models...', 75);
                const mlResult = await ApiService.runPCA(null, {
                    n_components: 2
                });
                steps.push({ step: 'ml', status: 'completed', result: mlResult });
            } else {
                steps.push({ step: 'ml', status: 'skipped' });
            }

            // STEP 7: Insights
            this.updateProgress('Step 7/8: Generating insights...', 85);
            const insightsResult = await ApiService.getInsights();
            steps.push({ step: 'insights', status: 'completed', result: insightsResult });

            // STEP 8: Report Generation
            this.updateProgress('Step 8/8: Creating report...', 95);
            const reportResult = await ApiService.generateReport(null, reportMode);
            steps.push({ step: 'report', status: 'completed', result: reportResult });

            // Complete
            this.updateProgress('Pipeline completed successfully!', 100);
            ApiService.showSuccess('All pipeline steps completed');

            return {
                status: 'completed',
                steps
            };

        } catch (error) {
            this.updateProgress('Pipeline failed', 0);
            ApiService.showError(`Pipeline step failed: ${error.message}`);

            return {
                status: 'failed',
                steps,
                error: error.message
            };
        }
    },

    /**
     * =====================================================
     * MINIMAL PIPELINE (NO FORECAST/ML)
     * =====================================================
     */

    async runMinimalPipeline() {
        try {
            this.updateProgress('Starting minimal pipeline...', 0);

            // STEP 1: Cleaning
            this.updateProgress('Step 1/3: Cleaning data...', 33);
            await ApiService.autoClean();

            // STEP 2: Analysis
            this.updateProgress('Step 2/3: Running analysis...', 66);
            const fileIds = ApiService.getFileIds();
            const columnsData = await ApiService.getColumns(fileIds[0]);
            await ApiService.runDescriptiveStats(null, columnsData.columns.slice(0, 10));

            // STEP 3: Insights
            this.updateProgress('Step 3/3: Generating insights...', 100);
            await ApiService.getInsights();

            ApiService.showSuccess('Minimal pipeline completed');

        } catch (error) {
            ApiService.showError(`Minimal pipeline failed: ${error.message}`);
            throw error;
        }
    },

    /**
     * =====================================================
     * UI PROGRESS HELPERS
     * =====================================================
     */

    updateProgress(text, percentage) {
        console.log(`[Pipeline] ${percentage}% - ${text}`);

        // Update progress bar
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
        }

        // Update progress text
        const progressText = document.querySelector('.progress-text');
        if (progressText) {
            progressText.textContent = text;
        }

        // Update step indicators
        this.updateStepIndicator(text);
    },

    updateStepIndicator(stepText) {
        const indicators = document.querySelectorAll('.step-indicator');

        // Map step text to indicator classes
        const stepMap = {
            'schema': 0,
            'cleaning': 1,
            'weighting': 2,
            'analysis': 3,
            'forecasting': 4,
            'ml': 5,
            'insights': 6,
            'report': 7
        };

        // Find which step we're on
        let currentStep = -1;
        for (const [key, index] of Object.entries(stepMap)) {
            if (stepText.toLowerCase().includes(key)) {
                currentStep = index;
                break;
            }
        }

        // Update indicator states
        indicators.forEach((indicator, index) => {
            if (index < currentStep) {
                indicator.classList.add('completed');
                indicator.classList.remove('active');
            } else if (index === currentStep) {
                indicator.classList.add('active');
                indicator.classList.remove('completed');
            } else {
                indicator.classList.remove('active', 'completed');
            }
        });
    },

    /**
     * =====================================================
     * NAVIGATION HELPERS
     * =====================================================
     */

    goToNextStep(currentPage) {
        const flowMap = {
            'index': 'schema.html',
            'schema': 'cleaning.html',
            'cleaning': 'weighting.html',
            'weighting': 'analysis.html',
            'analysis': 'forecasting.html',
            'forecasting': 'ml.html',
            'ml': 'insight.html',
            'insight': 'report.html'
        };

        // Mark current step complete
        const stepMap = {
            'index.html': 'upload',
            'schema.html': 'schema',
            'cleaning.html': 'cleaning',
            'weighting.html': 'weighting',
            'analysis.html': 'analysis',
            'forecasting.html': 'forecasting',
            'ml.html': 'ml',
            'insight.html': 'insights'
        };

        const stepName = stepMap[currentPage];
        if (stepName && typeof markStepComplete === 'function') {
            markStepComplete(stepName);
        }

        const nextPage = flowMap[currentPage];
        if (nextPage) {
            window.location.href = nextPage;
        }
    },

    goToPreviousStep(currentPage) {
        const reverseFlowMap = {
            'schema': 'index.html',
            'cleaning': 'schema.html',
            'weighting': 'cleaning.html',
            'analysis': 'weighting.html',
            'forecasting': 'analysis.html',
            'ml': 'forecasting.html',
            'insight': 'ml.html',
            'report': 'insight.html'
        };

        const prevPage = reverseFlowMap[currentPage];
        if (prevPage) {
            window.location.href = prevPage;
        }
    },

    /**
     * =====================================================
     * VALIDATION HELPERS
     * =====================================================
     */

    validateFileIds() {
        const fileIds = ApiService.getFileIds();

        if (fileIds.length === 0) {
            ApiService.showError('No files uploaded. Please upload files first.');
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 2000);
            return false;
        }

        return true;
    },

    displayFileInfo() {
        const fileIds = ApiService.getFileIds();
        const fileInfoElement = document.querySelector('.file-info');

        if (fileInfoElement && fileIds.length > 0) {
            fileInfoElement.innerHTML = `
                <div class="alert alert-info">
                    <strong>Active Files:</strong> ${fileIds.length} file(s) loaded
                    <button class="btn btn-sm btn-secondary" onclick="PipelineService.showFileDetails()">
                        View Details
                    </button>
                </div>
            `;
        }
    },

    showFileDetails() {
        const fileIds = ApiService.getFileIds();
        const detailsHtml = fileIds.map((fid, idx) =>
            `<li>${idx + 1}. File ID: ${fid}</li>`
        ).join('');

        ApiService.showToast(`Files loaded:\n${detailsHtml}`, 'info');
    },

    /**
     * =====================================================
     * PROGRESS TRACKING WITH PERCENTAGES
     * =====================================================
     */

    /**
     * Update step with label and percentage
     * @param {string} label - Step label (e.g., "Schema Mapping...")
     * @param {number} percent - Progress percentage (0-100)
     */
    async updateStep(label, percent) {
        // Update status text
        const statusElement = document.getElementById("pipeline-status");
        if (statusElement) {
            statusElement.innerText = label;
        }

        // Update progress bar using app.js function
        if (typeof setProgress === 'function') {
            setProgress(percent, label);
        } else {
            // Fallback to direct update
            this.updateProgress(label, percent);
        }

        // Show progress container if hidden
        if (typeof showProgress === 'function') {
            showProgress();
        }
    },

    /**
     * =====================================================
     * REAL-TIME PIPELINE STATUS POLLING
     * =====================================================
     */

    /**
     * Poll pipeline status at regular intervals
     * @param {string} fileId - File ID to check status for
     * @param {number} intervalMs - Polling interval in milliseconds
     * @returns {number} Interval ID for stopping polling
     */
    pollPipelineStatus(fileId, intervalMs = 2000) {
        let pollCount = 0;
        const maxPolls = 150; // 5 minutes at 2-second intervals

        const intervalId = setInterval(async () => {
            pollCount++;

            try {
                const res = await ApiService.getPipelineStatus(fileId);

                // Update progress based on response
                if (res && res.stage) {
                    const label = res.current_stage_label || res.stage;
                    const percent = res.progress_percent || 0;

                    this.updateStep(label, percent);

                    // Stop polling if completed or failed
                    if (res.stage === 'completed' || res.stage === 'failed' || percent >= 100) {
                        clearInterval(intervalId);

                        if (res.stage === 'completed') {
                            if (typeof showToastSuccess === 'function') {
                                showToastSuccess('Pipeline completed successfully!');
                            }
                        } else if (res.stage === 'failed') {
                            if (typeof showToastError === 'function') {
                                showToastError('Pipeline failed. Check logs.');
                            }
                        }
                    }
                }

                // Safety: Stop polling after max attempts
                if (pollCount >= maxPolls) {
                    clearInterval(intervalId);
                    console.warn('Pipeline polling timeout after 5 minutes');
                }

            } catch (error) {
                console.error('Error polling pipeline status:', error);
                // Don't stop polling on error - might be transient
            }
        }, intervalMs);

        return intervalId; // Return so caller can stop polling if needed
    },

    /**
     * Stop polling by interval ID
     * @param {number} intervalId - Interval ID returned from pollPipelineStatus
     */
    stopPolling(intervalId) {
        if (intervalId) {
            clearInterval(intervalId);
            console.log('Pipeline polling stopped');
        }
    },

    /**
     * =====================================================
     * STEP PROGRESS PERCENTAGES
     * =====================================================
     */

    /**
     * Get progress percentage for each workflow step
     */
    getStepPercentages() {
        return {
            upload: 5,
            schema: 15,
            cleaning: 25,
            weighting: 40,
            analysis: 60,
            forecasting: 75,
            ml: 85,
            insights: 90,
            report: 100
        };
    },

    /**
     * Update progress for a named step
     * @param {string} stepName - Step name (upload, schema, cleaning, etc.)
     */
    updateStepByName(stepName) {
        const percentages = this.getStepPercentages();
        const percent = percentages[stepName] || 0;
        const label = this.getStepLabel(stepName);

        this.updateStep(label, percent);
    },

    /**
     * Get human-readable label for step
     * @param {string} stepName - Step name
     * @returns {string} Formatted label
     */
    getStepLabel(stepName) {
        const labels = {
            upload: 'Uploading files...',
            schema: 'Schema mapping...',
            cleaning: 'Cleaning data...',
            weighting: 'Calculating weights...',
            analysis: 'Running analysis...',
            forecasting: 'Forecasting...',
            ml: 'Machine learning...',
            insights: 'Generating insights...',
            report: 'Creating report...'
        };

        return labels[stepName] || stepName;
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Validate file IDs for pages that require them
    const currentPage = window.location.pathname.split('/').pop().replace('.html', '');
    const requiresFileIds = ['schema', 'cleaning', 'weighting', 'analysis', 'forecast', 'ml', 'insight', 'report'];

    if (requiresFileIds.includes(currentPage)) {
        if (!PipelineService.validateFileIds()) {
            return;
        }
        PipelineService.displayFileInfo();
    }
});
