/**
 * Dashboard Service
 * Manages dashboard data loading, statistics, and pipeline tracking
 */

const DashboardService = {
    /**
     * Initialize dashboard on page load
     */
    async init() {
        await this.loadOverview();
        await this.loadRecentFiles();
        await this.loadQuickCharts();
        this.updatePipelineProgress();
    },

    /**
     * Load overview statistics
     * Counts completed steps, files processed, reports generated
     */
    async loadOverview() {
        const stats = this.calculateStats();

        const statsHTML = `
            <div class="stat-card">
                <div class="stat-icon">📁</div>
                <div class="stat-value">${stats.totalFiles}</div>
                <div class="stat-label">Files Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">✅</div>
                <div class="stat-value">${stats.completedSteps}</div>
                <div class="stat-label">Steps Completed</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div class="stat-value">${stats.analysisRuns}</div>
                <div class="stat-label">Analyses Run</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📄</div>
                <div class="stat-value">${stats.reportsGenerated}</div>
                <div class="stat-label">Reports Generated</div>
            </div>
        `;

        document.getElementById('overview-stats').innerHTML = statsHTML;
    },

    /**
     * Calculate statistics from localStorage
     */
    calculateStats() {
        const fileIds = ApiService.getFileIds();
        const completedSteps = this.countCompletedSteps();

        return {
            totalFiles: fileIds.length,
            completedSteps: completedSteps,
            analysisRuns: parseInt(localStorage.getItem('analysis_count') || '0'),
            reportsGenerated: parseInt(localStorage.getItem('report_count') || '0')
        };
    },

    /**
     * Count completed pipeline steps
     */
    countCompletedSteps() {
        const steps = [
            'step1_complete',
            'step2_complete',
            'step3_complete',
            'step4_complete',
            'step5_complete',
            'step6_complete',
            'step7_complete',
            'step8_complete',
            'step9_complete'
        ];

        return steps.filter(step => localStorage.getItem(step) === 'true').length;
    },

    /**
     * Load recent files with status indicators
     */
    async loadRecentFiles() {
        const fileIds = ApiService.getFileIds();
        const container = document.getElementById('recent-files-list');

        if (fileIds.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📭</div>
                    <p>No files uploaded yet</p>
                    <button onclick="goTo('index.html')" class="btn btn-primary">Upload Your First File</button>
                </div>
            `;
            return;
        }

        let filesHTML = '';

        for (const fileId of fileIds.slice(-5).reverse()) {
            const status = this.getFileStatus(fileId);
            const timestamp = this.getFileTimestamp(fileId);

            filesHTML += `
                <div class="file-item">
                    <div class="file-info">
                        <div class="file-name">📄 Dataset ${fileId.substring(0, 8)}</div>
                        <div class="file-id">${fileId}</div>
                        <div class="file-timestamp">${timestamp}</div>
                    </div>
                    <span class="badge badge-${status.class}">${status.label}</span>
                </div>
            `;
        }

        container.innerHTML = filesHTML;
    },

    /**
     * Determine file processing status
     */
    getFileStatus(fileId) {
        const currentFileId = localStorage.getItem('currentFileId');

        // Check completion flags
        if (localStorage.getItem('step9_complete') === 'true') {
            return { class: 'report', label: 'Report' };
        }
        if (localStorage.getItem('step7_complete') === 'true') {
            return { class: 'ml', label: 'ML' };
        }
        if (localStorage.getItem('step6_complete') === 'true') {
            return { class: 'forecast', label: 'Forecast' };
        }
        if (localStorage.getItem('step5_complete') === 'true') {
            return { class: 'analysis', label: 'Analysis' };
        }
        if (localStorage.getItem('step4_complete') === 'true') {
            return { class: 'weighted', label: 'Weighted' };
        }
        if (localStorage.getItem('step3_complete') === 'true') {
            return { class: 'cleaned', label: 'Cleaned' };
        }
        if (localStorage.getItem('step2_complete') === 'true') {
            return { class: 'schema', label: 'Schema' };
        }

        return { class: 'uploaded', label: 'Uploaded' };
    },

    /**
     * Get file timestamp from localStorage
     */
    getFileTimestamp(fileId) {
        const timestamp = localStorage.getItem(`file_${fileId}_timestamp`);
        if (timestamp) {
            const date = new Date(parseInt(timestamp));
            return date.toLocaleString();
        }
        return 'Recently';
    },

    /**
     * Load quick analytics charts
     * Shows mini summaries of key metrics
     */
    async loadQuickCharts() {
        const container = document.getElementById('quick-charts');
        const fileIds = ApiService.getFileIds();

        if (fileIds.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <p>No analytics available yet</p>
                </div>
            `;
            return;
        }

        try {
            // Try to load cached analysis results
            const currentFileId = localStorage.getItem('currentFileId') || fileIds[0];
            const cachedResults = localStorage.getItem(`analysis_${currentFileId}`);

            if (cachedResults) {
                const data = JSON.parse(cachedResults);
                this.renderMiniCharts(data);
            } else {
                // Load fresh data from API
                const response = await ApiService.runDescriptiveStats(currentFileId);
                this.renderMiniCharts(response);

                // Cache the results
                localStorage.setItem(`analysis_${currentFileId}`, JSON.stringify(response));
            }
        } catch (error) {
            console.error('Error loading quick charts:', error);
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">⚠️</div>
                    <p>Unable to load analytics</p>
                    <button onclick="DashboardService.loadQuickCharts()" class="btn btn-secondary">Retry</button>
                </div>
            `;
        }
    },

    /**
     * Render mini chart summaries
     */
    renderMiniCharts(data) {
        const container = document.getElementById('quick-charts');

        // Extract key metrics
        const results = data.results || data;

        let chartsHTML = '';

        // Show statistics for first few numeric columns
        if (results && typeof results === 'object') {
            const columns = Object.keys(results).slice(0, 6);

            for (const col of columns) {
                const stats = results[col];
                if (stats && typeof stats === 'object') {
                    const mean = stats.mean || stats.Mean || 'N/A';
                    const median = stats.median || stats['50%'] || 'N/A';
                    const std = stats.std || stats['Std Dev'] || 'N/A';

                    chartsHTML += `
                        <div class="mini-chart">
                            <div class="mini-chart-title">${col}</div>
                            <div class="mini-chart-value">${this.formatNumber(mean)}</div>
                            <div class="mini-chart-subtitle">Mean: ${this.formatNumber(mean)} | Median: ${this.formatNumber(median)}</div>
                        </div>
                    `;
                }
            }
        }

        if (chartsHTML === '') {
            chartsHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <p>Run analysis to see insights</p>
                    <button onclick="goTo('analysis.html')" class="btn btn-primary">Go to Analysis</button>
                </div>
            `;
        }

        container.innerHTML = chartsHTML;
    },

    /**
     * Format number for display
     */
    formatNumber(value) {
        if (value === null || value === undefined || value === 'N/A') {
            return 'N/A';
        }
        const num = parseFloat(value);
        if (isNaN(num)) {
            return 'N/A';
        }
        return num.toFixed(2);
    },

    /**
     * Update pipeline progress bar
     */
    updatePipelineProgress() {
        const completedSteps = this.countCompletedSteps();
        const totalSteps = 9;
        const percentage = Math.round((completedSteps / totalSteps) * 100);

        const fill = document.getElementById('pipeline-progress-fill');
        const label = document.getElementById('progress-label');
        const pct = document.getElementById('progress-percentage');

        if (fill) {
            fill.style.width = percentage + '%';
        }

        if (label) {
            if (completedSteps === 0) {
                label.textContent = 'Not started';
            } else if (completedSteps === totalSteps) {
                label.textContent = 'Pipeline complete! 🎉';
            } else {
                label.textContent = `Step ${completedSteps} of ${totalSteps}`;
            }
        }

        if (pct) {
            pct.textContent = percentage + '%';
        }
    },

    /**
     * Restart pipeline
     * Clears all progress flags
     */
    restartPipeline() {
        const confirm = window.confirm(
            'Are you sure you want to restart the pipeline? This will clear all progress but keep your files.'
        );

        if (!confirm) return;

        // Clear step completion flags
        for (let i = 1; i <= 9; i++) {
            localStorage.removeItem(`step${i}_complete`);
        }

        // Reset counters
        localStorage.setItem('analysis_count', '0');
        localStorage.setItem('report_count', '0');

        // Show success message
        showToast('Pipeline restarted successfully', 'success');

        // Reload dashboard
        setTimeout(() => {
            window.location.reload();
        }, 1000);
    },

    /**
     * Download all outputs as ZIP
     * Calls backend API to generate ZIP file
     */
    async downloadAllOutputs() {
        const fileIds = ApiService.getFileIds();

        if (fileIds.length === 0) {
            showToast('No files to download', 'error');
            return;
        }

        try {
            showToast('Preparing download...', 'info');

            const currentFileId = localStorage.getItem('currentFileId') || fileIds[0];

            // Call report download endpoint
            const response = await fetch(`http://127.0.0.1:8000/api/report/download/${currentFileId}`);

            if (!response.ok) {
                throw new Error('Download failed');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `statflow_outputs_${currentFileId.substring(0, 8)}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            showToast('Download started', 'success');
        } catch (error) {
            console.error('Download error:', error);
            showToast('Download failed. Please try again.', 'error');
        }
    }
};

// Auto-initialize dashboard when DOM loads
document.addEventListener('DOMContentLoaded', async () => {
    await DashboardService.init();
});