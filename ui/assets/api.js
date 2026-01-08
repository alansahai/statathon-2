/**
 * StatFlow AI - API Service Layer
 * 
 * Unified interface for all backend API calls with:
 * - Multi-file support (1-5 files)
 * - LocalStorage persistence for file_ids
 * - Consistent error handling
 * - Loading state management
 * - Toast notifications
 * 
 * UI Flow:
 * index.html → schema.html → cleaning.html → weighting.html
 * → analysis.html → (forecast/ml optional) → insights.html → report.html
 */

const API_BASE = "http://127.0.0.1:8000/api";

const ApiService = {
    /**
     * =====================================================
     * FILE ID MANAGEMENT (LocalStorage)
     * =====================================================
     */

    setFileIds(fileIds) {
        if (!Array.isArray(fileIds)) {
            fileIds = [fileIds];
        }
        localStorage.setItem('statflow_file_ids', JSON.stringify(fileIds));
        console.log('[API] File IDs stored:', fileIds);
    },

    getFileIds() {
        const stored = localStorage.getItem('statflow_file_ids');
        if (!stored) return [];
        try {
            return JSON.parse(stored);
        } catch (e) {
            console.error('[API] Failed to parse stored file_ids:', e);
            return [];
        }
    },

    clearFileIds() {
        localStorage.removeItem('statflow_file_ids');
        console.log('[API] File IDs cleared');
    },

    hasFileIds() {
        return this.getFileIds().length > 0;
    },

    /**
     * =====================================================
     * CORE REQUEST HANDLER
     * =====================================================
     */

    async request(path, method = 'GET', body = null, isFormData = false) {
        const url = `${API_BASE}${path}`;

        const options = {
            method,
            headers: {}
        };

        // Add body for non-GET requests
        if (body && method !== 'GET') {
            if (isFormData) {
                options.body = body; // FormData sets its own Content-Type
            } else {
                options.headers['Content-Type'] = 'application/json';
                options.body = JSON.stringify(body);
            }
        }

        try {
            this.setLoading(true);
            console.log(`[API] ${method} ${path}`, body ? body : '');

            const response = await fetch(url, options);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || `API Error: ${response.status}`);
            }

            console.log(`[API] Response:`, data);
            this.setLoading(false);

            return data;

        } catch (error) {
            this.setLoading(false);
            console.error(`[API] Request failed:`, error);
            this.showError(error.message);
            throw error;
        }
    },

    /**
     * =====================================================
     * STEP 01: UPLOAD
     * =====================================================
     */

    async uploadSingleFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const result = await this.request('/upload/single', 'POST', formData, true);

        if (result.status === 'success' && result.file_ids) {
            this.setFileIds(result.file_ids);
            this.showSuccess(`File uploaded: ${file.name}`);
        }

        return result;
    },

    async uploadMultipleFiles(files) {
        const formData = new FormData();

        for (const file of files) {
            formData.append('files', file);
        }

        const result = await this.request('/upload/multiple', 'POST', formData, true);

        if (result.status === 'success' && result.file_ids) {
            this.setFileIds(result.file_ids);
            this.showSuccess(`${result.file_ids.length} files uploaded successfully`);
        }

        return result;
    },

    /**
     * =====================================================
     * STEP 02: SCHEMA MAPPING
     * =====================================================
     */

    async getColumns(fileId = null) {
        const fid = fileId || this.getFileIds()[0];
        if (!fid) {
            throw new Error('No file_id available. Please upload a file first.');
        }

        return await this.request(`/schema/columns/${fid}`, 'GET');
    },

    async saveSchemaMapping(fileId, mapping) {
        const fid = fileId || this.getFileIds()[0];
        if (!fid) {
            throw new Error('No file_id available');
        }

        const result = await this.request(`/schema/save/${fid}`, 'POST', { mapping });

        if (result.status === 'success') {
            this.showSuccess('Schema mapping saved successfully');
        }

        return result;
    },

    async applySchemaMapping(fileId = null) {
        const fid = fileId || this.getFileIds()[0];
        if (!fid) {
            throw new Error('No file_id available');
        }

        const result = await this.request(`/schema/apply/${fid}`, 'POST');

        if (result.status === 'success') {
            this.showSuccess('Schema applied successfully');
        }

        return result;
    },

    /**
     * =====================================================
     * STEP 03: CLEANING
     * =====================================================
     */

    async detectIssues(fileIds = null) {
        const fids = fileIds || this.getFileIds();

        const result = await this.request('/cleaning/detect-issues', 'POST', {
            file_ids: fids
        });

        return result;
    },

    async autoClean(fileIds = null) {
        const fids = fileIds || this.getFileIds();

        const result = await this.request('/cleaning/auto-clean', 'POST', {
            file_ids: fids
        });

        if (result.status === 'success') {
            this.showSuccess('Data cleaned successfully');
        }

        return result;
    },

    async manualClean(fileIds = null, rules = {}) {
        const fids = fileIds || this.getFileIds();

        const result = await this.request('/cleaning/manual-clean', 'POST', {
            file_ids: fids,
            ...rules
        });

        if (result.status === 'success') {
            this.showSuccess('Manual cleaning applied');
        }

        return result;
    },

    /**
     * =====================================================
     * STEP 04: WEIGHTING
     * =====================================================
     */

    async calculateWeights(fileIds = null, method = 'base', params = {}) {
        const fids = fileIds || this.getFileIds();

        const result = await this.request('/weighting/calculate', 'POST', {
            file_ids: fids,
            method,
            ...params
        });

        if (result.status === 'success') {
            this.showSuccess('Weights calculated successfully');
        }

        return result;
    },

    async validateWeights(fileIds = null) {
        const fids = fileIds || this.getFileIds();

        return await this.request('/weighting/validate', 'POST', {
            file_ids: fids
        });
    },

    async trimWeights(fileIds = null, minWeight = 0.3, maxWeight = 3.0) {
        const fids = fileIds || this.getFileIds();

        const result = await this.request('/weighting/trim', 'POST', {
            file_ids: fids,
            min_weight: minWeight,
            max_weight: maxWeight
        });

        if (result.status === 'success') {
            this.showSuccess('Weights trimmed successfully');
        }

        return result;
    },

    /**
     * =====================================================
     * STEP 05: ANALYSIS
     * =====================================================
     */

    async runDescriptiveStats(fileIds = null, columns = [], weightColumn = null) {
        const fids = fileIds || this.getFileIds();

        return await this.request('/analysis/descriptive', 'POST', {
            file_ids: fids,
            columns,
            weight_column: weightColumn
        });
    },

    async runCrosstab(fileIds = null, rowVar, colVar, weightColumn = null) {
        const fids = fileIds || this.getFileIds();

        return await this.request('/analysis/crosstab', 'POST', {
            file_ids: fids,
            row_var: rowVar,
            col_var: colVar,
            weight_column: weightColumn
        });
    },

    async runStatisticalTest(fileIds = null, params = {}) {
        const fids = fileIds || this.getFileIds();

        return await this.request('/analysis/test', 'POST', {
            file_ids: fids,
            ...params
        });
    },

    /**
     * =====================================================
     * STEP 06: FORECASTING
     * =====================================================
     */

    async runForecast(fileIds = null, params = {}) {
        const fids = fileIds || this.getFileIds();

        const result = await this.request('/forecasting/run', 'POST', {
            file_ids: fids,
            method: params.method || 'auto',
            periods: params.periods || 12,
            ...params
        });

        if (result.status === 'success') {
            this.showSuccess('Forecast generated successfully');
        }

        return result;
    },

    async decomposeForecast(fileIds = null, params = {}) {
        const fids = fileIds || this.getFileIds();

        return await this.request('/forecasting/decompose', 'POST', {
            file_ids: fids,
            ...params
        });
    },

    /**
     * =====================================================
     * STEP 07: MACHINE LEARNING
     * =====================================================
     */

    async runClassification(fileIds = null, params = {}) {
        const fids = fileIds || this.getFileIds();

        return await this.request('/ml/classify', 'POST', {
            file_ids: fids,
            ...params
        });
    },

    async runRegression(fileIds = null, params = {}) {
        const fids = fileIds || this.getFileIds();

        return await this.request('/ml/regress', 'POST', {
            file_ids: fids,
            ...params
        });
    },

    async runClustering(fileIds = null, params = {}) {
        const fids = fileIds || this.getFileIds();

        return await this.request('/ml/cluster', 'POST', {
            file_ids: fids,
            ...params
        });
    },

    async runPCA(fileIds = null, params = {}) {
        const fids = fileIds || this.getFileIds();

        return await this.request('/ml/pca', 'POST', {
            file_ids: fids,
            ...params
        });
    },

    /**
     * =====================================================
     * STEP 08: INSIGHTS
     * =====================================================
     */

    async getInsights(fileIds = null, params = {}) {
        const fids = fileIds || this.getFileIds();

        return await this.request('/insight/full', 'POST', {
            file_ids: fids,
            ...params
        });
    },

    /**
     * =====================================================
     * STEP 10: REPORT GENERATION
     * =====================================================
     */

    async generateReport(fileIds = null, mode = 'separate', reportType = 'full') {
        const fids = fileIds || this.getFileIds();

        const result = await this.request('/report/generate', 'POST', {
            file_ids: fids,
            report_type: reportType,
            format: 'pdf',
            mode: mode
        });

        if (result.status === 'success') {
            this.showSuccess(`Report generated successfully (${mode} mode)`);
        }

        return result;
    },

    /**
     * =====================================================
     * PIPELINE ORCHESTRATION
     * =====================================================
     */

    async runFullPipeline(fileIds = null, options = {}) {
        const fids = fileIds || this.getFileIds();

        const result = await this.request('/pipeline/run-full', 'POST', {
            file_ids: fids,
            include_forecast: options.includeForecast || false,
            include_ml: options.includeML || false,
            report_mode: options.reportMode || 'separate'
        });

        if (result.status === 'pipeline_completed') {
            this.showSuccess('Pipeline completed successfully!');
        }

        return result;
    },

    async runMinimalPipeline(fileIds = null) {
        const fids = fileIds || this.getFileIds();

        return await this.request('/pipeline/run-minimal', 'POST', {
            file_ids: fids
        });
    },

    async getPipelineStatus(fileId = null) {
        const fid = fileId || this.getFileIds()[0];

        if (fid) {
            return await this.request(`/pipeline/status/${fid}`, 'GET');
        } else {
            return await this.request('/pipeline/status', 'GET');
        }
    },

    /**
     * =====================================================
     * UI HELPERS
     * =====================================================
     */

    setLoading(state) {
        const loader = document.querySelector('.loading');
        const overlay = document.querySelector('.loading-overlay');

        if (loader) {
            loader.style.display = state ? 'block' : 'none';
        }

        if (overlay) {
            overlay.style.display = state ? 'flex' : 'none';
        }

        // Disable all buttons during loading
        const buttons = document.querySelectorAll('button');
        buttons.forEach(btn => {
            btn.disabled = state;
        });
    },

    showSuccess(message) {
        this.showToast(message, 'success');
    },

    showError(message) {
        this.showToast(message, 'error');
    },

    showToast(message, type = 'info') {
        // Remove existing toasts
        const existing = document.querySelectorAll('.toast');
        existing.forEach(t => t.remove());

        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;

        // Add to body
        document.body.appendChild(toast);

        // Auto remove after 4 seconds
        setTimeout(() => {
            toast.classList.add('toast-fade');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    },

    /**
     * Load demo dataset and auto-navigate to schema page
     * @returns {Promise<Object>} Upload response with file_ids
     */
    async loadDemoDataset() {
        try {
            this.setLoading(true);

            // Fetch sample CSV from backend (remove /api from base URL for static files)
            const staticBase = API_BASE.replace('/api', '');
            const response = await fetch(`${staticBase}/static/sample.csv`);
            if (!response.ok) {
                throw new Error('Sample dataset not available');
            }

            const csvText = await response.text();
            const blob = new Blob([csvText], { type: 'text/csv' });
            const file = new File([blob], 'demo_survey_data.csv', { type: 'text/csv' });

            // Upload the demo file
            const formData = new FormData();
            formData.append('files', file);
            formData.append('user_id', 'demo_user');

            const uploadResponse = await fetch(`${API_BASE}/upload/multi`, {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                throw new Error('Failed to upload demo dataset');
            }

            const result = await uploadResponse.json();
            this.setFileIds(result.file_ids);

            this.showSuccess('✓ Demo dataset loaded successfully!');

            // Auto-redirect to schema page
            setTimeout(() => {
                window.location.href = 'schema.html';
            }, 1500);

            return result;
        } catch (error) {
            console.error('Demo load error:', error);
            this.showError('Failed to load demo dataset: ' + error.message);
            throw error;
        } finally {
            this.setLoading(false);
        }
    }
};

/**
 * Test backend connection
 */
async function testBackendConnection() {
    try {
        const res = await fetch(`${API_BASE}/upload/ping`);
        const data = await res.json();
        console.log('✓ Backend connected:', data);
        return true;
    } catch (err) {
        console.error('✗ Cannot reach backend:', err);
        return false;
    }
}

// Auto-check for file_ids on page load
document.addEventListener('DOMContentLoaded', () => {
    // Test backend connection
    testBackendConnection();
    const fileIds = ApiService.getFileIds();
    console.log('[API] Initialized. File IDs:', fileIds.length > 0 ? fileIds : 'None');
});
