// =======================================================================
// analysis.js – Comprehensive Statistical Analysis Module
// Integrates with backend: /api/v1/analysis/*
// =======================================================================

console.log('[analysis.js] Loading module...');

let currentFileId = null;
let currentColumns = [];

// Initialize on page load
window.addEventListener("DOMContentLoaded", async () => {
    console.log('[analysis.js] DOM loaded, initializing...');

    // Get file ID from URL or sessionStorage
    const urlParams = new URLSearchParams(window.location.search);
    currentFileId = urlParams.get('file') || sessionStorage.getItem('uploadedFileId');

    console.log('[analysis.js] Current file ID:', currentFileId);

    if (!currentFileId) {
        showError("No file selected. Please go back to weighting step.");
        return;
    }

    // Initialize UI
    initializeUI();

    // Load file information
    await loadFileInfo();

    // Check if analysis already exists
    await checkAnalysisStatus();

    // Setup navigation
    setupNavigation();
});

function initializeUI() {
    const container = document.querySelector('.analysis-container');
    if (!container) return;

    container.innerHTML = `
        <div class="hero-analysis">
            <h1>📊 Statistical Analysis</h1>
            <p>Comprehensive descriptive statistics and visualizations</p>
        </div>

        <!-- File Info Banner -->
        <div id="fileInfo" style="margin-bottom: 2rem; font-size: 1.1rem;"></div>

        <!-- Quick Start Card -->
        <div class="card" style="margin-bottom: 2rem; background: linear-gradient(135deg, #6f42c1 0%, #5a32a3 100%); color: white;">
            <h3 style="color: white; margin-bottom: 1rem;">🚀 Quick Start</h3>
            <p style="line-height: 1.6; opacity: 0.9;">
                Start by running descriptive statistics to understand your data's basic characteristics. 
                Then explore relationships using crosstabs and regression, or compare groups with subgroup analysis.
            </p>
            <button id="runQuickAnalysisBtn" class="btn" style="background: white; color: #6f42c1; margin-top: 1rem;">
                ⚡ Run Quick Analysis
            </button>
        </div>

        <!-- Analysis Tabs -->
        <div class="card">
            <div class="tabs" style="display: flex; gap: 0.5rem; border-bottom: 2px solid #e0e0e0; margin-bottom: 1.5rem; flex-wrap: wrap;">
                <button class="tab-btn active" data-tab="descriptive">📈 Descriptive</button>
                <button class="tab-btn" data-tab="crosstab">📋 Crosstab</button>
                <button class="tab-btn" data-tab="regression">📉 Regression</button>
                <button class="tab-btn" data-tab="subgroup">👥 Subgroup</button>
            </div>

            <!-- DESCRIPTIVE TAB -->
            <div class="tab-content" id="descriptive-tab" style="display: block;">
                <div class="info-section" style="background: #f0f9ff; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #3b82f6;">
                    <h4 style="margin: 0 0 1rem 0; color: #3b82f6;">ℹ️ What This Does</h4>
                    <p style="margin: 0; color: #666; line-height: 1.6;">
                        Descriptive statistics summarize your data using measures like mean (average), median (middle value), 
                        standard deviation (spread), and range. Get a quick snapshot of your data's patterns.
                    </p>
                </div>

                <button id="descriptiveBtn" class="btn btn-primary">
                    📊 Calculate Descriptive Statistics
                </button>

                <div id="descriptiveResults" style="margin-top: 1.5rem;"></div>
            </div>

            <!-- CROSSTAB TAB -->
            <div class="tab-content" id="crosstab-tab" style="display: none;">
                <div class="info-section" style="background: #f0fdf4; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #22c55e;">
                    <h4 style="margin: 0 0 1rem 0; color: #22c55e;">ℹ️ What This Does</h4>
                    <p style="margin: 0; color: #666; line-height: 1.6;">
                        Crosstabulation shows relationships between two categorical variables, displaying how often 
                        different combinations occur together. Perfect for finding patterns across groups.
                    </p>
                </div>

                <p style="color: #666; margin-bottom: 1rem;">Feature coming soon - analyze relationships between categorical variables</p>
            </div>

            <!-- REGRESSION TAB -->
            <div class="tab-content" id="regression-tab" style="display: none;">
                <div class="info-section" style="background: #fef3c7; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #f59e0b;">
                    <h4 style="margin: 0 0 1rem 0; color: #f59e0b;">ℹ️ What This Does</h4>
                    <p style="margin: 0; color: #666; line-height: 1.6;">
                        Regression analysis predicts one variable based on others. Use OLS for continuous outcomes 
                        or logistic regression for yes/no outcomes.
                    </p>
                </div>

                <p style="color: #666; margin-bottom: 1rem;">Feature coming soon - model relationships between variables</p>
            </div>

            <!-- SUBGROUP TAB -->
            <div class="tab-content" id="subgroup-tab" style="display: none;">
                <div class="info-section" style="background: #fce7f3; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #ec4899;">
                    <h4 style="margin: 0 0 1rem 0; color: #ec4899;">ℹ️ What This Does</h4>
                    <p style="margin: 0; color: #666; line-height: 1.6;">
                        Subgroup analysis compares statistics across different groups. Find out if patterns differ 
                        by demographic, geographic, or other categorical characteristics.
                    </p>
                </div>

                <p style="color: #666; margin-bottom: 1rem;">Feature coming soon - compare statistics across groups</p>
            </div>
        </div>

        <!-- Navigation Buttons -->
        <div class="nav-buttons" style="margin-top: 2rem;">
            <button id="backBtn" class="btn btn-secondary">← Back to Weighting</button>
            <button id="continueBtn" class="btn btn-primary">Continue to Insights →</button>
        </div>
    `;

    // Setup tabs
    setupTabs();

    // Setup button handlers
    document.getElementById("descriptiveBtn")?.addEventListener("click", runDescriptive);
    document.getElementById("runQuickAnalysisBtn")?.addEventListener("click", runDescriptive);
}

function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;

            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            tabContents.forEach(content => {
                content.style.display = 'none';
            });

            document.getElementById(`${targetTab}-tab`).style.display = 'block';
        });
    });
}

function setupNavigation() {
    const backBtn = document.getElementById("backBtn");
    const continueBtn = document.getElementById("continueBtn");

    if (backBtn) {
        backBtn.addEventListener("click", () => {
            window.location.href = `weighting.html?file=${currentFileId}`;
        });
    }

    if (continueBtn) {
        continueBtn.addEventListener("click", () => {
            sessionStorage.setItem('uploadedFileId', currentFileId);
            window.location.href = `insight.html?file=${currentFileId}`;
        });
    }
}

async function loadFileInfo() {
    try {
        const fileInfoDiv = document.getElementById("fileInfo");
        if (fileInfoDiv) {
            fileInfoDiv.innerHTML = `
                <div class="card" style="background: #f8f9fa; padding: 1rem;">
                    <strong>📁 Current File:</strong> ${currentFileId}
                </div>
            `;
        }
    } catch (error) {
        console.error('[analysis.js] Error loading file info:', error);
    }
}

async function checkAnalysisStatus() {
    try {
        const response = await fetch(`${window.API_BASE}/analysis/operations-log/${currentFileId}`);

        if (response.ok) {
            const data = await response.json();
            if (data.operations && data.operations.length > 0) {
                showSuccess(`✅ Found ${data.operations.length} previous analysis operation(s)`);
            }
        }
    } catch (error) {
        console.log('[analysis.js] No previous analysis found (this is normal)');
    }
}

async function runDescriptive() {
    const resultsDiv = document.getElementById("descriptiveResults");

    if (!resultsDiv) return;

    try {
        showLoading(true);
        resultsDiv.innerHTML = '<p style="color: #666;">📊 Calculating statistics...</p>';

        const response = await fetch(`${window.API_BASE}/analysis/descriptive`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                file_id: currentFileId,
                columns: [] // Empty array means analyze all numeric columns
            })
        });

        const data = await response.json();
        console.log('[analysis.js] Descriptive response:', data);

        if (data.status !== "success") {
            throw new Error(data.error || "Analysis failed");
        }

        displayDescriptiveResults(data, resultsDiv);
        showSuccess("✅ Analysis completed successfully!");

    } catch (error) {
        resultsDiv.innerHTML = `<div class="alert alert-error">❌ Error: ${error.message}</div>`;
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

function displayDescriptiveResults(data, container) {
    let html = '<div class="results-section">';
    html += '<h3 style="margin: 1.5rem 0 1rem 0; color: #6f42c1;">📊 Descriptive Statistics</h3>';

    if (data.descriptive_stats && Object.keys(data.descriptive_stats).length > 0) {
        html += '<div style="overflow-x: auto;">';
        html += '<table class="stats-table" style="width: 100%; border-collapse: collapse; margin-top: 1rem;">';
        html += '<thead><tr style="background: #6f42c1; color: white;">';
        html += '<th style="padding: 0.75rem; text-align: left; border: 1px solid #dee2e6;">Variable</th>';
        html += '<th style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">Count</th>';
        html += '<th style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">Mean</th>';
        html += '<th style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">Median</th>';
        html += '<th style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">Std Dev</th>';
        html += '<th style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">Min</th>';
        html += '<th style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">Max</th>';
        html += '</tr></thead><tbody>';

        Object.entries(data.descriptive_stats).forEach(([col, stats]) => {
            html += '<tr style="border-bottom: 1px solid #dee2e6;">';
            html += `<td style="padding: 0.75rem; border: 1px solid #dee2e6;"><strong>${col}</strong></td>`;
            html += `<td style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">${stats.count || 'N/A'}</td>`;
            html += `<td style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">${formatNumber(stats.mean)}</td>`;
            html += `<td style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">${formatNumber(stats.median)}</td>`;
            html += `<td style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">${formatNumber(stats.std)}</td>`;
            html += `<td style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">${formatNumber(stats.min)}</td>`;
            html += `<td style="padding: 0.75rem; text-align: right; border: 1px solid #dee2e6;">${formatNumber(stats.max)}</td>`;
            html += '</tr>';
        });

        html += '</tbody></table></div>';
    } else {
        html += '<p style="color: #666;">No numeric variables found to analyze.</p>';
    }

    // Add frequencies if available
    if (data.frequencies && Object.keys(data.frequencies).length > 0) {
        html += '<h3 style="margin: 2rem 0 1rem 0; color: #6f42c1;">📋 Frequency Distributions</h3>';
        html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">';

        Object.entries(data.frequencies).forEach(([col, freqData]) => {
            html += '<div class="card" style="background: #f8f9fa;">';
            html += `<h4 style="margin-bottom: 0.5rem;">${col}</h4>`;
            html += '<div style="max-height: 200px; overflow-y: auto;">';
            html += '<table style="width: 100%; font-size: 0.9rem;">';

            if (freqData && typeof freqData === 'object') {
                Object.entries(freqData).slice(0, 10).forEach(([value, count]) => {
                    html += `<tr><td style="padding: 0.25rem;">${value}</td><td style="text-align: right; padding: 0.25rem;">${count}</td></tr>`;
                });

                if (Object.keys(freqData).length > 10) {
                    html += `<tr><td colspan="2" style="padding: 0.25rem; color: #666; font-style: italic;">... and ${Object.keys(freqData).length - 10} more</td></tr>`;
                }
            }

            html += '</table></div></div>';
        });

        html += '</div>';
    }

    html += '</div>';
    container.innerHTML = html;
}

function formatNumber(value) {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') {
        return value.toFixed(2);
    }
    return value;
}

function showLoading(show) {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

function showError(message) {
    console.error('[analysis.js]', message);
    alert(`Error: ${message}`);
}

function showSuccess(message) {
    console.log('[analysis.js]', message);
}

console.log('[analysis.js] Module loaded successfully');