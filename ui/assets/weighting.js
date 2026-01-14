// =======================================================================
// weighting.js – Comprehensive Survey Weighting Module
// Integrates with backend: /api/v1/weighting/*
// =======================================================================

console.log('[weighting.js] Loading module...');

let currentFileId = null;

// Initialize on page load
window.addEventListener("DOMContentLoaded", async () => {
    console.log('[weighting.js] DOM loaded, initializing...');

    // Get file ID from URL or sessionStorage
    const urlParams = new URLSearchParams(window.location.search);
    currentFileId = urlParams.get('file') || sessionStorage.getItem('uploadedFileId');

    console.log('[weighting.js] Current file ID:', currentFileId);

    if (!currentFileId) {
        // Initialize UI even without file ID so error is visible
        initializeUI();
        showError("No file selected. Please upload a file first.");
        return;
    }

    // Initialize UI
    initializeUI();

    // Load file information
    await loadFileInfo();

    // Check if weights already exist
    await checkWeightingStatus();

    // Setup navigation
    setupNavigation();
});

function initializeUI() {
    // Setup Run Weighting button
    const runBtn = document.getElementById("runWeightBtn");
    if (runBtn) {
        runBtn.addEventListener("click", () => runWeighting());
    }

    // Initialize results container with tabs
    const resultsContainer = document.getElementById("results-container");
    if (resultsContainer) {
        resultsContainer.innerHTML = `
            <!-- Educational Header -->
            <div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 2rem;">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <span style="font-size: 3rem;">⚖️</span>
                    <div>
                        <h2 style="margin: 0; color: white;">Understanding Survey Weighting</h2>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Make your survey data representative of the target population</p>
                    </div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.15); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <p style="margin: 0; line-height: 1.6;">
                        <strong>What is weighting?</strong> Survey weighting adjusts your sample data to match the characteristics of your target population. 
                        This corrects for sampling biases and ensures your results are statistically representative.
                    </p>
                </div>
            </div>
        
            <div class="card" style="margin-top: 2rem;">
                <div class="tabs" style="display: flex; gap: 1rem; border-bottom: 2px solid #e0e0e0; margin-bottom: 1.5rem;">
                    <button class="tab-btn active" data-tab="calculate">📊 Calculate</button>
                    <button class="tab-btn" data-tab="validate">✓ Validate</button>
                    <button class="tab-btn" data-tab="trim">✂️ Trim</button>
                    <button class="tab-btn" data-tab="diagnostics">🔍 Diagnostics</button>
                </div>
                
                <div class="tab-content" id="calculate-tab" style="display: block;">
                    <!-- What This Does Section -->
                    <div class="info-section" style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #667eea;">
                        <h4 style="margin: 0 0 1rem 0; color: #667eea; display: flex; align-items: center; gap: 0.5rem;">
                            <span>ℹ️</span> What This Step Does
                        </h4>
                        <p style="margin: 0 0 1rem 0; color: #666; line-height: 1.6;">
                            Weight calculation adjusts the importance of each survey response to ensure your sample accurately represents your target population. 
                            This corrects for over-represented or under-represented groups in your data.
                        </p>
                        <div style="background: white; padding: 1rem; border-radius: 6px; margin-top: 1rem;">
                            <strong style="display: block; margin-bottom: 0.5rem; color: #333;">📋 How It Works:</strong>
                            <ul style="margin: 0.5rem 0 0 1.5rem; color: #666; line-height: 1.8;">
                                <li><strong>Base Weights:</strong> Creates simple weights (uniform or from probabilities)</li>
                                <li><strong>Post-Stratification:</strong> Adjusts weights to match known population distributions</li>
                                <li><strong>Raking:</strong> Iteratively fits weights to multiple control totals simultaneously</li>
                            </ul>
                        </div>
                    </div>
                
                    <h3 style="margin-bottom: 1rem;">⚖️ Calculate Survey Weights</h3>
                    <p style="color: #666; margin-bottom: 1.5rem;">Choose a weighting method to calculate representative weights for your survey data.</p>
                    
                    <div class="form-group">
                        <label for="weightMethod">Weighting Method:</label>
                        <select id="weightMethod" class="form-control">
                            <option value="base">📊 Base Weights (Recommended for beginners)</option>
                            <option value="poststrat">🎯 Post-Stratification (Match population targets)</option>
                            <option value="raking">🔄 Raking (Advanced - Multiple controls)</option>
                        </select>
                    </div>
                    
                    <div id="methodOptions" style="margin-top: 1.5rem;"></div>
                    
                    <button id="calculateBtn" class="btn btn-primary" style="margin-top: 1.5rem;">
                        <span>🔄 Calculate Weights</span>
                    </button>
                    
                    <div id="calculateResults" style="margin-top: 1.5rem;"></div>
                </div>
                
                <div class="tab-content" id="validate-tab" style="display: none;">
                    <!-- What This Does Section -->
                    <div class="info-section" style="background: #f0f9ff; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #3b82f6;">
                        <h4 style="margin: 0 0 1rem 0; color: #3b82f6; display: flex; align-items: center; gap: 0.5rem;">
                            <span>ℹ️</span> What This Step Does
                        </h4>
                        <p style="margin: 0 0 1rem 0; color: #666; line-height: 1.6;">
                            Weight validation checks the quality and reliability of your calculated weights. It identifies potential issues like extreme values, 
                            negative weights, or missing data that could affect your analysis results.
                        </p>
                        <div style="background: white; padding: 1rem; border-radius: 6px; margin-top: 1rem;">
                            <strong style="display: block; margin-bottom: 0.5rem; color: #333;">✅ Quality Checks:</strong>
                            <ul style="margin: 0.5rem 0 0 1.5rem; color: #666; line-height: 1.8;">
                                <li>No zero or negative weights</li>
                                <li>No missing or infinite values</li>
                                <li>Reasonable coefficient of variation (CV < 1.0)</li>
                                <li>Distribution characteristics within acceptable ranges</li>
                            </ul>
                        </div>
                    </div>
                
                    <h3>✓ Validate Weights</h3>
                    <p style="color: #666; margin-bottom: 1.5rem;">Check the quality and distribution of calculated weights to ensure they're suitable for analysis.</p>
                    
                    <button id="validateBtn" class="btn btn-primary">
                        <span>✓ Run Validation</span>
                    </button>
                    
                    <div id="validateResults" style="margin-top: 1.5rem;"></div>
                </div>
                
                <div class="tab-content" id="trim-tab" style="display: none;">
                    <!-- What This Does Section -->
                    <div class="info-section" style="background: #fffbeb; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #f59e0b;">
                        <h4 style="margin: 0 0 1rem 0; color: #f59e0b; display: flex; align-items: center; gap: 0.5rem;">
                            <span>ℹ️</span> What This Step Does
                        </h4>
                        <p style="margin: 0 0 1rem 0; color: #666; line-height: 1.6;">
                            Weight trimming caps extreme weights to improve stability and reduce the influence of outlier cases. 
                            Weights outside your specified range are adjusted to the minimum or maximum threshold.
                        </p>
                        <div style="background: white; padding: 1rem; border-radius: 6px; margin-top: 1rem;">
                            <strong style="display: block; margin-bottom: 0.5rem; color: #333;">⚠️ When to Use:</strong>
                            <ul style="margin: 0.5rem 0 0 1.5rem; color: #666; line-height: 1.8;">
                                <li>Weights have extremely high or low values</li>
                                <li>Coefficient of variation (CV) is too high</li>
                                <li>A few cases are dominating weighted estimates</li>
                                <li>You want more stable, conservative estimates</li>
                            </ul>
                        </div>
                    </div>
                
                    <h3>✂️ Trim Weights</h3>
                    <p style="color: #666; margin-bottom: 1.5rem;">Trim extreme weights to improve stability. Common ranges: 0.3 to 3.0 or 0.5 to 2.0</p>
                    
                    <div class="form-group">
                        <label for="minWeight">Minimum Weight:</label>
                        <input type="number" id="minWeight" class="form-control" value="0.3" step="0.1" min="0">
                    </div>
                    
                    <div class="form-group">
                        <label for="maxWeight">Maximum Weight:</label>
                        <input type="number" id="maxWeight" class="form-control" value="3.0" step="0.1" min="0">
                    </div>
                    
                    <button id="trimBtn" class="btn btn-primary" style="margin-top: 1rem;">
                        <span>✂️ Trim Weights</span>
                    </button>
                    
                    <div id="trimResults" style="margin-top: 1.5rem;"></div>
                </div>
                
                <div class="tab-content" id="diagnostics-tab" style="display: none;">
                    <!-- What This Does Section -->
                    <div class="info-section" style="background: #f0fdf4; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #22c55e;">
                        <h4 style="margin: 0 0 1rem 0; color: #22c55e; display: flex; align-items: center; gap: 0.5rem;">
                            <span>ℹ️</span> What This Step Does
                        </h4>
                        <p style="margin: 0 0 1rem 0; color: #666; line-height: 1.6;">
                            Diagnostics provide comprehensive statistics about your weights including distribution characteristics, 
                            effective sample size, and design effects. Use these to understand the impact of weighting on your analysis.
                        </p>
                        <div style="background: white; padding: 1rem; border-radius: 6px; margin-top: 1rem;">
                            <strong style="display: block; margin-bottom: 0.5rem; color: #333;">📊 Key Metrics:</strong>
                            <ul style="margin: 0.5rem 0 0 1.5rem; color: #666; line-height: 1.8;">
                                <li><strong>Effective Sample Size:</strong> Equivalent sample size after weighting (higher is better)</li>
                                <li><strong>Design Effect (DEFF):</strong> Variance inflation due to weighting (closer to 1.0 is better)</li>
                                <li><strong>CV:</strong> Coefficient of variation - weight variability (< 0.5 is good)</li>
                                <li><strong>Percentiles:</strong> Distribution of weights across your sample</li>
                            </ul>
                        </div>
                    </div>
                
                    <h3>📊 Weight Diagnostics</h3>
                    <p style="color: #666; margin-bottom: 1.5rem;">View detailed statistics and diagnostics to assess weight quality and understand their impact.</p>
                    
                    <button id="diagnosticsBtn" class="btn btn-primary">
                        <span>📊 Load Diagnostics</span>
                    </button>
                    
                    <div id="diagnosticsResults" style="margin-top: 1.5rem;"></div>
                </div>
            </div>
        `;

        // Setup tab switching
        setupTabs();

        // Setup button handlers
        document.getElementById("calculateBtn")?.addEventListener("click", calculateWeights);
        document.getElementById("validateBtn")?.addEventListener("click", validateWeights);
        document.getElementById("trimBtn")?.addEventListener("click", trimWeights);
        document.getElementById("diagnosticsBtn")?.addEventListener("click", loadDiagnostics);

        // Setup method change handler
        document.getElementById("weightMethod")?.addEventListener("change", updateMethodOptions);
        updateMethodOptions(); // Initialize
    }
}

function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;

            // Update button states
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update content visibility
            tabContents.forEach(content => {
                content.style.display = content.id === `${targetTab}-tab` ? 'block' : 'none';
            });
        });
    });
}

function updateMethodOptions() {
    const method = document.getElementById("weightMethod")?.value;
    const optionsDiv = document.getElementById("methodOptions");

    if (!optionsDiv) return;

    if (method === "poststrat") {
        optionsDiv.innerHTML = `
            <div class="info-box" style="background: #e3f2fd; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #1976d2;">
                <h5 style="margin: 0 0 0.75rem 0; color: #1976d2; display: flex; align-items: center; gap: 0.5rem;">
                    <span>🎯</span> Post-Stratification Method
                </h5>
                <p style="margin: 0 0 1rem 0; color: #555; line-height: 1.6;">
                    This method adjusts survey weights to match known population distributions across different groups (strata). 
                    For example, if your survey has 30% males but the population is 48% male, it will increase the weight of male responses.
                </p>
                <div style="background: white; padding: 1rem; border-radius: 6px;">
                    <strong style="display: block; margin-bottom: 0.5rem; color: #333;">🔍 What happens to your data:</strong>
                    <ol style="margin: 0.5rem 0 0 1.25rem; color: #666; line-height: 1.8;">
                        <li>Identifies groups in your stratification column (e.g., age groups, regions)</li>
                        <li>Calculates the proportion of each group in your sample</li>
                        <li>Adjusts weights so each group matches the population proportion</li>
                        <li>Ensures weighted results represent the true population structure</li>
                    </ol>
                </div>
            </div>
            
            <div class="form-group">
                <label for="strataColumn">Strata Column (optional):</label>
                <input type="text" id="strataColumn" class="form-control" placeholder="e.g., age_group, region, gender">
                <small style="color: #666;">Leave empty for auto-detection. Column should contain categorical groups.</small>
            </div>
        `;
    } else if (method === "raking") {
        optionsDiv.innerHTML = `
            <div class="info-box" style="background: #fff3e0; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #f57c00;">
                <h5 style="margin: 0 0 0.75rem 0; color: #f57c00; display: flex; align-items: center; gap: 0.5rem;">
                    <span>🔄</span> Raking (Iterative Proportional Fitting)
                </h5>
                <p style="margin: 0 0 1rem 0; color: #555; line-height: 1.6;">
                    Raking is an advanced method that adjusts weights to match multiple population controls simultaneously. 
                    It iteratively adjusts weights across different dimensions until all target proportions are met.
                </p>
                <div style="background: white; padding: 1rem; border-radius: 6px;">
                    <strong style="display: block; margin-bottom: 0.5rem; color: #333;">🔍 What happens to your data:</strong>
                    <ol style="margin: 0.5rem 0 0 1.25rem; color: #666; line-height: 1.8;">
                        <li>Takes control totals for multiple dimensions (e.g., gender AND region)</li>
                        <li>Adjusts weights for first dimension to match targets</li>
                        <li>Then adjusts for second dimension while preserving first</li>
                        <li>Repeats until all dimensions match within tolerance</li>
                        <li>Results in weights that balance multiple population characteristics</li>
                    </ol>
                    <div style="margin-top: 0.75rem; padding: 0.5rem; background: #fff8e1; border-radius: 4px;">
                        <strong style="color: #f57c00;">💡 Example:</strong> Match both gender distribution (48% M, 52% F) 
                        AND regional distribution (20% North, 30% South, etc.) simultaneously.
                    </div>
                </div>
            </div>
            
            <div class="form-group">
                <label>Control Totals (JSON format):</label>
                <textarea id="controlTotals" class="form-control" rows="6" placeholder='{"gender": {"male": 500, "female": 500}, "region": {"north": 300, "south": 400, "east": 300}}'></textarea>
                <small style="color: #666;">Format: {"column_name": {"category": target_count, ...}}</small>
            </div>
        `;
    } else {
        optionsDiv.innerHTML = `
            <div class="info-box" style="background: #e8f5e9; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #388e3c;">
                <h5 style="margin: 0 0 0.75rem 0; color: #388e3c; display: flex; align-items: center; gap: 0.5rem;">
                    <span>📊</span> Base Weights Method
                </h5>
                <p style="margin: 0 0 1rem 0; color: #555; line-height: 1.6;">
                    The simplest weighting method. Creates uniform weights (all equal) or inverse probability weights if you have 
                    sampling probabilities. This is recommended for beginners or when you don't have population control totals.
                </p>
                <div style="background: white; padding: 1rem; border-radius: 6px;">
                    <strong style="display: block; margin-bottom: 0.5rem; color: #333;">🔍 What happens to your data:</strong>
                    <ol style="margin: 0.5rem 0 0 1.25rem; color: #666; line-height: 1.8;">
                        <li><strong>Uniform:</strong> All responses get equal weight (weight = 1.0)</li>
                        <li><strong>Inverse Probability:</strong> If you have sampling probabilities, weights = 1/probability</li>
                        <li>Ensures each response contributes equally to analysis</li>
                        <li>No adjustments for population characteristics</li>
                    </ol>
                    <div style="margin-top: 0.75rem; padding: 0.5rem; background: #e8f5e9; border-radius: 4px;">
                        <strong style="color: #388e3c;">✅ Best for:</strong> Initial exploration, simple random samples, 
                        or when population distributions are unknown.
                    </div>
                </div>
            </div>
        `;
    }
}

async function loadFileInfo() {
    const fileName = sessionStorage.getItem('uploadedFileName') || 'Unknown file';
    const fileInfoDiv = document.getElementById("fileInfo");

    if (fileInfoDiv) {
        fileInfoDiv.innerHTML = `
            <div style="background: #f5f5f5; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196F3;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 2rem;">📄</span>
                    <div>
                        <strong style="font-size: 1.1rem;">${fileName}</strong>
                        <p style="margin: 0.25rem 0 0 0; color: #666; font-size: 0.9rem;">File ID: ${currentFileId}</p>
                    </div>
                </div>
                <div id="weightingStatusBadge" style="margin-top: 0.5rem;"></div>
            </div>
        `;
    }
}

async function checkWeightingStatus() {
    try {
        const response = await fetch(`${window.API_BASE}/weighting/diagnostics/${currentFileId}`);

        const statusBadge = document.getElementById("weightingStatusBadge");
        if (!statusBadge) return;

        if (response.ok) {
            const data = await response.json();
            statusBadge.innerHTML = `
                <div style="display: inline-block; background: #4caf50; color: white; padding: 0.3rem 0.8rem; border-radius: 4px; font-size: 0.85rem;">
                    ✓ Weights Calculated
                </div>
            `;

            // Auto-load diagnostics to show user the existing weights
            setTimeout(() => {
                const diagnosticsTab = document.querySelector('[data-tab="diagnostics"]');
                if (diagnosticsTab) diagnosticsTab.click();
                loadDiagnostics();
            }, 500);
        } else {
            statusBadge.innerHTML = `
                <div style="display: inline-block; background: #ff9800; color: white; padding: 0.3rem 0.8rem; border-radius: 4px; font-size: 0.85rem;">
                    ⚠ No Weights - Please Calculate
                </div>
            `;
        }
    } catch (error) {
        console.log('[weighting.js] No existing weights found:', error);
    }
}

function setupNavigation() {
    const backBtn = document.getElementById("backBtn");
    const continueBtn = document.getElementById("continueBtn");

    if (backBtn) {
        backBtn.addEventListener("click", () => {
            window.location.href = `cleaning.html?file=${currentFileId}`;
        });
    }

    if (continueBtn) {
        // Enable continue button once weighting is done
        continueBtn.disabled = false;
        continueBtn.addEventListener("click", () => {
            // Save file ID for next step
            sessionStorage.setItem('uploadedFileId', currentFileId);
            window.location.href = `analysis.html?file=${currentFileId}`;
        });
    }
}

// ==================== API Functions ====================

async function runWeighting() {
    // Backward compatibility - runs base weighting
    await calculateWeights();
}

async function calculateWeights() {
    const method = document.getElementById("weightMethod")?.value || "base";
    const resultsDiv = document.getElementById("calculateResults");

    if (!resultsDiv) return;

    showLoading(true);
    resultsDiv.innerHTML = '<p style="color: #666;">Calculating weights...</p>';

    try {
        const payload = {
            file_id: currentFileId,
            method: method
        };

        // Add method-specific parameters
        if (method === "poststrat") {
            const strataColumn = document.getElementById("strataColumn")?.value;
            if (strataColumn) payload.strata_column = strataColumn;
        } else if (method === "raking") {
            const controlTotalsStr = document.getElementById("controlTotals")?.value;
            if (controlTotalsStr) {
                try {
                    payload.control_totals = JSON.parse(controlTotalsStr);
                } catch (e) {
                    throw new Error("Invalid JSON format for control totals");
                }
            }
        }

        console.log('[weighting.js] Calculating weights with payload:', payload);

        const response = await fetch(`${window.API_BASE}/weighting/calculate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        console.log('[weighting.js] Calculate response:', data);

        if (!response.ok || data.status !== "success") {
            throw new Error(data.detail || data.errors?.[currentFileId] || "Weight calculation failed");
        }

        displayCalculateResults(data, resultsDiv);

        // Update status badge
        const statusBadge = document.getElementById("weightingStatusBadge");
        if (statusBadge) {
            statusBadge.innerHTML = `
                <div style="display: inline-block; background: #4caf50; color: white; padding: 0.3rem 0.8rem; border-radius: 4px; font-size: 0.85rem;">
                    ✓ Weights Calculated
                </div>
            `;
        }

        // Auto-run validation to show quality checks
        console.log('[weighting.js] Auto-running validation...');
        setTimeout(() => validateWeights(), 1000);

    } catch (err) {
        console.error('[weighting.js] Calculate error:', err);
        resultsDiv.innerHTML = `
            <div class="alert alert-danger">
                <strong>Error:</strong> ${err.message}
            </div>
        `;
    } finally {
        showLoading(false);
    }
}

async function validateWeights() {
    const resultsDiv = document.getElementById("validateResults");
    if (!resultsDiv) return;

    showLoading(true);
    resultsDiv.innerHTML = '<p style="color: #666;">Validating weights...</p>';

    try {
        const response = await fetch(`${window.API_BASE}/weighting/validate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ file_id: currentFileId })
        });

        const data = await response.json();
        console.log('[weighting.js] Validate response:', data);

        if (!response.ok) {
            throw new Error(data.detail || "Validation failed");
        }

        displayValidateResults(data, resultsDiv);

    } catch (err) {
        console.error('[weighting.js] Validate error:', err);
        resultsDiv.innerHTML = `
            <div class="alert alert-danger">
                <strong>Error:</strong> ${err.message}
            </div>
        `;
    } finally {
        showLoading(false);
    }
}

async function trimWeights() {
    const minWeight = parseFloat(document.getElementById("minWeight")?.value || 0.3);
    const maxWeight = parseFloat(document.getElementById("maxWeight")?.value || 3.0);
    const resultsDiv = document.getElementById("trimResults");

    if (!resultsDiv) return;

    showLoading(true);
    resultsDiv.innerHTML = '<p style="color: #666;">Trimming weights...</p>';

    try {
        const response = await fetch(`${window.API_BASE}/weighting/trim`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                file_id: currentFileId,
                min_w: minWeight,
                max_w: maxWeight
            })
        });

        const data = await response.json();
        console.log('[weighting.js] Trim response:', data);

        if (!response.ok) {
            throw new Error(data.detail || "Trim failed");
        }

        displayTrimResults(data, resultsDiv);

    } catch (err) {
        console.error('[weighting.js] Trim error:', err);
        resultsDiv.innerHTML = `
            <div class="alert alert-danger">
                <strong>Error:</strong> ${err.message}
            </div>
        `;
    } finally {
        showLoading(false);
    }
}

async function loadDiagnostics() {
    const resultsDiv = document.getElementById("diagnosticsResults");
    if (!resultsDiv) return;

    showLoading(true);
    resultsDiv.innerHTML = '<p style="color: #666;">Loading diagnostics...</p>';

    try {
        const response = await fetch(`${window.API_BASE}/weighting/diagnostics/${currentFileId}`);
        const data = await response.json();
        console.log('[weighting.js] Diagnostics response:', data);

        if (!response.ok) {
            throw new Error(data.detail || "Failed to load diagnostics");
        }

        displayDiagnostics(data, resultsDiv);

    } catch (err) {
        console.error('[weighting.js] Diagnostics error:', err);
        resultsDiv.innerHTML = `
            <div class="alert alert-danger">
                <strong>Error:</strong> ${err.message}
            </div>
        `;
    } finally {
        showLoading(false);
    }
}

// ==================== Display Functions ====================

function displayCalculateResults(data, container) {
    const result = data.results?.[currentFileId];
    if (!result) {
        container.innerHTML = '<div class="alert alert-warning">No results available. Please calculate weights first.</div>';
        return;
    }

    let html = `
        <div class="alert alert-success">
            <strong>✅ Weights calculated successfully!</strong>
            <p style="margin: 0.5rem 0 0 0;">Method: <strong>${result.method}</strong></p>
        </div>
    `;

    // Summary statistics
    if (result.result?.summary) {
        const summary = result.result.summary;
        html += `
            <div class="card" style="margin-top: 1rem; background: #f9f9f9;">
                <h4>📊 Weight Summary</h4>
                <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    ${summary.mean ? `<div class="stat-card"><strong>Mean:</strong> ${summary.mean.toFixed(4)}</div>` : ''}
                    ${summary.min ? `<div class="stat-card"><strong>Min:</strong> ${summary.min.toFixed(4)}</div>` : ''}
                    ${summary.max ? `<div class="stat-card"><strong>Max:</strong> ${summary.max.toFixed(4)}</div>` : ''}
                    ${summary.sum ? `<div class="stat-card"><strong>Sum:</strong> ${summary.sum.toFixed(2)}</div>` : ''}
                </div>
            </div>
        `;
    }

    // Next steps
    html += `
        <div class="card" style="margin-top: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <h4 style="color: white; margin-bottom: 0.5rem;">✨ What's Next?</h4>
            <p style="margin: 0.5rem 0;">Your weights have been calculated! You can now:</p>
            <ul style="margin: 0.5rem 0 0 1.5rem; line-height: 1.8;">
                <li>Click the <strong>Validate</strong> tab to check weight quality</li>
                <li>Click the <strong>Diagnostics</strong> tab to view detailed statistics</li>
                <li>Click the <strong>Trim</strong> tab to adjust extreme weights (optional)</li>
                <li>Click <strong>Continue →</strong> below to proceed to analysis</li>
            </ul>
        </div>
    `;

    // Auto-actions
    if (result.auto_actions && result.auto_actions.length > 0) {
        html += `
            <div class="card" style="margin-top: 1rem; background: #e3f2fd;">
                <h4>🔧 Auto-Actions Performed</h4>
                <ul style="margin: 0.5rem 0 0 1.5rem;">
                    ${result.auto_actions.map(action => `<li>${action}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // Warnings
    if (result.warnings && result.warnings.length > 0) {
        html += `
            <div class="card" style="margin-top: 1rem; background: #fff3e0;">
                <h4>⚠️ Warnings</h4>
                <ul style="margin: 0.5rem 0 0 1.5rem;">
                    ${result.warnings.map(warning => `<li>${warning}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    container.innerHTML = html;
}

function displayValidateResults(data, container) {
    const result = data.results?.[currentFileId];
    if (!result) {
        container.innerHTML = '<div class="alert alert-warning">No validation results available</div>';
        return;
    }

    // Handle both nested and flat result structures
    const validation = result.validation || result;
    const isValid = validation.problems?.length === 0 || validation.status === 'pass';

    let html = `
        <div class="alert ${isValid ? 'alert-success' : 'alert-warning'}">
            <strong>${isValid ? '✅ Weights are valid!' : '⚠️ Validation issues detected'}</strong>
        </div>
    `;

    // Statistics
    if (validation.statistics) {
        const stats = validation.statistics;
        html += `
            <div class="card" style="margin-top: 1rem; background: #f9f9f9;">
                <h4>📊 Weight Statistics</h4>
                <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div class="stat-card"><strong>Count:</strong> ${stats.count || validation.n_valid || 'N/A'}</div>
                    <div class="stat-card"><strong>Mean:</strong> ${stats.mean?.toFixed(4) || 'N/A'}</div>
                    <div class="stat-card"><strong>Std Dev:</strong> ${stats.std?.toFixed(4) || 'N/A'}</div>
                    <div class="stat-card"><strong>Min:</strong> ${stats.min?.toFixed(4) || 'N/A'}</div>
                    <div class="stat-card"><strong>Max:</strong> ${stats.max?.toFixed(4) || 'N/A'}</div>
                    <div class="stat-card"><strong>CV:</strong> ${stats.cv?.toFixed(4) || 'N/A'}</div>
                </div>
            </div>
        `;
    }

    // Problems
    if (validation.problems && validation.problems.length > 0) {
        html += `
            <div class="card" style="margin-top: 1rem; background: #ffebee;">
                <h4>❌ Issues Found</h4>
                <ul style="margin: 0.5rem 0 0 1.5rem;">
                    ${validation.problems.map(problem => `<li>${problem}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    container.innerHTML = html;
}

function displayTrimResults(data, container) {
    const result = data.results?.[currentFileId];
    if (!result) {
        container.innerHTML = '<div class="alert alert-warning">No results available. Please calculate weights first.</div>';
        return;
    }

    let html = `
        <div class="alert alert-success">
            <strong>✅ Weights trimmed successfully!</strong>
        </div>
    `;

    // Summary
    if (result.summary) {
        const summary = result.summary;
        html += `
            <div class="card" style="margin-top: 1rem; background: #f9f9f9;">
                <h4>📊 Trim Summary</h4>
                <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div class="stat-card"><strong>Trimmed:</strong> ${summary.trimmed_count} (${summary.trimmed_pct?.toFixed(2)}%)</div>
                    <div class="stat-card"><strong>New Mean:</strong> ${summary.new_mean?.toFixed(4) || 'N/A'}</div>
                    <div class="stat-card"><strong>New Min:</strong> ${summary.new_min?.toFixed(4) || 'N/A'}</div>
                    <div class="stat-card"><strong>New Max:</strong> ${summary.new_max?.toFixed(4) || 'N/A'}</div>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function displayDiagnostics(data, container) {
    // Backend returns {diagnostics: {...}} directly, not nested in file_id
    const diagnostics = data.diagnostics || data.diagnostics?.[currentFileId];
    if (!diagnostics) {
        container.innerHTML = '<div class="alert alert-warning">No diagnostics available. Please calculate weights first.</div>';
        return;
    }

    let html = '<div class="card" style="background: #f9f9f9;"><h4>📊 Weight Diagnostics</h4>';

    // Statistics
    if (diagnostics.statistics) {
        const stats = diagnostics.statistics;
        html += `
            <div style="margin-top: 1rem;">
                <h5>Statistics</h5>
                <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div class="stat-card"><strong>Count:</strong> ${stats.count}</div>
                    <div class="stat-card"><strong>Mean:</strong> ${stats.mean?.toFixed(4) || 'N/A'}</div>
                    <div class="stat-card"><strong>Std:</strong> ${stats.std?.toFixed(4) || 'N/A'}</div>
                    <div class="stat-card"><strong>Min:</strong> ${stats.min?.toFixed(4) || 'N/A'}</div>
                    <div class="stat-card"><strong>Max:</strong> ${stats.max?.toFixed(4) || 'N/A'}</div>
                    <div class="stat-card"><strong>CV:</strong> ${stats.cv?.toFixed(4) || 'N/A'}</div>
                </div>
            </div>
        `;
    }

    // Quality checks
    if (diagnostics.quality_checks) {
        const checks = diagnostics.quality_checks;
        html += `
            <div style="margin-top: 1.5rem;">
                <h5>Quality Checks</h5>
                <ul style="list-style: none; padding: 0;">
                    ${Object.entries(checks).map(([key, value]) => `
                        <li style="padding: 0.5rem; background: white; margin: 0.5rem 0; border-radius: 4px;">
                            <strong>${key.replace(/_/g, ' ')}:</strong> ${value ? '✅' : '❌'}
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }

    html += '</div>';
    container.innerHTML = html;
}

// ==================== Utility Functions ====================

function showLoading(show) {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

function showError(message) {
    const container = document.querySelector('.container');
    if (container) {
        container.innerHTML = `
            <div class="alert alert-danger" style="margin-top: 2rem;">
                <strong>⚠️ Error:</strong> ${message}
                <hr style="margin: 1rem 0; border-color: rgba(0,0,0,0.1);">
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    <strong>💡 Tip:</strong> Make sure you've uploaded a file first. 
                    <a href="index.html" style="color: #d32f2f; text-decoration: underline;">Go to Upload Page</a>
                </p>
            </div>
        `;
    }
}

console.log('[weighting.js] Module loaded successfully');
