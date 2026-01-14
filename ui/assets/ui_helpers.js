/**
 * StatFlow AI - UI Helper Module
 * Centralized UI utilities for manual workflow progression
 * 
 * Features:
 * - Manual step control with Continue buttons
 * - Centralized error/success banners
 * - Loading state management
 * - API request wrapper with error handling
 * - Graceful failure recovery

if (!window.API_BASE) console.error("API_BASE not ready");
 * - URL parameter management
 * - Theme management system (Dark/Light mode)
 */

// =============================================================================
// THEME SERVICE - Dark/Light Mode Management
// =============================================================================

const ThemeService = {
    STORAGE_KEY: 'statflow_theme',
    THEMES: {
        LIGHT: 'light',
        DARK: 'dark'
    },

    /**
     * Initialize theme system
     * Applies saved theme or defaults to light mode
     */
    init() {
        const savedTheme = this.getCurrentTheme();
        this.setTheme(savedTheme);

        // Listen for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
                if (!localStorage.getItem(this.STORAGE_KEY)) {
                    this.setTheme(e.matches ? this.THEMES.DARK : this.THEMES.LIGHT);
                }
            });
        }

        console.log(`✅ Theme initialized: ${savedTheme}`);
    },

    /**
     * Get current theme from localStorage or system preference
     * @returns {string} 'light' or 'dark'
     */
    getCurrentTheme() {
        const saved = localStorage.getItem(this.STORAGE_KEY);
        if (saved) return saved;

        // Check system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return this.THEMES.DARK;
        }

        return this.THEMES.LIGHT;
    },

    /**
     * Set theme and persist to localStorage
     * @param {string} theme - 'light' or 'dark'
     */
    setTheme(theme) {
        const validTheme = theme === this.THEMES.DARK ? this.THEMES.DARK : this.THEMES.LIGHT;

        // Remove both classes
        document.body.classList.remove('light-mode', 'dark-mode');

        // Add new theme class
        document.body.classList.add(`${validTheme}-mode`);

        // Save to localStorage
        localStorage.setItem(this.STORAGE_KEY, validTheme);

        // Update theme toggle button if exists
        this.updateThemeToggleUI(validTheme);

        // Trigger custom event for charts to update
        window.dispatchEvent(new CustomEvent('themeChange', { detail: { theme: validTheme } }));

        console.log(`🎨 Theme set to: ${validTheme}`);
    },

    /**
     * Toggle between light and dark themes
     */
    toggleTheme() {
        const currentTheme = this.getCurrentTheme();
        const newTheme = currentTheme === this.THEMES.LIGHT ? this.THEMES.DARK : this.THEMES.LIGHT;
        this.setTheme(newTheme);
    },

    /**
     * Update theme toggle button UI
     * @param {string} theme - Current theme
     */
    updateThemeToggleUI(theme) {
        const toggleBtn = document.getElementById('themeToggle');
        if (toggleBtn) {
            const icon = toggleBtn.querySelector('.theme-icon');
            if (icon) {
                icon.textContent = theme === this.THEMES.DARK ? '☀️' : '🌙';
            }
            toggleBtn.setAttribute('aria-label', `Switch to ${theme === this.THEMES.DARK ? 'light' : 'dark'} mode`);
            toggleBtn.setAttribute('title', `Switch to ${theme === this.THEMES.DARK ? 'light' : 'dark'} mode`);
        }
    },

    /**
     * Get chart colors for current theme
     * @returns {object} Colors object with theme-appropriate values
     */
    getChartColors() {
        const theme = this.getCurrentTheme();

        if (theme === this.THEMES.DARK) {
            return {
                gridColor: 'rgba(255, 255, 255, 0.1)',
                textColor: '#e5e7eb',
                tooltipBg: 'rgba(31, 41, 55, 0.95)',
                tooltipText: '#f3f4f6',
                borderColor: 'rgba(255, 255, 255, 0.2)'
            };
        }

        return {
            gridColor: 'rgba(0, 0, 0, 0.05)',
            textColor: '#6b7280',
            tooltipBg: 'rgba(0, 0, 0, 0.8)',
            tooltipText: '#ffffff',
            borderColor: 'rgba(0, 0, 0, 0.1)'
        };
    }
};

// =============================================================================
// 1️⃣ URL / PARAM HELPERS
// =============================================================================

/**
 * Get query parameter from current URL
 * @param {string} param - Parameter name to retrieve
 * @returns {string|null} Parameter value or null
 */
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

/**
 * Get file_id from URL using WorkflowService
 * @returns {string|null} File ID or null
 */
function getFileId() {
    return window.WorkflowService ? window.WorkflowService.getFileId() : getQueryParam('file');
}

/**
 * Redirect to a page with error message in URL
 * @param {string} page - Target page (e.g., 'schema.html')
 * @param {string} message - Error message to pass
 */
function redirectWithError(page, message) {
    const encodedMessage = encodeURIComponent(message);
    window.location.href = `${page}?error=${encodedMessage}`;
}

/**
 * Apply error from URL parameter and display banner
 * Checks for ?error=message in URL and shows error banner
 */
function applyErrorFromURL() {
    const errorMsg = getQueryParam('error');
    if (errorMsg) {
        showErrorBanner(decodeURIComponent(errorMsg), { autoHide: false });
        // Clean URL without reloading page
        const url = new URL(window.location);
        url.searchParams.delete('error');
        window.history.replaceState({}, '', url);
    }
}

// =============================================================================
// 2️⃣ UI BANNERS
// =============================================================================

/**
 * Show success banner with auto-hide
 * @param {string} message - Success message to display
 * @param {boolean} autoHide - Whether to auto-hide after 3 seconds (default: true)
 */
function showSuccessBanner(message, autoHide = true) {
    // Remove any existing banners
    const existingBanners = document.querySelectorAll('.alert');
    existingBanners.forEach(banner => banner.remove());

    const banner = document.createElement('div');
    banner.className = 'alert alert-success';
    banner.innerHTML = `<span>✅ ${message}</span>`;
    banner.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        background: #d4edda;
        color: #155724;
        padding: 1rem 1.5rem;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-width: 500px;
        animation: slideInRight 0.3s ease-out;
    `;

    document.body.appendChild(banner);

    if (autoHide) {
        setTimeout(() => {
            banner.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => banner.remove(), 300);
        }, 3000);
    }
}

/**
 * Show error banner with optional back button
 * @param {string} message - Error message to display
 * @param {Object} options - Configuration options
 * @param {boolean} options.showBackButton - Show back navigation button
 * @param {string} options.backPage - Page to navigate back to
 * @param {boolean} options.autoHide - Auto-hide after 5 seconds (default: false)
 */
function showErrorBanner(message, options = {}) {
    const { showBackButton = false, backPage = null, autoHide = false } = options;

    // Remove any existing banners
    const existingBanners = document.querySelectorAll('.alert');
    existingBanners.forEach(banner => banner.remove());

    const banner = document.createElement('div');
    banner.className = 'alert alert-error';

    let html = `<span>⚠️ ${message}</span>`;

    if (showBackButton && backPage) {
        const fileId = getFileId();
        const backUrl = fileId && window.WorkflowService
            ? window.WorkflowService.buildUrl(backPage, fileId)
            : backPage;

        html += `
            <button 
                onclick="window.location.href='${backUrl}'" 
                style="
                    margin-left: 1rem;
                    padding: 0.5rem 1rem;
                    background: #721c24;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 0.9rem;
                "
            >
                ← Back to ${backPage.replace('.html', '')}
            </button>
        `;
    }

    banner.innerHTML = html;
    banner.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        background: #f8d7da;
        color: #721c24;
        padding: 1rem 1.5rem;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-width: 500px;
        animation: slideInRight 0.3s ease-out;
        display: flex;
        align-items: center;
        justify-content: space-between;
    `;

    document.body.appendChild(banner);

    if (autoHide) {
        setTimeout(() => {
            banner.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => banner.remove(), 300);
        }, 5000);
    }
}

// =============================================================================
// 3️⃣ LOADING HANDLERS
// =============================================================================

/**
 * Show loading spinner in container
 * @param {string} containerId - ID of container element
 * @param {string} message - Loading message (default: "Loading...")
 */
function showLoading(containerId, message = "Loading...") {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Container ${containerId} not found`);
        return;
    }

    const loadingHTML = `
        <div class="loading-box" style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            background: #f8f9fa;
            border-radius: 8px;
            border: 2px dashed #dee2e6;
        ">
            <div class="spinner" style="
                width: 50px;
                height: 50px;
                border: 5px solid #e9ecef;
                border-top-color: var(--primary-color, #007bff);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            "></div>
            <p style="
                margin-top: 1rem;
                color: #6c757d;
                font-size: 1rem;
            ">${message}</p>
        </div>
    `;

    container.innerHTML = loadingHTML;

    // Add spin animation if not already in document
    if (!document.getElementById('spinner-keyframes')) {
        const style = document.createElement('style');
        style.id = 'spinner-keyframes';
        style.textContent = `
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOutRight {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
}

/**
 * Hide loading spinner and clear container
 * @param {string} containerId - ID of container element
 */
function hideLoading(containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Container ${containerId} not found`);
        return;
    }
    container.innerHTML = '';
}

// =============================================================================
// 4️⃣ BUTTON STATE HANDLERS
// =============================================================================

/**
 * Disable a button
 * @param {string} buttonId - ID of button element
 */
function disableButton(buttonId) {
    const button = document.getElementById(buttonId);
    if (button) {
        button.disabled = true;
        button.style.opacity = '0.5';
        button.style.cursor = 'not-allowed';
    }
}

/**
 * Enable a button
 * @param {string} buttonId - ID of button element
 */
function enableButton(buttonId) {
    const button = document.getElementById(buttonId);
    if (button) {
        button.disabled = false;
        button.style.opacity = '1';
        button.style.cursor = 'pointer';
    }
}

// =============================================================================
// 5️⃣ CONTINUE BUTTON LOGIC
// =============================================================================

/**
 * Initialize Continue button for manual progression
 * @param {string} nextPageName - Name of next page (e.g., 'cleaning.html')
 */
function initContinueButton(nextPageName) {
    const continueBtn = document.getElementById('continueBtn');
    if (!continueBtn) {
        console.warn('Continue button not found (expected id="continueBtn")');
        return;
    }

    // Initially disable button
    disableButton('continueBtn');

    // Set up click handler
    continueBtn.addEventListener('click', function () {
        // Validate file_id exists
        const fileId = getFileId();
        if (!fileId) {
            showErrorBanner('No file ID found. Please start from the upload page.', {
                showBackButton: true,
                backPage: 'index.html'
            });
            return;
        }

        // Check if button is disabled
        if (continueBtn.disabled) {
            showErrorBanner('Please wait for the current operation to complete.', {
                autoHide: true
            });
            return;
        }

        // Navigate to next page using WorkflowService
        if (window.WorkflowService) {
            const nextUrl = window.WorkflowService.buildUrl(nextPageName, fileId);
            window.location.href = nextUrl;
        } else {
            // Fallback if WorkflowService not available
            window.location.href = `${nextPageName}?file=${encodeURIComponent(fileId)}`;
        }
    });
}

/**
 * Enable Continue button after successful operation
 * @param {string} buttonId - ID of continue button (default: 'continueBtn')
 */
function enableContinueButton(buttonId = 'continueBtn') {
    enableButton(buttonId);
    const button = document.getElementById(buttonId);
    if (button) {
        button.classList.add('btn-ready');
        button.style.animation = 'pulse 0.5s ease-in-out';
    }
}

// =============================================================================
// 6️⃣ CENTRALIZED API WRAPPER
// =============================================================================

/**
 * Centralized API request wrapper with error handling
 * @param {string} method - HTTP method (GET, POST, PUT, DELETE)
 * @param {string} url - API endpoint URL
 * @param {Object} data - Request body data (optional)
 * @param {Object} options - Additional options
 * @param {Object} options.headers - Custom headers
 * @param {number} options.timeout - Request timeout in ms (default: 30000)
 * @returns {Promise<{ok: boolean, data: any, error: string|null, statusCode: number}>}
 */
async function apiRequest(method, url, data = null, options = {}) {
    const { headers = {}, timeout = 30000 } = options;

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        const fetchOptions = {
            method: method.toUpperCase(),
            headers: {
                'Content-Type': 'application/json',
                ...headers
            },
            signal: controller.signal
        };

        // Add body for POST/PUT/PATCH requests
        if (data && ['POST', 'PUT', 'PATCH'].includes(method.toUpperCase())) {
            fetchOptions.body = JSON.stringify(data);
        }

        console.log("Calling:", url);
        const response = await fetch(url, fetchOptions);
        clearTimeout(timeoutId);

        return await parseApiResponse(response);

    } catch (error) {
        // Network error or timeout
        if (error.name === 'AbortError') {
            return {
                ok: false,
                data: null,
                error: 'Request timeout. Please check your connection and try again.',
                statusCode: 0
            };
        }

        return {
            ok: false,
            data: null,
            error: error.message || 'Network error. Please check your connection.',
            statusCode: 0
        };
    }
}

// =============================================================================
// 7️⃣ GENERIC RESPONSE PARSER
// =============================================================================

/**
 * Parse API response consistently
 * @param {Response} response - Fetch API response object
 * @returns {Promise<{ok: boolean, data: any, message: string|null, statusCode: number}>}
 */
async function parseApiResponse(response) {
    const statusCode = response.status;
    let data = null;
    let message = null;

    try {
        // Try to parse as JSON
        const text = await response.text();
        if (text) {
            try {
                data = JSON.parse(text);
            } catch (parseError) {
                // Not JSON, treat as plain text
                data = { message: text };
            }
        }
    } catch (error) {
        console.error('Failed to read response body:', error);
    }

    // Extract message from various response formats
    if (data) {
        message = data.message || data.error || data.detail || null;
    }

    // Determine success based on status code
    const ok = statusCode >= 200 && statusCode < 300;

    if (!ok && !message) {
        // Provide default error messages
        if (statusCode === 404) {
            message = 'Resource not found';
        } else if (statusCode === 500) {
            message = 'Server error. Please try again later.';
        } else if (statusCode === 401 || statusCode === 403) {
            message = 'Access denied';
        } else {
            message = `Request failed with status ${statusCode}`;
        }
    }

    return {
        ok,
        data,
        message,
        statusCode
    };
}

// =============================================================================
// 8️⃣ PAGE BOOTSTRAP (PER PAGE SETUP)
// =============================================================================

/**
 * Bootstrap page with standard setup tasks
 * - Initialize theme system
 * - Validate file_id exists
 * - Apply error from URL
 * - Display file info
 * - Show progress
 */
function bootstrapPage() {
    // 0. Initialize theme system first
    ThemeService.init();

    // 1. Check for file_id (skip for index.html)
    const currentPage = window.WorkflowService
        ? window.WorkflowService.getCurrentPage()
        : window.location.pathname.split('/').pop();

    if (currentPage !== 'index.html') {
        const fileId = getFileId();
        if (!fileId) {
            // Redirect to upload page
            showErrorBanner('No file ID found. Redirecting to upload page...', { autoHide: true });
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 2000);
            return;
        }
    }

    // 2. Apply error from URL if present
    applyErrorFromURL();

    // 3. Display file info using WorkflowService
    if (window.WorkflowService && window.WorkflowService.displayFileInfo) {
        window.WorkflowService.displayFileInfo();
    } else {
        displayFileInfo();
    }

    // 4. Show progress
    if (window.WorkflowService && window.WorkflowService.showProgress) {
        window.WorkflowService.showProgress();
    } else {
        showProgress();
    }

    console.log('✅ Page bootstrapped successfully');
}

// =============================================================================
// 9️⃣ FILE INFO DISPLAY
// =============================================================================

/**
 * Display file information in designated element
 * @param {string} elementId - ID of element to display info in (default: 'fileInfo')
 */
function displayFileInfo(elementId = 'fileInfo') {
    const fileId = getFileId();
    const element = document.getElementById(elementId);

    if (!element) {
        console.warn(`Element ${elementId} not found`);
        return;
    }

    if (fileId) {
        element.innerHTML = `
            <span style="
                display: inline-block;
                background: #e8f4fd;
                color: var(--primary-color, #007bff);
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-size: 0.9rem;
                border: 1px solid rgba(0,123,255,0.2);
            ">
                📁 Active File: <strong>${fileId}</strong>
            </span>
        `;
    } else {
        element.innerHTML = '<span style="color: #6c757d;">No file selected</span>';
    }
}

// =============================================================================
// 🔟 PROGRESS DISPLAY
// =============================================================================

/**
 * Show workflow progress percentage
 * @param {string} elementId - ID of element to display progress (default: 'progressBar')
 */
function showProgress(elementId = 'progressBar') {
    const element = document.getElementById(elementId);
    if (!element) {
        return;
    }

    const currentPage = window.WorkflowService
        ? window.WorkflowService.getCurrentPage()
        : window.location.pathname.split('/').pop();

    const progress = window.WorkflowService
        ? window.WorkflowService.getWorkflowProgress(currentPage)
        : 0;

    element.innerHTML = `
        <div style="
            width: 100%;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            height: 24px;
            position: relative;
        ">
            <div style="
                width: ${progress}%;
                background: linear-gradient(90deg, var(--primary-color, #007bff), #0056b3);
                height: 100%;
                transition: width 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 0.85rem;
                font-weight: bold;
            ">
                ${progress}%
            </div>
        </div>
    `;
}

// =============================================================================
// 1️⃣2️⃣ ADVANCED UI EXPERIENCE COMPONENTS
// =============================================================================

/**
 * Animated Loading Overlay
 */
function showLoadingOverlay(message = 'Processing...') {
    let overlay = document.getElementById('global-loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'global-loading-overlay';
        overlay.innerHTML = `
            <div class="loading-overlay-backdrop">
                <div class="loading-overlay-content">
                    <div class="loading-spinner-large"></div>
                    <p class="loading-overlay-text">${message}</p>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
    } else {
        overlay.querySelector('.loading-overlay-text').textContent = message;
        overlay.style.display = 'flex';
    }
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('global-loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

/**
 * Page Transition Fade-in/out
 */
function initPageTransitions() {
    document.body.classList.add('page-fade-in');

    document.addEventListener('click', (e) => {
        const link = e.target.closest('a[href$=".html"]');
        if (link && !link.hasAttribute('target')) {
            e.preventDefault();
            const href = link.getAttribute('href');
            document.body.classList.add('page-fade-out');
            setTimeout(() => {
                window.location.href = href;
            }, 300);
        }
    });
}

/**
 * Progress Timeline (7-step pipeline)
 */
function renderProgressTimeline(currentStep) {
    const steps = [
        { id: 'upload', label: 'Upload', icon: '📤' },
        { id: 'schema', label: 'Schema', icon: '🔧' },
        { id: 'cleaning', label: 'Clean', icon: '🧹' },
        { id: 'weighting', label: 'Weight', icon: '⚖️' },
        { id: 'analysis', label: 'Analyze', icon: '📊' },
        { id: 'insight', label: 'Insights', icon: '🧠' },
        { id: 'report', label: 'Report', icon: '📄' }
    ];

    const timelineContainer = document.getElementById('progress-timeline');
    if (!timelineContainer) return;

    const currentIndex = steps.findIndex(s => s.id === currentStep);

    let html = '<div class="timeline-wrapper">';
    steps.forEach((step, index) => {
        const status = index < currentIndex ? 'completed'
            : index === currentIndex ? 'active'
                : 'pending';

        html += `
            <div class="timeline-step ${status}" data-step="${step.id}">
                <div class="timeline-icon">${step.icon}</div>
                <div class="timeline-label">${step.label}</div>
                ${index < steps.length - 1 ? '<div class="timeline-connector"></div>' : ''}
            </div>
        `;
    });
    html += '</div>';

    timelineContainer.innerHTML = html;
}

/**
 * Breadcrumb Navigation Component
 */
function renderBreadcrumbs(currentPage) {
    const breadcrumbContainer = document.getElementById('breadcrumb-nav');
    if (!breadcrumbContainer) return;

    const pages = window.WorkflowService?.WORKFLOW_ORDER || [];
    const currentIndex = pages.indexOf(currentPage);

    let html = '<nav class="breadcrumb-wrapper"><ol class="breadcrumb-list">';
    html += '<li class="breadcrumb-item"><a href="index.html">🏠 Home</a></li>';

    for (let i = 0; i <= currentIndex; i++) {
        const page = pages[i];
        const label = page.replace('.html', '').replace(/^\w/, c => c.toUpperCase());

        if (i === currentIndex) {
            html += `<li class="breadcrumb-item active">${label}</li>`;
        } else {
            const fileId = window.WorkflowService?.getFileId() || '';
            html += `<li class="breadcrumb-item"><a href="${page}?file=${fileId}">${label}</a></li>`;
        }
    }

    html += '</ol></nav>';
    breadcrumbContainer.innerHTML = html;
}

/**
 * Session Persistence Module
 */
function saveToSession(key, value) {
    try {
        sessionStorage.setItem(`statflow_${key}`, JSON.stringify(value));
    } catch (e) {
        console.warn('Session storage failed:', e);
    }
}

function loadFromSession(key) {
    try {
        const data = sessionStorage.getItem(`statflow_${key}`);
        return data ? JSON.parse(data) : null;
    } catch (e) {
        console.warn('Session retrieval failed:', e);
        return null;
    }
}

/**
 * Auto-restore file_id across pages
 */
function autoRestoreFileId() {
    let fileId = window.WorkflowService?.getFileId();

    if (!fileId) {
        fileId = loadFromSession('last_file_id');
        if (fileId) {
            console.log('Restored file_id from session:', fileId);
            const url = new URL(window.location);
            url.searchParams.set('file', fileId);
            window.history.replaceState({}, '', url);
        }
    } else {
        saveToSession('last_file_id', fileId);
    }

    return fileId;
}

/**
 * LocalStorage Cache for latest results
 */
function cacheResult(key, data, ttl = 3600000) {
    const cacheEntry = {
        data: data,
        timestamp: Date.now(),
        ttl: ttl
    };
    localStorage.setItem(`statflow_cache_${key}`, JSON.stringify(cacheEntry));
}

function getCachedResult(key) {
    const cached = localStorage.getItem(`statflow_cache_${key}`);
    if (!cached) return null;

    try {
        const entry = JSON.parse(cached);
        if (Date.now() - entry.timestamp > entry.ttl) {
            localStorage.removeItem(`statflow_cache_${key}`);
            return null;
        }
        return entry.data;
    } catch (e) {
        return null;
    }
}

function clearCache(key) {
    if (key) {
        localStorage.removeItem(`statflow_cache_${key}`);
    } else {
        Object.keys(localStorage).forEach(k => {
            if (k.startsWith('statflow_cache_')) {
                localStorage.removeItem(k);
            }
        });
    }
}

/**
 * Error Recovery Component
 */
function renderErrorRecovery(error, retryCallback) {
    return `
        <div class="error-recovery-container">
            <div class="error-recovery-header">
                <span class="error-icon">⚠️</span>
                <h3>Something Went Wrong</h3>
            </div>
            <div class="error-recovery-body">
                <p class="error-message">${error.message || 'An unexpected error occurred'}</p>
                <div class="error-actions">
                    <button class="btn btn-primary" onclick="(${retryCallback.toString()})()">
                        🔄 Retry
                    </button>
                    <button class="btn btn-secondary" onclick="window.location.reload()">
                        ↻ Refresh Page
                    </button>
                    <button class="btn btn-outline" onclick="window.WorkflowService?.goToUpload()">
                        ← Start Over
                    </button>
                </div>
                <details class="error-details">
                    <summary>Technical Details</summary>
                    <pre>${error.stack || error.toString()}</pre>
                </details>
            </div>
        </div>
    `;
}

/**
 * Retry Action UI Block
 */
async function retryWithExponentialBackoff(fn, maxRetries = 3, initialDelay = 1000) {
    let lastError;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;

            if (attempt < maxRetries - 1) {
                const delay = initialDelay * Math.pow(2, attempt);
                showWarningBanner(`Attempt ${attempt + 1} failed. Retrying in ${delay / 1000}s...`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    throw lastError;
}

/**
 * Enable Back Button
 */
function enableBackButton(callback) {
    const backBtn = document.getElementById('backBtn');
    if (backBtn) {
        backBtn.addEventListener('click', callback);
        backBtn.style.display = 'inline-block';
    }
}

/**
 * Show File Info Banner
 */
function showFileInfo(fileId, pageName) {
    const fileInfoContainer = document.getElementById('fileInfo');
    if (fileInfoContainer && fileId) {
        fileInfoContainer.innerHTML = `
            <div style="background: #e8f4fd; padding: 1rem 1.5rem; border-radius: 8px; border-left: 4px solid var(--primary-color); margin-bottom: 1.5rem;">
                <strong>📁 File ID:</strong> ${fileId} | <strong>📄 Page:</strong> ${pageName}
            </div>
        `;
    }
}

/**
 * Show Warning Banner
 */
function showWarningBanner(message) {
    const banner = document.createElement('div');
    banner.className = 'banner warning-banner';
    banner.innerHTML = `
        <span class="banner-icon">⚠️</span>
        <span class="banner-message">${message}</span>
    `;
    banner.style.cssText = 'background: #fef3c7; color: #92400e; border-left: 4px solid #f59e0b; padding: 1rem; margin-bottom: 1rem; border-radius: 8px; display: flex; align-items: center; gap: 0.5rem;';

    const container = document.querySelector('.container') || document.body;
    container.insertBefore(banner, container.firstChild);

    setTimeout(() => banner.remove(), 5000);
}

/**
 * Insert assistant component into page
 */
async function insertAssistantComponent() {
    try {
        const url = 'components/assistant.html';
        console.log("Calling:", url);
        const response = await fetch(url);
        const html = await response.text();
        document.body.insertAdjacentHTML('beforeend', html);
        console.log('✅ Assistant component loaded');
    } catch (error) {
        console.error('Failed to load assistant component:', error);
    }
}

/**
 * Scroll chat container to bottom
 * @param {string} containerId - ID of chat container
 */
function scrollChatToBottom(containerId = 'assistantChat') {
    const container = document.getElementById(containerId);
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

/**
 * Smooth slide animation for elements
 * @param {HTMLElement} element - Element to animate
 * @param {string} direction - 'in' or 'out'
 */
function smoothSlide(element, direction = 'in') {
    if (!element) return;

    if (direction === 'in') {
        element.style.transform = 'translateX(0)';
        element.style.opacity = '1';
    } else {
        element.style.transform = 'translateX(100%)';
        element.style.opacity = '0';
    }
}

/**
 * Update pipeline step status
 * @param {string} step - Step name
 * @param {string} state - Status state (pending, running, success, error)
 */
function updateStepStatus(step, state) {
    const stepElement = document.querySelector(`.pipeline-step[data-step="${step}"]`);
    if (!stepElement) return;

    stepElement.classList.remove('pending', 'active', 'success', 'error');
    stepElement.classList.add(state === 'running' ? 'active' : state);

    const statusBadge = stepElement.querySelector('.step-status');
    if (statusBadge) {
        statusBadge.setAttribute('data-status', state);
    }
}

/**
 * Update main progress bar
 * @param {number} percent - Progress percentage (0-100)
 */
function updateMainProgress(percent) {
    const progressBar = document.getElementById('progressBarFill');
    const progressText = document.getElementById('overallProgress');

    if (progressBar) {
        progressBar.style.width = `${percent}%`;
    }

    if (progressText) {
        progressText.textContent = `${percent}%`;
    }
}

/**
 * Append log to pipeline console
 * @param {string} text - Log message
 * @param {string} type - Log type (info, success, error, warning)
 */
function appendPipelineLog(text, type = 'info') {
    const console = document.getElementById('pipelineLogConsole');
    if (!console) return;

    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${type}`;
    logEntry.innerHTML = `
        <span class="log-timestamp">[${timestamp}]</span>
        <span class="log-message">${text}</span>
    `;

    console.appendChild(logEntry);
    scrollLogsToEnd();
}

/**
 * Scroll logs console to end
 */
function scrollLogsToEnd() {
    const console = document.getElementById('pipelineLogConsole');
    if (console) {
        console.scrollTop = console.scrollHeight;
    }
}

/**
 * Update live status from WebSocket message
 * @param {object} msg - WebSocket message object
 */
function updateLiveStatus(msg) {
    const { event, step, status, message } = msg;

    if (step && status) {
        updateStepStatus(step, status);
    }

    if (message) {
        appendLiveLog(message, status === 'error' ? 'error' : 'info');
    }
}

/**
 * Append live log message with animation
 * @param {string} message - Log message
 * @param {string} level - Log level (info, success, error, warning)
 */
function appendLiveLog(message, level = 'info') {
    const liveLogsContainer = document.getElementById('wsLiveLogs');
    if (!liveLogsContainer) return;

    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${level} ws-log-live`;
    logEntry.innerHTML = `
        <span class="log-timestamp">[${timestamp}]</span>
        <span class="log-message">${message}</span>
    `;

    liveLogsContainer.appendChild(logEntry);

    // Auto-scroll
    liveLogsContainer.scrollTop = liveLogsContainer.scrollHeight;

    // Remove animation class after animation completes
    setTimeout(() => {
        logEntry.classList.remove('ws-log-live');
    }, 1000);

    // Keep only last 50 logs
    while (liveLogsContainer.children.length > 50) {
        liveLogsContainer.removeChild(liveLogsContainer.firstChild);
    }
}

/**
 * Update live progress with animation
 * @param {number} percent - Progress percentage (0-100)
 */
function updateLiveProgress(percent) {
    updateMainProgress(percent);

    // Add pulse effect to progress bar
    const progressBar = document.getElementById('progressBarFill');
    if (progressBar) {
        progressBar.classList.add('ws-badge-live');
        setTimeout(() => {
            progressBar.classList.remove('ws-badge-live');
        }, 500);
    }
}

/**
 * Flash step with animation
 * @param {string} stepName - Name of the step to flash
 */
function flashStep(stepName) {
    const stepElement = document.querySelector(`.pipeline-step[data-step="${stepName}"]`);
    if (!stepElement) return;

    stepElement.classList.add('ws-step-flash');

    setTimeout(() => {
        stepElement.classList.remove('ws-step-flash');
    }, 1000);
}

// =============================================================================
// 1️⃣1️⃣ EXPORT ALL FUNCTIONS GLOBALLY
// =============================================================================

window.UIHelper = {
    // URL helpers
    getQueryParam,
    getFileId,
    redirectWithError,
    applyErrorFromURL,

    // Banners
    showSuccessBanner,
    showErrorBanner,
    showWarningBanner,

    // Loading
    showLoading,
    hideLoading,
    showLoadingOverlay,
    hideLoadingOverlay,

    // Button state
    disableButton,
    enableButton,
    enableContinueButton,
    enableBackButton,

    // Continue button
    initContinueButton,

    // API
    apiRequest,
    parseApiResponse,

    // Page setup
    bootstrapPage,

    // Display
    displayFileInfo,
    showProgress,
    showFileInfo,

    // Advanced UI
    initPageTransitions,
    renderProgressTimeline,
    renderBreadcrumbs,
    saveToSession,
    loadFromSession,
    autoRestoreFileId,
    cacheResult,
    getCachedResult,
    clearCache,
    renderErrorRecovery,
    retryWithExponentialBackoff,

    // Assistant helpers
    insertAssistantComponent,
    scrollChatToBottom,
    smoothSlide,

    // Pipeline helpers
    updateStepStatus,
    updateMainProgress,
    appendPipelineLog,
    scrollLogsToEnd,

    // WebSocket helpers
    updateLiveStatus,
    appendLiveLog,
    updateLiveProgress,
    flashStep
};

// Export ThemeService
window.ThemeService = ThemeService;

// Log successful load
console.log('✅ UIHelper module loaded successfully');
console.log('✅ ThemeService loaded successfully');

// Auto-bootstrap on DOMContentLoaded (optional - can be disabled by setting data-no-auto-bootstrap on body)
document.addEventListener('DOMContentLoaded', function () {
    if (!document.body.hasAttribute('data-no-auto-bootstrap')) {
        // Small delay to ensure WorkflowService is loaded
        setTimeout(() => {
            bootstrapPage();
        }, 100);
    }
});
