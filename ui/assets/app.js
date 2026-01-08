/* ========================================
   StatFlow AI - Global Utilities
   Modern, reusable JavaScript utilities
   ======================================== */

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

/* ========================================
   FETCH WRAPPER - Enhanced API calls
   ======================================== */

/**
 * Enhanced fetch wrapper with loading states, error handling, and retries
 * @param {string} url - API endpoint
 * @param {Object} options - Fetch options
 * @param {Object} config - Additional configuration
 * @returns {Promise} Response data
 */
async function api(url, options = {}, config = {}) {
    const {
        showLoading = false,
        loadingElement = null,
        retry = 0,
        timeout = 30000
    } = config;

    // Show loading spinner if requested
    if (showLoading && loadingElement) {
        showSpinner(loadingElement);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        },
        signal: controller.signal,
        ...options
    };

    // Build full URL
    const fullUrl = url.startsWith('http') ? url : `${API_BASE_URL}${url}`;

    try {
        const response = await fetch(fullUrl, defaultOptions);
        clearTimeout(timeoutId);

        if (!response.ok) {
            // Try to parse error response
            let errorMessage;
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || `HTTP ${response.status}: ${response.statusText}`;
            } catch {
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }

        // Parse response based on content type
        const contentType = response.headers.get('content-type');
        let data;

        if (contentType?.includes('application/json')) {
            data = await response.json();
        } else if (contentType?.includes('text')) {
            data = await response.text();
        } else {
            data = await response.blob();
        }

        return data;

    } catch (error) {
        clearTimeout(timeoutId);

        // Retry logic
        if (retry > 0 && error.name !== 'AbortError') {
            console.log(`Retrying... (${retry} attempts left)`);
            await new Promise(resolve => setTimeout(resolve, 1000));
            return api(url, options, { ...config, retry: retry - 1 });
        }

        // Handle different error types
        if (error.name === 'AbortError') {
            throw new Error('Request timeout - please try again');
        }

        console.error('API Error:', error);
        throw error;

    } finally {
        // Hide loading spinner
        if (showLoading && loadingElement) {
            hideSpinner(loadingElement);
        }
    }
}

// Convenience methods for common HTTP verbs
const apiGet = (url, config = {}) => api(url, { method: 'GET' }, config);
const apiPost = (url, data, config = {}) => api(url, { method: 'POST', body: JSON.stringify(data) }, config);
const apiPut = (url, data, config = {}) => api(url, { method: 'PUT', body: JSON.stringify(data) }, config);
const apiDelete = (url, config = {}) => api(url, { method: 'DELETE' }, config);

/* ========================================
   URL PARAMETERS - Query string helpers
   ======================================== */

/**
 * Get URL parameter by name
 * @param {string} name - Parameter name
 * @param {string} defaultValue - Default value if not found
 * @returns {string|null} Parameter value
 */
function getUrlParam(name, defaultValue = null) {
    const params = new URLSearchParams(window.location.search);
    return params.get(name) || defaultValue;
}

/**
 * Get all URL parameters as object
 * @returns {Object} All parameters
 */
function getUrlParams() {
    const params = new URLSearchParams(window.location.search);
    const result = {};
    for (const [key, value] of params.entries()) {
        result[key] = value;
    }
    return result;
}

/**
 * Set URL parameter without page reload
 * @param {string} name - Parameter name
 * @param {string} value - Parameter value
 * @param {boolean} replace - Replace history state instead of push
 */
function setUrlParam(name, value, replace = false) {
    const url = new URL(window.location);
    url.searchParams.set(name, value);

    if (replace) {
        window.history.replaceState({}, '', url);
    } else {
        window.history.pushState({}, '', url);
    }
}

/**
 * Remove URL parameter
 * @param {string} name - Parameter name
 */
function removeUrlParam(name) {
    const url = new URL(window.location);
    url.searchParams.delete(name);
    window.history.replaceState({}, '', url);
}

/**
 * Build query string from object
 * @param {Object} params - Parameters object
 * @returns {string} Query string
 */
function buildQueryString(params) {
    return new URLSearchParams(params).toString();
}

/* ========================================
   LOADING SPINNER - Dynamic spinners
   ======================================== */

/**
 * Show loading spinner in element
 * @param {string|HTMLElement} target - Element ID or element
 * @param {Object} options - Spinner options
 */
function showSpinner(target, options = {}) {
    const {
        size = 'medium',
        color = 'primary',
        text = 'Loading...',
        overlay = false
    } = options;

    const element = typeof target === 'string' ? document.getElementById(target) : target;
    if (!element) return;

    const sizeMap = {
        small: '20px',
        medium: '40px',
        large: '60px'
    };

    const colorMap = {
        primary: 'var(--primary-600)',
        success: 'var(--success-600)',
        warning: 'var(--warning-600)',
        danger: 'var(--danger-600)'
    };

    const spinnerSize = sizeMap[size] || sizeMap.medium;
    const spinnerColor = colorMap[color] || colorMap.primary;

    const spinnerHTML = `
        <div class="spinner-container" style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            padding: 2rem;
            ${overlay ? 'position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255,255,255,0.9); z-index: 1000;' : ''}
        ">
            <div class="spinner" style="
                width: ${spinnerSize};
                height: ${spinnerSize};
                border: 3px solid rgba(0,0,0,0.1);
                border-top-color: ${spinnerColor};
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            "></div>
            ${text ? `<p style="color: var(--gray-600); font-size: 0.9375rem; margin: 0;">${text}</p>` : ''}
        </div>
    `;

    element.innerHTML = spinnerHTML;
}

/**
 * Hide loading spinner
 * @param {string|HTMLElement} target - Element ID or element
 */
function hideSpinner(target) {
    const element = typeof target === 'string' ? document.getElementById(target) : target;
    if (!element) return;

    const spinner = element.querySelector('.spinner-container');
    if (spinner) {
        spinner.remove();
    }
}

/**
 * Show full-page loading overlay
 * @param {string} text - Loading text
 */
function showPageLoader(text = 'Loading...') {
    const overlay = document.createElement('div');
    overlay.id = 'page-loader-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(4px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    `;

    overlay.innerHTML = `
        <div style="
            background: white;
            padding: 2rem 3rem;
            border-radius: 1rem;
            box-shadow: var(--shadow-2xl);
            text-align: center;
        ">
            <div style="
                width: 60px;
                height: 60px;
                border: 4px solid rgba(0,0,0,0.1);
                border-top-color: var(--primary-600);
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
                margin: 0 auto 1.5rem;
            "></div>
            <p style="
                color: var(--gray-700);
                font-size: 1.125rem;
                font-weight: 600;
                margin: 0;
            ">${text}</p>
        </div>
    `;

    document.body.appendChild(overlay);
}

/**
 * Hide full-page loading overlay
 */
function hidePageLoader() {
    const overlay = document.getElementById('page-loader-overlay');
    if (overlay) {
        overlay.remove();
    }
}

/* ========================================
   TOAST NOTIFICATIONS - User feedback
   ======================================== */

let toastContainer = null;

/**
 * Initialize toast container
 */
function initToastContainer() {
    if (toastContainer) return;

    toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    toastContainer.style.cssText = `
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 10000;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        max-width: 400px;
    `;
    document.body.appendChild(toastContainer);
}

/**
 * Show toast notification
 * @param {string} message - Toast message
 * @param {Object} options - Toast options
 */
function toast(message, options = {}) {
    const {
        type = 'info',
        duration = 3000,
        position = 'top-right',
        dismissible = true
    } = options;

    initToastContainer();

    const toastId = `toast-${Date.now()}`;
    const typeStyles = {
        success: {
            bg: 'var(--success-50)',
            border: 'var(--success-500)',
            icon: '✓',
            color: 'var(--success-700)'
        },
        error: {
            bg: 'var(--danger-50)',
            border: 'var(--danger-500)',
            icon: '✗',
            color: 'var(--danger-700)'
        },
        warning: {
            bg: 'var(--warning-50)',
            border: 'var(--warning-500)',
            icon: '⚠',
            color: 'var(--warning-700)'
        },
        info: {
            bg: 'var(--info-50)',
            border: 'var(--info-500)',
            icon: 'ℹ',
            color: 'var(--info-700)'
        }
    };

    const style = typeStyles[type] || typeStyles.info;

    const toastElement = document.createElement('div');
    toastElement.id = toastId;
    toastElement.style.cssText = `
        background: ${style.bg};
        border-left: 4px solid ${style.border};
        color: ${style.color};
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        box-shadow: var(--shadow-lg);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        min-width: 300px;
        animation: slideIn 0.3s ease-out;
        cursor: ${dismissible ? 'pointer' : 'default'};
    `;

    toastElement.innerHTML = `
        <span style="font-size: 1.25rem; font-weight: bold;">${style.icon}</span>
        <span style="flex: 1; font-weight: 500;">${message}</span>
        ${dismissible ? '<span style="font-size: 1.25rem; opacity: 0.5;">×</span>' : ''}
    `;

    // Add click to dismiss
    if (dismissible) {
        toastElement.addEventListener('click', () => dismissToast(toastId));
    }

    toastContainer.appendChild(toastElement);

    // Auto dismiss
    if (duration > 0) {
        setTimeout(() => dismissToast(toastId), duration);
    }

    return toastId;
}

/**
 * Dismiss toast by ID
 * @param {string} toastId - Toast ID
 */
function dismissToast(toastId) {
    const toastElement = document.getElementById(toastId);
    if (toastElement) {
        toastElement.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => toastElement.remove(), 300);
    }
}

// Convenience toast methods
const toastSuccess = (message, options = {}) => toast(message, { ...options, type: 'success' });
const toastError = (message, options = {}) => toast(message, { ...options, type: 'error' });
const toastWarning = (message, options = {}) => toast(message, { ...options, type: 'warning' });
const toastInfo = (message, options = {}) => toast(message, { ...options, type: 'info' });

/* ========================================
   LEGACY COMPATIBILITY
   ======================================== */

// Helper Functions
function getFileId() {
    return localStorage.getItem('currentFileId') || '';
}

function setFileId(fileId) {
    localStorage.setItem('currentFileId', fileId);
}

// Legacy API call wrapper (for backwards compatibility)
async function apiCall(endpoint, options = {}) {
    return api(endpoint, options);
}

// Format JSON for Display
function formatJSON(data) {
    return JSON.stringify(data, null, 2);
}

// Legacy loading methods
function showLoader(elementId) {
    showSpinner(elementId, { text: 'Loading...' });
}

function hideLoader(elementId) {
    hideSpinner(elementId);
}

// Display Success Message
function showSuccess(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<p class="success">✓ ${message}</p>`;
    }
}

// Display Error Message
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<p class="error">✗ ${message}</p>`;
    }
}

// Display Info Message
function showInfo(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<p class="info">${message}</p>`;
    }
}


/* ========================================
   ADDITIONAL UTILITIES
   ======================================== */

// Load Navigation Bar
async function loadNavbar() {
    try {
        const navContainer = document.getElementById('navbar-container');
        if (!navContainer) return;

        const response = await fetch('components/navbar.html');
        const navHTML = await response.text();
        navContainer.innerHTML = navHTML;

        // Highlight current page
        const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        const navLinks = navContainer.querySelectorAll('nav a');
        navLinks.forEach(link => {
            if (link.getAttribute('href') === currentPage) {
                link.classList.add('active');
            }
        });

        // Display file ID if available
        const fileIdDisplay = document.getElementById('nav-file-id');
        if (fileIdDisplay) {
            const fileId = getFileId();
            fileIdDisplay.textContent = fileId ? `File: ${fileId}` : 'No file loaded';
        }
    } catch (error) {
        console.error('Error loading navbar:', error);
    }
}

// Parse CSV to Array
function parseCSV(text) {
    const lines = text.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];

    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim()) {
            const values = lines[i].split(',').map(v => v.trim());
            const row = {};
            headers.forEach((header, idx) => {
                row[header] = values[idx];
            });
            data.push(row);
        }
    }

    return { headers, data };
}

// Download File
function downloadFile(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Validate File ID
function validateFileId(fileId) {
    if (!fileId || fileId.trim() === '') {
        toastError('Please enter a valid File ID');
        return false;
    }
    return true;
}

// Format Date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

// Debounce Function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle Function
function throttle(func, limit) {
    let inThrottle;
    return function (...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Deep clone object
function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

// Generate unique ID
function generateId(prefix = 'id') {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Format number with commas
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

// Truncate text
function truncate(text, length = 100, suffix = '...') {
    if (text.length <= length) return text;
    return text.substring(0, length) + suffix;
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Copy to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        toastSuccess('Copied to clipboard!');
        return true;
    } catch (error) {
        console.error('Copy failed:', error);
        toastError('Failed to copy to clipboard');
        return false;
    }
}

// LocalStorage with expiry
const storage = {
    set(key, value, expiryMinutes = null) {
        const item = {
            value: value,
            expiry: expiryMinutes ? Date.now() + (expiryMinutes * 60 * 1000) : null
        };
        localStorage.setItem(key, JSON.stringify(item));
    },

    get(key) {
        const itemStr = localStorage.getItem(key);
        if (!itemStr) return null;

        try {
            const item = JSON.parse(itemStr);

            if (item.expiry && Date.now() > item.expiry) {
                localStorage.removeItem(key);
                return null;
            }

            return item.value;
        } catch {
            return itemStr;
        }
    },

    remove(key) {
        localStorage.removeItem(key);
    },

    clear() {
        localStorage.clear();
    }
};

/* ========================================
   CSS ANIMATIONS (Inject into page)
   ======================================== */

function injectAnimations() {
    if (document.getElementById('app-animations')) return;

    const style = document.createElement('style');
    style.id = 'app-animations';
    style.textContent = `
        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(100%);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes slideOut {
            from {
                opacity: 1;
                transform: translateX(0);
            }
            to {
                opacity: 0;
                transform: translateX(100%);
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    `;
    document.head.appendChild(style);
}

/* ========================================
   INITIALIZATION
   ======================================== */

// Initialize App on DOM Load
document.addEventListener('DOMContentLoaded', () => {
    // Inject animations
    injectAnimations();

    // Load navbar
    loadNavbar();

    // Check for stored file ID
    const storedFileId = getFileId();
    if (storedFileId) {
        console.log('Loaded File ID:', storedFileId);
    }

    // Add global error handler
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled promise rejection:', event.reason);
        toastError('An unexpected error occurred');
    });
});

/* ========================================
   EXPORTS - Global namespace
   ======================================== */

// Export for use in other scripts
window.StatFlowAPI = {
    // Core
    API_BASE_URL,

    // API Methods
    api,
    apiGet,
    apiPost,
    apiPut,
    apiDelete,
    apiCall, // Legacy

    // URL Parameters
    getUrlParam,
    getUrlParams,
    setUrlParam,
    removeUrlParam,
    buildQueryString,

    // Loading Spinners
    showSpinner,
    hideSpinner,
    showPageLoader,
    hidePageLoader,
    showLoader, // Legacy
    hideLoader, // Legacy

    // Toast Notifications
    toast,
    toastSuccess,
    toastError,
    toastWarning,
    toastInfo,
    dismissToast,

    // File Management
    getFileId,
    setFileId,
    validateFileId,

    // Utilities
    formatJSON,
    formatDate,
    formatNumber,
    debounce,
    throttle,
    deepClone,
    generateId,
    truncate,
    escapeHtml,
    copyToClipboard,
    parseCSV,
    downloadFile,

    // Storage
    storage,

    // Legacy Display Methods
    showSuccess,
    showError,
    showInfo
};

console.log('✓ StatFlow API initialized');
