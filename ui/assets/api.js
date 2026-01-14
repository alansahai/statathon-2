// ======================================================
//  GLOBAL API CONFIG FOR STATFLOW UI
// ======================================================
console.log("[API] Initializing...");

// Define global API base
window.API_BASE = "http://localhost:8000/api/v1";

// Define global health endpoint
window.API_HEALTH = "http://localhost:8000/health";

// Helper function to construct API URLs
function apiUrl(path) {
    if (!path.startsWith('/')) {
        path = '/' + path;
    }
    return window.API_BASE + path;
}

// Make helper globally available
window.apiUrl = apiUrl;

// Debug log
console.log("[API] API_BASE set to:", window.API_BASE);
