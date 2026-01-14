// =======================================================================
// upload.js – FINAL MVP VERSION (StatFlow)
// Uses backend /api/upload and stores fileId + fileName
// =======================================================================

console.log("[upload.js] Starting upload module...");

window.addEventListener("DOMContentLoaded", () => {
    initUploadUI();
});

function initUploadUI() {
    const input = document.getElementById("fileInput");
    const btn = document.getElementById("uploadBtn");

    if (!input || !btn) {
        console.error("[upload.js] Missing input or button");
        return;
    }

    btn.addEventListener("click", uploadFile);
    console.log("✅ Upload UI ready");
}

async function uploadFile() {
    const input = document.getElementById("fileInput");
    const file = input.files[0];

    if (!file) {
        alert("Select a file before uploading.");
        return;
    }

    const form = new FormData();
    form.append("file", file);

    const url = `${window.API_BASE}/upload/single`;
    console.log("[upload.js] Calling:", url);

    try {
        const res = await fetch(url, { method: "POST", body: form });
        const data = await res.json();

        console.log("[upload.js] Response:", data);

        if (!data.status || data.status !== "success") {
            throw new Error(data.error || "Upload failed");
        }

        // Extract file_id from backend response
        const fileId = data.file_ids && data.file_ids[0];
        const filename = file.name;

        if (!fileId) {
            throw new Error("No file_id returned from server");
        }

        // Store in sessionStorage
        sessionStorage.setItem("uploadedFileId", fileId);
        sessionStorage.setItem("uploadedFileName", filename);

        console.log("[upload.js] Stored in sessionStorage:")
        console.log("  - uploadedFileId:", fileId);
        console.log("  - uploadedFileName:", filename);

        // Redirect to schema mapping with file_id in URL
        window.location.href = `/ui/schema.html?file=${fileId}`;

    } catch (err) {
        console.error("Upload failed:", err);
        alert("Upload failed: " + err.message);
    }
}
