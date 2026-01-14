// =======================================================================
// report.js – FINAL MVP VERSION
// Downloads PDF report from /api/v1/report/generate
// =======================================================================

window.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("generateReportBtn") || document.getElementById("generateBtn");
    if (btn) {
        btn.addEventListener("click", generateReport);
    }
});

async function generateReport() {
    const fileId = sessionStorage.getItem("uploadedFileId");

    const url = `${window.API_BASE}/report/generate`;

    const payload = { file_id: fileId };

    const statusEl = document.getElementById("statusMessage");
    if (statusEl) statusEl.innerText = "Generating report...";

    try {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        console.log("[report.js] Response:", data);

        if (data.status !== "success") throw new Error(data.error || "Report generation failed");

        if (statusEl) statusEl.innerText = "Report generated successfully!";

        // If there's a download URL, create a download link
        if (data.report_id || data.download_url) {
            const reportId = data.report_id;
            const downloadUrl = data.download_url || `${window.API_BASE}/report/download/${reportId}`;

            const link = document.createElement("a");
            link.href = downloadUrl;
            link.download = `report_${reportId}.pdf`;
            link.click();
        }

    } catch (err) {
        if (statusEl) statusEl.innerText = "Error: " + err.message;
        console.error("Report generation failed:", err);
    }
}
