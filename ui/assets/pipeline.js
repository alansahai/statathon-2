// =======================================================================
// pipeline.js – FINAL MVP VERSION (StatFlow)
// Uses backend /api/run_pipeline to run everything in one shot
// =======================================================================

window.addEventListener("DOMContentLoaded", () => {
    document.getElementById("runPipelineBtn").addEventListener("click", runPipeline);
});

async function runPipeline() {
    const filename = sessionStorage.getItem("uploadedFileId");

    const url = `${window.API_BASE}/run_pipeline`;
    const payload = { filename: filename };

    try {
        document.getElementById("pipelineLog").innerText = "Running pipeline...";

        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        console.log("[pipeline.js] Response:", data);

        if (data.status !== "success") throw new Error(data.error);

        document.getElementById("pipelineLog").innerText =
            "Pipeline completed successfully.";

    } catch (err) {
        document.getElementById("pipelineLog").innerText = "Error: " + err.message;
    }
}
