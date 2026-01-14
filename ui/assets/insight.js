/**
 * insight.js - AI Insights with Chart.js Integration
 */

if (!window.API_BASE) console.error("API_BASE not ready");
window.insightCharts = {};

const InsightModule = {
    fileId: null,
    columns: [],

    async init(fileId) {
        this.fileId = fileId;
        try {
            UIHelper.showLoadingOverlay?.('Generating AI insights...');
            await this.loadColumns();
            await this.runFullInsightGeneration();
            this.initializeChartControls();
            UIHelper.hideLoadingOverlay?.();
            UIHelper.enableContinueButton?.('continueBtn');
            document.getElementById('continueBtn').onclick = () => {
                window.location.href = `report.html?file_id=${fileId}`;
            };
        } catch (error) {
            console.error('Insight init error:', error);
            UIHelper.hideLoadingOverlay?.();
            UIHelper.showErrorBanner?.('Failed to generate insights: ' + error.message);
        }
    },

    async loadColumns() {
        try {
            const url = `${window.API_BASE}/upload/columns/${this.fileId}`;
            console.log("Calling:", url);
            const response = await fetch(url);
            const data = await response.json();
            if (data.columns) this.columns = data.columns;
        } catch (error) {
            console.error('Column loading error:', error);
        }
    },

    async runFullInsightGeneration() {
        try {
            const url = `${window.API_BASE}/insight/full`;
            console.log("Calling:", url);
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_id: this.fileId })
            });
            const data = await response.json();
            if (data.ok && data.data) {
                this.renderInsights(data.data);
                UIHelper.showSuccessBanner?.('Insights generated successfully!');
            } else {
                throw new Error(data.message || 'Insight generation failed');
            }
        } catch (error) {
            console.error('Insight generation error:', error);
            throw error;
        }
    },

    renderInsights(insightData) {
        const insights = insightData.insights || insightData;
        const numericInsights = insights.numeric_insights || [];
        const categoricalInsights = insights.categorical_insights || [];
        const narrative = insights.narrative || '';

        let html = `
            <div class="card mb-4">
                ${narrative ? `
                    <div style="background: #e3f2fd; padding: 1.5rem; border-radius: var(--radius); border-left: 4px solid var(--accent); margin-bottom: 1.5rem;">
                        <h3 style="margin-bottom: 0.75rem;">📝 Executive Summary</h3>
                        <p style="line-height: 1.6; margin: 0;">${narrative}</p>
                    </div>
                ` : ''}

                ${numericInsights.length > 0 ? `
                    <div class="mb-4">
                        <h3 class="mb-3">📊 Numeric Insights</h3>
                        <div style="display: grid; gap: 1rem;">
                            ${numericInsights.map((insight, i) => `
                                <div style="background: ${i % 2 === 0 ? '#f8f9fa' : '#e8f4fd'}; padding: 1rem; border-radius: var(--radius); border-left: 3px solid ${i % 2 === 0 ? '#28a745' : '#007bff'};">
                                    <div style="display: flex; gap: 0.75rem; align-items: start;">
                                        <span style="font-size: 1.5rem;">📈</span>
                                        <div>
                                            <div style="font-weight: bold; margin-bottom: 0.25rem;">${insight.variable || 'Variable'}</div>
                                            <div style="color: #6c757d; font-size: 0.95rem;">${insight.insight || insight.description || insight}</div>
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}

                ${categoricalInsights.length > 0 ? `
                    <div class="mb-4">
                        <h3 class="mb-3">📋 Categorical Insights</h3>
                        <div style="display: grid; gap: 1rem;">
                            ${categoricalInsights.map((insight, i) => `
                                <div style="background: ${i % 2 === 0 ? '#fff3cd' : '#d4edda'}; padding: 1rem; border-radius: var(--radius); border-left: 3px solid ${i % 2 === 0 ? '#ffc107' : '#28a745'};">
                                    <div style="display: flex; gap: 0.75rem; align-items: start;">
                                        <span style="font-size: 1.5rem;">📊</span>
                                        <div>
                                            <div style="font-weight: bold; margin-bottom: 0.25rem;">${insight.variable || 'Category'}</div>
                                            <div style="color: #6c757d; font-size: 0.95rem;">${insight.insight || insight.description || insight}</div>
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;

        document.getElementById('insightResults').innerHTML = html;
    },

    initializeChartControls() {
        const select = document.getElementById('insightColumn');
        this.columns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            select.appendChild(option);
        });

        select.addEventListener('change', (e) => {
            if (e.target.value) this.loadInsightChart(e.target.value);
        });
    },

    async loadInsightChart(column) {
        const loading = document.getElementById('insightLoading');
        const canvas = document.getElementById('insightChart');

        try {
            loading.style.display = 'block';
            const url = `${window.API_BASE}/dashboard/bar/${this.fileId}/${column}`;
            console.log("Calling:", url);
            const response = await fetch(url);
            const data = await response.json();
            loading.style.display = 'none';

            if (!data.ok) throw new Error(data.message);

            if (window.insightCharts.main) window.insightCharts.main.destroy();

            const ctx = canvas.getContext('2d');
            window.insightCharts.main = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.data.labels || [],
                    datasets: [{
                        label: column,
                        data: data.data.values || [],
                        backgroundColor: 'rgba(139, 92, 246, 0.7)',
                        borderColor: 'rgba(139, 92, 246, 1)',
                        borderWidth: 2,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: { backgroundColor: 'rgba(0,0,0,0.8)', padding: 12, cornerRadius: 6 } },
                    scales: {
                        y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: '#6b7280' } },
                        x: { grid: { display: false }, ticks: { color: '#6b7280', maxRotation: 45 } }
                    }
                }
            });
        } catch (error) {
            console.error('Chart error:', error);
            loading.style.display = 'none';
            UIHelper.showErrorBanner?.('Failed to load chart: ' + error.message);
        }
    }
};

window.InsightModule = InsightModule;
