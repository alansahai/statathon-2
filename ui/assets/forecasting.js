/**
 * forecasting.js - Time Series Forecasting with Chart.js
 */

if (!window.API_BASE) console.error("API_BASE not ready");
window.forecastCharts = {};

const ForecastingModule = {
    fileId: null,
    columns: [],

    async init(fileId) {
        this.fileId = fileId;
        try {
            await this.loadColumns();
            this.initializeControls();
        } catch (error) {
            console.error('Forecasting init error:', error);
            UIHelper.showErrorBanner?.('Failed to initialize forecasting: ' + error.message);
        }
    },

    async loadColumns() {
        try {
            const response = await fetch(`${window.API_BASE}/upload/columns/${this.fileId}`);
            const data = await response.json();
            if (data.columns) this.columns = data.columns;
        } catch (error) {
            console.error('Column loading error:', error);
        }
    },

    initializeControls() {
        const timeSelect = document.getElementById('timeColumn');
        const targetSelect = document.getElementById('targetColumn');

        this.columns.forEach(col => {
            timeSelect.appendChild(new Option(col, col));
            targetSelect.appendChild(new Option(col, col));
        });

        document.getElementById('runForecastBtn').addEventListener('click', () => this.runForecast());
    },

    async runForecast() {
        const timeColumn = document.getElementById('timeColumn').value;
        const targetColumn = document.getElementById('targetColumn').value;
        const periods = parseInt(document.getElementById('forecastPeriods').value);
        const model = document.getElementById('forecastModel').value;

        if (!timeColumn || !targetColumn) {
            UIHelper.showErrorBanner?.('Please select both time column and target variable');
            return;
        }

        try {
            UIHelper.showLoadingOverlay?.('Generating forecast...');

            const response = await fetch(`${window.API_BASE}/forecasting/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_id: this.fileId,
                    time_column: timeColumn,
                    target_column: targetColumn,
                    periods: periods,
                    model: model
                })
            });

            const data = await response.json();
            UIHelper.hideLoadingOverlay?.();

            if (!data.ok) {
                throw new Error(data.message || 'Forecast failed');
            }

            this.renderForecastResults(data.data, targetColumn);
            UIHelper.showSuccessBanner?.('Forecast generated successfully!');
        } catch (error) {
            console.error('Forecast error:', error);
            UIHelper.hideLoadingOverlay?.();
            UIHelper.showErrorBanner?.('Forecast failed: ' + error.message);
        }
    },

    renderForecastResults(forecastData, targetName) {
        document.getElementById('forecastResults').style.display = 'block';

        // Render metrics
        document.getElementById('forecastMetrics').innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                <div style="text-align: center; padding: 1rem; background: var(--hover-bg); border-radius: var(--radius);">
                    <div style="font-size: 0.875rem; color: var(--text-secondary);">MAE</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: var(--accent);">${forecastData.mae?.toFixed(2) || 'N/A'}</div>
                </div>
                <div style="text-align: center; padding: 1rem; background: var(--hover-bg); border-radius: var(--radius);">
                    <div style="font-size: 0.875rem; color: var(--text-secondary);">RMSE</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: var(--accent);">${forecastData.rmse?.toFixed(2) || 'N/A'}</div>
                </div>
                <div style="text-align: center; padding: 1rem; background: var(--hover-bg); border-radius: var(--radius);">
                    <div style="font-size: 0.875rem; color: var(--text-secondary);">MAPE</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: var(--accent);">${forecastData.mape?.toFixed(2) || 'N/A'}%</div>
                </div>
            </div>
        `;

        // Render chart
        this.renderForecastChart(forecastData, targetName);
    },

    renderForecastChart(forecastData, targetName) {
        const canvas = document.getElementById('forecastChart');

        if (window.forecastCharts.main) {
            window.forecastCharts.main.destroy();
        }

        const ctx = canvas.getContext('2d');
        window.forecastCharts.main = new Chart(ctx, {
            type: 'line',
            data: {
                labels: forecastData.dates || [],
                datasets: [
                    {
                        label: 'Actual',
                        data: forecastData.actual || [],
                        borderColor: 'rgba(59, 130, 246, 1)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        pointRadius: 3,
                        fill: true
                    },
                    {
                        label: 'Forecast',
                        data: forecastData.forecast || [],
                        borderColor: 'rgba(236, 72, 153, 1)',
                        backgroundColor: 'rgba(236, 72, 153, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 3,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: { color: '#111827', font: { size: 12 }, padding: 15 }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        padding: 12,
                        cornerRadius: 6
                    },
                    title: {
                        display: true,
                        text: `Time Series Forecast: ${targetName}`,
                        color: '#111827',
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: { color: 'rgba(0, 0, 0, 0.05)' },
                        ticks: { color: '#6b7280' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#6b7280', maxRotation: 45, minRotation: 0 }
                    }
                }
            }
        });
    }
};

window.ForecastingModule = ForecastingModule;
