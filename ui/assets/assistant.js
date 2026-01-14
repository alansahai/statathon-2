/**
 * AI Assistant Module
 * Handles AI-powered natural language queries and chart generation
 */

if (!window.API_BASE) console.error("API_BASE not ready");

const Assistant = {
    // State
    isOpen: false,
    currentFileId: null,
    messageHistory: [],
    columns: [],

    /**
     * Initialize the assistant
     */
    init() {
        console.log('🤖 Initializing AI Assistant...');

        // Get DOM elements
        this.elements = {
            toggle: document.getElementById('assistantToggle'),
            panel: document.getElementById('assistantPanel'),
            close: document.getElementById('assistantClose'),
            chat: document.getElementById('assistantChat'),
            input: document.getElementById('assistantInput'),
            send: document.getElementById('assistantSend'),
            typing: document.getElementById('assistantTyping'),
            suggestions: document.getElementById('assistantSuggestions')
        };

        // Check if elements exist
        if (!this.elements.toggle || !this.elements.panel) {
            console.warn('⚠️ Assistant elements not found. Make sure assistant.html is loaded.');
            return;
        }

        // Bind events
        this.bindEvents();

        // Load current file ID
        this.loadFileContext();

        // Generate dynamic suggestions if columns are available
        this.loadColumns();

        console.log('✅ AI Assistant initialized');
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Toggle panel
        this.elements.toggle.addEventListener('click', () => this.togglePanel());
        this.elements.close.addEventListener('click', () => this.togglePanel());

        // Send message
        this.elements.send.addEventListener('click', () => this.sendMessage());
        this.elements.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.elements.input.addEventListener('input', (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
        });

        // Suggestion chips
        this.elements.suggestions.addEventListener('click', (e) => {
            if (e.target.classList.contains('suggestion-chip')) {
                const prompt = e.target.getAttribute('data-prompt');
                this.elements.input.value = prompt;
                this.sendMessage();
            }
        });
    },

    /**
     * Toggle assistant panel
     */
    togglePanel() {
        this.isOpen = !this.isOpen;
        this.elements.panel.classList.toggle('open', this.isOpen);

        if (this.isOpen) {
            this.elements.input.focus();
            UIHelper.scrollChatToBottom(this.elements.chat);
        }
    },

    /**
     * Load file context from session/URL
     */
    loadFileContext() {
        // Try to get file_id from URL params
        const urlParams = new URLSearchParams(window.location.search);
        const fileName = sessionStorage.getItem('uploadedFileName');
        console.log("[assistant.js] Loaded file:", fileName);
        const fileId = urlParams.get('file_id') || fileName;

        if (fileId) {
            this.currentFileId = fileId;
            console.log(`📁 Assistant loaded file context: ${fileId}`);
        }
    },

    /**
     * Load columns for the current file
     */
    async loadColumns() {
        if (!this.currentFileId) return;

        try {
            // Check if WorkflowService is available
            if (typeof WorkflowService === 'undefined' || !WorkflowService.preloadAssistantData) {
                console.warn('WorkflowService not available, skipping column loading');
                return;
            }

            const data = await WorkflowService.preloadAssistantData(this.currentFileId);
            this.columns = data.columns || [];

            if (this.columns.length > 0) {
                this.generateSuggestions();
            }
        } catch (error) {
            console.error('Failed to load columns:', error);
        }
    },

    /**
     * Generate dynamic suggestions based on columns
     */
    generateSuggestions() {
        if (this.columns.length === 0) return;

        const suggestionsHtml = [
            `<button class="suggestion-chip" data-prompt="Show distribution of ${this.columns[0]}">📊 Distribution of ${this.columns[0]}</button>`,
            `<button class="suggestion-chip" data-prompt="Analyze correlation between ${this.columns[0]} and ${this.columns[1] || this.columns[0]}">📈 Correlation analysis</button>`,
            `<button class="suggestion-chip" data-prompt="Show trends over time">⏱️ Time trends</button>`,
            `<button class="suggestion-chip" data-prompt="Find outliers in ${this.columns[0]}">🔍 Find outliers</button>`
        ];

        this.elements.suggestions.innerHTML = `
            <h4>💡 Quick Actions</h4>
            <div class="suggestion-chips">
                ${suggestionsHtml.join('')}
            </div>
        `;
    },

    /**
     * Send a message to the AI
     */
    async sendMessage() {
        const message = this.elements.input.value.trim();
        if (!message) return;

        // Disable input
        this.elements.input.disabled = true;
        this.elements.send.disabled = true;

        // Add user message to chat
        this.appendUserMessage(message);

        // Clear input
        this.elements.input.value = '';
        this.elements.input.style.height = 'auto';

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Send to backend
            const url = `${window.API_BASE}/nlq/query`;
            console.log("Calling:", url);
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: message,
                    file_id: this.currentFileId || 'demo'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            // Hide typing indicator
            this.hideTypingIndicator();

            // Handle response
            this.handleBackendResponse(data);

        } catch (error) {
            console.error('Assistant error:', error);
            this.hideTypingIndicator();
            this.appendAIMessage(`Sorry, I encountered an error: ${error.message}. Please make sure the backend server is running.`);
        } finally {
            // Re-enable input
            this.elements.input.disabled = false;
            this.elements.send.disabled = false;
            this.elements.input.focus();
        }
    },

    /**
     * Append user message to chat
     */
    appendUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'assistant-message user';
        messageElement.innerHTML = `
            <div class="assistant-avatar">👤</div>
            <div class="assistant-bubble">
                <p>${this.escapeHtml(message)}</p>
            </div>
        `;

        this.elements.chat.appendChild(messageElement);
        UIHelper.scrollChatToBottom(this.elements.chat);

        // Add to history
        this.messageHistory.push({
            role: 'user',
            content: message,
            timestamp: new Date().toISOString()
        });
    },

    /**
     * Append AI message to chat
     */
    appendAIMessage(message, chartData = null, followUpSuggestions = null) {
        const messageElement = document.createElement('div');
        messageElement.className = 'assistant-message ai';

        let chartHtml = '';
        if (chartData) {
            const chartId = 'chart-' + Date.now();
            chartHtml = `
                <div class="assistant-chart-container">
                    <canvas id="${chartId}"></canvas>
                </div>
            `;
        }

        let suggestionsHtml = '';
        if (followUpSuggestions && followUpSuggestions.length > 0) {
            const suggestionButtons = followUpSuggestions.map(s =>
                `<button class="suggestion-chip" data-prompt="${this.escapeHtml(s)}">${this.escapeHtml(s)}</button>`
            ).join('');
            suggestionsHtml = `
                <div style="margin-top: 1rem;">
                    <small style="color: var(--text-secondary); display: block; margin-bottom: 0.5rem;">💡 Follow-up questions:</small>
                    <div class="suggestion-chips">${suggestionButtons}</div>
                </div>
            `;
        }

        messageElement.innerHTML = `
            <div class="assistant-avatar">🤖</div>
            <div class="assistant-bubble">
                ${this.formatMessage(message)}
                ${chartHtml}
                ${suggestionsHtml}
            </div>
        `;

        this.elements.chat.appendChild(messageElement);
        UIHelper.scrollChatToBottom(this.elements.chat);

        // Render chart if present
        if (chartData) {
            const chartId = messageElement.querySelector('canvas').id;
            this.renderChart(chartId, chartData);
        }

        // Add to history
        this.messageHistory.push({
            role: 'assistant',
            content: message,
            chartData: chartData,
            timestamp: new Date().toISOString()
        });
    },

    /**
     * Format message text (simple markdown-like formatting)
     */
    formatMessage(text) {
        // Split by double newlines for paragraphs
        const paragraphs = text.split('\n\n');
        return paragraphs.map(p => {
            // Bold text
            let formatted = p.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            // Code inline
            formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');
            return `<p>${formatted}</p>`;
        }).join('');
    },

    /**
     * Show typing indicator
     */
    showTypingIndicator() {
        this.elements.typing.classList.add('active');
        UIHelper.scrollChatToBottom(this.elements.chat);
    },

    /**
     * Hide typing indicator
     */
    hideTypingIndicator() {
        this.elements.typing.classList.remove('active');
    },

    /**
     * Handle backend response
     */
    handleBackendResponse(data) {
        const aiText = data.ai_text || data.response || 'I received your message but have no response to show.';
        const chartData = data.chart_data || null;
        const followUp = data.follow_up_suggestions || null;

        this.appendAIMessage(aiText, chartData, followUp);
    },

    /**
     * Render a chart
     */
    renderChart(canvasId, chartData) {
        setTimeout(() => {
            const canvas = document.getElementById(canvasId);
            if (!canvas) {
                console.error('Canvas not found:', canvasId);
                return;
            }

            const ctx = canvas.getContext('2d');

            // Get theme colors
            const colors = window.ThemeService ? ThemeService.getChartColors() : {
                primary: '#3b82f6',
                secondary: '#8b5cf6',
                success: '#10b981',
                warning: '#f59e0b',
                danger: '#ef4444'
            };

            const chartConfig = this.createChart(chartData, colors);

            new Chart(ctx, chartConfig);
        }, 100);
    },

    /**
     * Create Chart.js configuration
     */
    createChart(chartData, colors) {
        const type = chartData.type || 'bar';

        const config = {
            type: type,
            data: {
                labels: chartData.labels || [],
                datasets: [{
                    label: chartData.label || 'Data',
                    data: chartData.data || [],
                    backgroundColor: this.getChartColors(colors, chartData.data?.length || 0),
                    borderColor: colors.primary,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        enabled: true
                    }
                },
                scales: type !== 'pie' && type !== 'doughnut' ? {
                    y: {
                        beginAtZero: true
                    }
                } : {}
            }
        };

        return config;
    },

    /**
     * Get chart colors array
     */
    getChartColors(colors, count) {
        const colorArray = [
            colors.primary,
            colors.secondary,
            colors.success,
            colors.warning,
            colors.danger
        ];

        const result = [];
        for (let i = 0; i < count; i++) {
            result.push(colorArray[i % colorArray.length]);
        }

        return result;
    },

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

// Export to window
window.Assistant = Assistant;

console.log('✅ Assistant module loaded');
