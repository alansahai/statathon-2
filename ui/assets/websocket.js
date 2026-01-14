/**
 * WebSocket Client for Real-Time Pipeline Updates
 * Manages WebSocket connections and event handling for live pipeline status
 */

class PipelineWebSocket {
    constructor() {
        this.ws = null;
        this.fileId = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        this.pingInterval = null;
        this.isIntentionallyClosed = false;

        // Event handlers
        this.handlers = {
            message: [],
            error: [],
            close: [],
            open: []
        };
    }

    /**
     * Connect to WebSocket server
     * @param {string} fileId - File ID for pipeline tracking
     * @returns {Promise<void>}
     */
    connect(fileId) {
        return new Promise((resolve, reject) => {
            this.fileId = fileId;
            this.isIntentionallyClosed = false;

            const wsUrl = `ws://localhost:8000/ws/pipeline/${fileId}`;
            console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);

            try {
                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = (event) => {
                    console.log('✅ WebSocket connected');
                    this.reconnectAttempts = 0;
                    this.startPing();
                    this.triggerHandlers('open', event);
                    resolve();
                };

                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        console.log('📨 WebSocket message:', data);
                        this.triggerHandlers('message', data);
                    } catch (error) {
                        console.error('Failed to parse WebSocket message:', error);
                    }
                };

                this.ws.onerror = (error) => {
                    console.error('❌ WebSocket error:', error);
                    this.triggerHandlers('error', error);
                    reject(error);
                };

                this.ws.onclose = (event) => {
                    console.log('🔌 WebSocket closed:', event.code, event.reason);
                    this.stopPing();
                    this.triggerHandlers('close', event);

                    // Attempt reconnection if not intentionally closed
                    if (!this.isIntentionallyClosed && this.reconnectAttempts < this.maxReconnectAttempts) {
                        this.reconnect();
                    }
                };

            } catch (error) {
                console.error('Failed to create WebSocket:', error);
                reject(error);
            }
        });
    }

    /**
     * Reconnect to WebSocket server
     */
    reconnect() {
        this.reconnectAttempts++;
        console.log(`🔄 Reconnecting... (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        setTimeout(() => {
            if (this.fileId && !this.isIntentionallyClosed) {
                this.connect(this.fileId).catch(error => {
                    console.error('Reconnection failed:', error);
                });
            }
        }, this.reconnectDelay * this.reconnectAttempts);
    }

    /**
     * Disconnect from WebSocket server
     */
    disconnect() {
        this.isIntentionallyClosed = true;
        this.stopPing();

        if (this.ws) {
            this.ws.close(1000, 'Client disconnect');
            this.ws = null;
        }

        this.fileId = null;
        console.log('🔌 WebSocket disconnected');
    }

    /**
     * Send message to WebSocket server
     * @param {object} message - Message object to send
     */
    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket is not connected');
        }
    }

    /**
     * Start ping interval to keep connection alive
     */
    startPing() {
        this.pingInterval = setInterval(() => {
            this.send({ type: 'ping' });
        }, 30000); // Ping every 30 seconds
    }

    /**
     * Stop ping interval
     */
    stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    /**
     * Register message handler
     * @param {function} handler - Handler function for incoming messages
     */
    onMessage(handler) {
        this.handlers.message.push(handler);
    }

    /**
     * Register error handler
     * @param {function} handler - Handler function for errors
     */
    onError(handler) {
        this.handlers.error.push(handler);
    }

    /**
     * Register close handler
     * @param {function} handler - Handler function for connection close
     */
    onClose(handler) {
        this.handlers.close.push(handler);
    }

    /**
     * Register open handler
     * @param {function} handler - Handler function for connection open
     */
    onOpen(handler) {
        this.handlers.open.push(handler);
    }

    /**
     * Trigger all handlers for a specific event
     * @param {string} event - Event name
     * @param {any} data - Event data
     */
    triggerHandlers(event, data) {
        if (this.handlers[event]) {
            this.handlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in ${event} handler:`, error);
                }
            });
        }
    }

    /**
     * Check if WebSocket is connected
     * @returns {boolean}
     */
    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }

    /**
     * Get connection state
     * @returns {string}
     */
    getState() {
        if (!this.ws) return 'CLOSED';

        switch (this.ws.readyState) {
            case WebSocket.CONNECTING:
                return 'CONNECTING';
            case WebSocket.OPEN:
                return 'OPEN';
            case WebSocket.CLOSING:
                return 'CLOSING';
            case WebSocket.CLOSED:
                return 'CLOSED';
            default:
                return 'UNKNOWN';
        }
    }
}

// Export to window
window.PipelineWebSocket = PipelineWebSocket;

console.log('✅ PipelineWebSocket module loaded');
