/**
 * WebSocket Client - Real-time data connection
 */
class TradingWebSocket {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 5000;
        this.maxReconnectAttempts = 10;
        this.reconnectAttempts = 0;
        this.handlers = {};
        this.isConnected = false;
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${protocol}//${window.location.host}/ws`;

        try {
            this.ws = new WebSocket(url);

            this.ws.onopen = () => {
                console.log('[WS] Đã kết nối');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this._updateStatus(true);
                this._emit('connected');
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this._emit(data.type, data);
                } catch (e) {
                    console.error('[WS] Lỗi parse:', e);
                }
            };

            this.ws.onclose = () => {
                console.log('[WS] Ngắt kết nối');
                this.isConnected = false;
                this._updateStatus(false);
                this._emit('disconnected');
                this._reconnect();
            };

            this.ws.onerror = (error) => {
                console.error('[WS] Lỗi:', error);
            };
        } catch (e) {
            console.error('[WS] Không thể kết nối:', e);
            this._reconnect();
        }
    }

    _reconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[WS] Thử kết nối lại (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => this.connect(), this.reconnectInterval);
        }
    }

    _updateStatus(connected) {
        const dot = document.querySelector('.status-dot');
        const text = document.querySelector('.status-text');
        if (dot && text) {
            if (connected) {
                dot.classList.add('connected');
                text.textContent = 'Đã kết nối';
            } else {
                dot.classList.remove('connected');
                text.textContent = 'Đang kết nối lại...';
            }
        }
    }

    on(event, handler) {
        if (!this.handlers[event]) this.handlers[event] = [];
        this.handlers[event].push(handler);
    }

    _emit(event, data) {
        if (this.handlers[event]) {
            this.handlers[event].forEach(handler => handler(data));
        }
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Global instance
const tradingWS = new TradingWebSocket();
