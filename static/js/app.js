/**
 * Main Application Logic - Dashboard Controller
 */

// ============================================================
// STATE
// ============================================================
const state = {
    currentSymbol: 'BTC/USDT',
    currentTimeframe: '1h',
    isLoading: false,
    portfolio: null,
    prices: {},
};

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    setupEventListeners();
    startClock();
    connectWebSocket();
    loadInitialData();
});

function initCharts() {
    initPriceChart();
    initVolumeChart();
    initEquityChart();
}

function setupEventListeners() {
    // Symbol buttons
    document.querySelectorAll('.symbol-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.symbol-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentSymbol = btn.dataset.symbol;
            document.getElementById('chart-title').textContent = state.currentSymbol;
            loadMarketData();
            loadAnalysis();
            fetchAvailableModels(); // Lấy danh sách model khi đổi symbol
        });
    });

    // Timeframe buttons
    document.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentTimeframe = btn.dataset.tf;
            loadMarketData();
        });
    });

    // Tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const panel = btn.closest('.panel');
            panel.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            panel.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`${btn.dataset.tab}-content`).classList.add('active');
        });
    });
}

function startClock() {
    function updateTime() {
        const now = new Date();
        const el = document.getElementById('header-time');
        if (el) {
            el.textContent = now.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }
    }
    updateTime();
    setInterval(updateTime, 1000);
}

function connectWebSocket() {
    tradingWS.on('price_update', (data) => {
        updatePrices(data.data);
        if (data.portfolio) {
            updatePortfolioUI(data.portfolio);
        }
    });

    tradingWS.on('signal_update', (data) => {
        updateSignals(data.data);
    });

    tradingWS.on('connected', () => {
        addSignalLog('info', 'Đã kết nối WebSocket');
    });

    tradingWS.connect();
}

// ============================================================
// DATA LOADING
// ============================================================
async function loadInitialData() {
    showLoading('Đang tải dữ liệu...');
    try {
        await Promise.all([
            loadMarketData(),
            loadAnalysis(),
            loadPortfolio(),
            fetchAvailableModels(), // Get models
        ]);
        addSignalLog('info', 'Hệ thống đã sẵn sàng');
    } catch (e) {
        console.error('Lỗi tải dữ liệu:', e);
        addSignalLog('info', 'Lỗi kết nối - thử tải lại trang');
    }
    hideLoading();
}

async function loadMarketData() {
    const [base, quote] = state.currentSymbol.split('/');
    try {
        const resp = await fetch(`/api/market/${base}/${quote}?timeframe=${state.currentTimeframe}&limit=200`);
        if (resp.ok) {
            const data = await resp.json();
            updatePriceChart(data);
            updateCurrentPrice(data);
        }
    } catch (e) {
        console.error('Lỗi load market data:', e);
    }
}

async function loadAnalysis() {
    const [base, quote] = state.currentSymbol.split('/');
    try {
        const resp = await fetch(`/api/analysis/${base}/${quote}`);
        if (resp.ok) {
            const data = await resp.json();
            updateAnalysisUI(data);
        }
    } catch (e) {
        console.error('Lỗi load analysis:', e);
    }
}

async function loadPortfolio() {
    try {
        const resp = await fetch('/api/portfolio');
        if (resp.ok) {
            const data = await resp.json();
            state.portfolio = data;
            updatePortfolioUI(data);
        }
    } catch (e) {
        console.error('Lỗi load portfolio:', e);
    }
}

// ============================================================
// UI UPDATE FUNCTIONS
// ============================================================
function updateCurrentPrice(marketData) {
    if (!marketData || !marketData.data || marketData.data.length === 0) return;

    const latest = marketData.data[marketData.data.length - 1];
    const prev = marketData.data.length > 1 ? marketData.data[marketData.data.length - 2] : latest;
    const change = ((latest.close - prev.close) / prev.close * 100);

    // Update ticker
    updateTickerItem(marketData.symbol, latest.close, change);
}

function updatePrices(pricesData) {
    if (!pricesData) return;

    state.prices = pricesData;
    const tickerEl = document.getElementById('market-ticker');
    tickerEl.innerHTML = '';

    Object.entries(pricesData).forEach(([symbol, info]) => {
        const price = info.last || 0;
        const change = info.change || 0;
        updateTickerItem(symbol, price, change);
    });
}

function updateTickerItem(symbol, price, change) {
    const tickerEl = document.getElementById('market-ticker');

    let existingItem = tickerEl.querySelector(`[data-ticker="${symbol}"]`);
    if (!existingItem) {
        existingItem = document.createElement('div');
        existingItem.className = 'ticker-item';
        existingItem.dataset.ticker = symbol;
        tickerEl.appendChild(existingItem);
    }

    const shortSymbol = symbol.split('/')[0];
    const priceStr = price >= 1000 ? `$${price.toLocaleString(undefined, {maximumFractionDigits: 0})}` : `$${price.toFixed(2)}`;
    const changeClass = change >= 0 ? 'positive' : 'negative';
    const changeStr = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;

    existingItem.innerHTML = `
        <span class="ticker-symbol">${shortSymbol}</span>
        <span class="ticker-price">${priceStr}</span>
        <span class="ticker-change ${changeClass}">${changeStr}</span>
    `;
}

function updateAnalysisUI(analysis) {
    if (!analysis) return;

    // Update signal
    const signal = analysis.signal || {};
    const signalText = signal.action || 'GIỮ';
    const confidence = signal.confidence || 0;

    // AI Signal circle
    const circle = document.getElementById('ai-signal-circle');
    const textEl = document.getElementById('ai-signal-text');
    circle.className = 'ai-signal-circle';

    if (signalText === 'MUA') {
        circle.classList.add('buy');
        textEl.style.color = '#10b981';
    } else if (signalText === 'BÁN') {
        circle.classList.add('sell');
        textEl.style.color = '#ef4444';
    } else {
        circle.classList.add('hold');
        textEl.style.color = '#f59e0b';
    }
    textEl.textContent = signalText;

    // Confidence
    document.getElementById('confidence-fill').style.width = `${confidence}%`;
    document.getElementById('confidence-value').textContent = `${confidence.toFixed(1)}%`;

    // Top stat signal
    const sigEl = document.getElementById('current-signal');
    sigEl.textContent = signalText;
    sigEl.className = `stat-value signal-${signalText === 'MUA' ? 'buy' : signalText === 'BÁN' ? 'sell' : 'hold'}`;
    document.getElementById('signal-confidence').textContent = `Độ tin cậy: ${confidence.toFixed(0)}%`;

    // AI trained status
    const statusEl = document.getElementById('ai-status');
    if (analysis.ai_trained) {
        statusEl.innerHTML = '<span class="ai-badge trained">✓ Đã train</span>';
        
        // Update model metrics if available
        if (analysis.ai_info && analysis.ai_info.models) {
            updateModelMetrics(analysis.ai_info.models);
        }
    } else {
        statusEl.innerHTML = '<span class="ai-badge untrained">⚠ Chưa train</span>';
        // Reset metrics
        document.getElementById('acc-xgb').textContent = `--`;
        document.getElementById('acc-rf').textContent = `--`;
        document.getElementById('acc-lstm').textContent = `--`;
    }

    // Technical indicators
    updateIndicators(analysis.technical || {});

    // Add signal to log
    if (signalText !== 'GIỮ') {
        addSignalLog(
            signalText === 'MUA' ? 'buy' : 'sell',
            `${state.currentSymbol}: ${signalText} | ${signal.reason || ''} | Conf: ${confidence.toFixed(0)}%`
        );
    }
}

function updateIndicators(tech) {
    // RSI
    updateGaugeIndicator('rsi', tech.rsi, 0, 100, (v) => {
        if (v < 30) return { label: 'QUÁ BÁN', cls: 'bullish' };
        if (v > 70) return { label: 'QUÁ MUA', cls: 'bearish' };
        return { label: 'TRUNG LẬP', cls: 'neutral' };
    });

    // MACD
    const macdVal = tech.macd_histogram || 0;
    document.getElementById('val-macd').textContent = macdVal.toFixed(2);
    const macdBar = document.getElementById('bar-macd');
    if (macdVal >= 0) {
        macdBar.style.left = '50%';
        macdBar.style.width = `${Math.min(Math.abs(macdVal) * 5, 50)}%`;
        macdBar.style.background = '#10b981';
        setSigText('sig-macd', 'TĂNG', 'bullish');
    } else {
        const w = Math.min(Math.abs(macdVal) * 5, 50);
        macdBar.style.left = `${50 - w}%`;
        macdBar.style.width = `${w}%`;
        macdBar.style.background = '#ef4444';
        setSigText('sig-macd', 'GIẢM', 'bearish');
    }

    // Bollinger Bands
    const bbUpper = tech.bb_upper || 0;
    const bbLower = tech.bb_lower || 0;
    const price = tech.price || 0;
    if (bbUpper > 0 && bbLower > 0) {
        const bbRange = bbUpper - bbLower;
        const bbPos = bbRange > 0 ? ((price - bbLower) / bbRange * 100) : 50;
        updateGaugeIndicator('bb', bbPos.toFixed(0), 0, 100, (v) => {
            if (v < 20) return { label: 'DƯỚI DẢI', cls: 'bullish' };
            if (v > 80) return { label: 'TRÊN DẢI', cls: 'bearish' };
            return { label: 'TRONG DẢI', cls: 'neutral' };
        });
        document.getElementById('val-bb').textContent = `${bbPos.toFixed(0)}%`;
    }

    // ADX
    updateGaugeIndicator('adx', tech.adx, 0, 60, (v) => {
        if (v > 25) return { label: 'XU HƯỚNG MẠNH', cls: 'bullish' };
        return { label: 'SIDEWAY', cls: 'neutral' };
    });

    // Stochastic
    updateGaugeIndicator('stoch', tech.stoch_k, 0, 100, (v) => {
        if (v < 20) return { label: 'QUÁ BÁN', cls: 'bullish' };
        if (v > 80) return { label: 'QUÁ MUA', cls: 'bearish' };
        return { label: 'TRUNG LẬP', cls: 'neutral' };
    });

    // MFI
    updateGaugeIndicator('mfi', tech.mfi, 0, 100, (v) => {
        if (v < 20) return { label: 'QUÁ BÁN', cls: 'bullish' };
        if (v > 80) return { label: 'QUÁ MUA', cls: 'bearish' };
        return { label: 'TRUNG LẬP', cls: 'neutral' };
    });

    // Momentum Score
    const momentum = tech.momentum_score || 0;
    document.getElementById('val-momentum').textContent = momentum.toFixed(0);
    const mBar = document.getElementById('momentum-bar');
    if (momentum >= 0) {
        mBar.style.left = '50%';
        mBar.style.width = `${Math.min(Math.abs(momentum), 50)}%`;
        mBar.style.background = 'linear-gradient(90deg, #06b6d4, #10b981)';
        document.getElementById('val-momentum').style.color = '#10b981';
    } else {
        const w = Math.min(Math.abs(momentum), 50);
        mBar.style.left = `${50 - w}%`;
        mBar.style.width = `${w}%`;
        mBar.style.background = 'linear-gradient(90deg, #ef4444, #f59e0b)';
        document.getElementById('val-momentum').style.color = '#ef4444';
    }
}

function updateGaugeIndicator(id, value, min, max, classify) {
    const val = parseFloat(value) || 0;
    document.getElementById(`val-${id}`).textContent = val.toFixed(1);

    const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
    const gauge = document.getElementById(`gauge-${id}`);
    if (gauge) {
        gauge.style.width = `${pct}%`;
        gauge.className = 'gauge-fill';
        if (pct > 70) gauge.classList.add('overbought');
        else if (pct < 30) gauge.classList.add('oversold');
        else gauge.classList.add('neutral');
    }

    const result = classify(val);
    setSigText(`sig-${id}`, result.label, result.cls);
}

function setSigText(id, text, cls) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = text;
        el.className = `ind-signal ${cls}`;
    }
}

function updatePortfolioUI(portfolio) {
    if (!portfolio) return;
    state.portfolio = portfolio;

    // Stats
    const equityEl = document.getElementById('total-equity');
    const pnlEl = document.getElementById('total-pnl');

    equityEl.textContent = `$${(portfolio.total_equity || 10000).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;

    const pnl = portfolio.total_pnl || 0;
    const pnlPct = portfolio.total_pnl_pct || 0;
    pnlEl.textContent = `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}`;
    pnlEl.style.color = pnl >= 0 ? '#10b981' : '#ef4444';

    const changeEl = document.getElementById('equity-change');
    changeEl.textContent = `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%`;
    changeEl.className = `stat-change ${pnlPct >= 0 ? 'positive' : 'negative'}`;

    const pnlChangeEl = document.getElementById('pnl-change');
    pnlChangeEl.textContent = `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%`;
    pnlChangeEl.className = `stat-change ${pnl >= 0 ? 'positive' : 'negative'}`;

    document.getElementById('open-positions').textContent = portfolio.open_positions || 0;
    document.getElementById('total-trades').textContent = portfolio.total_trades || 0;

    // Positions
    updatePositionsList(portfolio.positions || []);

    // Trade history
    updateTradeHistory(portfolio.recent_trades || []);
}

function updatePositionsList(positions) {
    const container = document.getElementById('positions-list');
    if (positions.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📭</span>
                <span>Chưa có vị thế nào</span>
            </div>`;
        return;
    }

    container.innerHTML = positions.map(p => {
        const invested = p.entry_price * p.size;
        const pnlColor = p.unrealized_pnl >= 0 ? '#10b981' : '#ef4444';
        
        return `
        <div class="position-item ${p.side}">
            <div class="pos-col-symbol">
                <span class="pos-symbol">${p.symbol}</span>
                <span class="pos-side-badge ${p.side}">${p.side === 'buy' ? 'MUA' : 'BÁN'}</span>
            </div>
            <div class="pos-col-info">
                <div class="pos-row-sub">
                    <span>Vốn: <b class="mono">&dollar;${invested.toLocaleString(undefined, {maximumFractionDigits: 1})}</b></span>
                    <span>Vào: <b class="mono">&dollar;${p.entry_price.toLocaleString()}</b></span>
                </div>
                <div class="pos-row-sub">
                    <span>Cắt lỗ: <b class="mono" style="color:#ef4444">&dollar;${p.stop_loss.toLocaleString()}</b></span>
                    <span>Chốt lời: <b class="mono" style="color:#10b981">&dollar;${p.take_profit.toLocaleString()}</b></span>
                </div>
            </div>
            <div class="pos-col-pnl" style="color: ${pnlColor}">
                <div class="pnl-value">${p.unrealized_pnl >= 0 ? '+' : ''}&dollar;${p.unrealized_pnl.toFixed(2)}</div>
                <div class="pnl-pct">${p.unrealized_pnl_pct.toFixed(2)}%</div>
            </div>
            <button class="pos-close-btn-round" onclick="closeTrade('${p.symbol}')" title="Đóng vị thế">✕</button>
        </div>`;
    }).join('');
}

function updateTradeHistory(trades) {
    const container = document.getElementById('trades-table');
    if (trades.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📋</span>
                <span>Chưa có giao dịch</span>
            </div>`;
        return;
    }

    container.innerHTML = [...trades].reverse().map(t => {
        const invested = t.entry_price * t.size;
        return `
        <div class="trade-row" style="flex-direction: column; align-items: stretch; gap: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display:flex; gap:8px; align-items:center;">
                    <span class="pos-side-badge ${t.side}">${t.side === 'buy' ? 'MUA' : 'BÁN'}</span>
                    <span class="pos-symbol">${t.symbol}</span>
                </div>
                <div class="pnl-value" style="color: ${t.pnl >= 0 ? '#10b981' : '#ef4444'};">
                    ${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}
                </div>
            </div>
            <div style="display: flex; gap: 15px; font-size: 0.75rem; color: var(--text-secondary);">
                <span>Vốn: <b class="mono">&dollar;${invested.toLocaleString(undefined, {maximumFractionDigits: 1})}</b></span>
                <span>Vào: <b class="mono">&dollar;${t.entry_price.toLocaleString()}</b> → <b>&dollar;${t.exit_price.toLocaleString()}</b></span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: var(--text-secondary);">
                <div style="display: flex; gap: 15px;">
                    <span>Cắt lỗ: <b class="mono" style="opacity: 0.8">&dollar;${t.stop_loss ? t.stop_loss.toLocaleString() : '0'}</b></span>
                    <span>Chốt lời: <b class="mono" style="opacity: 0.8">&dollar;${t.take_profit ? t.take_profit.toLocaleString() : '0'}</b></span>
                </div>
                <span style="opacity: 0.6; font-style: italic;">Lý do: ${t.reason}</span>
            </div>
        </div>
    `}).join('');
}

function updateSignals(signalsData) {
    Object.entries(signalsData).forEach(([symbol, signal]) => {
        if (signal.action !== 'GIỮ') {
            addSignalLog(
                signal.action === 'MUA' ? 'buy' : 'sell',
                `${symbol}: ${signal.action} | Tin cậy: ${signal.confidence.toFixed(0)}%`
            );
        }

        // Update signal display if this is the current symbol
        if (symbol === state.currentSymbol) {
            const sigEl = document.getElementById('current-signal');
            sigEl.textContent = signal.action;
            sigEl.className = `stat-value signal-${signal.action === 'MUA' ? 'buy' : signal.action === 'BÁN' ? 'sell' : 'hold'}`;
        }
    });
}

// ============================================================
// ACTIONS
// ============================================================
async function trainModel() {
    const [base, quote] = state.currentSymbol.split('/');
    const btn = document.getElementById('btn-train');
    const daysSelector = document.getElementById('train-days');
    const days = daysSelector ? daysSelector.value : 90;
    btn.disabled = true;
    btn.textContent = '⏳ Đang luyện...';

    showLoading(`Đang khởi tạo huấn luyện AI model cho ${state.currentSymbol} với ${days} ngày... (Quá trình này có thể tốn vài phút)`);
    addSignalLog('info', `Bắt đầu train model ${state.currentSymbol} (${days} ngày dữ liệu)...`);

    try {
        const resp = await fetch(`/api/train/${base}/${quote}?days=${days}`, { method: 'POST' });
        const data = await resp.json();

        if (data.error) {
            showToast(data.error, 'error');
            addSignalLog('info', `Lỗi train: ${data.error}`);
        } else {
            showToast('Train model thành công!', 'success');
            addSignalLog('info', `Train ${state.currentSymbol} hoàn tất - ${data.data_points} mẫu. Tên: ${data.model_name}`);

            // Update model predictions display
            if (data.metrics) {
                updateModelMetrics(data.metrics);
            }

            // Reload analysis and models
            await loadAnalysis();
            await fetchAvailableModels();
        }
    } catch (e) {
        showToast('Lỗi kết nối server', 'error');
    }

    btn.disabled = false;
    btn.textContent = 'Huấn luyện';
    hideLoading();
}

async function fetchAvailableModels() {
    const [base, quote] = state.currentSymbol.split('/');
    const select = document.getElementById('load-model-select');
    if (!select) return;
    
    try {
        const resp = await fetch(`/api/models/${base}/${quote}`);
        if (resp.ok) {
            const data = await resp.json();
            select.innerHTML = '<option value="">Chọn model...</option>';
            if (data.models && data.models.length > 0) {
                data.models.forEach(model => {
                    const opt = document.createElement('option');
                    opt.value = model;
                    opt.textContent = model.replace(state.currentSymbol.replace('/', '_') + '_', '');
                    select.appendChild(opt);
                });
            } else {
                select.innerHTML = '<option value="">Chưa có model nào</option>';
            }
        }
    } catch (e) {
        console.error("Lỗi lấy danh sách models", e);
        select.innerHTML = '<option value="">Lỗi kết nối</option>';
    }
}

async function loadSelectedModel() {
    const select = document.getElementById('load-model-select');
    const modelName = select.value;
    if (!modelName) {
        showToast('Vui lòng chọn model để tải', 'warning');
        return;
    }

    const [base, quote] = state.currentSymbol.split('/');
    const btn = document.getElementById('btn-load-model');
    btn.disabled = true;
    btn.textContent = '...';

    try {
        const resp = await fetch(`/api/models/load/${base}/${quote}?model_name=${encodeURIComponent(modelName)}`, { method: 'POST' });
        const data = await resp.json();

        if (data.error) {
            showToast(data.error, 'error');
        } else {
            showToast(`Tải thành công: ${modelName}`, 'success');
            addSignalLog('info', `Đã tải model AI: ${modelName}`);
            await loadAnalysis();
        }
    } catch (e) {
        showToast('Lỗi kết nối khi tải model', 'error');
    }

    btn.disabled = false;
    btn.textContent = 'Tải';
}

async function deleteSelectedModel() {
    const select = document.getElementById('load-model-select');
    const modelName = select.value;
    if (!modelName) {
        showToast('Vui lòng chọn model để xóa', 'warning');
        return;
    }

    if (!confirm(`Bạn có chắc muốn xóa model ${modelName} không? Hành động này không thể hoàn tác.`)) {
        return;
    }

    const [base, quote] = state.currentSymbol.split('/');
    const btn = document.getElementById('btn-delete-model');
    btn.disabled = true;

    try {
        const resp = await fetch(`/api/models/${base}/${quote}?model_name=${encodeURIComponent(modelName)}`, { method: 'DELETE' });
        const data = await resp.json();

        if (data.error) {
            showToast(data.error, 'error');
        } else {
            showToast(`Xóa thành công: ${modelName}`, 'success');
            addSignalLog('info', `Đã xóa model AI: ${modelName}`);
            await fetchAvailableModels(); // Cập nhật lại dropdown
        }
    } catch (e) {
        showToast('Lỗi kết nối khi xóa model', 'error');
    }

    btn.disabled = false;
}

function updateModelMetrics(metrics) {
    if (metrics.xgboost && !metrics.xgboost.error) {
        const acc = (metrics.xgboost.accuracy * 100).toFixed(1);
        const f1 = (metrics.xgboost.f1 * 100).toFixed(1);
        document.getElementById('acc-xgb').textContent = `CX: ${acc}% | F1: ${f1}%`;
    }
    if (metrics.random_forest && !metrics.random_forest.error) {
        const acc = (metrics.random_forest.accuracy * 100).toFixed(1);
        const f1 = (metrics.random_forest.f1 * 100).toFixed(1);
        document.getElementById('acc-rf').textContent = `CX: ${acc}% | F1: ${f1}%`;
    }
    if (metrics.lstm && !metrics.lstm.error) {
        const acc = (metrics.lstm.accuracy * 100).toFixed(1);
        const f1 = (metrics.lstm.f1 * 100).toFixed(1);
        document.getElementById('acc-lstm').textContent = `CX: ${acc}% | F1: ${f1}%`;
    }
}

async function openTrade(side) {
    try {
        const resp = await fetch(`/api/trade/open?symbol=${encodeURIComponent(state.currentSymbol)}&side=${side}`, {
            method: 'POST'
        });
        const data = await resp.json();

        if (data.success) {
            const pos = data.position;
            const invested = (pos.entry_price * pos.size).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            showToast(`Đã mở lệnh ${side === 'buy' ? '▲ MUA' : '▼ BÁN'} ${state.currentSymbol}`, 'success');
            addSignalLog(side === 'buy' ? 'buy' : 'sell',
                `MỞ LỆNH: ${side === 'buy' ? 'MUA' : 'BÁN'} ${state.currentSymbol} @ $${pos.entry_price.toLocaleString()} | Vốn: $${invested} | Cắt lỗ: $${pos.stop_loss.toLocaleString()} | Chốt lời: $${pos.take_profit.toLocaleString()}`);
            await loadPortfolio();
        } else {
            showToast(data.error || 'Không thể mở lệnh', 'error');
            if (data.warnings) {
                data.warnings.forEach(w => addSignalLog('info', `⚠ ${w}`));
            }
        }
    } catch (e) {
        showToast('Lỗi kết nối', 'error');
    }
}

async function closeTrade(symbol) {
    const [base, quote] = symbol.split('/');
    try {
        const resp = await fetch(`/api/trade/close/${base}/${quote}`, { method: 'POST' });
        const data = await resp.json();

        if (data.success) {
            const pnl = data.trade.pnl;
            showToast(`Đã đóng ${symbol}: ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}`,
                pnl >= 0 ? 'success' : 'warning');
            addSignalLog(pnl >= 0 ? 'buy' : 'sell',
                `ĐÓNG LỆNH: ${symbol} | P&L: ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}`);
            await loadPortfolio();
        } else {
            showToast(data.error || 'Lỗi đóng lệnh', 'error');
        }
    } catch (e) {
        showToast('Lỗi kết nối', 'error');
    }
}

async function runBacktest() {
    const [base, quote] = state.currentSymbol.split('/');
    const days = document.getElementById('backtest-days').value;
    const strategy = document.getElementById('backtest-strategy').value;

    const btn = document.getElementById('btn-backtest');
    btn.disabled = true;
    btn.textContent = '⏳ Đang chạy...';

    showLoading(`Đang backtest ${state.currentSymbol} (${days} ngày)...`);
    addSignalLog('info', `Bắt đầu backtest ${state.currentSymbol} - ${days} ngày - ${strategy}`);

    try {
        const resp = await fetch(
            `/api/backtest/${base}/${quote}?days=${days}&strategy=${strategy}`,
            { method: 'POST' }
        );
        const data = await resp.json();

        if (data.error) {
            showToast(data.error, 'error');
        } else {
            displayBacktestResults(data);
            addSignalLog('info',
                `Backtest hoàn tất: Lợi nhuận ${data.total_return}% | Thắng ${data.win_rate}% | ${data.total_trades} giao dịch`);
        }
    } catch (e) {
        showToast('Lỗi backtest', 'error');
    }

    btn.disabled = false;
    btn.textContent = '▶ Chạy';
    hideLoading();
}

function displayBacktestResults(results) {
    document.getElementById('backtest-empty').style.display = 'none';
    document.getElementById('backtest-results').style.display = 'block';

    const totalReturn = results.total_return || 0;
    const isProfit = totalReturn >= 0;

    document.getElementById('bt-stats').innerHTML = `
        <div class="bt-stat">
            <div class="bt-stat-label">Lợi Nhuận</div>
            <div class="bt-stat-value" style="color: ${isProfit ? '#10b981' : '#ef4444'}">
                ${isProfit ? '+' : ''}${totalReturn.toFixed(2)}%
            </div>
        </div>
        <div class="bt-stat">
            <div class="bt-stat-label">Tỉ lệ thắng</div>
            <div class="bt-stat-value">${(results.win_rate || 0).toFixed(1)}%</div>
        </div>
        <div class="bt-stat">
            <div class="bt-stat-label">Hệ số lợi nhuận</div>
            <div class="bt-stat-value">${(results.profit_factor || 0).toFixed(2)}</div>
        </div>
        <div class="bt-stat">
            <div class="bt-stat-label">Tổng giao dịch</div>
            <div class="bt-stat-value">${results.total_trades || 0}</div>
        </div>
        <div class="bt-stat">
            <div class="bt-stat-label">Sụt giảm tối đa</div>
            <div class="bt-stat-value" style="color: #ef4444">
                -${(results.portfolio_metrics?.max_drawdown || 0).toFixed(2)}%
            </div>
        </div>
        <div class="bt-stat">
            <div class="bt-stat-label">Chỉ số Sharpe</div>
            <div class="bt-stat-value">${(results.portfolio_metrics?.sharpe_ratio || 0).toFixed(2)}</div>
        </div>
        <div class="bt-stat">
            <div class="bt-stat-label">Thắng Trung bình</div>
            <div class="bt-stat-value" style="color: #10b981">$${(results.avg_win || 0).toFixed(2)}</div>
        </div>
        <div class="bt-stat">
            <div class="bt-stat-label">Thua Trung bình</div>
            <div class="bt-stat-value" style="color: #ef4444">$${(results.avg_loss || 0).toFixed(2)}</div>
        </div>
    `;

    // Equity curve
    if (results.equity_curve && results.equity_curve.length > 0) {
        updateEquityChart(results.equity_curve, results.initial_capital);
    }
}

// ============================================================
// UTILITIES
// ============================================================
function addSignalLog(type, message) {
    const log = document.getElementById('signal-log');
    const time = new Date().toLocaleTimeString('vi-VN', {
        hour: '2-digit', minute: '2-digit', second: '2-digit'
    });

    const entry = document.createElement('div');
    entry.className = `signal-entry ${type}`;
    entry.innerHTML = `
        <span class="signal-time">${time}</span>
        <span class="signal-text">${message}</span>
    `;

    log.insertBefore(entry, log.firstChild);

    // Keep max 50 entries
    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

function clearSignalLog() {
    const log = document.getElementById('signal-log');
    log.innerHTML = '';
    addSignalLog('info', 'Đã xóa nhật ký');
}

function showLoading(text) {
    const overlay = document.getElementById('loading-overlay');
    document.getElementById('loading-text').textContent = text || 'Đang tải...';
    overlay.classList.add('active');
    state.isLoading = true;
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
    state.isLoading = false;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ============================================================
// DATABRICKS & MLFLOW
// ============================================================
async function checkDatabricksStatus() {
    const badge = document.getElementById('db-status-badge');
    const openBtn = document.getElementById('btn-db-open');

    try {
        const resp = await fetch('/api/databricks/status');
        const data = await resp.json();

        if (!data.enabled) {
            badge.textContent = 'Chưa cấu hình';
            badge.className = 'db-status-badge disabled';
            setDbDot('db-dot-sql', false);
            setDbDot('db-dot-mlflow', false);
            setDbDot('db-dot-delta', false);
            document.getElementById('db-sql-status').textContent = 'Disabled';
            document.getElementById('db-mlflow-status').textContent = 'Disabled';
            document.getElementById('db-delta-status').textContent = 'Disabled';
            return;
        }

        // SQL Warehouse
        const sqlOk = data.sql_warehouse && data.sql_warehouse.connected;
        setDbDot('db-dot-sql', sqlOk);
        document.getElementById('db-sql-status').textContent = sqlOk ? 'Connected' : 'Lỗi';

        // Set Databricks workspace link
        if (data.sql_warehouse && data.sql_warehouse.host) {
            openBtn.href = `https://${data.sql_warehouse.host}`;
        }

        // MLflow
        const mlflowOk = data.mlflow && data.mlflow.initialized;
        setDbDot('db-dot-mlflow', mlflowOk);
        document.getElementById('db-mlflow-status').textContent = mlflowOk ? 'Ready' : 'Chưa sẵn sàng';

        // Delta Lake
        const deltaSync = data.data_sync;
        if (deltaSync && deltaSync.tables) {
            const tableCount = Object.values(deltaSync.tables).filter(t => t.exists).length;
            const totalRows = Object.values(deltaSync.tables).reduce((sum, t) => sum + (t.row_count || 0), 0);
            setDbDot('db-dot-delta', tableCount > 0);
            document.getElementById('db-delta-status').textContent = `${tableCount} bảng | ${totalRows.toLocaleString()} rows`;
        } else {
            setDbDot('db-dot-delta', false);
            document.getElementById('db-delta-status').textContent = 'Chưa setup';
        }

        // Overall badge
        if (sqlOk) {
            badge.textContent = 'Đã kết nối';
            badge.className = 'db-status-badge connected';
        } else {
            badge.textContent = 'Lỗi kết nối';
            badge.className = 'db-status-badge disconnected';
        }

        // Auto-load experiments if MLflow is ready
        if (mlflowOk) {
            loadMLflowExperiments();
        }

    } catch (e) {
        badge.textContent = 'Lỗi kết nối';
        badge.className = 'db-status-badge disconnected';
        console.error('Databricks status error:', e);
    }
}

function setDbDot(id, online) {
    const dot = document.getElementById(id);
    if (dot) {
        dot.className = `db-dot ${online ? 'online' : 'offline'}`;
    }
}

async function syncDataToDatabricks() {
    const symbol = document.getElementById('db-sync-symbol').value;
    const days = document.getElementById('db-sync-days').value;
    const btn = document.getElementById('btn-sync-data');
    const resultEl = document.getElementById('db-sync-result');

    btn.disabled = true;
    btn.classList.add('db-syncing');
    btn.textContent = '⏳ Đang sync...';
    resultEl.textContent = `Đang đồng bộ ${symbol} (${days} ngày)...`;
    resultEl.className = 'db-sync-result';

    try {
        const resp = await fetch(
            `/api/databricks/sync-data?symbol=${encodeURIComponent(symbol)}&timeframe=1h&days=${days}`,
            { method: 'POST' }
        );
        const data = await resp.json();

        if (data.success) {
            resultEl.textContent = `✅ Đã sync ${data.rows_inserted} rows cho ${symbol} lúc ${new Date(data.sync_time).toLocaleTimeString('vi-VN')}`;
            resultEl.className = 'db-sync-result success';
            showToast(`Sync thành công: ${data.rows_inserted} rows`, 'success');
            addSignalLog('info', `[Databricks] Sync OHLCV ${symbol}: ${data.rows_inserted} rows`);
        } else {
            resultEl.textContent = `❌ Lỗi: ${data.error}`;
            resultEl.className = 'db-sync-result error';
            showToast(`Sync lỗi: ${data.error}`, 'error');
        }
    } catch (e) {
        resultEl.textContent = `❌ Lỗi kết nối server`;
        resultEl.className = 'db-sync-result error';
        showToast('Lỗi kết nối khi sync', 'error');
    }

    btn.disabled = false;
    btn.classList.remove('db-syncing');
    btn.textContent = 'Sync OHLCV';
}

async function syncTradesToDatabricks() {
    const btn = document.getElementById('btn-sync-trades');
    const resultEl = document.getElementById('db-sync-result');

    btn.disabled = true;
    btn.textContent = '⏳...';

    try {
        const resp = await fetch('/api/databricks/sync-trades', { method: 'POST' });
        const data = await resp.json();

        if (data.success) {
            resultEl.textContent = `✅ Đã sync ${data.rows_inserted} trades`;
            resultEl.className = 'db-sync-result success';
            showToast(`Sync trades thành công: ${data.rows_inserted} giao dịch`, 'success');
        } else {
            resultEl.textContent = `❌ ${data.error}`;
            resultEl.className = 'db-sync-result error';
        }
    } catch (e) {
        resultEl.textContent = '❌ Lỗi kết nối';
        resultEl.className = 'db-sync-result error';
    }

    btn.disabled = false;
    btn.textContent = 'Sync Trades';
}

async function loadMLflowExperiments() {
    const container = document.getElementById('db-experiments-table');

    try {
        const resp = await fetch('/api/databricks/experiments?limit=10');
        const data = await resp.json();

        if (!data.enabled || !data.runs || data.runs.length === 0) {
            container.innerHTML = `
                <div class="empty-state" style="padding: 16px;">
                    <span class="empty-icon">📊</span>
                    <span>Chưa có experiment. Train model để tạo experiment đầu tiên.</span>
                </div>`;
            return;
        }

        renderExperimentsTable(data.runs, container);

    } catch (e) {
        console.error('MLflow experiments error:', e);
        container.innerHTML = `
            <div class="empty-state" style="padding: 16px;">
                <span>Lỗi tải experiments</span>
            </div>`;
    }
}

function renderExperimentsTable(runs, container) {
    let html = `
        <table class="db-exp-table">
            <thead>
                <tr>
                    <th>Run</th>
                    <th>Symbol</th>
                    <th>Days</th>
                    <th>XGB Acc</th>
                    <th>RF Acc</th>
                    <th>LSTM Acc</th>
                    <th>Ensemble</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>`;

    runs.forEach(run => {
        const xgbAcc = run.xgboost_accuracy ? (run.xgboost_accuracy * 100).toFixed(1) : '--';
        const rfAcc = run.random_forest_accuracy ? (run.random_forest_accuracy * 100).toFixed(1) : '--';
        const lstmAcc = run.lstm_accuracy ? (run.lstm_accuracy * 100).toFixed(1) : '--';
        const ensembleAcc = run.ensemble_avg_accuracy ? (run.ensemble_avg_accuracy * 100).toFixed(1) : '--';

        const metricClass = (val) => {
            if (val === '--') return '';
            return parseFloat(val) >= 55.0 ? 'db-metric-good' : (parseFloat(val) < 50 ? 'db-metric-bad' : '');
        };

        html += `
            <tr>
                <td class="db-run-name" title="${run.run_name || run.run_id}">${run.run_name || run.run_id.substring(0, 8)}</td>
                <td>${run.symbol || '--'}</td>
                <td>${run.days || '--'}</td>
                <td class="${metricClass(xgbAcc)}">${xgbAcc}%</td>
                <td class="${metricClass(rfAcc)}">${rfAcc}%</td>
                <td class="${metricClass(lstmAcc)}">${lstmAcc}%</td>
                <td class="${metricClass(ensembleAcc)}" style="font-weight:700">${ensembleAcc}%</td>
                <td>${run.status === 'FINISHED' ? '✅' : '⏳'}</td>
            </tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

// Auto-refresh data every 30 seconds
setInterval(() => {
    if (!state.isLoading) {
        loadMarketData();
    }
}, 30000);

// Refresh analysis every 60 seconds
setInterval(() => {
    if (!state.isLoading) {
        loadAnalysis();
    }
}, 60000);

// Check Databricks status on load (delayed so main data loads first)
setTimeout(() => checkDatabricksStatus(), 3000);

