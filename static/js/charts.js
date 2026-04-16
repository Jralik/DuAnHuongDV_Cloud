/**
 * Charts Module - Chart.js configurations
 */

// Chart.js global defaults
Chart.defaults.color = '#9ca3af';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";

let priceChart = null;
let volumeChart = null;
let equityChart = null;

/**
 * Initialize price chart (line chart for OHLC data)
 */
function initPriceChart() {
    const ctx = document.getElementById('price-chart');
    if (!ctx) return;

    if (priceChart) priceChart.destroy();

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Giá',
                    data: [],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.05)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBackgroundColor: '#06b6d4',
                },
                {
                    label: 'SMA 20',
                    data: [],
                    borderColor: 'rgba(245, 158, 11, 0.6)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: 'EMA 21',
                    data: [],
                    borderColor: 'rgba(139, 92, 246, 0.6)',
                    borderWidth: 1,
                    borderDash: [3, 3],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: 'BB Upper',
                    data: [],
                    borderColor: 'rgba(107, 114, 128, 0.3)',
                    borderWidth: 1,
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: 'BB Lower',
                    data: [],
                    borderColor: 'rgba(107, 114, 128, 0.3)',
                    backgroundColor: 'rgba(107, 114, 128, 0.04)',
                    borderWidth: 1,
                    fill: '-1',
                    tension: 0.3,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        boxWidth: 8,
                        boxHeight: 2,
                        padding: 12,
                        font: { size: 10 },
                        usePointStyle: true,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    titleColor: '#e5e7eb',
                    bodyColor: '#9ca3af',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            let val = context.parsed.y;
                            if (val >= 1000) return `${context.dataset.label}: $${val.toLocaleString()}`;
                            return `${context.dataset.label}: $${val.toFixed(2)}`;
                        }
                    }
                },
            },
            scales: {
                x: {
                    display: true,
                    grid: { display: false },
                    ticks: {
                        maxTicksLimit: 10,
                        font: { size: 10 },
                        maxRotation: 0,
                    }
                },
                y: {
                    display: true,
                    position: 'right',
                    grid: {
                        color: 'rgba(255,255,255,0.03)',
                    },
                    ticks: {
                        font: { size: 10, family: "'JetBrains Mono', monospace" },
                        callback: (v) => v >= 1000 ? `$${(v/1000).toFixed(1)}K` : `$${v.toFixed(0)}`,
                    }
                }
            },
            animation: {
                duration: 400,
                easing: 'easeOutQuart',
            }
        }
    });
}

/**
 * Initialize volume chart
 */
function initVolumeChart() {
    const ctx = document.getElementById('volume-chart');
    if (!ctx) return;

    if (volumeChart) volumeChart.destroy();

    volumeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Volume',
                data: [],
                backgroundColor: [],
                borderWidth: 0,
                barPercentage: 0.85,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    callbacks: {
                        label: (ctx) => `Vol: ${ctx.parsed.y.toFixed(2)}`
                    }
                },
            },
            scales: {
                x: { display: false },
                y: {
                    display: true,
                    position: 'right',
                    grid: { display: false },
                    ticks: {
                        font: { size: 9, family: "'JetBrains Mono', monospace" },
                        maxTicksLimit: 3,
                    }
                }
            },
            animation: { duration: 300 }
        }
    });
}

/**
 * Update price chart with market data
 */
function updatePriceChart(marketData) {
    if (!priceChart || !marketData || !marketData.data) return;

    const data = marketData.data;
    const labels = data.map(d => {
        const date = new Date(d.time);
        return `${date.getHours().toString().padStart(2,'0')}:${date.getMinutes().toString().padStart(2,'0')}`;
    });

    const closes = data.map(d => d.close);
    const opens = data.map(d => d.open);

    // Calculate simple SMA 20 & EMA 21 client-side for display
    const sma20 = calcSMA(closes, 20);
    const ema21 = calcEMA(closes, 21);
    const { upper: bbUpper, lower: bbLower } = calcBB(closes, 20, 2);

    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = closes;
    priceChart.data.datasets[1].data = sma20;
    priceChart.data.datasets[2].data = ema21;
    priceChart.data.datasets[3].data = bbUpper;
    priceChart.data.datasets[4].data = bbLower;
    priceChart.update('none');

    // Update volume chart
    if (volumeChart) {
        const volumes = data.map(d => d.volume);
        const colors = data.map(d =>
            d.close >= d.open
                ? 'rgba(16, 185, 129, 0.5)'
                : 'rgba(239, 68, 68, 0.5)'
        );

        volumeChart.data.labels = labels;
        volumeChart.data.datasets[0].data = volumes;
        volumeChart.data.datasets[0].backgroundColor = colors;
        volumeChart.update('none');
    }
}

/**
 * Initialize equity curve chart for backtest results
 */
function initEquityChart() {
    const ctx = document.getElementById('equity-chart');
    if (!ctx) return;

    if (equityChart) equityChart.destroy();

    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Equity',
                data: [],
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.2,
                pointRadius: 0,
            }, {
                label: 'Baseline',
                data: [],
                borderColor: 'rgba(107, 114, 128, 0.4)',
                borderWidth: 1,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: { boxWidth: 8, boxHeight: 2, font: { size: 10 } }
                },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}`
                    }
                },
            },
            scales: {
                x: { display: false },
                y: {
                    position: 'right',
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: {
                        font: { size: 10, family: "'JetBrains Mono', monospace" },
                        callback: (v) => `$${v.toLocaleString()}`
                    }
                }
            },
            animation: { duration: 600 }
        }
    });
}

/**
 * Update equity chart with backtest results
 */
function updateEquityChart(equityCurve, initialCapital) {
    if (!equityChart) initEquityChart();

    const labels = equityCurve.map((_, i) => i);
    const baseline = equityCurve.map(() => initialCapital);

    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = equityCurve;
    equityChart.data.datasets[1].data = baseline;

    // Color based on profit/loss
    const lastVal = equityCurve[equityCurve.length - 1];
    if (lastVal >= initialCapital) {
        equityChart.data.datasets[0].borderColor = '#10b981';
        equityChart.data.datasets[0].backgroundColor = 'rgba(16, 185, 129, 0.1)';
    } else {
        equityChart.data.datasets[0].borderColor = '#ef4444';
        equityChart.data.datasets[0].backgroundColor = 'rgba(239, 68, 68, 0.1)';
    }

    equityChart.update();
}

/* ---- Helper calculations ---- */

function calcSMA(data, period) {
    const result = new Array(data.length).fill(null);
    for (let i = period - 1; i < data.length; i++) {
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) sum += data[j];
        result[i] = sum / period;
    }
    return result;
}

function calcEMA(data, period) {
    const result = new Array(data.length).fill(null);
    const k = 2 / (period + 1);
    // Seed with SMA
    let sum = 0;
    for (let i = 0; i < period && i < data.length; i++) sum += data[i];
    result[period - 1] = sum / period;
    for (let i = period; i < data.length; i++) {
        result[i] = data[i] * k + result[i - 1] * (1 - k);
    }
    return result;
}

function calcBB(data, period, stdDev) {
    const sma = calcSMA(data, period);
    const upper = new Array(data.length).fill(null);
    const lower = new Array(data.length).fill(null);

    for (let i = period - 1; i < data.length; i++) {
        let sumSq = 0;
        for (let j = i - period + 1; j <= i; j++) {
            sumSq += Math.pow(data[j] - sma[i], 2);
        }
        const std = Math.sqrt(sumSq / period);
        upper[i] = sma[i] + std * stdDev;
        lower[i] = sma[i] - std * stdDev;
    }
    return { upper, lower };
}
