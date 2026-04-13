/**
 * FlowEdge Council Dashboard — Chart.js visualizations
 */

function renderHealthChart(trendData) {
    const ctx = document.getElementById('healthChart');
    if (!ctx || !trendData || trendData.length === 0) return;

    // Reverse so oldest is first (left side of chart)
    const data = [...trendData].reverse();

    const labels = data.map(d => d.review_date);
    const healthScores = data.map(d => d.overall_health);
    const winRates = data.map(d => (d.win_rate * 100));
    const pnls = data.map(d => d.pnl);

    // Color each health bar by status
    const healthColors = data.map(d => {
        if (d.overall_health >= 75) return 'rgba(34, 197, 94, 0.7)';
        if (d.overall_health >= 50) return 'rgba(234, 179, 8, 0.7)';
        return 'rgba(239, 68, 68, 0.7)';
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Health Score',
                    data: healthScores,
                    backgroundColor: healthColors,
                    borderRadius: 4,
                    yAxisID: 'y',
                    order: 2,
                },
                {
                    label: 'Win Rate %',
                    data: winRates,
                    type: 'line',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: 'rgba(99, 102, 241, 1)',
                    tension: 0.3,
                    yAxisID: 'y',
                    order: 1,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: {
                    labels: { color: '#9ca3af', font: { size: 11 } },
                },
                tooltip: {
                    backgroundColor: '#1a1d29',
                    titleColor: '#fff',
                    bodyColor: '#d1d5db',
                    borderColor: '#2a2d3a',
                    borderWidth: 1,
                    callbacks: {
                        afterBody: function(context) {
                            const idx = context[0].dataIndex;
                            const d = data[idx];
                            return [
                                `Status: ${d.status}`,
                                `P&L: $${d.pnl.toLocaleString()}`,
                                `Trades: ${d.trades}`,
                            ];
                        }
                    }
                },
            },
            scales: {
                x: {
                    ticks: { color: '#6b7280', font: { size: 10 } },
                    grid: { color: 'rgba(42, 45, 58, 0.5)' },
                },
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        color: '#6b7280',
                        font: { size: 10 },
                        callback: v => v + '%',
                    },
                    grid: { color: 'rgba(42, 45, 58, 0.5)' },
                },
            },
        },
    });
}

// Auto-refresh dashboard every 5 minutes
if (window.location.pathname === '/') {
    setTimeout(() => window.location.reload(), 5 * 60 * 1000);
}
