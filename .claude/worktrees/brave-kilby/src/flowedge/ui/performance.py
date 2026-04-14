"""Performance dashboard — P&L charts, trade history, model analytics."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

perf_router = APIRouter(prefix="/performance", tags=["performance"])


@perf_router.get("/", response_class=HTMLResponse)
async def performance_dashboard() -> HTMLResponse:
    """Serve the performance tracking dashboard."""
    return HTMLResponse(content=PERF_HTML)


@perf_router.get("/data")
async def performance_data() -> dict:  # type: ignore[type-arg]
    """Get the latest performance report data."""
    from flowedge.scanner.performance.simulator import load_report

    report = load_report()
    if report is None:
        return {"status": "no_data", "message": "Run a simulation first"}
    return report.model_dump(mode="json")


@perf_router.post("/run")
async def run_simulation() -> dict:  # type: ignore[type-arg]
    """Run the historical simulation now."""
    from flowedge.scanner.performance.simulator import run_historical_simulation

    report = await run_historical_simulation()
    return {
        "status": "complete",
        "total_return_pct": report.total_return_pct,
        "total_return_dollars": report.total_return_dollars,
        "ending_value": report.ending_value,
        "total_trades": report.total_trades,
        "win_rate": report.win_rate,
    }


PERF_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PHANTOM — Performance Analytics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
:root {
  --bg:#06060b;--bg2:#0c0c14;--bg3:#12121c;--border:#1a1a2e;--border2:#252540;
  --text:#c8c8d4;--text2:#888898;--text3:#55556a;
  --accent:#6366f1;--green:#22c55e;--green2:#4ade80;--red:#ef4444;--red2:#f87171;
  --cyan:#06b6d4;--cyan2:#22d3ee;--yellow:#eab308;--purple:#a855f7;
  --font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:var(--font);background:var(--bg);color:var(--text);}
.header{background:rgba(12,12,20,0.8);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);
  padding:16px 32px;display:flex;justify-content:space-between;align-items:center;}
.header h1{font-size:18px;font-weight:600;color:#fff;}
.header .sub{font-size:11px;color:var(--text3);font-family:var(--mono);}
button{background:linear-gradient(135deg,var(--accent),#4f46e5);color:#fff;border:none;
  padding:10px 20px;border-radius:8px;cursor:pointer;font-weight:600;font-size:13px;
  font-family:var(--font);}
button:hover{transform:translateY(-1px);box-shadow:0 4px 20px rgba(99,102,241,0.15);}
button:disabled{opacity:0.5;cursor:not-allowed;}
.main{padding:24px 32px;}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:24px;}
.kpi{background:rgba(12,12,20,0.7);backdrop-filter:blur(8px);border:1px solid var(--border);
  border-radius:12px;padding:20px;text-align:center;}
.kpi .label{font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;
  font-family:var(--mono);}
.kpi .value{font-size:28px;font-weight:700;margin-top:6px;font-family:var(--mono);}
.kpi .value.green{color:var(--green2);} .kpi .value.red{color:var(--red2);}
.kpi .value.cyan{color:var(--cyan2);} .kpi .value.yellow{color:var(--yellow);}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.card{background:rgba(12,12,20,0.7);backdrop-filter:blur(8px);border:1px solid var(--border);
  border-radius:12px;padding:20px;}
.card.full{grid-column:1/-1;}
.card h2{font-size:11px;font-weight:600;color:var(--text3);text-transform:uppercase;
  letter-spacing:1.5px;margin-bottom:16px;font-family:var(--mono);}
canvas{max-height:280px;}
table{width:100%;border-collapse:collapse;font-size:12px;font-family:var(--mono);}
th{text-align:left;padding:8px;color:var(--text3);font-size:10px;border-bottom:1px solid var(--border);
  text-transform:uppercase;letter-spacing:1px;}
td{padding:8px;border-bottom:1px solid rgba(26,26,46,0.5);}
.win{color:var(--green2);} .loss{color:var(--red2);}
.loading{text-align:center;padding:60px;color:var(--text3);}
.nav{padding:8px 32px;background:rgba(12,12,20,0.6);border-bottom:1px solid var(--border);
  display:flex;gap:16px;font-size:13px;}
.nav a{color:var(--text2);text-decoration:none;padding:6px 0;border-bottom:2px solid transparent;}
.nav a:hover,.nav a.active{color:#fff;border-bottom-color:var(--accent);}
@media(max-width:800px){.grid{grid-template-columns:1fr;}}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>PHANTOM Performance</h1>
    <div class="sub">$1,000 simulated bot &middot; Started January 1, 2026</div>
  </div>
  <div style="display:flex;gap:12px;align-items:center;">
    <button onclick="runSim()" id="runBtn">Run Simulation</button>
    <a href="/dashboard/" style="color:var(--text2);font-size:13px;">&#8592; Scanner</a>
  </div>
</div>
<div class="nav">
  <a href="/dashboard/">Scanner</a>
  <a href="/charts/">Analytics</a>
  <a href="/performance/" class="active">Performance</a>
  <a href="/models/">Models</a>
</div>
<div class="main">
  <div class="kpi-row" id="kpis"><div class="loading">Loading performance data...</div></div>
  <div class="grid" id="charts">
    <div class="card full"><h2>Portfolio Value Over Time</h2><canvas id="equityChart"></canvas></div>
    <div class="card"><h2>Daily P&L</h2><canvas id="dailyPnlChart"></canvas></div>
    <div class="card"><h2>Win Rate by Score Bucket</h2><canvas id="bucketChart"></canvas></div>
    <div class="card full"><h2>Monthly Returns</h2><canvas id="monthlyChart"></canvas></div>
    <div class="card full" id="modelCard" style="display:none;">
      <h2>Model Accuracy &amp; Risk Metrics</h2>
      <div id="modelMetrics" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;"></div>
    </div>
    <div class="card full" id="tickerCard" style="display:none;">
      <h2>Performance by Ticker</h2>
      <div id="tickerTable" style="max-height:300px;overflow-y:auto;"></div>
    </div>
    <div class="card full"><h2>Trade History</h2><div id="tradeTable" style="max-height:400px;overflow-y:auto;"></div></div>
  </div>
</div>
<script>
let charts = {};
function dc(id){if(charts[id]){charts[id].destroy();delete charts[id];}}

async function loadData() {
  try {
    const resp = await fetch('/performance/data');
    const data = await resp.json();
    if (data.status === 'no_data') {
      document.getElementById('kpis').innerHTML = '<div class="loading">No data yet. Click "Run Simulation" to start.</div>';
      return;
    }
    renderKPIs(data);
    renderEquityChart(data);
    renderDailyPnl(data);
    renderBuckets(data);
    renderMonthly(data);
    renderModelAccuracy(data);
    renderTickerBreakdown(data);
    renderTrades(data);
  } catch(e) { console.error('Load failed:', e); }
}

function renderKPIs(d) {
  const ret = d.total_return_pct;
  const retColor = ret >= 0 ? 'green' : 'red';
  const dollarColor = d.total_return_dollars >= 0 ? 'green' : 'red';
  document.getElementById('kpis').innerHTML = `
    <div class="kpi"><div class="label">Starting Capital</div><div class="value">$${d.starting_capital.toLocaleString()}</div></div>
    <div class="kpi"><div class="label">Current Value</div><div class="value ${dollarColor}">$${d.ending_value.toLocaleString(undefined,{minimumFractionDigits:2})}</div></div>
    <div class="kpi"><div class="label">Total Return</div><div class="value ${retColor}">${ret >= 0 ? '+' : ''}${ret.toFixed(1)}%</div></div>
    <div class="kpi"><div class="label">P&L Dollars</div><div class="value ${dollarColor}">${d.total_return_dollars >= 0 ? '+' : ''}$${d.total_return_dollars.toFixed(2)}</div></div>
    <div class="kpi"><div class="label">Total Trades</div><div class="value cyan">${d.total_trades}</div></div>
    <div class="kpi"><div class="label">Win Rate</div><div class="value ${d.win_rate >= 0.5 ? 'green' : 'yellow'}">${(d.win_rate*100).toFixed(1)}%</div></div>
    <div class="kpi"><div class="label">Profit Factor</div><div class="value cyan">${d.profit_factor.toFixed(2)}</div></div>
    <div class="kpi"><div class="label">Max Drawdown</div><div class="value red">-${d.max_drawdown_pct.toFixed(1)}%</div></div>
  `;
}

function renderEquityChart(d) {
  if (!d.daily_snapshots?.length) return;
  const labels = d.daily_snapshots.map(s => s.date);
  const values = d.daily_snapshots.map(s => s.portfolio_value);
  dc('equityChart');
  charts['equityChart'] = new Chart(document.getElementById('equityChart'), {
    type:'line',
    data:{labels, datasets:[{
      label:'Portfolio Value ($)',data:values,
      borderColor:'#6366f1',backgroundColor:'rgba(99,102,241,0.08)',
      fill:true,tension:0.3,pointRadius:0,borderWidth:2,
    },{
      label:'Starting Capital',data:labels.map(()=>d.starting_capital),
      borderColor:'rgba(255,255,255,0.15)',borderDash:[5,5],pointRadius:0,borderWidth:1,
    }]},
    options:{plugins:{legend:{labels:{color:'#888'}}},
      scales:{x:{ticks:{color:'#555',maxTicksLimit:10},grid:{color:'#1a1a2e'}},
              y:{ticks:{color:'#555',callback:v=>'$'+v},grid:{color:'#1a1a2e'}}}}
  });
}

function renderDailyPnl(d) {
  if (!d.daily_snapshots?.length) return;
  const labels = d.daily_snapshots.map(s => s.date);
  const values = d.daily_snapshots.map(s => s.daily_pnl);
  const colors = values.map(v => v >= 0 ? 'rgba(74,222,128,0.7)' : 'rgba(248,113,113,0.7)');
  dc('dailyPnlChart');
  charts['dailyPnlChart'] = new Chart(document.getElementById('dailyPnlChart'), {
    type:'bar',
    data:{labels, datasets:[{label:'Daily P&L',data:values,backgroundColor:colors}]},
    options:{plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#555',maxTicksLimit:8},grid:{display:false}},
              y:{ticks:{color:'#555',callback:v=>'$'+v},grid:{color:'#1a1a2e'}}}}
  });
}

function renderBuckets(d) {
  if (!d.by_score_bucket) return;
  const labels = Object.keys(d.by_score_bucket);
  const winRates = labels.map(k => (d.by_score_bucket[k].win_rate * 100));
  const counts = labels.map(k => d.by_score_bucket[k].trades);
  dc('bucketChart');
  charts['bucketChart'] = new Chart(document.getElementById('bucketChart'), {
    type:'bar',
    data:{labels:labels.map(l=>'Score '+l), datasets:[
      {label:'Win Rate %',data:winRates,backgroundColor:'rgba(99,102,241,0.7)',yAxisID:'y'},
      {label:'Trade Count',data:counts,backgroundColor:'rgba(6,182,212,0.5)',yAxisID:'y1'},
    ]},
    options:{plugins:{legend:{labels:{color:'#888'}}},
      scales:{x:{ticks:{color:'#555'},grid:{display:false}},
              y:{ticks:{color:'#555',callback:v=>v+'%'},grid:{color:'#1a1a2e'},max:100},
              y1:{position:'right',ticks:{color:'#555'},grid:{display:false}}}}
  });
}

function renderMonthly(d) {
  if (!d.monthly_returns?.length) return;
  const labels = d.monthly_returns.map(m => m.month);
  const returns = d.monthly_returns.map(m => m.return_pct);
  const colors = returns.map(v => v >= 0 ? 'rgba(74,222,128,0.7)' : 'rgba(248,113,113,0.7)');
  dc('monthlyChart');
  charts['monthlyChart'] = new Chart(document.getElementById('monthlyChart'), {
    type:'bar',
    data:{labels, datasets:[{label:'Monthly Return %',data:returns,backgroundColor:colors}]},
    options:{plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#555'},grid:{display:false}},
              y:{ticks:{color:'#555',callback:v=>v+'%'},grid:{color:'#1a1a2e'}}}}
  });
}

function renderModelAccuracy(d) {
  if (!d.model_accuracy) return;
  const m = d.model_accuracy;
  const card = document.getElementById('modelCard');
  card.style.display = 'block';
  const metricItem = (label, value, color) => `
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center;">
      <div style="font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:0.5px;font-family:var(--mono);">${label}</div>
      <div style="font-size:20px;font-weight:700;margin-top:4px;color:${color};font-family:var(--mono);">${value}</div>
    </div>`;
  document.getElementById('modelMetrics').innerHTML = [
    metricItem('Direction Accuracy', (m.direction_accuracy*100).toFixed(1)+'%', m.direction_accuracy>=0.5?'var(--green2)':'var(--red2)'),
    metricItem('Sharpe Ratio', m.sharpe_ratio.toFixed(2), m.sharpe_ratio>=1?'var(--green2)':m.sharpe_ratio>=0?'var(--yellow)':'var(--red2)'),
    metricItem('Sortino Ratio', m.sortino_ratio.toFixed(2), m.sortino_ratio>=1?'var(--green2)':'var(--yellow)'),
    metricItem('Calmar Ratio', m.calmar_ratio.toFixed(2), m.calmar_ratio>=1?'var(--green2)':'var(--yellow)'),
    metricItem('Expectancy', '$'+m.expectancy.toFixed(2), m.expectancy>=0?'var(--green2)':'var(--red2)'),
    metricItem('Avg Score (W)', m.avg_score_winners.toFixed(0), 'var(--green2)'),
    metricItem('Avg Score (L)', m.avg_score_losers.toFixed(0), 'var(--red2)'),
    metricItem('Score Gap', m.score_separation.toFixed(1), m.score_separation>5?'var(--green2)':'var(--yellow)'),
    metricItem('Hi-Score WR', (m.high_score_win_rate*100).toFixed(1)+'%', m.high_score_win_rate>=0.5?'var(--green2)':'var(--yellow)'),
    metricItem('Lo-Score WR', (m.low_score_win_rate*100).toFixed(1)+'%', 'var(--text2)'),
    metricItem('Max Win Streak', m.consecutive_wins_max, 'var(--green2)'),
    metricItem('Max Loss Streak', m.consecutive_losses_max, 'var(--red2)'),
  ].join('');
}

function renderTickerBreakdown(d) {
  if (!d.by_ticker?.length) return;
  const card = document.getElementById('tickerCard');
  card.style.display = 'block';
  let html = '<table><tr><th>Ticker</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>Total P&L</th><th>Avg P&L%</th><th>Best</th><th>Worst</th></tr>';
  for (const t of d.by_ticker) {
    const cls = t.total_pnl_dollars >= 0 ? 'win' : 'loss';
    html += `<tr>
      <td style="font-weight:600;">${t.ticker}</td>
      <td>${t.total_trades}</td>
      <td class="win">${t.wins}</td>
      <td class="loss">${t.losses}</td>
      <td>${(t.win_rate*100).toFixed(1)}%</td>
      <td class="${cls}">${t.total_pnl_dollars>=0?'+':''}$${t.total_pnl_dollars.toFixed(2)}</td>
      <td class="${cls}">${t.avg_pnl_pct>=0?'+':''}${t.avg_pnl_pct.toFixed(1)}%</td>
      <td class="win">+${t.best_trade_pct.toFixed(1)}%</td>
      <td class="loss">${t.worst_trade_pct.toFixed(1)}%</td>
    </tr>`;
  }
  html += '</table>';
  document.getElementById('tickerTable').innerHTML = html;
}

function renderTrades(d) {
  if (!d.trades?.length) return;
  const closed = d.trades.filter(t => t.result !== 'open').reverse();
  let html = '<table><tr><th>ID</th><th>Ticker</th><th>Entry</th><th>Exit</th><th>Score</th><th>P&L $</th><th>P&L %</th><th>Days</th><th>Reason</th></tr>';
  for (const t of closed.slice(0, 50)) {
    const cls = t.pnl_dollars >= 0 ? 'win' : 'loss';
    html += `<tr>
      <td>${t.trade_id}</td><td style="font-weight:600;">${t.ticker}</td>
      <td>${t.entry_date}</td><td>${t.exit_date||'-'}</td>
      <td>${t.nexus_score}</td>
      <td class="${cls}">${t.pnl_dollars>=0?'+':''}$${t.pnl_dollars.toFixed(2)}</td>
      <td class="${cls}">${t.pnl_pct>=0?'+':''}${t.pnl_pct.toFixed(1)}%</td>
      <td>${t.hold_days}d</td><td style="font-size:11px;color:var(--text3);">${t.exit_reason}</td>
    </tr>`;
  }
  html += '</table>';
  document.getElementById('tradeTable').innerHTML = html;
}

async function runSim() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true; btn.textContent = 'Running...';
  try {
    await fetch('/performance/run', {method:'POST'});
    await loadData();
  } catch(e) { alert('Simulation failed: ' + e.message); }
  btn.disabled = false; btn.textContent = 'Run Simulation';
}

loadData();
</script>
</body>
</html>"""
