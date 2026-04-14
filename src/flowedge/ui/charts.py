"""Enhanced dashboard with charts — IV surface, GEX heatmap, flow timeline."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

charts_router = APIRouter(prefix="/charts", tags=["charts"])


@charts_router.get("/", response_class=HTMLResponse)
async def charts_dashboard() -> HTMLResponse:
    """Serve the enhanced charts dashboard."""
    return HTMLResponse(content=CHARTS_HTML)


CHARTS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FlowEdge NEXUS — Analytics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0a0f; color: #e0e0e0; }
  .header { background: #111118; padding: 16px 24px; border-bottom: 1px solid #222;
            display: flex; justify-content: space-between; align-items: center; }
  .header h1 { font-size: 20px; color: #fff; }
  .header .sub { color: #666; font-size: 13px; }
  .controls { padding: 12px 24px; background: #111118; border-bottom: 1px solid #222;
              display: flex; gap: 12px; align-items: center; }
  input { background: #1a1a24; border: 1px solid #333; color: #fff; padding: 8px 12px;
          border-radius: 6px; font-family: inherit; }
  button { background: #2563eb; color: #fff; border: none; padding: 8px 16px;
           border-radius: 6px; cursor: pointer; font-weight: 500; font-family: inherit; }
  button:hover { background: #3b82f6; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px 24px; }
  .card { background: #111118; border: 1px solid #222; border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 14px; color: #888; margin-bottom: 12px; text-transform: uppercase;
             letter-spacing: 0.5px; }
  .card.full { grid-column: 1 / -1; }
  canvas { max-height: 300px; }
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
             gap: 8px; margin-bottom: 16px; }
  .metric { background: #0a0a0f; border-radius: 6px; padding: 10px; text-align: center; }
  .metric .label { font-size: 10px; color: #666; text-transform: uppercase; }
  .metric .value { font-size: 20px; font-weight: 700; margin-top: 2px; }
  .metric .value.green { color: #4ade80; }
  .metric .value.red { color: #ef4444; }
  .metric .value.blue { color: #60a5fa; }
  .metric .value.yellow { color: #fbbf24; }
  .legend { display: flex; gap: 16px; margin-top: 8px; font-size: 11px; color: #888; }
  .legend span { display: flex; align-items: center; gap: 4px; }
  .legend .dot { width: 8px; height: 8px; border-radius: 50%; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>NEXUS Analytics</h1>
    <div class="sub">SPECTER \u00b7 ORACLE \u00b7 SENTINEL \u00b7 VORTEX \u00b7 PULSE</div>
  </div>
</div>
<div class="controls">
  <input type="text" id="ticker" placeholder="TSLA" value="TSLA" style="width:100px;">
  <button onclick="loadAll()">Analyze</button>
</div>

<div class="grid">
  <div class="card full" id="metricsCard">
    <h2>Signal Overview</h2>
    <div class="metrics" id="metricsGrid"></div>
  </div>

  <div class="card">
    <h2>ORACLE \u2014 IV Term Structure</h2>
    <canvas id="ivChart"></canvas>
  </div>

  <div class="card">
    <h2>VORTEX \u2014 GEX by Strike</h2>
    <canvas id="gexChart"></canvas>
  </div>

  <div class="card">
    <h2>SPECTER \u2014 Flow Premium Timeline</h2>
    <canvas id="flowChart"></canvas>
  </div>

  <div class="card">
    <h2>PULSE \u2014 Signal Radar</h2>
    <canvas id="radarChart"></canvas>
  </div>
</div>

<script>
let charts = {};

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

async function loadAll() {
  const ticker = document.getElementById('ticker').value.toUpperCase();
  await Promise.all([loadIV(ticker), loadGEX(ticker), loadFlow(ticker), loadRadar(ticker)]);
}

async function loadIV(ticker) {
  try {
    const resp = await fetch('/api/v1/scanner/iv/' + ticker);
    const data = await resp.json();
    if (!data || !data.term_structure) return;

    const labels = data.term_structure.map(p => p.days_to_expiration + 'd');
    const values = data.term_structure.map(p => (p.iv * 100).toFixed(1));

    destroyChart('ivChart');
    charts['ivChart'] = new Chart(document.getElementById('ivChart'), {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Implied Volatility %',
          data: values,
          borderColor: '#60a5fa',
          backgroundColor: 'rgba(96,165,250,0.1)',
          fill: true, tension: 0.3, pointRadius: 4,
        }]
      },
      options: {
        plugins: { legend: { labels: { color: '#888' }}},
        scales: {
          x: { ticks: { color: '#666' }, grid: { color: '#1a1a24' }},
          y: { ticks: { color: '#666' }, grid: { color: '#1a1a24' }},
        }
      }
    });

    // Update metrics
    const m = document.getElementById('metricsGrid');
    m.innerHTML = `
      <div class="metric"><div class="label">IV Rank</div><div class="value ${data.iv_rank.iv_rank < 30 ? 'green' : 'yellow'}">${data.iv_rank.iv_rank.toFixed(1)}%</div></div>
      <div class="metric"><div class="label">Current IV</div><div class="value blue">${(data.iv_rank.current_iv).toFixed(1)}%</div></div>
      <div class="metric"><div class="label">Regime</div><div class="value">${data.regime.toUpperCase()}</div></div>
      <div class="metric"><div class="label">Cheap Premium</div><div class="value ${data.is_cheap_premium ? 'green' : 'red'}">${data.is_cheap_premium ? 'YES' : 'NO'}</div></div>
      <div class="metric"><div class="label">Term Structure</div><div class="value">${data.is_contango ? 'Contango' : 'BACKWARDATION'}</div></div>
      <div class="metric"><div class="label">ORACLE Score</div><div class="value blue">${data.strength.toFixed(1)}</div></div>
    `;
  } catch(e) { console.error('IV load failed:', e); }
}

async function loadGEX(ticker) {
  try {
    const resp = await fetch('/api/v1/scanner/gex/' + ticker);
    const data = await resp.json();
    if (!data || !data.key_levels) return;

    const levels = data.key_levels.slice(0, 20);
    const labels = levels.map(l => '$' + l.strike.toFixed(0));
    const callGamma = levels.map(l => l.call_gamma);
    const putGamma = levels.map(l => -l.put_gamma);

    destroyChart('gexChart');
    charts['gexChart'] = new Chart(document.getElementById('gexChart'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Call GEX', data: callGamma, backgroundColor: 'rgba(74,222,128,0.7)' },
          { label: 'Put GEX', data: putGamma, backgroundColor: 'rgba(239,68,68,0.7)' },
        ]
      },
      options: {
        indexAxis: 'y',
        plugins: { legend: { labels: { color: '#888' }}},
        scales: {
          x: { ticks: { color: '#666' }, grid: { color: '#1a1a24' }},
          y: { ticks: { color: '#666', font: { size: 10 }}, grid: { color: '#1a1a24' }},
        }
      }
    });
  } catch(e) { console.error('GEX load failed:', e); }
}

async function loadFlow(ticker) {
  try {
    const resp = await fetch('/api/v1/scanner/flow/' + ticker);
    const data = await resp.json();
    if (!data || !data.length || !data[0].alerts) return;

    const alerts = data[0].alerts.slice(0, 50);
    const calls = alerts.filter(a => a.option_type === 'call');
    const puts = alerts.filter(a => a.option_type === 'put');

    const callPremiums = calls.map((a, i) => ({ x: i, y: a.premium }));
    const putPremiums = puts.map((a, i) => ({ x: i, y: a.premium }));

    destroyChart('flowChart');
    charts['flowChart'] = new Chart(document.getElementById('flowChart'), {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Calls', data: callPremiums, backgroundColor: 'rgba(74,222,128,0.6)', pointRadius: 4 },
          { label: 'Puts', data: putPremiums, backgroundColor: 'rgba(239,68,68,0.6)', pointRadius: 4 },
        ]
      },
      options: {
        plugins: { legend: { labels: { color: '#888' }}},
        scales: {
          x: { display: false },
          y: { ticks: { color: '#666', callback: v => '$' + (v/1000).toFixed(0) + 'k' }, grid: { color: '#1a1a24' }},
        }
      }
    });
  } catch(e) { console.error('Flow load failed:', e); }
}

async function loadRadar(ticker) {
  try {
    const resp = await fetch('/api/v1/scanner/scan', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tickers: [ticker], min_score: 0})
    });
    const data = await resp.json();
    if (!data.opportunities || !data.opportunities.length) return;
    const opp = data.opportunities[0];

    destroyChart('radarChart');
    charts['radarChart'] = new Chart(document.getElementById('radarChart'), {
      type: 'radar',
      data: {
        labels: ['SPECTER', 'ORACLE', 'SENTINEL', 'Momentum', 'Direction'],
        datasets: [{
          label: ticker,
          data: [opp.uoa_score, opp.iv_score, opp.catalyst_score, opp.composite_score, opp.composite_score],
          backgroundColor: 'rgba(96,165,250,0.2)',
          borderColor: '#60a5fa',
          pointBackgroundColor: '#60a5fa',
        }]
      },
      options: {
        plugins: { legend: { labels: { color: '#888' }}},
        scales: {
          r: {
            min: 0, max: 10,
            ticks: { color: '#666', backdropColor: 'transparent' },
            grid: { color: '#222' },
            pointLabels: { color: '#aaa' },
          }
        }
      }
    });
  } catch(e) { console.error('Radar load failed:', e); }
}
</script>
</body>
</html>"""
