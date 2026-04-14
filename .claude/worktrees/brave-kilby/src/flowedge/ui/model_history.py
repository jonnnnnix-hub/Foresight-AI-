"""Model performance history dashboard — tracks all models over time.

Shows:
1. Win rate over time per model (line chart)
2. Portfolio return over time per model (line chart)
3. Profit factor over time per model (line chart)
4. Model comparison table (latest metrics side-by-side)
5. Backfill button to import historical runs

Route: /models/
API:   /models/data — returns timeline data for all models
       /models/backfill — imports existing backtest files
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

models_router = APIRouter(prefix="/models", tags=["models"])


@models_router.get("/", response_class=HTMLResponse)
async def models_dashboard() -> HTMLResponse:
    """Serve the model performance history dashboard."""
    return HTMLResponse(content=MODELS_HTML)


@models_router.get("/data")
async def models_data(model: str | None = None) -> dict:  # type: ignore[type-arg]
    """Get model performance timeline data.

    Args:
        model: Filter to a specific model name. None = all models.
    """
    from flowedge.scanner.backtest.run_history import (
        get_available_models,
        get_model_comparison,
        get_model_timeline,
        load_index,
    )

    if model:
        timeline = get_model_timeline(model)
        return {"model": model, "timeline": timeline}

    comparison = get_model_comparison()
    index = load_index()

    return {
        "models": get_available_models(),
        "timelines": comparison,
        "summary": index.get("models", {}),
        "total_runs": len(index.get("runs", [])),
    }


@models_router.post("/backfill")
async def backfill_history() -> dict:  # type: ignore[type-arg]
    """Import existing backtest result files into run history."""
    from flowedge.scanner.backtest.run_history import backfill_from_existing

    imported = backfill_from_existing()
    return {"status": "complete", "imported": imported}


# ── Dashboard HTML ───────────────────────────────────────────────────────────


MODELS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FLOWEDGE — Model Performance History</title>
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
button.secondary{background:var(--bg3);border:1px solid var(--border2);}
button.secondary:hover{background:var(--border2);}
.main{padding:24px 32px;}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:24px;}
.kpi{background:rgba(12,12,20,0.7);backdrop-filter:blur(8px);border:1px solid var(--border);
  border-radius:12px;padding:20px;text-align:center;}
.kpi .label{font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;
  font-family:var(--mono);}
.kpi .value{font-size:28px;font-weight:700;margin-top:6px;font-family:var(--mono);}
.kpi .value.green{color:var(--green2);} .kpi .value.red{color:var(--red2);}
.kpi .value.cyan{color:var(--cyan2);} .kpi .value.purple{color:var(--purple);}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.card{background:rgba(12,12,20,0.7);backdrop-filter:blur(8px);border:1px solid var(--border);
  border-radius:12px;padding:20px;}
.card.full{grid-column:1/-1;}
.card h2{font-size:11px;font-weight:600;color:var(--text3);text-transform:uppercase;
  letter-spacing:1.5px;margin-bottom:16px;font-family:var(--mono);}
canvas{max-height:300px;}
table{width:100%;border-collapse:collapse;font-size:12px;font-family:var(--mono);}
th{text-align:left;padding:10px 8px;color:var(--text3);font-size:10px;
  border-bottom:1px solid var(--border);text-transform:uppercase;letter-spacing:1px;}
td{padding:10px 8px;border-bottom:1px solid rgba(26,26,46,0.5);}
.win{color:var(--green2);} .loss{color:var(--red2);}
.loading{text-align:center;padding:60px;color:var(--text3);}
.nav{padding:8px 32px;background:rgba(12,12,20,0.6);border-bottom:1px solid var(--border);
  display:flex;gap:16px;font-size:13px;}
.nav a{color:var(--text2);text-decoration:none;padding:6px 0;border-bottom:2px solid transparent;}
.nav a:hover,.nav a.active{color:#fff;border-bottom-color:var(--accent);}
.model-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;
  font-family:var(--mono);font-weight:600;}
@media(max-width:800px){.grid{grid-template-columns:1fr;}}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Model Performance History</h1>
    <div class="sub" id="headerSub">Tracking model evolution over time</div>
  </div>
  <div style="display:flex;gap:12px;align-items:center;">
    <button class="secondary" onclick="backfill()" id="backfillBtn">Import History</button>
    <a href="/dashboard/" style="color:var(--text2);font-size:13px;">&#8592; Scanner</a>
  </div>
</div>
<div class="nav">
  <a href="/dashboard/">Scanner</a>
  <a href="/charts/">Analytics</a>
  <a href="/performance/">Performance</a>
  <a href="/models/" class="active">Models</a>
</div>
<div class="main">
  <div class="kpi-row" id="kpis">
    <div class="loading">Loading model data...</div>
  </div>
  <div class="grid" id="charts">
    <div class="card full">
      <h2>Win Rate Over Time</h2>
      <canvas id="wrChart"></canvas>
    </div>
    <div class="card">
      <h2>Portfolio Return Over Time</h2>
      <canvas id="retChart"></canvas>
    </div>
    <div class="card">
      <h2>Profit Factor Over Time</h2>
      <canvas id="pfChart"></canvas>
    </div>
    <div class="card full" id="compCard">
      <h2>Model Comparison</h2>
      <div id="compTable" style="max-height:400px;overflow-y:auto;"></div>
    </div>
  </div>
</div>
<script>
const MODEL_COLORS = {
  'precision_shares': '#6366f1',
  'hybrid_shares':    '#22c55e',
  'scalp_shares':     '#eab308',
  'scalp_real':       '#ef4444',
  'ensemble':         '#06b6d4',
  'rapid_intraday':   '#a855f7',
  'index_specialist': '#f97316',
  'unknown':          '#888898',
};
let charts = {};
function dc(id){if(charts[id]){charts[id].destroy();delete charts[id];}}
function mc(name){return MODEL_COLORS[name]||'#'+Math.floor(Math.random()*16777215).toString(16);}

async function loadData() {
  try {
    const resp = await fetch('/models/data');
    const data = await resp.json();
    if (!data.models || data.models.length === 0) {
      document.getElementById('kpis').innerHTML =
        '<div class="loading">No run history yet. Click "Import History" to backfill from existing backtests.</div>';
      return;
    }
    renderKPIs(data);
    renderWinRateChart(data);
    renderReturnChart(data);
    renderProfitFactorChart(data);
    renderComparisonTable(data);
    document.getElementById('headerSub').textContent =
      `${data.total_runs} runs across ${data.models.length} models`;
  } catch(e) { console.error('Load failed:', e); }
}

function renderKPIs(d) {
  const models = d.models || [];
  const summary = d.summary || {};
  let bestWR = 0, bestModel = '', totalRuns = d.total_runs || 0;
  for (const [name, stats] of Object.entries(summary)) {
    if (stats.best_win_rate > bestWR) { bestWR = stats.best_win_rate; bestModel = name; }
  }
  document.getElementById('kpis').innerHTML = `
    <div class="kpi"><div class="label">Total Runs</div>
      <div class="value cyan">${totalRuns}</div></div>
    <div class="kpi"><div class="label">Models Tracked</div>
      <div class="value purple">${models.length}</div></div>
    <div class="kpi"><div class="label">Best Win Rate</div>
      <div class="value green">${(bestWR*100).toFixed(1)}%</div></div>
    <div class="kpi"><div class="label">Best Model</div>
      <div class="value" style="font-size:16px;color:${mc(bestModel)};">${bestModel.replace(/_/g,' ')}</div></div>
  `;
}

function renderWinRateChart(d) {
  const datasets = [];
  for (const [model, entries] of Object.entries(d.timelines || {})) {
    if (entries.length < 1) continue;
    datasets.push({
      label: model.replace(/_/g, ' '),
      data: entries.map(e => ({x: e.timestamp.split('T')[0], y: (e.win_rate*100)})),
      borderColor: mc(model),
      backgroundColor: mc(model)+'22',
      tension: 0.3, pointRadius: 3, borderWidth: 2, fill: false,
    });
  }
  dc('wrChart');
  charts['wrChart'] = new Chart(document.getElementById('wrChart'), {
    type: 'line',
    data: {datasets},
    options: {
      plugins: {legend: {labels: {color: '#888', font: {size:11}}}},
      scales: {
        x: {type:'category', ticks:{color:'#555',maxTicksLimit:12}, grid:{color:'#1a1a2e'}},
        y: {ticks:{color:'#555',callback:v=>v+'%'}, grid:{color:'#1a1a2e'},
            suggestedMin:0, suggestedMax:100, title:{display:true,text:'Win Rate %',color:'#555'}},
      },
    },
  });
}

function renderReturnChart(d) {
  const datasets = [];
  for (const [model, entries] of Object.entries(d.timelines || {})) {
    if (entries.length < 1) continue;
    datasets.push({
      label: model.replace(/_/g, ' '),
      data: entries.map(e => ({x: e.timestamp.split('T')[0], y: e.portfolio_return_pct})),
      borderColor: mc(model),
      tension: 0.3, pointRadius: 3, borderWidth: 2, fill: false,
    });
  }
  dc('retChart');
  charts['retChart'] = new Chart(document.getElementById('retChart'), {
    type: 'line',
    data: {datasets},
    options: {
      plugins: {legend: {labels: {color: '#888', font: {size:11}}}},
      scales: {
        x: {type:'category', ticks:{color:'#555',maxTicksLimit:8}, grid:{color:'#1a1a2e'}},
        y: {ticks:{color:'#555',callback:v=>v+'%'}, grid:{color:'#1a1a2e'},
            title:{display:true,text:'Return %',color:'#555'}},
      },
    },
  });
}

function renderProfitFactorChart(d) {
  const datasets = [];
  for (const [model, entries] of Object.entries(d.timelines || {})) {
    if (entries.length < 1) continue;
    datasets.push({
      label: model.replace(/_/g, ' '),
      data: entries.map(e => ({x: e.timestamp.split('T')[0], y: e.profit_factor})),
      borderColor: mc(model),
      tension: 0.3, pointRadius: 3, borderWidth: 2, fill: false,
    });
  }
  dc('pfChart');
  charts['pfChart'] = new Chart(document.getElementById('pfChart'), {
    type: 'line',
    data: {datasets},
    options: {
      plugins: {legend: {labels: {color: '#888', font: {size:11}}}},
      scales: {
        x: {type:'category', ticks:{color:'#555',maxTicksLimit:8}, grid:{color:'#1a1a2e'}},
        y: {ticks:{color:'#555'}, grid:{color:'#1a1a2e'}, suggestedMin:0,
            title:{display:true,text:'Profit Factor',color:'#555'}},
      },
    },
  });
}

function renderComparisonTable(d) {
  const summary = d.summary || {};
  const models = Object.entries(summary).sort((a,b) => b[1].best_win_rate - a[1].best_win_rate);
  if (!models.length) return;
  let html = `<table>
    <tr><th>Model</th><th>Runs</th><th>Best WR</th><th>Latest WR</th>
    <th>Best Return</th><th>Latest Return</th><th>Trend</th></tr>`;
  for (const [name, stats] of models) {
    const trend = stats.latest_win_rate >= stats.best_win_rate * 0.95 ? '&#9650;' : '&#9660;';
    const trendColor = stats.latest_win_rate >= stats.best_win_rate * 0.95 ? 'var(--green2)' : 'var(--red2)';
    const wrClass = stats.latest_win_rate >= 0.5 ? 'win' : 'loss';
    const retClass = stats.latest_return_pct >= 0 ? 'win' : 'loss';
    html += `<tr>
      <td><span class="model-badge" style="background:${mc(name)}22;color:${mc(name)};">${name.replace(/_/g,' ')}</span></td>
      <td>${stats.runs}</td>
      <td class="win">${(stats.best_win_rate*100).toFixed(1)}%</td>
      <td class="${wrClass}">${(stats.latest_win_rate*100).toFixed(1)}%</td>
      <td class="win">${stats.best_return_pct>=0?'+':''}${stats.best_return_pct.toFixed(1)}%</td>
      <td class="${retClass}">${stats.latest_return_pct>=0?'+':''}${stats.latest_return_pct.toFixed(1)}%</td>
      <td style="color:${trendColor};font-size:14px;">${trend}</td>
    </tr>`;
  }
  html += '</table>';
  document.getElementById('compTable').innerHTML = html;
}

async function backfill() {
  const btn = document.getElementById('backfillBtn');
  btn.disabled = true; btn.textContent = 'Importing...';
  try {
    const resp = await fetch('/models/backfill', {method:'POST'});
    const data = await resp.json();
    btn.textContent = `Imported ${data.imported} runs`;
    setTimeout(() => { btn.textContent = 'Import History'; btn.disabled = false; }, 3000);
    await loadData();
  } catch(e) {
    btn.textContent = 'Error'; btn.disabled = false;
  }
}

loadData();
</script>
</body>
</html>"""
