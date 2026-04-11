"""Dashboard API — serves the scanner UI and real-time data."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@dashboard_router.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the scanner dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FlowEdge NEXUS — Scanner Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0a0f; color: #e0e0e0; }
  .header { background: #111118; padding: 16px 24px; border-bottom: 1px solid #222;
            display: flex; justify-content: space-between; align-items: center; }
  .header h1 { font-size: 20px; color: #fff; }
  .header .status { font-size: 13px; color: #888; }
  .header .status.live { color: #4ade80; }
  .controls { padding: 16px 24px; background: #111118; border-bottom: 1px solid #222;
              display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  input, select, button { font-family: inherit; font-size: 14px; }
  input { background: #1a1a24; border: 1px solid #333; color: #fff; padding: 8px 12px;
          border-radius: 6px; }
  button { background: #2563eb; color: #fff; border: none; padding: 8px 16px;
           border-radius: 6px; cursor: pointer; font-weight: 500; }
  button:hover { background: #3b82f6; }
  button.secondary { background: #333; }
  button.secondary:hover { background: #444; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
          padding: 16px 24px; }
  .card { background: #111118; border: 1px solid #222; border-radius: 8px;
          padding: 16px; }
  .card h2 { font-size: 15px; color: #999; margin-bottom: 12px; text-transform: uppercase;
             letter-spacing: 0.5px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px; color: #666; border-bottom: 1px solid #222;
       font-weight: 500; }
  td { padding: 8px; border-bottom: 1px solid #1a1a24; }
  .score { font-weight: 700; font-size: 15px; }
  .score.high { color: #4ade80; }
  .score.mid { color: #fbbf24; }
  .score.low { color: #666; }
  .bullish { color: #4ade80; }
  .bearish { color: #ef4444; }
  .neutral { color: #888; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-size: 11px; font-weight: 600; }
  .tag.cheap { background: #064e3b; color: #4ade80; }
  .tag.sweep { background: #1e3a5f; color: #60a5fa; }
  .tag.earnings { background: #4c1d95; color: #c084fc; }
  .detail { grid-column: 1 / -1; }
  .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                 gap: 12px; margin-top: 8px; }
  .metric { background: #0a0a0f; border-radius: 6px; padding: 12px; }
  .metric .label { font-size: 11px; color: #666; text-transform: uppercase; }
  .metric .value { font-size: 22px; font-weight: 700; margin-top: 4px; }
  .loading { text-align: center; padding: 40px; color: #666; }
  .error { color: #ef4444; padding: 12px; background: #1a0a0a; border-radius: 6px;
           margin: 8px 0; }
  .refresh-timer { font-size: 12px; color: #666; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="header">
  <h1>FlowEdge NEXUS</h1>
  <div class="status" id="status">Idle</div>
</div>

<div class="controls">
  <input type="text" id="tickers" placeholder="TSLA, NVDA, AAPL, META, AMZN"
         value="TSLA,NVDA,AAPL,META,AMZN,SPY,QQQ,AMD,GOOGL,MSFT" style="width: 400px;">
  <input type="number" id="minScore" value="3" min="0" max="10" step="0.5"
         style="width: 80px;" placeholder="Min">
  <button onclick="runScan()" id="scanBtn">Scan Now</button>
  <button class="secondary" onclick="toggleAutoRefresh()" id="autoBtn">
    Auto-refresh: OFF</button>
  <span class="refresh-timer" id="timer"></span>
</div>

<div class="grid" id="results">
  <div class="card detail">
    <div class="loading">Enter tickers and click Scan Now</div>
  </div>
</div>

<script>
let autoRefresh = false;
let refreshInterval = null;
let countdown = 0;
let countdownInterval = null;

async function runScan() {
  const btn = document.getElementById('scanBtn');
  const status = document.getElementById('status');
  const tickers = document.getElementById('tickers').value.split(',').map(t => t.trim().toUpperCase()).filter(Boolean);
  const minScore = parseFloat(document.getElementById('minScore').value) || 0;

  if (!tickers.length) return;

  btn.disabled = true;
  btn.textContent = 'Scanning...';
  status.textContent = 'Scanning...';
  status.className = 'status';

  try {
    const resp = await fetch('/api/v1/scanner/scan', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tickers, scan_types: ['uoa', 'iv', 'catalyst'], min_score: minScore})
    });
    const data = await resp.json();
    renderResults(data);
    status.textContent = `Live \u2022 ${new Date().toLocaleTimeString()}`;
    status.className = 'status live';
  } catch(e) {
    document.getElementById('results').innerHTML =
      '<div class="card detail"><div class="error">Scan failed: ' + e.message + '</div></div>';
    status.textContent = 'Error';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Scan Now';
  }
}

function renderResults(data) {
  const results = document.getElementById('results');
  if (!data.opportunities || !data.opportunities.length) {
    results.innerHTML = '<div class="card detail"><div class="loading">No opportunities above minimum score</div></div>';
    return;
  }

  let html = '<div class="card detail"><h2>Lotto Opportunities</h2><table>';
  html += '<tr><th>Ticker</th><th>NEXUS</th><th>SPECTER</th><th>ORACLE</th><th>SENTINEL</th><th>Direction</th><th>Signals</th></tr>';

  for (const opp of data.opportunities) {
    const scoreClass = opp.composite_score >= 6 ? 'high' : opp.composite_score >= 4 ? 'mid' : 'low';
    const dirClass = opp.suggested_direction;
    let tags = '';
    if (opp.iv_signal && opp.iv_signal.is_cheap_premium) tags += '<span class="tag cheap">CHEAP IV</span> ';
    if (opp.uoa_signal && opp.uoa_signal.signal_type === 'sweep_cluster') tags += '<span class="tag sweep">SWEEPS</span> ';
    if (opp.catalyst_signal && opp.catalyst_signal.days_to_nearest_catalyst) tags += '<span class="tag earnings">EARNINGS ' + opp.catalyst_signal.days_to_nearest_catalyst + 'd</span> ';

    html += '<tr>';
    html += '<td><strong>' + opp.ticker + '</strong></td>';
    html += '<td><span class="score ' + scoreClass + '">' + opp.composite_score.toFixed(1) + '</span></td>';
    html += '<td>' + opp.uoa_score.toFixed(1) + '</td>';
    html += '<td>' + opp.iv_score.toFixed(1) + '</td>';
    html += '<td>' + opp.catalyst_score.toFixed(1) + '</td>';
    html += '<td class="' + dirClass + '">' + opp.suggested_direction + '</td>';
    html += '<td>' + tags + '</td>';
    html += '</tr>';
  }
  html += '</table></div>';

  // Detail cards for top 3
  const top = data.opportunities.slice(0, 3);
  for (const opp of top) {
    html += '<div class="card"><h2>' + opp.ticker + '</h2>';
    html += '<div class="detail-grid">';
    html += metric('Score', opp.composite_score.toFixed(1) + '/10');
    html += metric('Direction', opp.suggested_direction);

    if (opp.iv_signal) {
      html += metric('IV Rank', opp.iv_signal.iv_rank.iv_rank.toFixed(1) + '%');
      html += metric('IV Regime', opp.iv_signal.regime);
    }
    if (opp.catalyst_signal && opp.catalyst_signal.days_to_nearest_catalyst !== null) {
      html += metric('Catalyst', opp.catalyst_signal.days_to_nearest_catalyst + ' days');
    }
    if (opp.uoa_signal) {
      html += metric('Flow Premium', '$' + (opp.uoa_signal.total_premium/1e6).toFixed(1) + 'M');
      html += metric('C/P Ratio', opp.uoa_signal.call_put_ratio.toFixed(1));
    }
    html += '</div>';

    if (opp.entry_criteria && opp.entry_criteria.length) {
      html += '<div style="margin-top:12px;font-size:12px;color:#4ade80;">';
      opp.entry_criteria.forEach(c => html += '\u2714 ' + c + '<br>');
      html += '</div>';
    }
    if (opp.risk_flags && opp.risk_flags.length) {
      html += '<div style="margin-top:8px;font-size:12px;color:#fbbf24;">';
      opp.risk_flags.forEach(f => html += '\u26a0 ' + f + '<br>');
      html += '</div>';
    }
    html += '</div>';
  }

  results.innerHTML = html;
}

function metric(label, value) {
  return '<div class="metric"><div class="label">' + label + '</div><div class="value">' + value + '</div></div>';
}

function toggleAutoRefresh() {
  autoRefresh = !autoRefresh;
  const btn = document.getElementById('autoBtn');
  if (autoRefresh) {
    btn.textContent = 'Auto-refresh: ON';
    btn.style.background = '#16a34a';
    runScan();
    countdown = 300;
    refreshInterval = setInterval(() => { countdown = 300; runScan(); }, 300000);
    countdownInterval = setInterval(() => {
      countdown--;
      document.getElementById('timer').textContent = 'Next scan in ' + countdown + 's';
    }, 1000);
  } else {
    btn.textContent = 'Auto-refresh: OFF';
    btn.style.background = '#333';
    clearInterval(refreshInterval);
    clearInterval(countdownInterval);
    document.getElementById('timer').textContent = '';
  }
}
</script>
</body>
</html>"""
