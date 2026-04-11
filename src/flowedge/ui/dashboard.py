"""NEXUS Dashboard — premium scanner UI with scan animations."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@dashboard_router.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the NEXUS scanner dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEXUS — FlowEdge Scanner</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg: #06060b; --bg2: #0c0c14; --bg3: #12121c;
  --border: #1a1a2e; --border2: #252540;
  --text: #c8c8d4; --text2: #888898; --text3: #55556a;
  --accent: #6366f1; --accent2: #818cf8; --accent-glow: rgba(99,102,241,0.15);
  --green: #22c55e; --green2: #4ade80; --green-glow: rgba(34,197,94,0.12);
  --red: #ef4444; --red2: #f87171;
  --yellow: #eab308; --yellow2: #facc15;
  --cyan: #06b6d4; --cyan2: #22d3ee; --cyan-glow: rgba(6,182,212,0.1);
  --purple: #a855f7; --purple2: #c084fc;
  --font: 'Inter', -apple-system, sans-serif;
  --mono: 'JetBrains Mono', monospace;
}

* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: var(--font); background: var(--bg); color: var(--text); overflow-x: hidden; }

/* ===== ANIMATED BACKGROUND ===== */
.bg-grid {
  position: fixed; inset: 0; z-index: 0; opacity: 0.03;
  background-image: linear-gradient(var(--border) 1px, transparent 1px),
                    linear-gradient(90deg, var(--border) 1px, transparent 1px);
  background-size: 60px 60px;
}
.bg-glow {
  position: fixed; width: 600px; height: 600px; border-radius: 50%;
  background: radial-gradient(circle, var(--accent-glow), transparent 70%);
  top: -200px; right: -200px; z-index: 0; animation: float 20s ease-in-out infinite;
}
.bg-glow2 {
  position: fixed; width: 400px; height: 400px; border-radius: 50%;
  background: radial-gradient(circle, var(--cyan-glow), transparent 70%);
  bottom: -100px; left: -100px; z-index: 0; animation: float 25s ease-in-out infinite reverse;
}
@keyframes float { 0%,100%{transform:translate(0,0)} 50%{transform:translate(30px,20px)} }

/* ===== HEADER ===== */
.header {
  position: relative; z-index: 10; background: rgba(12,12,20,0.8);
  backdrop-filter: blur(20px); border-bottom: 1px solid var(--border);
  padding: 16px 32px; display: flex; justify-content: space-between; align-items: center;
}
.logo { display: flex; align-items: center; gap: 12px; }
.logo-icon {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, var(--accent), var(--cyan));
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 14px; color: #fff; font-family: var(--mono);
}
.logo h1 { font-size: 18px; font-weight: 600; color: #fff; letter-spacing: -0.5px; }
.logo .sub { font-size: 11px; color: var(--text3); font-family: var(--mono); margin-top: 2px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.status-dot.live { background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse-dot 2s infinite; }
.status-dot.idle { background: var(--text3); }
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.4} }
#statusText { font-size: 12px; color: var(--text2); font-family: var(--mono); }

/* ===== CONTROLS ===== */
.controls {
  position: relative; z-index: 10; background: rgba(12,12,20,0.6);
  backdrop-filter: blur(12px); border-bottom: 1px solid var(--border);
  padding: 14px 32px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
}
input {
  background: var(--bg2); border: 1px solid var(--border2); color: #fff;
  padding: 10px 14px; border-radius: 8px; font-family: var(--mono); font-size: 13px;
  transition: border-color 0.2s, box-shadow 0.2s;
}
input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
button {
  background: linear-gradient(135deg, var(--accent), #4f46e5);
  color: #fff; border: none; padding: 10px 20px; border-radius: 8px;
  cursor: pointer; font-weight: 600; font-size: 13px; font-family: var(--font);
  transition: transform 0.1s, box-shadow 0.2s; position: relative; overflow: hidden;
}
button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px var(--accent-glow); }
button:active { transform: translateY(0); }
button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
button.secondary { background: var(--bg3); border: 1px solid var(--border2); }
button.secondary:hover { background: var(--border); }
button.secondary.active { background: var(--green); border-color: var(--green); }
.timer { font-size: 11px; color: var(--text3); font-family: var(--mono); }

/* ===== SCAN OVERLAY ===== */
.scan-overlay {
  position: fixed; inset: 0; z-index: 1000; background: rgba(6,6,11,0.92);
  backdrop-filter: blur(8px); display: none; flex-direction: column;
  align-items: center; justify-content: center;
}
.scan-overlay.active { display: flex; }
.scan-container { width: 520px; max-width: 90vw; text-align: center; }
.scan-title {
  font-size: 24px; font-weight: 700; color: #fff; margin-bottom: 8px;
  font-family: var(--mono);
}
.scan-subtitle { font-size: 13px; color: var(--text2); margin-bottom: 32px; }

/* Hexagon spinner */
.hex-spinner {
  width: 80px; height: 80px; margin: 0 auto 32px;
  position: relative;
}
.hex-spinner .ring {
  position: absolute; inset: 0; border: 2px solid transparent;
  border-top-color: var(--accent); border-radius: 50%;
  animation: spin 1.2s linear infinite;
}
.hex-spinner .ring:nth-child(2) {
  inset: 8px; border-top-color: var(--cyan);
  animation-duration: 1.8s; animation-direction: reverse;
}
.hex-spinner .ring:nth-child(3) {
  inset: 16px; border-top-color: var(--purple);
  animation-duration: 2.4s;
}
.hex-spinner .core {
  position: absolute; inset: 24px; border-radius: 50%;
  background: radial-gradient(circle, var(--accent), transparent);
  animation: pulse-core 2s ease-in-out infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse-core { 0%,100%{opacity:0.3;transform:scale(0.8)} 50%{opacity:1;transform:scale(1.1)} }

/* Progress bar */
.scan-progress {
  background: var(--bg3); border-radius: 6px; height: 6px;
  overflow: hidden; margin-bottom: 20px;
}
.scan-progress-bar {
  height: 100%; border-radius: 6px; transition: width 0.3s ease;
  background: linear-gradient(90deg, var(--accent), var(--cyan), var(--purple));
  background-size: 200% 100%; animation: shimmer 2s linear infinite;
}
@keyframes shimmer { to { background-position: -200% 0; } }

/* Engine steps */
.scan-steps { text-align: left; margin-bottom: 20px; }
.scan-step {
  display: flex; align-items: center; gap: 10px; padding: 8px 0;
  font-family: var(--mono); font-size: 12px; color: var(--text3);
  transition: color 0.3s;
}
.scan-step.active { color: var(--cyan2); }
.scan-step.done { color: var(--green2); }
.scan-step .icon { width: 20px; text-align: center; font-size: 14px; }
.scan-step .icon.spinner { animation: spin 1s linear infinite; display: inline-block; }
.scan-step .label { flex: 1; }
.scan-step .time { font-size: 11px; color: var(--text3); }

.scan-eta {
  font-family: var(--mono); font-size: 13px; color: var(--text2);
}

/* ===== MAIN GRID ===== */
.main { position: relative; z-index: 10; padding: 24px 32px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

/* ===== CARDS ===== */
.card {
  background: rgba(12,12,20,0.7); backdrop-filter: blur(8px);
  border: 1px solid var(--border); border-radius: 12px; padding: 20px;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.card:hover { border-color: var(--border2); box-shadow: 0 4px 30px rgba(0,0,0,0.3); }
.card.full { grid-column: 1 / -1; }
.card h2 {
  font-size: 11px; font-weight: 600; color: var(--text3); text-transform: uppercase;
  letter-spacing: 1.5px; margin-bottom: 16px; font-family: var(--mono);
}
.card h2 .engine-tag {
  display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px;
  background: var(--accent-glow); color: var(--accent2); margin-left: 8px;
}

/* ===== TABLE ===== */
table { width: 100%; border-collapse: collapse; }
th {
  text-align: left; padding: 10px 12px; color: var(--text3); font-size: 11px;
  font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
  border-bottom: 1px solid var(--border); font-family: var(--mono);
}
td { padding: 12px; border-bottom: 1px solid rgba(26,26,46,0.5); }
tr { transition: background 0.2s; }
tr:hover { background: rgba(99,102,241,0.04); }
.ticker-cell { font-weight: 700; font-size: 15px; color: #fff; font-family: var(--mono); }
.score-cell { font-weight: 700; font-size: 16px; font-family: var(--mono); }
.score-cell.high { color: var(--green2); text-shadow: 0 0 12px var(--green-glow); }
.score-cell.mid { color: var(--yellow2); }
.score-cell.low { color: var(--text3); }
.dir-bullish { color: var(--green2); font-weight: 600; }
.dir-bearish { color: var(--red2); font-weight: 600; }
.dir-neutral { color: var(--text3); }

/* Tags */
.tag {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 8px; border-radius: 6px; font-size: 10px;
  font-weight: 600; font-family: var(--mono); margin-right: 4px;
}
.tag.cheap { background: rgba(34,197,94,0.1); color: var(--green2); border: 1px solid rgba(34,197,94,0.2); }
.tag.sweep { background: rgba(6,182,212,0.1); color: var(--cyan2); border: 1px solid rgba(6,182,212,0.2); }
.tag.earnings { background: rgba(168,85,247,0.1); color: var(--purple2); border: 1px solid rgba(168,85,247,0.2); }
.tag.block { background: rgba(234,179,8,0.1); color: var(--yellow2); border: 1px solid rgba(234,179,8,0.2); }

/* Detail cards */
.detail-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px; margin-top: 12px;
}
.metric {
  background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px;
  transition: border-color 0.3s;
}
.metric:hover { border-color: var(--accent); }
.metric .label { font-size: 10px; color: var(--text3); text-transform: uppercase;
                 letter-spacing: 0.5px; font-family: var(--mono); }
.metric .value { font-size: 20px; font-weight: 700; margin-top: 4px; font-family: var(--mono); }
.metric .value.green { color: var(--green2); }
.metric .value.red { color: var(--red2); }
.metric .value.cyan { color: var(--cyan2); }
.metric .value.purple { color: var(--purple2); }
.metric .value.yellow { color: var(--yellow2); }

/* Entry/risk */
.signal-entry { margin-top: 12px; font-size: 12px; }
.signal-entry .good { color: var(--green2); }
.signal-entry .warn { color: var(--yellow2); }
.signal-line { padding: 3px 0; display: flex; gap: 6px; align-items: baseline; }

/* Intro */
.intro {
  grid-column: 1 / -1; text-align: center; padding: 80px 20px;
}
.intro-title { font-size: 32px; font-weight: 700; color: #fff; margin-bottom: 8px; }
.intro-sub { font-size: 14px; color: var(--text3); max-width: 500px; margin: 0 auto; }

/* Fade-in animation for results */
.fade-in { animation: fadeIn 0.5s ease-out; }
@keyframes fadeIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
.stagger-1 { animation-delay: 0.05s; }
.stagger-2 { animation-delay: 0.1s; }
.stagger-3 { animation-delay: 0.15s; }

@media (max-width: 800px) { .grid { grid-template-columns: 1fr; } .controls { padding: 12px 16px; } .main { padding: 16px; } }
</style>
</head>
<body>
<div class="bg-grid"></div>
<div class="bg-glow"></div>
<div class="bg-glow2"></div>

<!-- SCAN OVERLAY -->
<div class="scan-overlay" id="scanOverlay">
  <div class="scan-container">
    <div class="hex-spinner">
      <div class="ring"></div><div class="ring"></div><div class="ring"></div>
      <div class="core"></div>
    </div>
    <div class="scan-title" id="scanTitle">INITIALIZING NEXUS</div>
    <div class="scan-subtitle" id="scanSubtitle">Preparing scan engines...</div>
    <div class="scan-progress"><div class="scan-progress-bar" id="scanBar" style="width:0%"></div></div>
    <div class="scan-steps" id="scanSteps"></div>
    <div class="scan-eta" id="scanEta"></div>
  </div>
</div>

<!-- HEADER -->
<div class="header">
  <div class="logo">
    <div class="logo-icon">NX</div>
    <div>
      <h1>NEXUS</h1>
      <div class="sub">SPECTER &middot; ORACLE &middot; SENTINEL &middot; VORTEX &middot; PULSE</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <span class="status-dot idle" id="statusDot"></span>
    <span id="statusText">IDLE</span>
  </div>
</div>

<!-- CONTROLS -->
<div class="controls">
  <input type="text" id="tickers" placeholder="TSLA, NVDA, AAPL..."
         value="TSLA,NVDA,AAPL,META,AMZN,SPY,QQQ,AMD,GOOGL,MSFT" style="width:380px;">
  <input type="number" id="minScore" value="3" min="0" max="10" step="0.5" style="width:70px;" placeholder="Min">
  <button onclick="runScan()" id="scanBtn">SCAN</button>
  <button class="secondary" onclick="toggleAuto()" id="autoBtn">AUTO: OFF</button>
  <span class="timer" id="timer"></span>
</div>

<!-- MAIN -->
<div class="main">
  <div class="grid" id="results">
    <div class="intro fade-in">
      <div class="intro-title">Ready to scan</div>
      <div class="intro-sub">Enter tickers above and hit SCAN to activate SPECTER, ORACLE, and SENTINEL engines</div>
    </div>
  </div>
</div>

<script>
const ENGINES = [
  {id:'ingest', name:'Ingesting market data', engine:'DATA FEED', icon:'\u{1F4E1}', time:2},
  {id:'specter', name:'Analyzing options flow', engine:'SPECTER', icon:'\u{1F47B}', time:3},
  {id:'oracle', name:'Computing IV rank & regime', engine:'ORACLE', icon:'\u{1F52E}', time:2},
  {id:'sentinel', name:'Scanning catalysts & insiders', engine:'SENTINEL', icon:'\u{1F6E1}', time:3},
  {id:'nexus', name:'Fusing signals & scoring', engine:'NEXUS', icon:'\u26A1', time:1},
  {id:'rank', name:'Ranking opportunities', engine:'ARCHITECT', icon:'\u{1F3AF}', time:1},
];

let autoOn = false, autoIv = null, autoCount = null, cd = 0;

function showOverlay(tickers) {
  const ov = document.getElementById('scanOverlay');
  const steps = document.getElementById('scanSteps');
  ov.classList.add('active');
  document.getElementById('scanTitle').textContent = 'SCANNING ' + tickers.length + ' TARGETS';
  document.getElementById('scanSubtitle').textContent = tickers.join(', ');
  document.getElementById('scanBar').style.width = '0%';

  steps.innerHTML = ENGINES.map(e =>
    `<div class="scan-step" id="step-${e.id}">
      <span class="icon">\u25CB</span>
      <span class="label"><strong>${e.engine}</strong> \u2014 ${e.name}</span>
      <span class="time"></span>
    </div>`
  ).join('');

  animateSteps(tickers.length);
}

async function animateSteps(tickerCount) {
  const totalEst = ENGINES.reduce((s,e) => s + e.time, 0);
  let elapsed = 0;

  for (let i = 0; i < ENGINES.length; i++) {
    const e = ENGINES[i];
    const step = document.getElementById('step-' + e.id);
    const icon = step.querySelector('.icon');
    const timeEl = step.querySelector('.time');
    step.classList.add('active');
    icon.textContent = '\u25E0'; icon.classList.add('spinner');

    const remaining = totalEst - elapsed;
    document.getElementById('scanEta').textContent = `Est. ${remaining}s remaining \u2022 ${tickerCount} tickers \u00d7 ${ENGINES.length} engines`;

    await sleep(e.time * 400);
    elapsed += e.time;
    const pct = Math.min(95, (elapsed / totalEst) * 95);
    document.getElementById('scanBar').style.width = pct + '%';

    icon.classList.remove('spinner');
    icon.textContent = '\u2713';
    step.classList.remove('active');
    step.classList.add('done');
    timeEl.textContent = (e.time * 0.4).toFixed(1) + 's';
  }
}

function hideOverlay() {
  document.getElementById('scanBar').style.width = '100%';
  document.getElementById('scanEta').textContent = 'Complete';
  document.getElementById('scanTitle').textContent = 'SCAN COMPLETE';
  setTimeout(() => document.getElementById('scanOverlay').classList.remove('active'), 600);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function runScan() {
  const btn = document.getElementById('scanBtn');
  const dot = document.getElementById('statusDot');
  const txt = document.getElementById('statusText');
  const tickers = document.getElementById('tickers').value.split(',').map(t=>t.trim().toUpperCase()).filter(Boolean);
  const minScore = parseFloat(document.getElementById('minScore').value) || 0;
  if (!tickers.length) return;

  btn.disabled = true;
  dot.className = 'status-dot live';
  txt.textContent = 'SCANNING...';
  showOverlay(tickers);

  try {
    const resp = await fetch('/api/v1/scanner/scan', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({tickers, scan_types:['uoa','iv','catalyst'], min_score:minScore})
    });
    const data = await resp.json();
    hideOverlay();
    await sleep(700);
    renderResults(data);
    txt.textContent = 'LIVE \u2022 ' + new Date().toLocaleTimeString();
  } catch(e) {
    hideOverlay();
    document.getElementById('results').innerHTML =
      '<div class="card full fade-in"><div style="color:var(--red);padding:20px;">Scan failed: '+e.message+'</div></div>';
    txt.textContent = 'ERROR';
    dot.className = 'status-dot idle';
  } finally {
    btn.disabled = false;
  }
}

function renderResults(data) {
  const r = document.getElementById('results');
  if (!data.opportunities || !data.opportunities.length) {
    r.innerHTML = '<div class="intro fade-in"><div class="intro-title">No signals detected</div><div class="intro-sub">Try lowering the minimum score or adding more tickers</div></div>';
    return;
  }

  let html = '<div class="card full fade-in"><h2>Lotto Opportunities <span class="engine-tag">NEXUS</span></h2><table>';
  html += '<tr><th>Ticker</th><th style="text-align:center">SCORE</th><th>Direction</th><th>Signals</th><th>Top Contract</th></tr>';

  for (const o of data.opportunities) {
    const s100 = o.score_100 || Math.round(o.composite_score * 10);
    const sc = s100 >= 60 ? 'high' : s100 >= 40 ? 'mid' : 'low';
    const dc = o.suggested_direction === 'bullish' ? 'dir-bullish' : o.suggested_direction === 'bearish' ? 'dir-bearish' : 'dir-neutral';
    let tags = '';
    if (o.iv_signal?.is_cheap_premium) tags += '<span class="tag cheap">\u{1F4B0} CHEAP IV</span>';
    if (o.uoa_signal?.signal_type === 'sweep_cluster') tags += '<span class="tag sweep">\u{1F30A} SWEEPS</span>';
    if (o.uoa_signal?.signal_type === 'block_trade') tags += '<span class="tag block">\u{1F4B8} BLOCK</span>';
    if (o.catalyst_signal?.days_to_nearest_catalyst) tags += '<span class="tag earnings">\u{1F4C5} EARN '+o.catalyst_signal.days_to_nearest_catalyst+'d</span>';

    let contract = '<span style="color:var(--text3)">--</span>';
    if (o.contract_picks?.length) {
      const c = o.contract_picks[0];
      contract = `<span style="font-family:var(--mono);font-size:12px;">${c.option_type.toUpperCase()} $${c.strike} ${c.expiration}</span>`;
    }

    html += `<tr>
      <td class="ticker-cell">${o.ticker}</td>
      <td style="text-align:center"><span class="score-cell ${sc}">${s100}</span><span style="font-size:11px;color:var(--text3)">/100</span></td>
      <td class="${dc}">${o.suggested_direction.toUpperCase()}</td>
      <td>${tags}</td>
      <td>${contract}</td>
    </tr>`;
  }
  html += '</table></div>';

  data.opportunities.slice(0,3).forEach((o, i) => {
    const s100 = o.score_100 || Math.round(o.composite_score * 10);
    const scoreColor = s100 >= 60 ? 'green' : s100 >= 40 ? 'yellow' : 'red';
    html += `<div class="card fade-in stagger-${i+1}"><h2>${o.ticker} <span class="engine-tag">DETAIL</span></h2>`;
    html += '<div class="detail-grid">';
    html += m('NEXUS Score', s100 + '/100', scoreColor);
    html += m('Direction', o.suggested_direction.toUpperCase(), o.suggested_direction==='bullish'?'green':o.suggested_direction==='bearish'?'red':'');
    if (o.iv_signal) {
      html += m('IV Rank', o.iv_signal.iv_rank.iv_rank.toFixed(1)+'%', o.iv_signal.iv_rank.iv_rank < 30 ? 'green' : 'yellow');
      html += m('Regime', o.iv_signal.regime.toUpperCase(), '');
    }
    if (o.catalyst_signal?.days_to_nearest_catalyst != null) html += m('Catalyst', o.catalyst_signal.days_to_nearest_catalyst+'d', 'purple');
    if (o.uoa_signal) {
      html += m('Premium', '$'+(o.uoa_signal.total_premium/1e6).toFixed(1)+'M', 'cyan');
      html += m('C/P Ratio', o.uoa_signal.call_put_ratio.toFixed(1), o.uoa_signal.call_put_ratio > 2 ? 'green' : '');
    }
    html += '</div>';

    // Contract picks section
    if (o.contract_picks?.length) {
      html += '<div style="margin-top:16px;"><h2 style="margin-bottom:10px;">Recommended Contracts <span class="engine-tag">ARCHITECT</span></h2>';
      html += '<table style="font-size:12px;">';
      html += '<tr><th>Type</th><th>Strike</th><th>Exp</th><th>Vol</th><th>OI</th><th>Max Loss</th><th>Setup</th></tr>';
      o.contract_picks.forEach(c => {
        const typeColor = c.option_type === 'call' ? 'var(--green2)' : 'var(--red2)';
        html += `<tr>
          <td style="color:${typeColor};font-weight:600;font-family:var(--mono);">${c.option_type.toUpperCase()}</td>
          <td style="font-family:var(--mono);">$${c.strike}</td>
          <td style="font-family:var(--mono);">${c.expiration}</td>
          <td>${c.volume.toLocaleString()}</td>
          <td>${c.open_interest.toLocaleString()}</td>
          <td style="font-family:var(--mono);">$${c.max_loss_per_contract.toFixed(0)}</td>
          <td style="font-size:11px;color:var(--text2);">${c.reason}</td>
        </tr>`;
      });
      html += '</table></div>';
    }

    if (o.entry_criteria?.length) {
      html += '<div class="signal-entry" style="margin-top:12px;">';
      o.entry_criteria.forEach(c => html += `<div class="signal-line"><span class="good">\u2714</span> ${c}</div>`);
      html += '</div>';
    }
    if (o.risk_flags?.length) {
      html += '<div class="signal-entry" style="margin-top:6px;">';
      o.risk_flags.forEach(f => html += `<div class="signal-line"><span class="warn">\u26A0</span> ${f}</div>`);
      html += '</div>';
    }
    html += '</div>';
  });

  r.innerHTML = html;
}

function m(label, value, color) {
  return `<div class="metric"><div class="label">${label}</div><div class="value ${color}">${value}</div></div>`;
}

function toggleAuto() {
  autoOn = !autoOn;
  const btn = document.getElementById('autoBtn');
  if (autoOn) {
    btn.textContent = 'AUTO: ON'; btn.classList.add('active');
    runScan(); cd = 300;
    autoIv = setInterval(() => { cd = 300; runScan(); }, 300000);
    autoCount = setInterval(() => { cd--; document.getElementById('timer').textContent = cd + 's'; }, 1000);
  } else {
    btn.textContent = 'AUTO: OFF'; btn.classList.remove('active');
    clearInterval(autoIv); clearInterval(autoCount);
    document.getElementById('timer').textContent = '';
  }
}
</script>
</body>
</html>"""
