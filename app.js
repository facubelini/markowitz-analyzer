'use strict';

// ─── Constantes ───────────────────────────────────────────────
const RF     = 0.05;   // tasa libre de riesgo (5 %)
const DAYS   = 252;    // días hábiles / año
const N_SIM  = 5000;   // portafolios Monte Carlo
const YEARS  = 2;      // ventana histórica

const COLORS = [
  '#00d4aa','#ff6b6b','#4ecdc4','#45b7d1',
  '#ffd700','#a29bfe','#fd79a8','#6c5ce7',
  '#55efc4','#fdcb6e'
];

// ─── Estado global ────────────────────────────────────────────
let tickers  = ['AAPL','MSFT','GOOGL','AMZN'];
let results  = null;
const charts = {};

// ─── Utilidades ───────────────────────────────────────────────
const lerp   = (a, b, t) => a + (b - a) * t;
const clamp  = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmtPct = (v, d = 1) => (v * 100).toFixed(d) + '%';
const fmtN   = (v, d = 2) => v.toFixed(d);
const isValidTicker = t => /^[A-Z0-9]{1,5}(\.[A-Z]{1,2})?$/.test(t);

// Convierte un Ratio de Sharpe normalizado [0,1] en un color azul→teal→dorado
function sharpeColor(t) {
  t = clamp(t, 0, 1);
  if (t < 0.5) {
    const s = t * 2;
    return `rgba(${Math.round(lerp(30,0,s))},${Math.round(lerp(100,212,s))},${Math.round(lerp(200,170,s))},0.75)`;
  }
  const s = (t - 0.5) * 2;
  return `rgba(${Math.round(lerp(0,255,s))},${Math.round(lerp(212,210,s))},${Math.round(lerp(170,0,s))},0.75)`;
}

// Convierte correlación [-1,1] en color rojo→oscuro→teal
function corrColor(r) {
  r = clamp(r, -1, 1);
  if (r >= 0)
    return `rgb(${Math.round(lerp(40,0,r))},${Math.round(lerp(45,168,r))},${Math.round(lerp(62,118,r))})`;
  const a = -r;
  return `rgb(${Math.round(lerp(40,200,a))},${Math.round(lerp(45,48,a))},${Math.round(lerp(62,60,a))})`;
}

// ─── Obtención de datos ───────────────────────────────────────
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function fetchPrices(ticker) {
  const now   = Math.floor(Date.now() / 1000);
  const start = now - YEARS * 365 * 24 * 3600;
  const base  = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1d&period1=${start}&period2=${now}&events=div,splits&includeAdjustedClose=true`;

  let data;
  try {
    data = await getJSON(base);
  } catch (_) {
    // Fallback con proxy CORS
    data = await getJSON(`https://corsproxy.io/?${encodeURIComponent(base)}`);
  }

  if (data?.chart?.error) throw new Error(data.chart.error.description || 'No encontrado');

  const res = data?.chart?.result?.[0];
  if (!res) throw new Error('Sin datos');

  const ts     = res.timestamp || [];
  const prices = res.indicators?.adjclose?.[0]?.adjclose
               || res.indicators?.quote?.[0]?.close
               || [];

  const valid = ts
    .map((t, i) => ({ t, p: prices[i] }))
    .filter(x => x.p != null && x.p > 0);

  if (valid.length < 50) throw new Error('Datos insuficientes');
  return { ts: valid.map(x => x.t), prices: valid.map(x => x.p) };
}

// Alinea todas las series a fechas comunes
function alignSeries(dataMap) {
  const keys = Object.keys(dataMap);
  const sets  = keys.map(k => new Set(dataMap[k].ts.map(String)));

  let common = [...sets[0]];
  for (let i = 1; i < sets.length; i++) common = common.filter(v => sets[i].has(v));
  common = common.map(Number).sort((a, b) => a - b);

  const out = {};
  for (const k of keys) {
    const idx = new Map(dataMap[k].ts.map((t, i) => [String(t), i]));
    out[k] = common.map(t => dataMap[k].prices[idx.get(String(t))]);
  }
  return out;
}

// ─── Matemática financiera ────────────────────────────────────

// Retornos logarítmicos diarios: ln(Pₜ / Pₜ₋₁)
function logReturns(prices) {
  const r = [];
  for (let i = 1; i < prices.length; i++)
    r.push(Math.log(prices[i] / prices[i - 1]));
  return r;
}

function mean(arr) { return arr.reduce((s, v) => s + v, 0) / arr.length; }

function cov(a, b) {
  const ma = mean(a), mb = mean(b);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - ma) * (b[i] - mb);
  return s / (a.length - 1);
}

// Matriz de covarianza anualizada (× 252)
function buildCov(retMap, keys) {
  const n = keys.length;
  const C = Array.from({length: n}, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = i; j < n; j++) {
      const v = cov(retMap[keys[i]], retMap[keys[j]]) * DAYS;
      C[i][j] = C[j][i] = v;
    }
  return C;
}

// Matriz de correlación a partir de la covarianza
function buildCorr(C) {
  const n = C.length;
  const R = Array.from({length: n}, () => new Array(n).fill(0));
  const sd = C.map((row, i) => Math.sqrt(row[i]));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      R[i][j] = clamp(C[i][j] / (sd[i] * sd[j]), -1, 1);
  return R;
}

// Varianza del portafolio: wᵀ · C · w
function portVar(w, C) {
  let v = 0;
  for (let i = 0; i < w.length; i++)
    for (let j = 0; j < w.length; j++) v += w[i] * w[j] * C[i][j];
  return v;
}

// Retorno esperado del portafolio: Σ wᵢ · μᵢ
function portRet(w, mu) { return w.reduce((s, wi, i) => s + wi * mu[i], 0); }

// Pesos aleatorios con distribución Dirichlet (via exponenciales)
function randWeights(n) {
  const u = Array.from({length: n}, () => -Math.log(Math.random() + 1e-10));
  const s = u.reduce((a, b) => a + b, 0);
  return u.map(x => x / s);
}

// Simulación Monte Carlo: genera N portafolios aleatorios
function monteCarlo(mu, C, n) {
  const portfolios = [];
  for (let k = 0; k < n; k++) {
    const w   = randWeights(mu.length);
    const ret = portRet(w, mu);
    const vol = Math.sqrt(portVar(w, C));
    portfolios.push({ w, ret, vol, sharpe: (ret - RF) / vol });
  }
  return portfolios;
}

// Beta de cada activo vs SPY calculado de forma INDEPENDIENTE (alineación pairwise).
// Cada ticker se alinea solo con SPY, sin involucrar a los demás tickers.
// Así el beta de AAPL es siempre el mismo sin importar qué otros tickers haya.
// Retorna null para cada activo si SPY no está disponible.
function computeBetas(rawMap, tickerList, spyRaw) {
  if (!spyRaw) return tickerList.map(() => null);
  return tickerList.map(t => {
    try {
      // Alineación solo entre este activo y SPY
      const pair     = alignSeries({ asset: rawMap[t], spy: spyRaw });
      const assetRet = logReturns(pair.asset);
      const spyRet   = logReturns(pair.spy);
      const mktVar   = cov(spyRet, spyRet);
      if (mktVar === 0 || assetRet.length < 30) return null;
      return cov(assetRet, spyRet) / mktVar;
    } catch (_) { return null; }
  });
}

// ─── Chips de tickers ─────────────────────────────────────────
function renderChips() {
  const el = document.getElementById('ticker-chips');
  el.innerHTML = '';
  tickers.forEach((t, i) => {
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.id = `chip-${t}`;
    chip.innerHTML = `
      <span class="chip-color-dot" style="background:${COLORS[i % COLORS.length]}"></span>
      <span>${t}</span>
      <button class="chip-remove" aria-label="Eliminar ${t}">×</button>`;
    chip.querySelector('.chip-remove').onclick = () => { tickers = tickers.filter(x => x !== t); renderChips(); };
    el.appendChild(chip);
  });
  document.getElementById('analyze-btn').disabled = tickers.length < 2;
}

function addTicker(raw) {
  document.getElementById('ticker-error').classList.add('hidden');
  const parts = raw.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
  let added = 0;
  for (const t of parts) {
    if (!isValidTicker(t))    { showInputErr(`"${t}" no es un ticker válido.`); continue; }
    if (tickers.includes(t))  { showInputErr(`"${t}" ya está en el portafolio.`); continue; }
    if (tickers.length >= 10) { showInputErr('Máximo 10 tickers permitidos.'); break; }
    tickers.push(t);
    added++;
  }
  if (added) renderChips();
}

function showInputErr(msg) {
  const el = document.getElementById('ticker-error');
  el.textContent = msg;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 4000);
}

// ─── UI de carga ──────────────────────────────────────────────
function showLoading() {
  document.getElementById('loading-state').classList.remove('hidden');
  document.getElementById('results').classList.add('hidden');
  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('ticker-status-list').innerHTML = tickers.map(t => `
    <div class="ts-item" id="ts-${t}">
      <span class="ts-dot"></span>
      <span>${t}</span>
      <span class="ts-msg" style="color:var(--text-3);font-size:11px">esperando…</span>
    </div>`).join('');
}

function setTS(ticker, state, msg) {
  const el = document.getElementById(`ts-${ticker}`);
  if (!el) return;
  el.className = `ts-item ${state}`;
  el.querySelector('.ts-msg').textContent = msg;
}

function setStatus(msg) {
  const el = document.getElementById('loading-status');
  if (el) el.textContent = msg;
}

function hideLoading() {
  document.getElementById('loading-state').classList.add('hidden');
  document.getElementById('analyze-btn').disabled = tickers.length < 2;
}

// ─── Gráficos ─────────────────────────────────────────────────
function destroyChart(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }

function renderFrontier(portfolios, maxS, minV) {
  destroyChart('frontier');
  const sharpes = portfolios.map(p => p.sharpe);
  const lo = Math.min(...sharpes), hi = Math.max(...sharpes);

  const dots = portfolios.map(p => ({
    x: p.vol * 100, y: p.ret * 100,
    _w: p.w, _s: p.sharpe
  }));
  const cols = portfolios.map(p => sharpeColor((p.sharpe - lo) / (hi - lo || 1)));

  charts['frontier'] = new Chart(
    document.getElementById('frontier-chart').getContext('2d'),
    {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Portafolios',
            data: dots,
            backgroundColor: cols,
            pointRadius: 3.5,
            pointHoverRadius: 5,
            order: 3
          },
          {
            label: 'Máximo Sharpe',
            data: [{ x: maxS.vol * 100, y: maxS.ret * 100, _w: maxS.w, _s: maxS.sharpe }],
            backgroundColor: '#00e87a',
            borderColor: '#fff',
            borderWidth: 1.5,
            pointStyle: 'star',
            pointRadius: 13,
            pointHoverRadius: 17,
            order: 1
          },
          {
            label: 'Mínima Varianza',
            data: [{ x: minV.vol * 100, y: minV.ret * 100, _w: minV.w, _s: minV.sharpe }],
            backgroundColor: '#ff8c00',
            borderColor: '#fff',
            borderWidth: 1.5,
            pointStyle: 'triangle',
            pointRadius: 11,
            pointHoverRadius: 15,
            order: 2
          }
        ]
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: true },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1a1d27',
            borderColor: '#272b3d',
            borderWidth: 1,
            titleColor: '#dde3ef',
            bodyColor: '#8892a4',
            padding: 12,
            callbacks: {
              title: items => {
                const lbl = items[0].dataset.label;
                return lbl === 'Máximo Sharpe'   ? '★ Portafolio de Máximo Sharpe'
                     : lbl === 'Mínima Varianza' ? '▲ Portafolio de Mínima Varianza'
                     : 'Portafolio Simulado';
              },
              label: item => {
                const d = item.raw;
                return [
                  ` Retorno:      ${fmtPct(d.y / 100)}`,
                  ` Volatilidad:  ${fmtPct(d.x / 100)}`,
                  ` Ratio Sharpe: ${fmtN(d._s)}`,
                  ' ─────────────────────',
                  ...tickers.map((t, i) => ` ${t.padEnd(6)} ${fmtPct(d._w[i])}`)
                ];
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Volatilidad Anualizada (%)', color: '#525d72', font: { family: "'Inter'", size: 11 } },
            ticks: { color: '#525d72', font: { family: "'JetBrains Mono'", size: 10 }, callback: v => v.toFixed(0) + '%' },
            grid: { color: 'rgba(39,43,61,.6)' }
          },
          y: {
            title: { display: true, text: 'Retorno Anualizado (%)', color: '#525d72', font: { family: "'Inter'", size: 11 } },
            ticks: { color: '#525d72', font: { family: "'JetBrains Mono'", size: 10 }, callback: v => v.toFixed(0) + '%' },
            grid: { color: 'rgba(39,43,61,.6)' }
          }
        }
      }
    }
  );
}

// Gráfico de barras horizontales para pesos del portafolio
function renderWeightsBar(canvasId, w, tkrs) {
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const h = Math.max(90, tkrs.length * 38 + 30);
  canvas.parentElement.style.height = h + 'px';

  charts[canvasId] = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels: tkrs,
      datasets: [{
        data: w.map(v => v * 100),
        backgroundColor: tkrs.map((_, i) => COLORS[i % COLORS.length] + 'bb'),
        borderColor:     tkrs.map((_, i) => COLORS[i % COLORS.length]),
        borderWidth: 1.5,
        borderRadius: 4,
        barThickness: 22
      }]
    },
    options: {
      indexAxis: 'y',
      animation: { duration: 350 },
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1a1d27',
          borderColor: '#272b3d',
          borderWidth: 1,
          callbacks: { label: item => ` ${item.raw.toFixed(1)}%` }
        }
      },
      scales: {
        x: {
          max: 100,
          ticks: { color: '#525d72', font: { family: "'JetBrains Mono'", size: 10 }, callback: v => v + '%' },
          grid: { color: 'rgba(39,43,61,.4)' }
        },
        y: {
          ticks: { color: '#8892a4', font: { family: "'JetBrains Mono'", size: 11, weight: '600' } },
          grid: { display: false }
        }
      }
    }
  });
}

// ─── Mapa de calor de correlación ─────────────────────────────
function renderCorr(R, tkrs) {
  const n   = tkrs.length;
  const sz  = n <= 5 ? 50 : n <= 7 ? 44 : 38;
  const wrap = document.getElementById('corr-matrix');

  const table = document.createElement('table');
  table.className = 'corr-table';

  // Fila de encabezados
  const thead = table.createTHead();
  const hr    = thead.insertRow();
  hr.insertCell();
  tkrs.forEach(t => {
    const th = document.createElement('th');
    th.className = 'corr-col-label';
    th.textContent = t;
    hr.appendChild(th);
  });

  // Filas de datos
  const tbody = table.createTBody();
  for (let i = 0; i < n; i++) {
    const row = tbody.insertRow();
    const lc  = row.insertCell();
    lc.className = 'corr-row-label';
    lc.textContent = tkrs[i];
    for (let j = 0; j < n; j++) {
      const td = row.insertCell();
      const r  = R[i][j];
      td.innerHTML = `<div class="corr-cell" style="width:${sz}px;height:${sz}px;background:${corrColor(r)};color:#dde3ef" title="${tkrs[i]} vs ${tkrs[j]}: ${r.toFixed(3)}">${r.toFixed(2)}</div>`;
    }
  }

  wrap.innerHTML = '';
  wrap.appendChild(table);
}

// ─── Tarjetas de métricas ─────────────────────────────────────
function renderMetrics(port, containerId) {
  const retCol  = port.ret >= 0 ? 'c-green' : 'c-red';
  const shrpCol = port.sharpe >= 1 ? 'c-green' : port.sharpe >= 0 ? 'c-gold' : 'c-red';
  document.getElementById(containerId).innerHTML = `
    <div class="metric">
      <div class="metric-label">
        <span class="tip" data-tip="exp(μ_log × 252) − 1. Retorno anual compuesto simple: lo que rindió la inversión en el año, siempre acotado a más de −100%.">Retorno Anual</span>
      </div>
      <div class="metric-value ${retCol}">${fmtPct(port.ret)}</div>
    </div>
    <div class="metric">
      <div class="metric-label">
        <span class="tip" data-tip="Desviación estándar de retornos diarios × √252. Mide cuánto fluctúa el portafolio. Mayor volatilidad = mayor riesgo.">Volatilidad Anual</span>
      </div>
      <div class="metric-value c-white">${fmtPct(port.vol)}</div>
    </div>
    <div class="metric">
      <div class="metric-label">
        <span class="tip" data-tip="(Retorno − 5%) / Volatilidad. Mide el retorno por unidad de riesgo. Sharpe > 1 se considera bueno; > 2 es excelente.">Ratio de Sharpe</span>
      </div>
      <div class="metric-value ${shrpCol}">${fmtN(port.sharpe)}</div>
    </div>`;
}

// ─── Barra de resumen superior ────────────────────────────────
function renderSummary(maxS, minV, n) {
  document.getElementById('summary-bar').innerHTML = `
    <div class="sum-card">
      <div class="sum-label">Máximo Ratio de Sharpe</div>
      <div class="sum-value c-green">${fmtN(maxS.sharpe)}</div>
      <div class="sum-sub">${fmtPct(maxS.ret)} retorno · ${fmtPct(maxS.vol)} vol</div>
    </div>
    <div class="sum-card">
      <div class="sum-label">Mínima Volatilidad</div>
      <div class="sum-value c-white">${fmtPct(minV.vol)}</div>
      <div class="sum-sub">${fmtPct(minV.ret)} retorno · Sharpe ${fmtN(minV.sharpe)}</div>
    </div>
    <div class="sum-card">
      <div class="sum-label">Portafolios Simulados</div>
      <div class="sum-value c-gold">${n.toLocaleString('es-AR')}</div>
      <div class="sum-sub">Simulación Monte Carlo</div>
    </div>
    <div class="sum-card">
      <div class="sum-label">Activos Analizados</div>
      <div class="sum-value c-white">${tickers.length}</div>
      <div class="sum-sub">${tickers.join(' · ')}</div>
    </div>`;
}

// ─── Tabla de estadísticas ────────────────────────────────────
let tableData = [], sortCol = 'annReturn', sortAsc = false;

function renderTable(rows) { tableData = rows; sortRender(); }

function sortRender() {
  const sorted = [...tableData].sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    // Nulos siempre al final
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return sortAsc ? va - vb : vb - va;
  });
  document.getElementById('stats-tbody').innerHTML = sorted.map(s => `
    <tr>
      <td><div class="cell-ticker"><span class="tick-dot" style="background:${s.color}"></span>${s.ticker}</div></td>
      <td class="cell-num ${s.annReturn >= 0 ? 'c-green' : 'c-red'}">${fmtPct(s.annReturn)}</td>
      <td class="cell-num c-white">${fmtPct(s.annVol)}</td>
      <td class="cell-num ${s.sharpe >= 0 ? 'c-green' : 'c-red'}">${fmtN(s.sharpe)}</td>
      <td class="cell-num c-white">${s.beta != null ? fmtN(s.beta) : '<span style="opacity:.35">—</span>'}</td>
    </tr>`).join('');

  document.querySelectorAll('#stats-table th.sortable').forEach(th => {
    const col = th.dataset.col;
    th.classList.toggle('sort-active', col === sortCol);
    th.querySelector('.sort-arrow').textContent = col === sortCol ? (sortAsc ? '↑' : '↓') : '↕';
  });
}

function initTableSort() {
  document.querySelectorAll('#stats-table th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (col === sortCol) { sortAsc = !sortAsc; }
      else { sortCol = col; sortAsc = th.dataset.type === 'str'; }
      sortRender();
    });
  });
}

// ─── Pestañas ─────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`tab-${tab}`).classList.add('active');
    });
  });
}

// ─── Flujo principal de análisis ──────────────────────────────
async function analyze() {
  if (tickers.length < 2) { showInputErr('Agregá al menos 2 tickers.'); return; }

  showLoading();

  // 1 — Descarga de datos
  const raw    = {};
  const failed = [];
  setStatus('Obteniendo datos de Yahoo Finance…');

  await Promise.all(tickers.map(async t => {
    setTS(t, 'loading', 'descargando…');
    try {
      raw[t] = await fetchPrices(t);
      setTS(t, 'success', `${raw[t].prices.length} días`);
    } catch (e) {
      setTS(t, 'error', 'error');
      failed.push(t);
    }
  }));

  if (failed.length) {
    if (Object.keys(raw).length < 2) {
      hideLoading();
      showInputErr(`No se pudieron obtener datos de: ${failed.join(', ')}. Verificá la conexión o probá otros tickers.`);
      return;
    }
    failed.forEach(t => delete raw[t]);
    tickers = tickers.filter(t => !failed.includes(t));
    renderChips();
    showInputErr(`Se omitió ${failed.join(', ')} (error de descarga). Continuando con los demás.`);
  }

  // 2 — Descarga silenciosa de SPY como benchmark de mercado (S&P 500)
  setStatus('Descargando benchmark de mercado (SPY)…');
  let spyRaw = null;
  try { spyRaw = await fetchPrices('SPY'); } catch (_) { /* fallback a portafolio */ }

  // 3 — Alineación de series temporales (solo tickers del usuario)
  setStatus('Alineando series temporales…');
  const aligned = alignSeries(raw);

  // 4 — Retornos logarítmicos diarios
  setStatus('Calculando retornos logarítmicos…');
  const retMap = {};
  for (const t of tickers) retMap[t] = logReturns(aligned[t]);

  // 5 — Retornos medios anualizados y matriz de covarianza
  // exp(μ_log × 252) − 1  →  retorno anual compuesto simple (siempre > −100%)
  // NUNCA usar μ_log × 252 directamente: eso da retornos de capitalización continua
  // que pueden ser −164% (matemáticamente válido pero imposible como retorno simple).
  setStatus('Construyendo matriz de covarianza…');
  const mu = tickers.map(t => Math.exp(mean(retMap[t]) * DAYS) - 1);
  const C  = buildCov(retMap, tickers);
  const R  = buildCorr(C);
  const sd = tickers.map((_, i) => Math.sqrt(C[i][i]));

  // 6 — Beta vs SPY (alineación pairwise: cada activo vs SPY de forma independiente)
  //     Si SPY no está disponible, bt[i] = null → se muestra "—" en la tabla
  const bt = computeBetas(raw, tickers, spyRaw);

  // 7 — Simulación Monte Carlo
  setStatus(`Ejecutando Monte Carlo (${N_SIM.toLocaleString('es-AR')} portafolios)…`);
  const portfolios = monteCarlo(mu, C, N_SIM);

  const maxS = portfolios.reduce((b, p) => p.sharpe > b.sharpe ? p : b);
  const minV = portfolios.reduce((b, p) => p.vol   < b.vol   ? p : b);

  // Portafolio de pesos iguales (benchmark)
  const eqW  = tickers.map(() => 1 / tickers.length);
  const eqP  = { w: eqW, ret: portRet(eqW, mu), vol: Math.sqrt(portVar(eqW, C)) };
  eqP.sharpe = (eqP.ret - RF) / eqP.vol;

  results = { portfolios, maxS, minV, eqP, R, mu, C, sd, bt };

  // 8 — Renderizado
  setStatus('Generando gráficos…');
  hideLoading();
  document.getElementById('results').classList.remove('hidden');

  renderSummary(maxS, minV, N_SIM);
  renderFrontier(portfolios, maxS, minV);

  renderMetrics(maxS, 'metrics-max-sharpe');
  renderMetrics(minV, 'metrics-min-var');
  renderMetrics(eqP,  'metrics-equal');

  renderWeightsBar('wc-max-sharpe', maxS.w, tickers);
  renderWeightsBar('wc-min-var',    minV.w, tickers);
  renderWeightsBar('wc-equal',      eqP.w,  tickers);

  renderCorr(R, tickers);

  renderTable(tickers.map((t, i) => ({
    ticker:    t,
    annReturn: mu[i],
    annVol:    sd[i],
    sharpe:    (mu[i] - RF) / sd[i],
    beta:      bt[i],
    color:     COLORS[i % COLORS.length]
  })));
}

// ─── Inicialización ───────────────────────────────────────────
function init() {
  renderChips();
  initTabs();
  initTableSort();

  const input  = document.getElementById('ticker-input');
  const addBtn = document.getElementById('add-ticker-btn');
  const runBtn = document.getElementById('analyze-btn');

  addBtn.addEventListener('click', () => {
    const v = input.value.trim();
    if (v) { addTicker(v); input.value = ''; }
  });

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      const v = input.value.replace(/,\s*$/, '').trim();
      if (v) { addTicker(v); input.value = ''; }
    }
  });

  runBtn.addEventListener('click', analyze);
}

document.addEventListener('DOMContentLoaded', init);
