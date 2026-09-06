const $ = (selector) => document.querySelector(selector);
const svgNS = 'http://www.w3.org/2000/svg';

const state = {
  running: false,
  tick: 0,
  simSeconds: 6 * 3600 + 42 * 60,
  speed: 1,
  selectedTrain: null,
  activePreset: 'clean',
  forecast: null,
  scenario: {
    currentDelay: 18,
    trendPerEvent: 4,
    packetLoss: 0,
    staleMinutes: 0,
    outageMinutes: 0,
    inconsistentRate: 0,
    duplicateRate: 0,
  },
};

const presets = {
  clean: { label: 'Clean feed', currentDelay: 18, trendPerEvent: 4, packetLoss: 0, staleMinutes: 0, outageMinutes: 0, inconsistentRate: 0, duplicateRate: 0 },
  loss: { label: 'Packet loss · 15%', currentDelay: 46, trendPerEvent: 8, packetLoss: 15, staleMinutes: 0, outageMinutes: 0, inconsistentRate: 0, duplicateRate: 0 },
  stale: { label: 'Stale feed · 5 min', currentDelay: 74, trendPerEvent: 12, packetLoss: 0, staleMinutes: 5, outageMinutes: 0, inconsistentRate: 0, duplicateRate: 0 },
  outage: { label: 'Outage · 15 min', currentDelay: 142, trendPerEvent: 28, packetLoss: 0, staleMinutes: 0, outageMinutes: 15, inconsistentRate: 0, duplicateRate: 0 },
  combined: { label: 'Combined stress', currentDelay: 185, trendPerEvent: 36, packetLoss: 15, staleMinutes: 5, outageMinutes: 10, inconsistentRate: 8, duplicateRate: 4 },
};

const sliderMap = {
  currentDelay: ['current-delay', (value) => `${value} s`],
  trendPerEvent: ['trend', (value) => `${Number(value) >= 0 ? '+' : ''}${value} s`],
  packetLoss: ['packet-loss', (value) => `${value}%`],
  staleMinutes: ['stale', (value) => `${value} min`],
  outageMinutes: ['outage', (value) => `${value} min`],
  inconsistentRate: ['inconsistent', (value) => `${value}%`],
  duplicateRate: ['duplicate', (value) => `${value}%`],
};

const trains = [
  { id: 'IC 204', element: 'train-1', className: 'train-blue', phase: .08, speed: .00027, route: [[58, 112], [326, 72], [570, 136], [942, 76]], destination: 'East Gate', service: 'Intercity' },
  { id: 'R 118', element: 'train-2', className: 'train-amber', phase: .52, speed: .0002, route: [[58, 334], [342, 380], [590, 294], [942, 346]], destination: 'Lakeside', service: 'Regional' },
  { id: 'S 052', element: 'train-3', className: 'train-violet', phase: .3, speed: .00024, route: [[326, 72], [342, 380], [590, 294]], destination: 'Market', service: 'Shuttle' },
];

function svgElement(tag, attributes = {}) {
  const node = document.createElementNS(svgNS, tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function pointOnRoute(route, progress) {
  const lengths = route.slice(1).map((point, index) => {
    const previous = route[index];
    return Math.hypot(point[0] - previous[0], point[1] - previous[1]);
  });
  const total = lengths.reduce((sum, value) => sum + value, 0);
  let distance = progress * total;
  for (let index = 0; index < lengths.length; index += 1) {
    if (distance <= lengths[index]) {
      const start = route[index];
      const end = route[index + 1];
      const ratio = distance / lengths[index];
      return [start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio];
    }
    distance -= lengths[index];
  }
  return route[route.length - 1];
}

function buildTrains() {
  const layer = $('#train-layer');
  trains.forEach((train) => {
    const group = svgElement('g', { id: train.element, class: `train ${train.className}`, tabindex: '0', role: 'button', 'aria-label': `${train.id} ${train.service} train` });
    group.appendChild(svgElement('circle', { r: 9 }));
    const label = svgElement('text', { x: 14, y: -13 });
    label.textContent = train.id;
    group.appendChild(label);
    group.addEventListener('click', () => selectTrain(train));
    group.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') selectTrain(train); });
    layer.appendChild(group);
  });
}

function selectTrain(train) {
  state.selectedTrain = train;
  $('#selected-train').textContent = `${train.id} · ${train.service} → ${train.destination}`;
  addLog(`${train.id} selected for inspection`, 'info');
  document.querySelectorAll('.train').forEach((element) => element.classList.remove('selected'));
  $(`#${train.element}`).classList.add('selected');
}

function updateTrains() {
  const delayFactor = 1 - Math.min(0.38, Math.max(0, state.scenario.currentDelay / 1000));
  trains.forEach((train) => {
    const progress = (train.phase + state.tick * train.speed * state.speed * delayFactor) % 1;
    const [x, y] = pointOnRoute(train.route, progress);
    const element = $(`#${train.element}`);
    if (element) element.setAttribute('transform', `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
  });
}

function formatClock(seconds) {
  const whole = Math.floor(seconds) % 86400;
  const hours = Math.floor(whole / 3600).toString().padStart(2, '0');
  const minutes = Math.floor((whole % 3600) / 60).toString().padStart(2, '0');
  const secs = (whole % 60).toString().padStart(2, '0');
  return `${hours}:${minutes}:${secs}`;
}

function updateClock() {
  $('#sim-clock').textContent = formatClock(state.simSeconds);
  state.simSeconds += state.running ? state.speed : 0;
  state.tick += state.running ? .7 : 0;
  updateTrains();
}

function setSliderValues() {
  Object.entries(sliderMap).forEach(([key, [id, formatter]]) => {
    const input = $(`#${id}`);
    const output = $(`#${id}-value`);
    input.value = state.scenario[key];
    output.textContent = formatter(state.scenario[key]);
  });
}

function scenarioPayload() {
  return { scenario: { ...state.scenario } };
}

function markCustomScenario() {
  state.activePreset = null;
  document.querySelectorAll('.preset').forEach((button) => button.classList.remove('active'));
  $('#active-scenario').textContent = 'Custom scenario';
}

function applyPreset(name, shouldLog = true) {
  const preset = presets[name];
  if (!preset) return;
  state.activePreset = name;
  state.scenario = { ...preset };
  setSliderValues();
  document.querySelectorAll('.preset').forEach((button) => button.classList.toggle('active', button.dataset.preset === name));
  $('#active-scenario').textContent = preset.label;
  if (shouldLog) addLog(`${preset.label} loaded`, name === 'clean' ? 'info' : 'warning');
  requestForecast();
}

function localFallback() {
  const { currentDelay, trendPerEvent: trend, packetLoss, staleMinutes, outageMinutes } = state.scenario;
  const width = 18 + Math.abs(trend) * .22 + packetLoss * 2.3 + staleMinutes * 3.2 + outageMinutes * 3.8;
  const quantiles = Array.from({ length: 4 }, (_, index) => {
    const median = currentDelay + trend * (index + 1);
    const span = width * (1 + index * .08);
    return [median - span * 1.95, median - span * 1.55, median - span * .82, median, median + span * .82, median + span * 1.55, median + span * 1.95];
  });
  const qualityScore = Math.max(0, 100 - packetLoss * .28 - staleMinutes * 2.4 - outageMinutes * 1.6);
  return {
    mode: 'browser fallback', modelLabel: 'Persistence + uncertainty simulator', modelMessage: 'The API is unavailable; browser-local simulation is active.',
    quality: { missing_fraction: packetLoss / 100, stale_fraction: staleMinutes / 10, observation_age_saturating: Math.max(staleMinutes / 10, outageMinutes / 15), declared_delay_saturating: Math.min(1, currentDelay / 900), inconsistent_flag: state.scenario.inconsistentRate / 100, duplicate_flag: state.scenario.duplicateRate / 100, no_fresh_observation: outageMinutes >= 9 ? 1 : 0 },
    qualityScore, quantiles, medians: quantiles.map((row) => row[3]), spreads: quantiles.map((row) => row[6] - row[0]), latencyMs: .2, alert: qualityScore < 65, alertText: qualityScore < 65 ? 'Feed quality is degraded; treat the forecast as advisory.' : 'No immediate reliability alert.', router: { expert: null, probabilities: [] },
  };
}

let requestTimer;
function requestForecast() {
  clearTimeout(requestTimer);
  requestTimer = setTimeout(async () => {
    try {
      const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(scenarioPayload()) });
      if (!response.ok) throw new Error(`API ${response.status}`);
      state.forecast = await response.json();
    } catch (error) {
      state.forecast = localFallback();
    }
    renderForecastState();
  }, 90);
}

function renderForecastState() {
  const result = state.forecast;
  if (!result) return;
  const score = Number(result.qualityScore || 0);
  $('#mode-pill').innerHTML = `<span class="status-dot ${result.mode === 'R3S-MoE checkpoint' ? 'green' : ''}"></span> ${result.mode}`;
  $('#forecast-model').textContent = result.modelLabel;
  $('#confidence-value').innerHTML = `${score.toFixed(1)}<span>%</span>`;
  $('#confidence-caption').textContent = score >= 80 ? 'High · feed is fresh' : score >= 65 ? 'Moderate · monitor feed' : 'Low · advisory only';
  $('#median-value').innerHTML = `${Math.round(result.medians[0])}<span>s</span>`;
  $('#median-caption').textContent = `${state.scenario.trendPerEvent >= 0 ? '+' : ''}${state.scenario.trendPerEvent} s trend / event`;
  $('#spread-value').innerHTML = `${Math.round(result.spreads[0])}<span>s</span>`;
  $('#spread-caption').textContent = '90% forecast interval';
  $('#signal-value').textContent = result.alert ? 'Attention' : 'Nominal';
  $('#signal-caption').textContent = result.alertText;
  $('#signal-indicator').classList.toggle('alert', result.alert);
  $('#map-health').textContent = `${Math.round(score)}%`;
  $('#map-health-bar').style.width = `${score}%`;
  $('#map-health-bar').style.background = score < 65 ? 'var(--coral)' : score < 80 ? 'var(--amber)' : 'var(--green)';
  renderChart(result.quantiles);
  renderQuality(result.quality, score);
  addLog(result.alert ? 'Reliability alert: forecast uncertainty widened' : 'Forecast refreshed from current feed state', result.alert ? 'alert' : 'info');
}

function renderChart(quantiles) {
  const chart = $('#forecast-chart');
  chart.replaceChildren();
  const width = 720; const height = 270; const left = 48; const right = 18; const top = 18; const bottom = 38;
  const all = quantiles.flat();
  const min = Math.min(0, Math.floor(Math.min(...all) / 50) * 50);
  const max = Math.max(100, Math.ceil(Math.max(...all) / 50) * 50);
  const x = (index) => left + index * ((width - left - right) / 3);
  const y = (value) => top + (max - value) * ((height - top - bottom) / (max - min));
  const createPath = (values) => values.map((value, index) => `${index ? 'L' : 'M'} ${x(index)} ${y(value)}`).join(' ');
  for (let value = min; value <= max; value += Math.max(50, Math.round((max - min) / 4 / 10) * 10)) {
    chart.appendChild(svgElement('line', { x1: left, x2: width - right, y1: y(value), y2: y(value), class: 'chart-grid' }));
    const label = svgElement('text', { x: 8, y: y(value) + 4, class: 'chart-label' }); label.textContent = `${Math.round(value)}s`; chart.appendChild(label);
  }
  const low = quantiles.map((row) => `${x(quantiles.indexOf(row))},${y(row[0])}`).join(' ');
  const high = [...quantiles].reverse().map((row, reverseIndex) => `${x(quantiles.length - 1 - reverseIndex)},${y(row[6])}`).join(' ');
  chart.appendChild(svgElement('polygon', { points: `${low} ${high}`, class: 'chart-area' }));
  const innerLow = quantiles.map((row, index) => `${x(index)},${y(row[2])}`).join(' ');
  const innerHigh = [...quantiles].reverse().map((row, reverseIndex) => `${x(quantiles.length - 1 - reverseIndex)},${y(row[4])}`).join(' ');
  chart.appendChild(svgElement('polygon', { points: `${innerLow} ${innerHigh}`, class: 'chart-inner-area' }));
  chart.appendChild(svgElement('path', { d: createPath(quantiles.map((row) => row[3])), class: 'chart-line' }));
  quantiles.forEach((row, index) => {
    chart.appendChild(svgElement('circle', { cx: x(index), cy: y(row[3]), r: 5, class: 'chart-point' }));
    const label = svgElement('text', { x: x(index), y: height - 12, class: 'chart-label', 'text-anchor': 'middle' }); label.textContent = `H+${index + 1}`; chart.appendChild(label);
  });
}

function renderQuality(quality, score) {
  const labels = [['missing_fraction', 'Missing observations'], ['stale_fraction', 'Stale observations'], ['observation_age_saturating', 'Observation age'], ['declared_delay_saturating', 'Declared delay'], ['inconsistent_flag', 'Inconsistent updates'], ['duplicate_flag', 'Duplicate updates'], ['no_fresh_observation', 'No fresh observation']];
  const list = $('#quality-list');
  list.replaceChildren();
  labels.forEach(([key, label]) => {
    const value = Number(quality[key] || 0);
    const row = document.createElement('div'); row.className = 'quality-row';
    const name = document.createElement('span'); name.className = 'quality-name'; name.textContent = label;
    const track = document.createElement('span'); track.className = 'quality-track';
    const fill = document.createElement('i'); fill.className = 'quality-fill'; fill.style.width = `${Math.round(value * 100)}%`; fill.style.background = value > .55 ? 'var(--coral)' : value > .2 ? 'var(--amber)' : 'var(--green)'; track.appendChild(fill);
    const output = document.createElement('span'); output.className = 'quality-value'; output.textContent = `${Math.round(value * 100)}%`;
    row.append(name, track, output); list.appendChild(row);
  });
  $('#quality-score').textContent = `${Math.round(score)} / 100`;
  $('#quality-summary').textContent = score >= 80 ? 'Fresh observations are available across the corridor.' : score >= 65 ? 'Some feed imperfections are present; uncertainty is being widened.' : 'Multiple feed failures detected. Use the forecast for triage, not automatic action.';
}

function addLog(message, severity = 'info') {
  const log = $('#event-log');
  const item = document.createElement('div'); item.className = 'log-item';
  const time = document.createElement('span'); time.className = 'log-time'; time.textContent = formatClock(state.simSeconds);
  const dot = document.createElement('i'); dot.className = `log-dot ${severity === 'info' ? '' : severity}`;
  const text = document.createElement('span'); text.textContent = message;
  item.append(time, dot, text); log.prepend(item);
  while (log.children.length > 5) log.lastElementChild.remove();
  $('#last-event').textContent = message;
}

function bindControls() {
  document.querySelectorAll('.preset').forEach((button) => button.addEventListener('click', () => applyPreset(button.dataset.preset)));
  Object.entries(sliderMap).forEach(([key, [id, formatter]]) => {
    $(`#${id}`).addEventListener('input', (event) => {
      state.scenario[key] = Number(event.target.value);
      $(`#${id}-value`).textContent = formatter(state.scenario[key]);
      markCustomScenario();
      requestForecast();
    });
  });
  $('#play-button').addEventListener('click', () => {
    state.running = !state.running;
    $('#play-button').innerHTML = state.running ? '<span id="play-icon">Ⅱ</span> Pause simulation' : '<span id="play-icon">▶</span> Run simulation';
    addLog(state.running ? 'Simulation clock started' : 'Simulation clock paused', 'info');
  });
  $('#step-button').addEventListener('click', () => { state.simSeconds += 30; state.tick += 5; updateTrains(); addLog('Advanced simulation by 30 seconds', 'info'); });
  $('#reset-button').addEventListener('click', () => { state.running = false; state.tick = 0; state.simSeconds = 6 * 3600 + 42 * 60; applyPreset('clean'); addLog('Simulation reset to baseline', 'info'); });
  $('#clear-log').addEventListener('click', () => { $('#event-log').replaceChildren(); addLog('Event stream cleared', 'info'); });
}

buildTrains();
bindControls();
setSliderValues();
applyPreset('clean', false);
addLog('Simulation initialized with clean feed', 'info');
setInterval(updateClock, 1000 / 12);
