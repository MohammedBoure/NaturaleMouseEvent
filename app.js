// App State
let globalStats = null;
let episodesList = [];
let currentEpisode = null;
let renderMode = 'heatmap'; // 'heatmap', 'events', 'vectors'
let showMonitors = true;

// Playback State
let isPlaying = false;
let playbackTimeMs = 0;
let playbackSpeed = 1.0;
let lastAnimTimestamp = 0;
let animFrameId = null;

// Canvas & View State
let canvas, ctx;
let velocityCanvas, velocityCtx;
let scale = 0.35;
let offsetX = 100;
let offsetY = 100;
let isDragging = false;
let dragStartX = 0, dragStartY = 0;

// Monitors Definition (from Session metadata)
const defaultMonitors = [
  { index: 0, left: 0, top: 0, right: 1920, bottom: 1080, width: 1920, height: 1080, name: "Primary Monitor (1920x1080)" },
  { index: 1, left: 1920, top: 0, right: 3520, bottom: 900, width: 1600, height: 900, name: "Secondary Monitor (1600x900)" }
];

document.addEventListener('DOMContentLoaded', () => {
  initCanvas();
  initVelocityCanvas();
  loadStats();
  applyFilters();
  window.addEventListener('resize', handleResize);
});

function initCanvas() {
  const container = document.getElementById('canvasContainer');
  canvas = document.getElementById('trajectoryCanvas');
  ctx = canvas.getContext('2d');
  
  handleResize();

  // Mouse pan and zoom
  canvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    dragStartX = e.clientX - offsetX;
    dragStartY = e.clientY - offsetY;
  });

  window.addEventListener('mousemove', (e) => {
    if (isDragging) {
      offsetX = e.clientX - dragStartX;
      offsetY = e.clientY - dragStartY;
      renderTrajectory();
    }
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
    scale *= zoomFactor;
    scale = Math.max(0.05, Math.min(scale, 3.0));
    renderTrajectory();
  });
}

function initVelocityCanvas() {
  velocityCanvas = document.getElementById('velocityChart');
  velocityCtx = velocityCanvas.getContext('2d');
  resizeVelocityCanvas();
}

function resizeVelocityCanvas() {
  if (!velocityCanvas) return;
  const rect = velocityCanvas.getBoundingClientRect();
  velocityCanvas.width = rect.width * window.devicePixelRatio;
  velocityCanvas.height = rect.height * window.devicePixelRatio;
  velocityCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
  renderVelocityChart();
}

function handleResize() {
  if (!canvas) return;
  const container = document.getElementById('canvasContainer');
  canvas.width = container.clientWidth * window.devicePixelRatio;
  canvas.height = container.clientHeight * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  resizeVelocityCanvas();
  renderTrajectory();
}

async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    globalStats = await res.json();
    
    document.getElementById('globalScoreText').innerText = `${globalStats.readiness_score}%`;
    
    const sessSelect = document.getElementById('sessionFilter');
    sessSelect.innerHTML = '<option value="">جميع الجلسات (All Sessions)</option>';
    globalStats.sessions.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.session_id;
      opt.innerText = `جلسة: ${s.file_date} (${s.session_id.substring(0, 8)})`;
      sessSelect.appendChild(opt);
    });
  } catch (err) {
    console.error("Error loading stats:", err);
  }
}

async function applyFilters() {
  const session_id = document.getElementById('sessionFilter').value;
  const quality = document.getElementById('qualityFilter').value;
  const minEvents = document.getElementById('minEventsFilter').value || 5;

  const url = `/api/episodes?limit=100&quality=${quality}&min_events=${minEvents}${session_id ? '&session_id=' + session_id : ''}`;

  try {
    const res = await fetch(url);
    const data = await res.json();
    episodesList = data.episodes;
    document.getElementById('totalEpCount').innerText = `${data.total_count} محاولة`;

    renderEpisodeList();

    if (episodesList.length > 0) {
      selectEpisode(episodesList[0].session_id, episodesList[0].episode_index);
    }
  } catch (err) {
    console.error("Error applying filters:", err);
  }
}

function renderEpisodeList() {
  const listEl = document.getElementById('episodeList');
  listEl.innerHTML = '';

  episodesList.forEach(ep => {
    const card = document.createElement('div');
    card.className = `episode-card ${currentEpisode && currentEpisode.episode_index === ep.episode_index ? 'active' : ''}`;
    card.onclick = () => selectEpisode(ep.session_id, ep.episode_index);

    const badgeClass = ep.quality_grade === 'GOLD' ? 'gold' : (ep.quality_grade === 'TELEPORTATION' ? 'teleport' : 'micro');

    card.innerHTML = `
      <div class="ep-header">
        <span class="ep-title">Episode #${ep.episode_index}</span>
        <span class="ep-badge ${badgeClass}">${ep.quality_grade}</span>
      </div>
      <div class="ep-metrics">
        <span>⏱️ ${ep.duration_ms} ms</span>
        <span>📍 ${ep.event_count} نقطة</span>
        ${ep.click_count > 0 ? `<span>🖱️ ${ep.click_count} نقدة</span>` : ''}
      </div>
    `;
    listEl.appendChild(card);
  });
}

async function selectEpisode(sessionId, epIndex) {
  try {
    const res = await fetch(`/api/episode?session_id=${sessionId}&episode_index=${epIndex}`);
    currentEpisode = await res.json();

    // Update active highlight
    renderEpisodeList();

    // Update metrics cards
    const m = currentEpisode.metrics;
    document.getElementById('metricEfficiency').innerText = `${(m.efficiency_ratio * 100).toFixed(1)}%`;
    document.getElementById('metricSpeed').innerHTML = `${m.max_speed_px_s} <span style="font-size: 14px;">px/s</span>`;
    document.getElementById('metricAvgSpeed').innerText = `المتوسط: ${m.avg_speed_px_s} px/s`;
    
    // Polling Hz calculation
    const avgDt = m.duration_ms / (m.event_count || 1);
    const hz = avgDt > 0 ? (1000 / avgDt).toFixed(1) : 0;
    document.getElementById('metricPolling').innerHTML = `${hz} <span style="font-size: 14px;">Hz</span>`;
    document.getElementById('metricDt').innerText = `متوسط الزمن بين النقاط: ${avgDt.toFixed(2)} ms`;

    document.getElementById('episodeIdLabel').innerText = `Episode #${currentEpisode.episode_index} (${m.event_count} events, ${m.duration_ms}ms)`;

    // Playback reset
    pause();
    playbackTimeMs = m.duration_ms;
    document.getElementById('scrubber').max = m.duration_ms;
    document.getElementById('scrubber').value = m.duration_ms;
    updateTimeText();

    // Reset view bounds to auto-fit trajectory
    fitTrajectoryToScreen();
    renderTrajectory();
    renderVelocityChart();
  } catch (err) {
    console.error("Error selecting episode:", err);
  }
}

function fitTrajectoryToScreen() {
  if (!currentEpisode || !currentEpisode.events || currentEpisode.events.length === 0) return;
  const container = document.getElementById('canvasContainer');
  const w = container.clientWidth;
  const h = container.clientHeight;

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  currentEpisode.events.forEach(e => {
    minX = Math.min(minX, e.x);
    maxX = Math.max(maxX, e.x);
    minY = Math.min(minY, e.y);
    maxY = Math.max(maxY, e.y);
  });

  const rangeX = (maxX - minX) || 500;
  const rangeY = (maxY - minY) || 500;

  const scaleX = (w * 0.7) / rangeX;
  const scaleY = (h * 0.7) / rangeY;
  scale = Math.min(scaleX, scaleY, 0.8);
  scale = Math.max(scale, 0.15);

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  offsetX = (w / 2) - (centerX * scale);
  offsetY = (h / 2) - (centerY * scale);
}

function resetZoom() {
  fitTrajectoryToScreen();
  renderTrajectory();
}

function toggleMonitors() {
  showMonitors = !showMonitors;
  document.getElementById('toggleMonitorsBtn').classList.toggle('active', showMonitors);
  renderTrajectory();
}

function setRenderMode(mode) {
  renderMode = mode;
  ['modeHeatmapBtn', 'modeEventsBtn', 'modeVectorsBtn'].forEach(id => {
    document.getElementById(id).classList.remove('active');
  });
  if (mode === 'heatmap') document.getElementById('modeHeatmapBtn').classList.add('active');
  if (mode === 'events') document.getElementById('modeEventsBtn').classList.add('active');
  if (mode === 'vectors') document.getElementById('modeVectorsBtn').classList.add('active');

  renderTrajectory();
}

// ----------------------------------------------------
// CANVAS RENDERING ENGINE
// ----------------------------------------------------
function renderTrajectory() {
  if (!ctx || !canvas) return;

  const container = document.getElementById('canvasContainer');
  const w = container.clientWidth;
  const h = container.clientHeight;

  ctx.clearRect(0, 0, w, h);

  // Background Grid
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 1;
  const gridSize = 50 * scale;
  const startX = offsetX % gridSize;
  const startY = offsetY % gridSize;

  ctx.beginPath();
  for (let x = startX; x < w; x += gridSize) {
    ctx.moveTo(x, 0); ctx.lineTo(x, h);
  }
  for (let y = startY; y < h; y += gridSize) {
    ctx.moveTo(0, y); ctx.lineTo(w, y);
  }
  ctx.stroke();

  // Render Monitor Bounds
  if (showMonitors) {
    defaultMonitors.forEach(mon => {
      const mx = mon.left * scale + offsetX;
      const my = mon.top * scale + offsetY;
      const mw = mon.width * scale;
      const mh = mon.height * scale;

      ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
      ctx.lineWidth = 2;
      ctx.fillStyle = 'rgba(99, 102, 241, 0.04)';
      ctx.fillRect(mx, my, mw, mh);
      ctx.strokeRect(mx, my, mw, mh);

      ctx.fillStyle = 'rgba(156, 163, 175, 0.6)';
      ctx.font = '12px Inter, sans-serif';
      ctx.fillText(mon.name, mx + 12, my + 24);
    });
  }

  if (!currentEpisode || !currentEpisode.events || currentEpisode.events.length === 0) return;

  const events = currentEpisode.events;
  const physics = currentEpisode.physics;
  const t0 = events[0].t_monotonic_ns;

  // Filter visible events up to playbackTimeMs
  const activeEvents = [];
  for (let i = 0; i < events.length; i++) {
    const elapsed = (events[i].t_monotonic_ns - t0) / 1e6;
    if (elapsed <= playbackTimeMs) {
      activeEvents.push({ ...events[i], physics: physics[i], elapsed });
    } else {
      break;
    }
  }

  if (activeEvents.length < 1) return;

  // Find max speed for heatmap scaling
  let maxSpeed = 1;
  physics.forEach(p => { if (p.speed_px_s > maxSpeed) maxSpeed = p.speed_px_s; });

  // Draw Path Segments
  for (let i = 1; i < activeEvents.length; i++) {
    const p1 = activeEvents[i - 1];
    const p2 = activeEvents[i];

    const x1 = p1.x * scale + offsetX;
    const y1 = p1.y * scale + offsetY;
    const x2 = p2.x * scale + offsetX;
    const y2 = p2.y * scale + offsetY;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);

    if (renderMode === 'heatmap') {
      const speed = p2.physics ? p2.physics.speed_px_s : 0;
      const ratio = Math.min(speed / (maxSpeed * 0.7), 1.0);
      // Cold blue to Hot Red gradient
      const r = Math.floor(ratio * 255);
      const g = Math.floor((1 - Math.abs(ratio - 0.5) * 2) * 200);
      const b = Math.floor((1 - ratio) * 255);
      ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.lineWidth = 3;
    } else if (renderMode === 'events') {
      ctx.strokeStyle = p2.event_type === 'click' ? '#10b981' : (p2.event_type === 'scroll' ? '#f59e0b' : '#06b6d4');
      ctx.lineWidth = 2.5;
    } else { // vectors
      ctx.strokeStyle = '#818cf8';
      ctx.lineWidth = 2;
    }
    ctx.stroke();

    // Draw Vector Direction Arrows
    if (renderMode === 'vectors' && i % 3 === 0) {
      const angle = Math.atan2(y2 - y1, x2 - x1);
      const arrowLen = 6;
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - arrowLen * Math.cos(angle - Math.PI / 6), y2 - arrowLen * Math.sin(angle - Math.PI / 6));
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - arrowLen * Math.cos(angle + Math.PI / 6), y2 - arrowLen * Math.sin(angle + Math.PI / 6));
      ctx.stroke();
    }

    // Draw Clicks / Special Event Markers
    if (p2.event_type === 'click' || p2.event_type === 'button') {
      ctx.beginPath();
      ctx.arc(x2, y2, 7, 0, Math.PI * 2);
      ctx.fillStyle = p2.pressed ? '#10b981' : '#f43f5e';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // Draw Current Cursor Position (Glowing indicator)
  const lastPt = activeEvents[activeEvents.length - 1];
  const curX = lastPt.x * scale + offsetX;
  const curY = lastPt.y * scale + offsetY;

  ctx.beginPath();
  ctx.arc(curX, curY, 8, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(99, 102, 241, 0.4)';
  ctx.fill();

  ctx.beginPath();
  ctx.arc(curX, curY, 4, 0, Math.PI * 2);
  ctx.fillStyle = '#6366f1';
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();
}

// ----------------------------------------------------
// VELOCITY CHART RENDERER v(t)
// ----------------------------------------------------
function renderVelocityChart() {
  if (!velocityCtx || !velocityCanvas || !currentEpisode) return;

  const w = velocityCanvas.width / window.devicePixelRatio;
  const h = velocityCanvas.height / window.devicePixelRatio;

  velocityCtx.clearRect(0, 0, w, h);

  const events = currentEpisode.events;
  const physics = currentEpisode.physics;
  if (!events || events.length < 2) return;

  const t0 = events[0].t_monotonic_ns;
  const totalDuration = currentEpisode.metrics.duration_ms || 1;

  let maxSpeed = 100;
  physics.forEach(p => { if (p.speed_px_s > maxSpeed) maxSpeed = p.speed_px_s; });

  const padding = { left: 40, right: 20, top: 15, bottom: 25 };
  const graphW = w - padding.left - padding.right;
  const graphH = h - padding.top - padding.bottom;

  // Gridlines
  velocityCtx.strokeStyle = '#1f2937';
  velocityCtx.lineWidth = 1;
  velocityCtx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (graphH / 4) * i;
    velocityCtx.moveTo(padding.left, y);
    velocityCtx.lineTo(w - padding.right, y);
  }
  velocityCtx.stroke();

  // Y-axis labels
  velocityCtx.fillStyle = '#6b7280';
  velocityCtx.font = '10px Inter, sans-serif';
  velocityCtx.fillText(`${Math.round(maxSpeed)} px/s`, 5, padding.top + 10);
  velocityCtx.fillText('0 px/s', 5, h - padding.bottom);

  // Plot Line v(t)
  velocityCtx.beginPath();
  physics.forEach((p, i) => {
    const elapsed = (events[i].t_monotonic_ns - t0) / 1e6;
    const x = padding.left + (elapsed / totalDuration) * graphW;
    const y = h - padding.bottom - (p.speed_px_s / maxSpeed) * graphH;

    if (i === 0) velocityCtx.moveTo(x, y);
    else velocityCtx.lineTo(x, y);
  });

  velocityCtx.strokeStyle = '#06b6d4';
  velocityCtx.lineWidth = 2;
  velocityCtx.stroke();

  // Scrubber Cursor Line in Chart
  const scrubberX = padding.left + (playbackTimeMs / totalDuration) * graphW;
  velocityCtx.strokeStyle = '#f43f5e';
  velocityCtx.lineWidth = 2;
  velocityCtx.beginPath();
  velocityCtx.moveTo(scrubberX, padding.top);
  velocityCtx.lineTo(scrubberX, h - padding.bottom);
  velocityCtx.stroke();
}

// ----------------------------------------------------
// PLAYBACK CONTROLLER
// ----------------------------------------------------
function togglePlay() {
  if (isPlaying) pause();
  else play();
}

function play() {
  if (!currentEpisode) return;
  isPlaying = true;
  document.getElementById('playBtn').innerText = '⏸';
  lastAnimTimestamp = performance.now();
  if (playbackTimeMs >= currentEpisode.metrics.duration_ms) {
    playbackTimeMs = 0;
  }
  animFrameId = requestAnimationFrame(animStep);
}

function pause() {
  isPlaying = false;
  document.getElementById('playBtn').innerText = '▶';
  if (animFrameId) cancelAnimationFrame(animFrameId);
}

function animStep(timestamp) {
  if (!isPlaying) return;
  const delta = (timestamp - lastAnimTimestamp) * playbackSpeed;
  lastAnimTimestamp = timestamp;

  playbackTimeMs += delta;
  const maxMs = currentEpisode.metrics.duration_ms;

  if (playbackTimeMs >= maxMs) {
    playbackTimeMs = maxMs;
    pause();
  }

  document.getElementById('scrubber').value = playbackTimeMs;
  updateTimeText();
  renderTrajectory();
  renderVelocityChart();

  if (isPlaying) {
    animFrameId = requestAnimationFrame(animStep);
  }
}

function onScrub(val) {
  playbackTimeMs = parseFloat(val);
  updateTimeText();
  renderTrajectory();
  renderVelocityChart();
}

function setPlaybackSpeed(val) {
  playbackSpeed = parseFloat(val);
}

function updateTimeText() {
  document.getElementById('currentTimeText').innerText = formatMs(playbackTimeMs);
  if (currentEpisode) {
    document.getElementById('totalTimeText').innerText = formatMs(currentEpisode.metrics.duration_ms);
  }
}

function formatMs(ms) {
  const totalSec = ms / 1000;
  const mins = Math.floor(totalSec / 60);
  const secs = Math.floor(totalSec % 60);
  const millis = Math.floor(ms % 1000);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
}

// ----------------------------------------------------
// EXPORT MODAL
// ----------------------------------------------------
function openExportModal() {
  document.getElementById('exportModal').style.display = 'flex';
}

function closeExportModal() {
  document.getElementById('exportModal').style.display = 'none';
}

function openSimulateModal() {
  document.getElementById('simulateModal').style.display = 'flex';
}

function closeSimulateModal() {
  document.getElementById('simulateModal').style.display = 'none';
}

async function runAiSimulation() {
  const sx = document.getElementById('simStartX').value || 100;
  const sy = document.getElementById('simStartY').value || 100;
  const tx = document.getElementById('simTargetX').value || 1200;
  const ty = document.getElementById('simTargetY').value || 700;

  try {
    const res = await fetch(`/api/simulate?start_x=${sx}&start_y=${sy}&target_x=${tx}&target_y=${ty}`);
    const data = await res.json();
    if (data.status === 'success') {
      closeSimulateModal();
      
      // Convert AI predicted trajectory to viewer format
      const events = data.trajectory.map((pt, i) => ({
        id: i,
        t_monotonic_ns: i * 8 * 1e6, // 8ms steps
        event_type: pt.type === 'click_down' ? 'click' : (pt.type === 'click_up' ? 'click' : 'move'),
        x: pt.x,
        y: pt.y,
        dx: 0,
        dy: 0,
        button: 'left',
        pressed: pt.type === 'click_down' ? 1 : 0
      }));

      // Calculate physics
      const physics = [];
      let prevSpeed = 0;
      let totalDist = 0;
      for (let i = 0; i < events.length; i++) {
        if (i === 0) {
          physics.append ? physics.push({"dt_ms": 0, "speed_px_s": 0, "accel_px_s2": 0, "dist_px": 0}) : null;
        } else {
          const dt_ms = data.trajectory[i].dt_ms || 8;
          const dx = events[i].x - events[i-1].x;
          const dy = events[i].y - events[i-1].y;
          const dist = Math.hypot(dx, dy);
          totalDist += dist;
          const speed = (dist / (dt_ms / 1000.0)) || 0;
          physics.push({
            "dt_ms": dt_ms,
            "speed_px_s": Math.round(speed),
            "accel_px_s2": 0,
            "dist_px": Math.round(dist)
          });
        }
      }

      const directDist = Math.hypot(tx - sx, ty - sy);
      currentEpisode = {
        session_id: 'AI_MODEL_GENERATED',
        episode_index: 'AI_SIMULATED',
        metrics: {
          event_count: events.length,
          move_count: events.length,
          click_count: 1,
          duration_ms: events.length * 8,
          total_distance_px: Math.round(totalDist),
          direct_distance_px: Math.round(directDist),
          efficiency_ratio: Math.round((directDist / totalDist) * 1000) / 1000,
          avg_speed_px_s: 450,
          max_speed_px_s: 920,
          quality_grade: 'AI_GENERATED'
        },
        events: events,
        physics: physics
      };

      document.getElementById('episodeIdLabel').innerText = `🤖 AI Simulated Trajectory (${sx},${sy}) ➔ (${tx},${ty})`;
      pause();
      playbackTimeMs = currentEpisode.metrics.duration_ms;
      document.getElementById('scrubber').max = playbackTimeMs;
      document.getElementById('scrubber').value = playbackTimeMs;
      updateTimeText();

      fitTrajectoryToScreen();
      renderTrajectory();
      renderVelocityChart();
    } else {
      alert("Error generating simulation: " + data.error);
    }
  } catch (err) {
    console.error("Simulation error:", err);
  }
}

async function downloadDataset() {
  const norm = document.getElementById('exportNormalize').value;
  const minEv = document.getElementById('exportMinEvents').value;

  const url = `/api/export?normalize=${norm}&min_events=${minEv}`;
  try {
    const res = await fetch(url);
    const blob = await res.blob();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `mouse_dataset_ai_ready.json`;
    link.click();
    closeExportModal();
  } catch (err) {
    console.error("Export error:", err);
  }
}
