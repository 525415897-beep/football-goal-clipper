// frontend/player.js

// --- State ---
let state = {
  videoPath: null,
  roiLeft: null,   // {x, y, w, h} in pixel coords
  roiRight: null,
  candidates: [],  // {timestamp, kept: true}
  drawingROI: null, // 'left' | 'right' | null
  drawStart: null,
};

// --- DOM refs ---
const video = document.getElementById('video-player');
const canvas = document.getElementById('roi-overlay');
const ctx = canvas.getContext('2d');
const listEl = document.getElementById('timestamp-list');
const statusEl = document.getElementById('status');

const btnOpen = document.getElementById('btn-open');
const btnROILeft = document.getElementById('btn-set-roi-left');
const btnROIRight = document.getElementById('btn-set-roi-right');
const btnDetect = document.getElementById('btn-detect');
const btnAddManual = document.getElementById('btn-add-manual');
const btnExport = document.getElementById('btn-export');

// --- Canvas sizing ---
function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  const containerRect = video.parentElement.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  canvas.style.left = (rect.left - containerRect.left) + 'px';
  canvas.style.top = (rect.top - containerRect.top) + 'px';
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  drawROIs();
}

video.addEventListener('loadedmetadata', resizeCanvas);
window.addEventListener('resize', resizeCanvas);
video.addEventListener('resize', resizeCanvas);

// --- ROI drawing ---
function drawROIs() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.roiLeft) drawROI(state.roiLeft, '#3498db', '左球门');
  if (state.roiRight) drawROI(state.roiRight, '#e74c3c', '右球门');
}

function drawROI(r, color, label) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(r.x, r.y, r.w, r.h);
  ctx.fillStyle = color;
  ctx.font = '12px sans-serif';
  ctx.fillText(label, r.x + 4, r.y - 6);
}

canvas.addEventListener('mousedown', (e) => {
  if (!state.drawingROI) return;
  const rect = canvas.getBoundingClientRect();
  state.drawStart = {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
});

canvas.addEventListener('mouseup', (e) => {
  if (!state.drawingROI || !state.drawStart) return;
  const rect = canvas.getBoundingClientRect();
  const endX = (e.clientX - rect.left) * (canvas.width / rect.width);
  const endY = (e.clientY - rect.top) * (canvas.height / rect.height);

  const roi = {
    x: Math.min(state.drawStart.x, endX),
    y: Math.min(state.drawStart.y, endY),
    w: Math.abs(endX - state.drawStart.x),
    h: Math.abs(endY - state.drawStart.y),
  };

  if (state.drawingROI === 'left') state.roiLeft = roi;
  else state.roiRight = roi;

  state.drawingROI = null;
  state.drawStart = null;
  canvas.classList.remove('drawing');
  drawROIs();
  updateUI();
});

// --- Button handlers ---
btnOpen.addEventListener('click', async () => {
  const path = await window.api.selectVideo();
  if (!path) return;
  state.videoPath = path;
  video.src = path;
  statusEl.textContent = path.split('/').pop();
  state.roiLeft = null;
  state.roiRight = null;
  state.candidates = [];
  renderList();
  drawROIs();
  updateUI();
});

btnROILeft.addEventListener('click', () => {
  state.drawingROI = 'left';
  canvas.classList.add('drawing');
  statusEl.textContent = '请在视频画面上拖拽框选左侧球门区域';
});

btnROIRight.addEventListener('click', () => {
  state.drawingROI = 'right';
  canvas.classList.add('drawing');
  statusEl.textContent = '请在视频画面上拖拽框选右侧球门区域';
});

btnDetect.addEventListener('click', async () => {
  statusEl.textContent = '正在检测进球...';
  btnDetect.disabled = true;
  try {
    const result = await window.api.detectGoals({
      videoPath: state.videoPath,
      roiLeft: [state.roiLeft.x, state.roiLeft.y, state.roiLeft.w, state.roiLeft.h],
      roiRight: [state.roiRight.x, state.roiRight.y, state.roiRight.w, state.roiRight.h],
    });
    state.candidates = result.timestamps.map(t => ({ timestamp: t, kept: true }));
    renderList();
    statusEl.textContent = `检测完成，共 ${state.candidates.length} 个候选进球`;
  } catch (err) {
    statusEl.textContent = '检测失败: ' + err.message;
  }
  btnDetect.disabled = false;
  updateUI();
});

btnAddManual.addEventListener('click', () => {
  state.candidates.push({ timestamp: video.currentTime, kept: true });
  renderList();
  statusEl.textContent = '已手动添加当前时间点';
});

btnExport.addEventListener('click', async () => {
  const outputDir = await window.api.selectOutputDir();
  if (!outputDir) return;

  const kept = state.candidates.filter(c => c.kept).map(c => c.timestamp);
  statusEl.textContent = '正在导出...';
  btnExport.disabled = true;
  try {
    const result = await window.api.exportClips({
      videoPath: state.videoPath,
      timestamps: kept,
      outputDir: outputDir,
    });
    statusEl.textContent = `导出完成，${result.outputs.length} 个片段已保存`;
  } catch (err) {
    statusEl.textContent = '导出失败: ' + err.message;
  }
  btnExport.disabled = false;
});

// --- Timestamp list ---
function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function renderList() {
  listEl.innerHTML = '';
  state.candidates.forEach((c, i) => {
    const li = document.createElement('li');
    if (!c.kept) li.classList.add('removed');

    const timeSpan = document.createElement('span');
    timeSpan.className = 'time';
    timeSpan.textContent = formatTime(c.timestamp);

    const actions = document.createElement('span');
    actions.className = 'actions';

    const keepBtn = document.createElement('button');
    keepBtn.className = 'btn-keep' + (c.kept ? ' active' : '');
    keepBtn.textContent = c.kept ? '保留' : '已删';
    keepBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      c.kept = !c.kept;
      renderList();
    });

    const removeBtn = document.createElement('button');
    removeBtn.className = 'btn-remove' + (!c.kept ? ' active' : '');
    removeBtn.textContent = c.kept ? '删除' : '恢复';
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      c.kept = !c.kept;
      renderList();
    });

    actions.appendChild(keepBtn);
    actions.appendChild(removeBtn);
    li.appendChild(timeSpan);
    li.appendChild(actions);

    li.addEventListener('click', () => {
      video.currentTime = c.timestamp;
      video.play();
    });

    listEl.appendChild(li);
  });
}

// --- UI state ---
function updateUI() {
  const hasVideo = !!state.videoPath;
  const hasROI = state.roiLeft && state.roiRight;
  const hasCandidates = state.candidates.length > 0;

  btnROILeft.disabled = !hasVideo;
  btnROIRight.disabled = !hasVideo;
  btnDetect.disabled = !hasROI;
  btnAddManual.disabled = !hasVideo;
  btnExport.disabled = !hasCandidates;
}

updateUI();
