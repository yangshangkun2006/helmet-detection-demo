/* 安全帽检测 · 浏览器端推理（ONNX Runtime Web） */
(function () {
  'use strict';

  const SIZE = 640;
  const ANCHORS = 8400;
  const IOU_THRESHOLD = 0.7;
  const LABELS = ['hardhat', 'no_hardhat'];

  const fileInput = document.getElementById('file');
  const drop = document.getElementById('drop');
  const chosen = document.getElementById('chosen');
  const confInput = document.getElementById('conf');
  const confVal = document.getElementById('confVal');
  const runBtn = document.getElementById('run');
  const loading = document.getElementById('loading');
  const loadingText = document.getElementById('loadingText');
  const errBox = document.getElementById('err');
  const output = document.getElementById('output');
  const modelStatus = document.getElementById('modelStatus');

  let pendingFile = null;
  let pendingSample = null;
  let session = null;

  if (typeof ort === 'undefined') {
    errBox.textContent = 'ONNX Runtime 加载失败，请检查 vendor/ort.min.js 是否存在。';
    errBox.classList.add('on');
  } else {
    ort.env.wasm.wasmPaths = './vendor/';
    ort.env.wasm.numThreads = 1; // 使用非线程版 wasm，兼容 GitHub Pages
    ort.env.logLevel = 'error';
  }

  function setStatus(text) { modelStatus.textContent = text; }
  function updateRun() { runBtn.disabled = !(pendingFile || pendingSample); }

  function setFile(file) {
    pendingFile = file;
    pendingSample = null;
    chosen.textContent = file ? '已选择：' + file.name : '尚未选择文件';
    document.querySelectorAll('.sample').forEach(b => b.classList.remove('active'));
    updateRun();
  }
  function setSample(btn) {
    pendingSample = btn.dataset.sample;
    pendingFile = null;
    fileInput.value = '';
    chosen.textContent = '已选择示例：' + pendingSample;
    document.querySelectorAll('.sample').forEach(b => b.classList.toggle('active', b === btn));
    updateRun();
  }

  fileInput.addEventListener('change', e => setFile(e.target.files[0] || null));
  drop.addEventListener('click', () => fileInput.click());
  ['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => {
    e.preventDefault(); drop.classList.add('drag');
  }));
  ['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => {
    e.preventDefault(); drop.classList.remove('drag');
  }));
  drop.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });
  document.querySelectorAll('.sample').forEach(b => b.addEventListener('click', () => setSample(b)));
  confInput.addEventListener('input', () => { confVal.textContent = parseFloat(confInput.value).toFixed(2); });

  async function loadModel() {
    if (session) return session;
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime 不可用');
    setStatus('模型加载中…');
    session = await ort.InferenceSession.create('./best.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    setStatus('模型就绪');
    return session;
  }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('图片加载失败'));
      img.src = url;
    });
  }

  function preprocess(img) {
    const w = img.naturalWidth, h = img.naturalHeight;
    const src = document.createElement('canvas');
    src.width = w; src.height = h;
    src.getContext('2d').drawImage(img, 0, 0);

    const r = Math.min(SIZE / h, SIZE / w);
    const nw = Math.round(w * r), nh = Math.round(h * r);
    const px = Math.floor((SIZE - nw) / 2), py = Math.floor((SIZE - nh) / 2);

    const box = document.createElement('canvas');
    box.width = SIZE; box.height = SIZE;
    const bctx = box.getContext('2d');
    bctx.fillStyle = 'rgb(114,114,114)';
    bctx.fillRect(0, 0, SIZE, SIZE);
    bctx.drawImage(src, px, py, nw, nh);

    const data = bctx.getImageData(0, 0, SIZE, SIZE).data;
    const input = new Float32Array(3 * SIZE * SIZE);
    const plane = SIZE * SIZE;
    for (let i = 0, j = 0; i < plane; i++, j += 4) {
      input[i] = data[j] / 255.0;
      input[plane + i] = data[j + 1] / 255.0;
      input[2 * plane + i] = data[j + 2] / 255.0;
    }
    return { input, r, px, py, w, h };
  }

  function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const iw = Math.max(0, x2 - x1), ih = Math.max(0, y2 - y1);
    const inter = iw * ih;
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / Math.max(1e-6, areaA + areaB - inter);
  }

  function nms(items) {
    const order = items.map((_, i) => i).sort((a, b) => items[b].score - items[a].score);
    const keep = [];
    while (order.length) {
      const i = order.shift();
      keep.push(items[i]);
      for (let k = order.length - 1; k >= 0; k--) {
        if (iou(items[i], items[order[k]]) > IOU_THRESHOLD) order.splice(k, 1);
      }
    }
    return keep;
  }

  function postprocess(data, conf, meta) {
    const cands = [[], []];
    for (let i = 0; i < ANCHORS; i++) {
      const s0 = data[4 * ANCHORS + i];
      const s1 = data[5 * ANCHORS + i];
      const score = Math.max(s0, s1);
      if (score < conf) continue;
      const cls = s1 > s0 ? 1 : 0;
      const cx = data[i], cy = data[ANCHORS + i];
      const bw = data[2 * ANCHORS + i], bh = data[3 * ANCHORS + i];
      let x1 = (cx - bw / 2 - meta.px) / meta.r;
      let y1 = (cy - bh / 2 - meta.py) / meta.r;
      let x2 = (cx + bw / 2 - meta.px) / meta.r;
      let y2 = (cy + bh / 2 - meta.py) / meta.r;
      cands[cls].push({
        cls, score,
        x1: Math.max(0, Math.min(meta.w, x1)),
        y1: Math.max(0, Math.min(meta.h, y1)),
        x2: Math.max(0, Math.min(meta.w, x2)),
        y2: Math.max(0, Math.min(meta.h, y2))
      });
    }
    const boxes = nms(cands[0]).concat(nms(cands[1]));
    boxes.sort((a, b) => b.score - a.score);
    return boxes;
  }

  function draw(img, boxes) {
    const canvas = document.createElement('canvas');
    canvas.className = 'result-image';
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const scale = Math.max(1, img.naturalWidth / 640);
    const lw = Math.max(2, Math.round(1.8 * scale));
    const fontSize = Math.max(12, Math.round(13 * scale));
    ctx.font = fontSize + 'px sans-serif';
    ctx.textBaseline = 'bottom';
    boxes.forEach(b => {
      const color = b.cls === 1 ? '#ff6b78' : '#3ddc97';
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
      const text = LABELS[b.cls] + ' ' + b.score.toFixed(2);
      const tw = ctx.measureText(text).width;
      const ty = b.y1 > fontSize + 2 ? b.y1 - 2 : b.y2 + fontSize;
      ctx.fillStyle = 'rgba(8,16,18,.82)';
      ctx.fillRect(b.x1 - 2, ty - fontSize - 2, tw + 6, fontSize + 4);
      ctx.fillStyle = color;
      ctx.fillText(text, b.x1 + 1, ty);
    });
    return canvas;
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function render(img, boxes, meta, conf, elapsed, sourceName) {
    const nHelmet = boxes.filter(b => b.cls === 0).length;
    const nNo = boxes.filter(b => b.cls === 1).length;
    const rows = boxes.length ? boxes.map(b => {
      const bad = b.cls === 1;
      return '<tr><td><span class="tag ' + (bad ? 'bad' : 'ok') + '">' + LABELS[b.cls] + '</span></td>' +
        '<td>' + b.score.toFixed(2) + '</td><td>' +
        [b.x1, b.y1, b.x2, b.y2].map(v => Math.round(v)).join(', ') + '</td></tr>';
    }).join('') : '<tr><td colspan="3" style="color:var(--muted)">该阈值下未检出目标</td></tr>';

    output.innerHTML = '';
    output.appendChild(draw(img, boxes));
    const panel = document.createElement('div');
    panel.innerHTML =
      '<div class="stats">' +
        '<div class="stat ok"><div class="k">戴安全帽</div><div class="v">' + nHelmet + '</div></div>' +
        '<div class="stat bad"><div class="k">未戴帽</div><div class="v">' + nNo + '</div></div>' +
        '<div class="stat"><div class="k">总检测框</div><div class="v">' + boxes.length + '</div></div>' +
      '</div>' +
      '<div class="banner' + (nNo ? '' : ' safe') + '">' +
        (nNo ? '⚠ 检测到 ' + nNo + ' 个未戴帽目标 —— 这里可接告警业务逻辑' : '✓ 本图未检出未戴帽目标') +
      '</div>' +
      '<div class="det-head"><span>检测框明细（' + esc(sourceName) + '）</span><span>耗时 ' + elapsed + ' s · 阈值 ' + conf + '</span></div>' +
      '<div class="det-list"><table><thead><tr><th>类别</th><th>置信度</th><th>框坐标 x1,y1,x2,y2</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
    output.appendChild(panel);
  }

  runBtn.addEventListener('click', async () => {
    errBox.classList.remove('on');
    loading.classList.add('on');
    loadingText.textContent = session ? '正在推理…' : '首次加载模型（约 10 MB）…';
    runBtn.disabled = true;
    let objectUrl = null;
    try {
      const begun = performance.now();
      const model = await loadModel();
      const url = pendingFile ? (objectUrl = URL.createObjectURL(pendingFile)) : './samples/' + pendingSample;
      const img = await loadImage(url);
      const meta = preprocess(img);
      const tensor = new ort.Tensor('float32', meta.input, [1, 3, SIZE, SIZE]);
      const res = await model.run({ images: tensor });
      const boxes = postprocess(res.output0.data, parseFloat(confInput.value), meta);
      render(img, boxes, meta, confInput.value, ((performance.now() - begun) / 1000).toFixed(2),
             pendingFile ? pendingFile.name : pendingSample);
    } catch (e) {
      errBox.textContent = '检测失败：' + (e && e.message ? e.message : e);
      errBox.classList.add('on');
    } finally {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      loading.classList.remove('on');
      updateRun();
    }
  });
})();
