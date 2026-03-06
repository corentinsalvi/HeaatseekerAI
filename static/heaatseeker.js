const API = '';
let selectedFiles = [];

document.getElementById('fileInput').addEventListener('change', function() {
  handleFiles(Array.from(this.files));
});

const dropZone = document.getElementById('dropZone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  handleFiles(Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/')));
});

function handleFiles(files) {
  selectedFiles = [...selectedFiles, ...files];
  renderPreviews();
  document.getElementById('submitBtn').disabled = selectedFiles.length === 0;
  hide('errorBox'); hide('resultBox'); hide('status');
}

function renderPreviews() {
  const grid = document.getElementById('previewGrid');
  grid.innerHTML = '';
  selectedFiles.forEach(file => {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.title = file.name;
    grid.appendChild(img);
  });
}

async function submitUrl() {
  const url = document.getElementById('urlInput').value.trim();
  if (!url) { showError('Veuillez entrer une URL.'); return; }

  const btn = document.getElementById('urlBtn');
  btn.disabled = true;
  btn.textContent = '...';
  showOverlay('Scraping en cours…');
  hide('errorBox'); hide('resultBox');

  try {
    const resp = await fetch(`${API}/api/authenticate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await resp.json();
    hideOverlay();
    btn.disabled = false;
    btn.textContent = 'Analyser';
    data.error ? showError(data.error) : showResult(data);
  } catch(e) {
    hideOverlay();
    btn.disabled = false;
    btn.textContent = 'Analyser';
    showError('Impossible de contacter le serveur.');
  }
}

async function submitImages() {
  if (!selectedFiles.length) return;
  document.getElementById('submitBtn').disabled = true;
  showOverlay('Analyse en cours…');
  hide('errorBox'); hide('resultBox');

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append('images', f));

  try {
    const resp = await fetch(`${API}/api/authenticate-upload`, { method: 'POST', body: formData });
    const data = await resp.json();
    hideOverlay();
    document.getElementById('submitBtn').disabled = false;
    data.error ? showError(data.error) : showResult(data);
  } catch(e) {
    hideOverlay();
    document.getElementById('submitBtn').disabled = false;
    showError('Impossible de contacter le serveur.');
  }
}

function showResult(data) {
  const isLegit = data.result === 'LEGIT';
  const conf = data.avg_confidence;
  const low  = conf < 50;

  let verdictText, verdictClass;
  if (data.low_confidence) {
    verdictText = '?';
    verdictClass = 'verdict unknown';
  } else if (isLegit) {
    verdictText = 'LEGIT';
    verdictClass = 'verdict legit';
  } else {
    verdictText = 'FAKE';
    verdictClass = 'verdict fake';
  }
  document.getElementById('verdict').textContent = verdictText;
  document.getElementById('verdict').className = verdictClass;
  document.getElementById('confValue').textContent = conf.toFixed(1) + '%';

  const fill = document.getElementById('confFill');
  fill.className = 'confidence-bar-fill' + (low ? ' low' : '');
  setTimeout(() => fill.style.width = conf + '%', 50);

  document.getElementById('resultStats').innerHTML = `
    <div class="stat"><span class="stat-label">Total</span><span class="stat-value">${data.total}</span></div>
    <div class="stat"><span class="stat-label">Legit</span><span class="stat-value" style="color:var(--success)">${data.legit_count}</span></div>
    <div class="stat"><span class="stat-label">Fake</span><span class="stat-value" style="color:var(--danger)">${data.fake_count}</span></div>
    <div class="stat"><span class="stat-label">Confiance</span><span class="stat-value">${conf.toFixed(1)}%</span></div>
  `;

  document.getElementById('warnBox').className = 'warn' + (low ? ' visible' : '');

  const images = data.images_info || [];
  const grid = document.getElementById('gradcamGrid');
  grid.innerHTML = '';

  if (images.length > 0) {
    images.forEach((item, i) => {
      const div = document.createElement('div');
      div.className = 'gradcam-item';
      const badgeClass = item.vote === 'FAKE' ? 'badge-fake' : 'badge-legit';
      div.innerHTML = `
        <div class="gradcam-header">
          <span class="img-num">Image ${i + 1}</span>
          <span class="${badgeClass}">${item.vote}</span>
          <span class="score">${item.pred}%</span>
        </div>
        <div class="gradcam-images">
          <div class="gradcam-col">
            <div class="gradcam-label">Original</div>
            <img src="${item.original}" onerror="this.style.display='none'" alt=""/>
          </div>
          <div class="gradcam-col">
            <div class="gradcam-label">Grad-CAM</div>
            ${item.gradcam
              ? `<img src="${item.gradcam}" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'gradcam-unavailable',textContent:'Non disponible'}))" alt=""/>`
              : '<span class="gradcam-unavailable">Non disponible</span>'
            }
          </div>
        </div>`;
      grid.appendChild(div);
    });
    document.getElementById('gradcamSection').style.display = 'block';
  }

  show('resultBox');
}

async function logout() {
  await fetch('/api/logout', { method: 'POST' });
  window.location.href = '/login';
}

function showError(msg) {
  const el = document.getElementById('errorBox');
  el.textContent = msg;
  el.classList.add('visible');
}
function show(id) { document.getElementById(id).classList.add('visible'); }
function hide(id) { document.getElementById(id).classList.remove('visible'); }
function showOverlay(label) {
  document.getElementById('spinnerLabel').textContent = label || 'Analyse en cours…';
  document.getElementById('cardOverlay').classList.add('visible');
}
function hideOverlay() {
  document.getElementById('cardOverlay').classList.remove('visible');
}
