// Configuration Constants
const API_BASE_URL = 'http://127.0.0.1:5000';

// Global State
let activeModel = 'mnist'; // 'mnist' or 'imagenet'
let uploadedImageB64 = ''; // Holds the ImageNet input
let isDrawing = false;
let canvas, ctx;

// Select elements
const statusDot = document.querySelector('.status-dot');
const statusText = document.querySelector('.status-text');

const modelMnist = document.getElementById('model-mnist');
const modelImagenet = document.getElementById('model-imagenet');

const mnistContainer = document.getElementById('mnist-input-container');
const imagenetContainer = document.getElementById('imagenet-input-container');
const canvasClearBtn = document.getElementById('btn-clear-canvas');

const fileInput = document.getElementById('image-file-input');
const dropzone = document.getElementById('upload-dropzone');
const sampleItems = document.querySelectorAll('.sample-item');

const attackTypeSelect = document.getElementById('attack-type');
const epsilonGroup = document.getElementById('epsilon-group');
const epsilonRange = document.getElementById('epsilon-range');
const epsilonVal = document.getElementById('epsilon-val');
const pgdAdvancedFields = document.getElementById('pgd-advanced-fields');
const alphaRange = document.getElementById('alpha-range');
const alphaVal = document.getElementById('alpha-val');
const iterationsRange = document.getElementById('iterations-range');
const iterationsVal = document.getElementById('iterations-val');

const defenseTypeSelect = document.getElementById('defense-type');
const defenseParamGroup = document.getElementById('defense-param-group');
const defenseParamRange = document.getElementById('defense-param-range');
const defenseParamLabel = document.getElementById('defense-param-label');
const defenseParamVal = document.getElementById('defense-param-val');
const defenseParamDesc = document.getElementById('defense-param-desc');

const btnSimulate = document.getElementById('btn-simulate');

// Metric Elements
const metricAttackSuccess = document.getElementById('metric-attack-success').querySelector('.metric-value');
const metricPerturbationDist = document.getElementById('metric-perturbation-dist').querySelector('.metric-value');
const metricDefenseStatus = document.getElementById('metric-defense-status').querySelector('.metric-value');

// Viewer Elements
const viewOriginal = document.getElementById('view-original');
const viewPerturbation = document.getElementById('view-perturbation');
const viewAdversarial = document.getElementById('view-adversarial');
const viewDefended = document.getElementById('view-defended');

const predsOriginal = document.getElementById('preds-original');
const predsAdversarial = document.getElementById('preds-adversarial');
const predsDefended = document.getElementById('preds-defended');
const noiseDescription = document.getElementById('noise-description');
const noiseFill = document.getElementById('noise-fill');

// Check Backend Connection Status on startup
async function checkBackendStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/status`);
    const data = await response.json();
    if (data.status === 'online') {
      statusDot.className = 'status-dot online';
      statusText.textContent = 'Backend Online';
    } else {
      throw new Error();
    }
  } catch (error) {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Backend Offline - Start app.py';
  }
}

// Drawing logic (MNIST Canvas)
function initCanvas() {
  canvas = document.getElementById('mnist-canvas');
  ctx = canvas.getContext('2d');
  
  // Clear / Fill Black
  clearCanvas();
  
  // Events
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseout', stopDrawing);
  
  // Mobile touch support
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
  });
  
  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
  });
  
  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    const mouseEvent = new MouseEvent('mouseup', {});
    canvas.dispatchEvent(mouseEvent);
  });
}

function clearCanvas() {
  if (!ctx) return;
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function startDrawing(e) {
  isDrawing = true;
  draw(e);
}

function draw(e) {
  if (!isDrawing) return;
  
  const rect = canvas.getBoundingClientRect();
  // Get scale ratios in case element size differs from canvas attributes
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#ffffff'; // White ink for MNIST
  
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function stopDrawing() {
  isDrawing = false;
  ctx.beginPath();
}

// Set up Sliders Configuration depending on Attack/Defense selections
function initSliders() {
  // Attack Sliders
  epsilonRange.addEventListener('input', () => {
    epsilonVal.textContent = epsilonRange.value;
  });
  alphaRange.addEventListener('input', () => {
    alphaVal.textContent = alphaRange.value;
  });
  iterationsRange.addEventListener('input', () => {
    iterationsVal.textContent = iterationsRange.value;
  });
  
  attackTypeSelect.addEventListener('change', () => {
    const val = attackTypeSelect.value;
    if (val === 'none') {
      epsilonGroup.classList.add('hidden');
      pgdAdvancedFields.classList.add('hidden');
    } else if (val === 'fgsm') {
      epsilonGroup.classList.remove('hidden');
      pgdAdvancedFields.classList.add('hidden');
    } else if (val === 'pgd') {
      epsilonGroup.classList.remove('hidden');
      pgdAdvancedFields.classList.remove('hidden');
    } else if (val === 'patch') {
      epsilonGroup.classList.remove('hidden');
      pgdAdvancedFields.classList.add('hidden');
    }
  });

  // Defense Sliders
  defenseTypeSelect.addEventListener('change', () => {
    const val = defenseTypeSelect.value;
    if (val === 'none') {
      defenseParamGroup.classList.add('hidden');
    } else {
      defenseParamGroup.classList.remove('hidden');
      if (val === 'jpeg') {
        defenseParamLabel.textContent = 'JPEG Quality';
        defenseParamRange.min = '10';
        defenseParamRange.max = '95';
        defenseParamRange.step = '5';
        defenseParamRange.value = '30';
        defenseParamVal.textContent = '30';
        defenseParamDesc.textContent = 'Lower value means higher compression, filtering out noise but degrading image quality.';
      } else if (val === 'smoothing') {
        defenseParamLabel.textContent = 'Gaussian Kernel Size';
        defenseParamRange.min = '3';
        defenseParamRange.max = '9';
        defenseParamRange.step = '2';
        defenseParamRange.value = '3';
        defenseParamVal.textContent = '3';
        defenseParamDesc.textContent = 'A larger kernel blurs out adversarial noise but dampens object features.';
      } else if (val === 'bit_reduction') {
        defenseParamLabel.textContent = 'Bit Depth';
        defenseParamRange.min = '1';
        defenseParamRange.max = '6';
        defenseParamRange.step = '1';
        defenseParamRange.value = '3';
        defenseParamVal.textContent = '3';
        defenseParamDesc.textContent = 'Reduces the color channels resolution. Fewer bits wipes subtle gradients.';
      }
    }
  });

  defenseParamRange.addEventListener('input', () => {
    defenseParamVal.textContent = defenseParamRange.value;
  });
}

// ImageNet Dropzone and Sample selections
function initUploadPlayground() {
  // Model Toggles
  modelMnist.addEventListener('change', () => {
    activeModel = 'mnist';
    mnistContainer.classList.remove('hidden');
    imagenetContainer.classList.add('hidden');
  });
  
  modelImagenet.addEventListener('change', () => {
    activeModel = 'imagenet';
    mnistContainer.classList.add('hidden');
    imagenetContainer.classList.remove('hidden');
    // Load default corgi on first activation if empty
    if (!uploadedImageB64) {
      loadSampleImage('corgi');
    }
  });
  
  // Dropzone drag-drop
  dropzone.addEventListener('click', () => fileInput.click());
  
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleImageFile(file);
  });
  
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  
  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });
  
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleImageFile(file);
  });

  // Sample items selection
  sampleItems.forEach(item => {
    item.addEventListener('click', () => {
      sampleItems.forEach(i => i.classList.remove('selected'));
      item.classList.add('selected');
      const sampleName = item.dataset.sample;
      loadSampleImage(sampleName);
    });
  });
}

function handleImageFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    // Process and resize the image to a standard square layout for the web view
    const img = new Image();
    img.onload = () => {
      const resizeCanvas = document.createElement('canvas');
      resizeCanvas.width = 224;
      resizeCanvas.height = 224;
      const rCtx = resizeCanvas.getContext('2d');
      rCtx.drawImage(img, 0, 0, 224, 224);
      uploadedImageB64 = resizeCanvas.toDataURL('image/png');
      
      // Update UI preview
      viewOriginal.src = uploadedImageB64;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// Generate procedurally generated samples so the application is completely offline functional
function loadSampleImage(name) {
  // We can draw a simple shape to represent the sample on a temporary canvas, convert to Base64
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 224;
  tempCanvas.height = 224;
  const tCtx = tempCanvas.getContext('2d');
  
  // Fill background
  tCtx.fillStyle = '#1e293b';
  tCtx.fillRect(0, 0, 224, 224);
  
  if (name === 'corgi') {
    // Draw dog-like block shape
    tCtx.fillStyle = '#d97706'; // Golden brown
    tCtx.beginPath();
    tCtx.arc(112, 120, 50, 0, Math.PI * 2);
    tCtx.fill();
    // Ears
    tCtx.beginPath();
    tCtx.moveTo(80, 80); tCtx.lineTo(65, 40); tCtx.lineTo(95, 65); tCtx.fill();
    tCtx.beginPath();
    tCtx.moveTo(144, 80); tCtx.lineTo(159, 40); tCtx.lineTo(129, 65); tCtx.fill();
    // Eyes
    tCtx.fillStyle = '#000000';
    tCtx.beginPath(); tCtx.arc(92, 110, 6, 0, Math.PI * 2); tCtx.fill();
    tCtx.beginPath(); tCtx.arc(132, 110, 6, 0, Math.PI * 2); tCtx.fill();
    // Nose
    tCtx.beginPath(); tCtx.arc(112, 130, 8, 0, Math.PI * 2); tCtx.fill();
  } else if (name === 'zebra') {
    // Zebra stripes
    tCtx.fillStyle = '#ffffff';
    tCtx.fillRect(0, 0, 224, 224);
    tCtx.fillStyle = '#000000';
    for (let i = 0; i < 224; i += 30) {
      tCtx.beginPath();
      tCtx.moveTo(i, 0); tCtx.lineTo(i + 15, 0); tCtx.lineTo(i - 10, 224); tCtx.lineTo(i - 25, 224);
      tCtx.fill();
    }
  } else if (name === 'goldfish') {
    // Orange goldfish
    tCtx.fillStyle = '#ea580c'; // Orange
    tCtx.beginPath();
    tCtx.ellipse(112, 112, 60, 35, 0, 0, Math.PI * 2);
    tCtx.fill();
    // Tail fin
    tCtx.beginPath();
    tCtx.moveTo(52, 112); tCtx.lineTo(15, 80); tCtx.lineTo(30, 112); tCtx.lineTo(15, 144);
    tCtx.fill();
    // Eye
    tCtx.fillStyle = '#ffffff';
    tCtx.beginPath(); tCtx.arc(142, 105, 8, 0, Math.PI * 2); tCtx.fill();
    tCtx.fillStyle = '#000000';
    tCtx.beginPath(); tCtx.arc(144, 105, 4, 0, Math.PI * 2); tCtx.fill();
  } else if (name === 'daisy') {
    // Yellow center with white petals
    tCtx.fillStyle = '#ffffff';
    for (let i = 0; i < 8; i++) {
      const angle = (i * Math.PI) / 4;
      tCtx.beginPath();
      tCtx.ellipse(112 + Math.cos(angle)*50, 112 + Math.sin(angle)*50, 45, 15, angle, 0, Math.PI * 2);
      tCtx.fill();
    }
    tCtx.fillStyle = '#eab308'; // Yellow center
    tCtx.beginPath();
    tCtx.arc(112, 112, 30, 0, Math.PI * 2);
    tCtx.fill();
  }
  
  uploadedImageB64 = tempCanvas.toDataURL('image/png');
  viewOriginal.src = uploadedImageB64;
}

// Predictions Rendering
function renderPredictions(predictions, container) {
  container.innerHTML = '';
  if (!predictions || predictions.length === 0) {
    container.innerHTML = '<p class="placeholder-text">No predictions</p>';
    return;
  }
  
  predictions.forEach(pred => {
    const percent = (pred.confidence * 100).toFixed(1);
    const item = document.createElement('div');
    item.className = 'prediction-bar-container';
    item.innerHTML = `
      <div class="prediction-label" title="${pred.label}">${pred.label}</div>
      <div class="prediction-progress-bg">
        <div class="prediction-progress-fill" style="width: ${percent}%;"></div>
      </div>
      <div class="prediction-value">${percent}%</div>
    `;
    container.appendChild(item);
  });
}

// Trigger Backend API Request
async function runSimulation() {
  let imagePayload = '';
  
  if (activeModel === 'mnist') {
    // Get grayscale drawing canvas content
    imagePayload = canvas.toDataURL('image/png');
  } else {
    // Get loaded ImageNet image content
    imagePayload = uploadedImageB64;
  }
  
  if (!imagePayload) {
    alert('Please draw a digit or upload an image first.');
    return;
  }

  // Update UI into Loading State
  btnSimulate.textContent = 'Simulating Attack...';
  btnSimulate.disabled = true;
  btnSimulate.classList.add('secondary-btn');
  btnSimulate.classList.remove('primary-btn');
  
  const payload = {
    image: imagePayload,
    model_type: activeModel,
    attack_type: attackTypeSelect.value,
    epsilon: parseFloat(epsilonRange.value),
    alpha: parseFloat(alphaRange.value),
    iterations: parseInt(iterationsRange.value),
    defense_type: defenseTypeSelect.value,
    defense_param: parseFloat(defenseParamRange.value)
  };

  try {
    const response = await fetch(`${API_BASE_URL}/api/attack`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      throw new Error('API server returned error');
    }
    
    const result = await response.json();
    
    // Update Connection Indicator
    statusDot.className = 'status-dot online';
    statusText.textContent = 'Backend Online';

    // 1. Update Preview Images
    viewOriginal.src = imagePayload;
    viewAdversarial.src = result.adversarial_image;
    viewPerturbation.src = result.perturbation_image;
    viewDefended.src = result.defended_image;
    
    // 2. Update Charts/List Predictions
    renderPredictions(result.original_predictions, predsOriginal);
    renderPredictions(result.adversarial_predictions, predsAdversarial);
    renderPredictions(result.defended_predictions, predsDefended);
    
    // 3. Extract top values for metrics calculations
    const origTop = result.original_predictions[0];
    const advTop = result.adversarial_predictions[0];
    const defTop = result.defended_predictions[0];
    
    // Attack success condition: Top prediction class changes
    const attackSucceeded = (origTop.label !== advTop.label);
    
    // Metric 1: Attack Success
    if (payload.attack_type === 'none') {
      metricAttackSuccess.textContent = 'No Attack';
      metricAttackSuccess.className = 'metric-value status-neutral';
    } else if (attackSucceeded) {
      metricAttackSuccess.textContent = 'Successful';
      metricAttackSuccess.className = 'metric-value status-danger';
    } else {
      metricAttackSuccess.textContent = 'Resisted';
      metricAttackSuccess.className = 'metric-value status-safe';
    }
    
    // Metric 2: Epsilon Value
    metricPerturbationDist.textContent = payload.attack_type === 'none' ? '0.00' : payload.epsilon.toFixed(2);
    
    // Metric 3: Defense Status
    const defenseType = payload.defense_type;
    if (defenseType === 'none') {
      metricDefenseStatus.textContent = 'N/A';
      metricDefenseStatus.className = 'metric-value status-neutral';
    } else {
      // Defense recovery condition: Defended matches Original class AND it was successfully attacked
      if (!attackSucceeded) {
        metricDefenseStatus.textContent = 'Unnecessary';
        metricDefenseStatus.className = 'metric-value status-neutral';
      } else if (defTop.label === origTop.label) {
        metricDefenseStatus.textContent = 'Restored Label';
        metricDefenseStatus.className = 'metric-value status-safe';
      } else {
        metricDefenseStatus.textContent = 'Bypassed';
        metricDefenseStatus.className = 'metric-value status-danger';
      }
    }
    
    // Update Noise details descriptor
    if (payload.attack_type === 'none') {
      noiseDescription.textContent = 'No attack perturbation applied.';
      noiseFill.style.width = '0%';
    } else {
      noiseDescription.textContent = `Visualizing scaled L_inf pixel offsets (Epsilon: ${payload.epsilon}).`;
      noiseFill.style.width = `${(payload.epsilon / 0.3) * 100}%`;
    }
    
  } catch (error) {
    console.error(error);
    alert('Failed to connect to the backend server. Make sure "app.py" is running on port 5000.');
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Backend Offline';
  } finally {
    // Restore button state
    btnSimulate.textContent = 'Run Simulation';
    btnSimulate.disabled = false;
    btnSimulate.classList.add('primary-btn');
    btnSimulate.classList.remove('secondary-btn');
  }
}

// Initialise Elements
window.addEventListener('DOMContentLoaded', () => {
  checkBackendStatus();
  initCanvas();
  initSliders();
  initUploadPlayground();
  
  canvasClearBtn.addEventListener('click', clearCanvas);
  btnSimulate.addEventListener('click', runSimulation);
});
