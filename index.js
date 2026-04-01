// ============================================================
//  House Price Predictor — ONNX Runtime Inference
//  Uses onnxruntime-web (ORT) opset 15
// ============================================================

const ONNX_MODEL_PATH = 'house_price_model.onnx';
let session = null;

// ---- Load the ONNX model on page load ----------------------
async function loadModel() {
  try {
    updateStatus('loading');
    session = await ort.InferenceSession.create(ONNX_MODEL_PATH);
    updateStatus('ready');
    console.log('ONNX model loaded successfully.');
  } catch (err) {
    console.error('Failed to load ONNX model:', err);
    updateStatus('error');
  }
}

// ---- Run inference -----------------------------------------
async function predict() {
  if (!session) {
    alert('Model not loaded yet. Please wait.');
    return;
  }

  // Collect the 8 feature values from text inputs
  const featureIds = [
    'medinc', 'houseage', 'averooms', 'avebedrms',
    'population', 'aveoccup', 'latitude', 'longitude'
  ];

  const values = [];
  for (const id of featureIds) {
    const el = document.getElementById(id);
    const val = parseFloat(el.value);
    if (isNaN(val)) {
      showError(`Please enter a valid number for "${el.dataset.label}".`);
      return;
    }
    values.push(val);
    el.classList.remove('input-error');
  }

  hideError();
  showSpinner(true);

  try {
    // Build a Float32Array of shape [1, 8]
    const inputData  = new Float32Array(values);
    const inputTensor = new ort.Tensor('float32', inputData, [1, 8]);

    // Run inference
    const feeds   = { input: inputTensor };
    const results = await session.run(feeds);

    // Extract scalar output
    const rawValue = results['output'].data[0];          // in $100k units
    const dollars  = (rawValue * 100000).toFixed(0);
    const formatted = Number(dollars).toLocaleString('en-US', {
      style: 'currency', currency: 'USD', maximumFractionDigits: 0
    });

    displayResult(formatted, rawValue);
  } catch (err) {
    console.error('Inference error:', err);
    showError('Inference failed. Check the console for details.');
  } finally {
    showSpinner(false);
  }
}

// ---- Fill with a sample (Bay Area suburb) ------------------
function fillSample() {
  const sample = {
    medinc:     4.5,
    houseage:   25,
    averooms:   5.2,
    avebedrms:  1.1,
    population: 1200,
    aveoccup:   2.8,
    latitude:   37.4,
    longitude: -122.1
  };
  for (const [id, val] of Object.entries(sample)) {
    document.getElementById(id).value = val;
  }
  hideError();
  hideResult();
}

// ---- Clear all fields --------------------------------------
function clearFields() {
  const featureIds = [
    'medinc', 'houseage', 'averooms', 'avebedrms',
    'population', 'aveoccup', 'latitude', 'longitude'
  ];
  featureIds.forEach(id => {
    document.getElementById(id).value = '';
    document.getElementById(id).classList.remove('input-error');
  });
  hideError();
  hideResult();
}

// ---- UI helpers --------------------------------------------
function updateStatus(state) {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  if (state === 'loading') {
    dot.className  = 'status-dot loading';
    text.textContent = 'Loading model…';
  } else if (state === 'ready') {
    dot.className  = 'status-dot ready';
    text.textContent = 'Model ready';
  } else {
    dot.className  = 'status-dot error';
    text.textContent = 'Model failed to load';
  }
}

function displayResult(formatted, raw) {
  const box    = document.getElementById('result-box');
  const amount = document.getElementById('result-amount');
  const sub    = document.getElementById('result-sub');
  amount.textContent = formatted;
  sub.textContent    = `Raw model output: ${raw.toFixed(4)} ($100k units)`;
  box.classList.remove('hidden');
  box.classList.add('animate-in');
}

function hideResult() {
  const box = document.getElementById('result-box');
  box.classList.add('hidden');
  box.classList.remove('animate-in');
}

function showSpinner(show) {
  document.getElementById('spinner').style.display = show ? 'inline-block' : 'none';
  document.getElementById('predict-btn').disabled  = show;
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.classList.remove('hidden');
}

function hideError() {
  document.getElementById('error-msg').classList.add('hidden');
}

// ---- Boot --------------------------------------------------
window.addEventListener('DOMContentLoaded', loadModel);
