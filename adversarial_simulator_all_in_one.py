import os
import base64
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import urllib.request
import json
import ast
from PIL import Image
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ==============================================================================
# 1. CORE MACHINE LEARNING MODELS
# ==============================================================================

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 14 * 14)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def simple_blur(img):
    padded = np.pad(img, ((0,0), (1,1), (1,1)), mode='constant')
    blurred = np.zeros_like(img)
    for i in range(1, padded.shape[1]-1):
        for j in range(1, padded.shape[2]-1):
            blurred[0, i-1, j-1] = np.mean(padded[0, i-1:i+2, j-1:j+2])
    return blurred

def train_toy_mnist_model(model, filepath):
    print("Training toy MNIST model on synthetic shapes...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    x_train, y_train = [], []
    np.random.seed(42)
    for digit in range(10):
        for _ in range(80):
            img = np.zeros((1, 28, 28), dtype=np.float32)
            if digit == 0:
                for theta in np.linspace(0, 2 * np.pi, 25):
                    r = 7 + np.random.uniform(-0.5, 0.5)
                    cx, cy = int(14 + r * np.cos(theta)), int(14 + r * np.sin(theta))
                    if 0 <= cx < 28 and 0 <= cy < 28: img[0, cx, cy] = 1.0
            elif digit == 1: img[0, 4:24, 14] = 1.0
            elif digit == 2:
                img[0, 6, 10:18] = 1.0
                img[0, 6:13, 18] = 1.0
                for i in range(7): img[0, 13+i, 18-int(i*1.2)] = 1.0
                img[0, 20, 10:19] = 1.0
            elif digit == 3:
                img[0, 6, 10:18] = 1.0; img[0, 6:21, 18] = 1.0; img[0, 13, 11:18] = 1.0; img[0, 20, 10:18] = 1.0
            elif digit == 4:
                img[0, 6:14, 9] = 1.0; img[0, 13, 9:19] = 1.0; img[0, 6:22, 17] = 1.0
            elif digit == 5:
                img[0, 6, 9:19] = 1.0; img[0, 6:13, 9] = 1.0; img[0, 13, 9:19] = 1.0; img[0, 13:21, 18] = 1.0; img[0, 20, 9:19] = 1.0
            elif digit == 6:
                img[0, 6:21, 9] = 1.0; img[0, 13:21, 17] = 1.0; img[0, 13, 9:18] = 1.0; img[0, 20, 9:18] = 1.0
            elif digit == 7:
                img[0, 6, 9:19] = 1.0
                for i in range(15): img[0, 6+i, 18-int(i*0.6)] = 1.0
            elif digit == 8:
                img[0, 6, 10:18] = 1.0; img[0, 13, 10:18] = 1.0; img[0, 20, 10:18] = 1.0; img[0, 6:21, 9] = 1.0; img[0, 6:21, 18] = 1.0
            elif digit == 9:
                img[0, 6:14, 9] = 1.0; img[0, 6, 9:19] = 1.0; img[0, 13, 9:19] = 1.0; img[0, 6:22, 18] = 1.0
            img = simple_blur(img)
            img += np.random.normal(0, 0.05, img.shape)
            x_train.append(np.clip(img, 0.0, 1.0))
            y_train.append(digit)
    x_train = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print("MNIST toy model trained and saved.")

def get_mnist_model(weights_path="backend/data/mnist_toy.pth"):
    model = MNISTNet()
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        except Exception:
            train_toy_mnist_model(model, weights_path)
    else:
        train_toy_mnist_model(model, weights_path)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model

def get_imagenet_model():
    try:
        model = models.mobilenet_v2(pretrained=True)
        model.eval()
        for p in model.parameters(): p.requires_grad = False
        return model
    except Exception as e:
        print(f"ImageNet weights download skipped: {e}. Fallback to dummy mode.")
        class DummyImageNetNet(nn.Module):
            def __init__(self):
                super(DummyImageNetNet, self).__init__()
                self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1,1))
                self.fc = nn.Linear(10, 1000)
            def forward(self, x):
                x = self.pool(F.relu(self.conv(x))).view(x.size(0), -1)
                return self.fc(x)
        model = DummyImageNetNet()
        model.eval()
        for p in model.parameters(): p.requires_grad = False
        return model

def get_imagenet_classes(filepath="backend/data/imagenet_classes.json"):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f: return json.load(f)
        except Exception: pass
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    classes_url = "https://raw.githubusercontent.com/senthilthyagarajan/Imagenet-GPUDistribution/master/imagenet1000_clsidx_to_labels.txt"
    try:
        with urllib.request.urlopen(classes_url, timeout=5) as response:
            html = response.read().decode('utf-8')
            raw_dict = ast.literal_eval(html)
            classes_dict = {str(k): v for k, v in raw_dict.items()}
            with open(filepath, 'w') as f: json.dump(classes_dict, f)
            return classes_dict
    except Exception:
        fallback = {str(i): f"Class {i}" for i in range(1000)}
        fallback.update({"263": "corgi", "281": "tabby cat", "1": "goldfish", "985": "daisy"})
        return fallback

# ==============================================================================
# 2. ADVERSARIAL ATTACKS
# ==============================================================================

def fgsm_attack(model, image, label, epsilon):
    perturbed_image = image.clone().detach()
    perturbed_image.requires_grad = True
    output = model(perturbed_image)
    loss = nn.CrossEntropyLoss()(output, label)
    model.zero_grad()
    loss.backward()
    data_grad = perturbed_image.grad.data
    perturbed_image = perturbed_image + epsilon * data_grad.sign()
    return torch.clamp(perturbed_image, 0.0, 1.0).detach()

def pgd_attack(model, image, label, epsilon, alpha=0.01, iterations=10):
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()
    for _ in range(iterations):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(original_image + eta, min=0.0, max=1.0).detach()
    return perturbed_image

def patch_attack(image, patch_size_pct=0.25):
    perturbed_image = image.clone().detach()
    if len(image.shape) == 4:
        c, h, w = image.shape[1], image.shape[2], image.shape[3]
        patch_h, patch_w = int(h * patch_size_pct), int(w * patch_size_pct)
        y_start, x_start = (h - patch_h) // 2, (w - patch_w) // 2
        if c == 1:
            perturbed_image[0, 0, y_start:y_start+patch_h, x_start:x_start+patch_w] = 1.0
        else:
            perturbed_image[0, 0, y_start:y_start+patch_h, x_start:x_start+patch_w] = 1.0
            perturbed_image[0, 1, y_start:y_start+patch_h, x_start:x_start+patch_w] = 0.0
            perturbed_image[0, 2, y_start:y_start+patch_h, x_start:x_start+patch_w] = 1.0
    return perturbed_image

# ==============================================================================
# 3. DEFENSE FILTERS
# ==============================================================================

def jpeg_compression_defense(image_tensor, quality=30):
    is_batched = len(image_tensor.shape) == 4
    tensor = image_tensor[0] if is_batched else image_tensor
    c, h, w = tensor.shape
    numpy_img = tensor.cpu().numpy()
    if c == 1:
        numpy_img = (numpy_img[0] * 255).astype(np.uint8)
        pil_img = Image.fromarray(numpy_img, mode='L')
    else:
        numpy_img = (np.transpose(numpy_img, (1, 2, 0)) * 255).astype(np.uint8)
        pil_img = Image.fromarray(numpy_img, mode='RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed_pil = Image.open(buffer)
    compressed_numpy = np.array(compressed_pil).astype(np.float32) / 255.0
    if c == 1:
        compressed_tensor = torch.tensor(compressed_numpy).unsqueeze(0)
    else:
        compressed_tensor = torch.tensor(np.transpose(compressed_numpy, (2, 0, 1)))
    if is_batched: compressed_tensor = compressed_tensor.unsqueeze(0)
    return compressed_tensor

def spatial_smoothing_defense(image_tensor, kernel_size=3, method='gaussian'):
    is_batched = len(image_tensor.shape) == 4
    tensor = image_tensor if is_batched else image_tensor.unsqueeze(0)
    c = tensor.shape[1]
    if kernel_size == 3:
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], dtype=torch.float32)
    elif kernel_size == 5:
        kernel = torch.tensor([[1., 4., 7., 4., 1.], [4., 16., 26., 16., 4.], [7., 26., 41., 26., 7.], [4., 16., 26., 16., 4.], [1., 4., 7., 4., 1.]], dtype=torch.float32)
    else:
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32)
    kernel = kernel / kernel.sum()
    weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
    smoothed = F.conv2d(tensor, weight, padding=kernel_size//2, groups=c)
    if not is_batched: smoothed = smoothed.squeeze(0)
    return torch.clamp(smoothed, 0.0, 1.0)

def bit_depth_reduction_defense(image_tensor, bits=3):
    levels = 2 ** bits - 1
    return torch.clamp(torch.round(image_tensor * levels) / levels, 0.0, 1.0)

# ==============================================================================
# 4. FLASK SERVER & WEB API
# ==============================================================================

app = Flask(__name__)
CORS(app)

print("Initializing networks...")
mnist_model = get_mnist_model()
imagenet_model = get_imagenet_model()
imagenet_classes = get_imagenet_classes()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def forward_wrapper(model, x, is_mnist):
    if is_mnist: return model(x)
    return model((x - IMAGENET_MEAN) / IMAGENET_STD)

def base64_to_tensor(b64_string, is_mnist):
    if "," in b64_string: b64_string = b64_string.split(",")[1]
    image_bytes = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_bytes))
    if is_mnist:
        image = image.convert("L").resize((28, 28))
        tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        return tensor.unsqueeze(0).unsqueeze(0)
    image = image.convert("RGB").resize((224, 224))
    tensor = torch.tensor(np.transpose(np.array(image), (2, 0, 1)), dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0)

def tensor_to_base64(tensor):
    if len(tensor.shape) == 4: tensor = tensor.squeeze(0)
    c, h, w = tensor.shape
    numpy_arr = tensor.cpu().numpy()
    if c == 1:
        numpy_arr = (np.clip(numpy_arr[0], 0.0, 1.0) * 255.0).astype(np.uint8)
        image = Image.fromarray(numpy_arr, mode="L")
    else:
        numpy_arr = (np.clip(np.transpose(numpy_arr, (1, 2, 0)), 0.0, 1.0) * 255.0).astype(np.uint8)
        image = Image.fromarray(numpy_arr, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_top_predictions(logits, is_mnist, top_k=5):
    probabilities = F.softmax(logits, dim=1).squeeze(0)
    top_probs, top_indices = torch.topk(probabilities, top_k if not is_mnist else 10)
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        idx_str = str(idx.item())
        label = idx_str if is_mnist else imagenet_classes.get(idx_str, f"Index {idx_str}")
        predictions.append({"label": label.split(",")[0].strip(), "confidence": float(prob.item())})
    return predictions

# ==============================================================================
# 5. EMBEDDED FRONTEND FILES STATIC INJECTION
# ==============================================================================

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Adversarial ML Attack Simulator</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="app-container">
    <header class="app-header">
      <div class="header-logo">
        <span class="logo-icon">🛡️</span>
        <h1>Adversarial ML Simulator</h1>
      </div>
      <div class="connection-status" id="connection-status">
        <span class="status-dot offline"></span>
        <span class="status-text">Connecting...</span>
      </div>
    </header>

    <div class="main-layout">
      <aside class="control-sidebar">
        <section class="control-group">
          <h2>1. Select Target Model</h2>
          <div class="radio-toggle-group">
            <input type="radio" id="model-mnist" name="model_type" value="mnist" checked>
            <label for="model-mnist">
              <strong>MNIST CNN</strong>
              <span>Handwritten digits</span>
            </label>
            <input type="radio" id="model-imagenet" name="model_type" value="imagenet">
            <label for="model-imagenet">
              <strong>MobileNetV2</strong>
              <span>ImageNet General</span>
            </label>
          </div>
        </section>

        <section class="control-group" id="input-playground">
          <h2>2. Provide Input Image</h2>
          <div id="mnist-input-container">
            <p class="section-desc">Draw a digit (0-9) inside the canvas:</p>
            <div class="canvas-wrapper">
              <canvas id="mnist-canvas" width="280" height="280"></canvas>
            </div>
            <div class="canvas-actions">
              <button class="btn secondary-btn" id="btn-clear-canvas">Clear Canvas</button>
            </div>
          </div>

          <div id="imagenet-input-container" class="hidden">
            <p class="section-desc">Upload your own image or pick a sample:</p>
            <div class="upload-dropzone" id="upload-dropzone">
              <span class="upload-icon">📤</span>
              <p>Drag & drop image here or <span class="browse-link">browse</span></p>
              <input type="file" id="image-file-input" accept="image/*" class="hidden-file-input">
            </div>
            <div class="sample-images-grid">
              <div class="sample-item" data-sample="corgi" title="Corgi">🐶</div>
              <div class="sample-item" data-sample="zebra" title="Zebra">🦓</div>
              <div class="sample-item" data-sample="goldfish" title="Goldfish">🐠</div>
              <div class="sample-item" data-sample="daisy" title="Daisy">🌼</div>
            </div>
          </div>
        </section>

        <section class="control-group">
          <h2>3. Configure Attack</h2>
          <div class="field-group">
            <label for="attack-type">Attack Algorithm</label>
            <select id="attack-type">
              <option value="fgsm">FGSM (Fast Gradient Sign)</option>
              <option value="pgd">PGD (Projected Gradient Descent)</option>
              <option value="patch">Adversarial Patch</option>
              <option value="none">No Attack (Clean Run)</option>
            </select>
          </div>

          <div class="field-group slider-group" id="epsilon-group">
            <div class="slider-header">
              <label for="epsilon-range">Perturbation Strength (Epsilon &epsilon;)</label>
              <span class="value-display" id="epsilon-val">0.05</span>
            </div>
            <input type="range" id="epsilon-range" min="0.01" max="0.30" step="0.01" value="0.05">
            <span class="slider-desc" id="epsilon-desc">Controls the maximum size of pixel alterations.</span>
          </div>

          <div id="pgd-advanced-fields" class="hidden">
            <div class="field-group slider-group">
              <div class="slider-header">
                <label for="alpha-range">Step Size (Alpha &alpha;)</label>
                <span class="value-display" id="alpha-val">0.01</span>
              </div>
              <input type="range" id="alpha-range" min="0.002" max="0.05" step="0.002" value="0.01">
            </div>
            <div class="field-group slider-group">
              <div class="slider-header">
                <label for="iterations-range">Iterations</label>
                <span class="value-display" id="iterations-val">10</span>
              </div>
              <input type="range" id="iterations-range" min="5" max="30" step="1" value="10">
            </div>
          </div>
        </section>

        <section class="control-group">
          <h2>4. Apply Defense</h2>
          <div class="field-group">
            <label for="defense-type">Defense Preprocessing</label>
            <select id="defense-type">
              <option value="none">None (Vulnerable Model)</option>
              <option value="jpeg">JPEG Compression</option>
              <option value="smoothing">Spatial Smoothing (Gaussian)</option>
              <option value="bit_reduction">Bit-depth Reduction</option>
            </select>
          </div>

          <div class="field-group slider-group hidden" id="defense-param-group">
            <div class="slider-header">
              <label id="defense-param-label" for="defense-param-range">Defense Intensity</label>
              <span class="value-display" id="defense-param-val">30</span>
            </div>
            <input type="range" id="defense-param-range" min="1" max="100" step="1" value="30">
            <span class="slider-desc" id="defense-param-desc">Controls preprocessing intensity.</span>
          </div>
        </section>

        <button class="btn primary-btn active-cta" id="btn-simulate">Run Simulation</button>
      </aside>

      <main class="dashboard-content">
        <div class="metrics-bar">
          <div class="metric-card" id="metric-attack-success">
            <span class="metric-label">Attack Status</span>
            <span class="metric-value status-neutral">N/A</span>
          </div>
          <div class="metric-card" id="metric-perturbation-dist">
            <span class="metric-label">Perturbation (L_inf)</span>
            <span class="metric-value">0.00</span>
          </div>
          <div class="metric-card" id="metric-defense-status">
            <span class="metric-label">Defense Effect</span>
            <span class="metric-value status-neutral">N/A</span>
          </div>
        </div>

        <div class="results-grid">
          <div class="result-card">
            <h3>Original Image</h3>
            <div class="image-viewer-box">
              <img id="view-original" src="placeholder.png" alt="Original Input">
            </div>
            <div class="predictions-box" id="preds-original">
              <p class="placeholder-text">Run simulation to visualize predictions</p>
            </div>
          </div>

          <div class="result-card">
            <h3>Perturbation (Noise Map)</h3>
            <div class="image-viewer-box noise-bg">
              <img id="view-perturbation" src="placeholder.png" alt="Adversarial Noise">
            </div>
            <div class="noise-details-box">
              <div class="noise-strength-meter">
                <span class="noise-meter-fill" id="noise-fill"></span>
              </div>
              <p id="noise-description" class="placeholder-text">Adversarial modifications will show up here.</p>
            </div>
          </div>

          <div class="result-card highlighted-card">
            <h3>Adversarial Output</h3>
            <div class="image-viewer-box">
              <img id="view-adversarial" src="placeholder.png" alt="Adversarial Output">
            </div>
            <div class="predictions-box" id="preds-adversarial">
              <p class="placeholder-text">Run simulation to visualize predictions</p>
            </div>
          </div>

          <div class="result-card defended-card">
            <h3>Defended Output</h3>
            <div class="image-viewer-box">
              <img id="view-defended" src="placeholder.png" alt="Defended Output">
            </div>
            <div class="predictions-box" id="preds-defended">
              <p class="placeholder-text">Run simulation to visualize predictions</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  </div>

  <script src="app.js"></script>
</body>
</html>
"""

CSS_CONTENT = """
:root {
  --bg-primary: #0a0f1d;
  --bg-secondary: rgba(20, 28, 52, 0.6);
  --bg-tertiary: rgba(13, 20, 38, 0.85);
  --accent-cyan: #00f2fe;
  --accent-cyan-glow: rgba(0, 242, 254, 0.4);
  --accent-pink: #f35588;
  --accent-pink-glow: rgba(243, 85, 136, 0.4);
  --accent-green: #00e676;
  --accent-green-glow: rgba(0, 230, 118, 0.4);
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --border-color: rgba(255, 255, 255, 0.08);
  --border-focus: rgba(0, 242, 254, 0.5);
  --shadow-main: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  --font-display: 'Space Grotesk', 'Inter', sans-serif;
  --font-body: 'Inter', sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-body);
  min-height: 100vh;
  display: flex;
  justify-content: center;
  overflow-x: hidden;
}
.app-container {
  width: 100%;
  max-width: 1600px;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  padding: 20px;
}
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 24px;
  margin-bottom: 20px;
  background: rgba(15, 23, 42, 0.4);
  backdrop-filter: blur(10px);
  border: 1px solid var(--border-color);
  border-radius: 12px;
}
.header-logo { display: flex; align-items: center; gap: 12px; }
.logo-icon { font-size: 1.8rem; }
.header-logo h1 {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 1.5rem;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.connection-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.85rem;
  background: rgba(0, 0, 0, 0.2);
  padding: 6px 14px;
  border-radius: 20px;
  border: 1px solid var(--border-color);
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.status-dot.online { background-color: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
.status-dot.offline { background-color: var(--accent-pink); box-shadow: 0 0 8px var(--accent-pink); }
.status-text { color: var(--text-secondary); }
.main-layout { display: grid; grid-template-columns: 380px 1fr; gap: 20px; flex: 1; }
.control-sidebar { display: flex; flex-direction: column; gap: 20px; }
.control-group {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 20px;
  box-shadow: var(--shadow-main);
}
.control-group h2 {
  font-family: var(--font-display);
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--accent-cyan);
  margin-bottom: 15px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.section-desc { font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 12px; }
.radio-toggle-group { display: flex; gap: 8px; }
.radio-toggle-group input[type="radio"] { display: none; }
.radio-toggle-group label {
  flex: 1; display: flex; flex-direction: column; padding: 12px;
  border: 1px solid var(--border-color); border-radius: 10px; cursor: pointer;
  background: rgba(0, 0, 0, 0.2); transition: all 0.25s ease;
}
.radio-toggle-group label strong { font-size: 0.9rem; color: var(--text-primary); margin-bottom: 2px; }
.radio-toggle-group label span { font-size: 0.75rem; color: var(--text-secondary); }
.radio-toggle-group input[type="radio"]:checked + label {
  border-color: var(--accent-cyan); background: rgba(0, 242, 254, 0.08); box-shadow: 0 0 10px rgba(0, 242, 254, 0.15);
}
.canvas-wrapper {
  background: #020617; border: 2px solid var(--border-color); border-radius: 12px;
  overflow: hidden; width: 280px; height: 280px; margin: 0 auto 12px auto;
  cursor: crosshair; transition: border-color 0.3s;
}
.canvas-wrapper:hover { border-color: var(--border-focus); }
#mnist-canvas { display: block; }
.canvas-actions { display: flex; justify-content: center; }
.upload-dropzone {
  border: 2px dashed var(--border-color); border-radius: 12px; padding: 30px 15px;
  text-align: center; background: rgba(0, 0, 0, 0.15); cursor: pointer; transition: all 0.3s ease; margin-bottom: 12px;
}
.upload-dropzone:hover, .upload-dropzone.dragover { border-color: var(--accent-cyan); background: rgba(0, 242, 254, 0.04); }
.upload-icon { font-size: 2rem; margin-bottom: 10px; display: block; }
.upload-dropzone p { font-size: 0.85rem; color: var(--text-secondary); }
.browse-link { color: var(--accent-cyan); text-decoration: underline; font-weight: 500; }
.hidden-file-input { display: none; }
.sample-images-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.sample-item {
  font-size: 1.8rem; padding: 10px; border: 1px solid var(--border-color); border-radius: 10px;
  background: rgba(0, 0, 0, 0.2); text-align: center; cursor: pointer; transition: all 0.2s ease;
}
.sample-item:hover { transform: translateY(-2px); border-color: var(--accent-cyan); background: rgba(0, 242, 254, 0.05); }
.sample-item.selected { border-color: var(--accent-cyan); background: rgba(0, 242, 254, 0.1); box-shadow: 0 0 8px rgba(0, 242, 254, 0.2); }
.field-group { margin-bottom: 16px; }
.field-group label { display: block; font-size: 0.85rem; color: var(--text-primary); margin-bottom: 8px; font-weight: 500; }
select {
  width: 100%; padding: 10px 12px; background-color: #0d1326; border: 1px solid var(--border-color);
  border-radius: 8px; color: var(--text-primary); outline: none; font-size: 0.9rem; transition: border-color 0.2s;
}
select:focus { border-color: var(--accent-cyan); }
.slider-group { display: flex; flex-direction: column; }
.slider-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.slider-header label { margin-bottom: 0; }
.value-display {
  font-family: var(--font-display); font-size: 0.85rem; font-weight: 700; color: var(--accent-cyan);
  background: rgba(0, 242, 254, 0.1); padding: 2px 8px; border-radius: 4px;
}
input[type="range"] { -webkit-appearance: none; width: 100%; height: 6px; border-radius: 3px; background: #1e293b; outline: none; margin: 10px 0; }
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none; width: 16px; height: 16px; border-radius: 50%;
  background: var(--accent-cyan); cursor: pointer; box-shadow: 0 0 6px var(--accent-cyan-glow); transition: transform 0.1s;
}
input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.2); }
.slider-desc { font-size: 0.75rem; color: var(--text-secondary); }
.btn {
  width: 100%; padding: 12px; border-radius: 10px; font-weight: 600; font-size: 0.95rem;
  cursor: pointer; outline: none; transition: all 0.2s ease; font-family: var(--font-display);
}
.primary-btn {
  background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); border: none; color: #020617;
  box-shadow: 0 4px 15px rgba(0, 242, 254, 0.25);
}
.primary-btn:hover { box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4); transform: translateY(-1px); }
.secondary-btn { background: transparent; border: 1px solid var(--border-color); color: var(--text-secondary); }
.secondary-btn:hover { border-color: var(--text-primary); color: var(--text-primary); background: rgba(255, 255, 255, 0.03); }
.dashboard-content { display: flex; flex-direction: column; gap: 20px; }
.metrics-bar { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
.metric-card {
  background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 12px;
  padding: 15px 20px; display: flex; flex-direction: column; justify-content: center; box-shadow: var(--shadow-main);
}
.metric-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 4px; }
.metric-value { font-family: var(--font-display); font-size: 1.4rem; font-weight: 700; }
.status-safe { color: var(--accent-green); text-shadow: 0 0 8px var(--accent-green-glow); }
.status-danger { color: var(--accent-pink); text-shadow: 0 0 8px var(--accent-pink-glow); }
.status-neutral { color: var(--text-secondary); }
.results-grid { display: grid; grid-template-columns: repeat(2, 1fr); grid-template-rows: auto auto; gap: 20px; flex: 1; }
.result-card {
  background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 16px;
  padding: 20px; display: flex; flex-direction: column; box-shadow: var(--shadow-main); transition: all 0.3s;
}
.result-card h3 {
  font-family: var(--font-display); font-size: 1.05rem; font-weight: 600; margin-bottom: 15px;
  color: var(--text-primary); border-left: 3px solid var(--text-secondary); padding-left: 8px;
}
.result-card.highlighted-card { border-color: rgba(243, 85, 136, 0.3); box-shadow: 0 0 15px rgba(243, 85, 136, 0.05), var(--shadow-main); }
.result-card.highlighted-card h3 { color: var(--accent-pink); border-left-color: var(--accent-pink); }
.result-card.defended-card { border-color: rgba(0, 230, 118, 0.25); box-shadow: 0 0 15px rgba(0, 230, 118, 0.05), var(--shadow-main); }
.result-card.defended-card h3 { color: var(--accent-green); border-left-color: var(--accent-green); }
.image-viewer-box {
  background: #040814; border: 1px solid var(--border-color); border-radius: 12px;
  height: 220px; display: flex; align-items: center; justify-content: center; overflow: hidden; margin-bottom: 16px;
}
.image-viewer-box img { max-width: 100%; max-height: 100%; object-fit: contain; image-rendering: pixelated; }
.noise-bg {
  background-image: 
    linear-gradient(45deg, #060b19 25%, transparent 25%), 
    linear-gradient(-45deg, #060b19 25%, transparent 25%), 
    linear-gradient(45deg, transparent 75%, #060b19 75%), 
    linear-gradient(-45deg, transparent 75%, #060b19 75%);
  background-size: 20px 20px;
}
.noise-details-box { display: flex; flex-direction: column; gap: 12px; justify-content: center; align-items: center; flex: 1; }
.noise-strength-meter { width: 100%; height: 8px; background: #111827; border-radius: 10px; overflow: hidden; border: 1px solid var(--border-color); }
.noise-meter-fill { display: block; height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent-cyan) 0%, var(--accent-pink) 100%); border-radius: 10px; transition: width 0.4s ease; }
.placeholder-text { font-size: 0.85rem; color: var(--text-secondary); text-align: center; }
.predictions-box { display: flex; flex-direction: column; gap: 8px; flex: 1; }
.prediction-bar-container { display: flex; align-items: center; font-size: 0.8rem; gap: 10px; }
.prediction-label { width: 90px; color: var(--text-primary); font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.prediction-progress-bg { flex: 1; height: 8px; background-color: #0f172a; border-radius: 4px; overflow: hidden; border: 1px solid rgba(255,255,255,0.03); }
.prediction-progress-fill { height: 100%; border-radius: 4px; transition: width 0.6s cubic-bezier(0.1, 0.8, 0.2, 1); background-color: var(--accent-cyan); }
.highlighted-card .prediction-progress-fill { background-color: var(--accent-pink); }
.defended-card .prediction-progress-fill { background-color: var(--accent-green); }
.prediction-value { width: 40px; text-align: right; font-family: var(--font-display); font-weight: 600; color: var(--text-secondary); }
.hidden { display: none !important; }
@media (max-width: 1200px) { .main-layout { grid-template-columns: 1fr; } }
@media (max-width: 768px) { .results-grid { grid-template-columns: 1fr; } }
"""

JS_CONTENT = """
const API_BASE_URL = window.location.origin;
let activeModel = 'mnist';
let uploadedImageB64 = '';
let isDrawing = false;
let canvas, ctx;

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

const metricAttackSuccess = document.getElementById('metric-attack-success').querySelector('.metric-value');
const metricPerturbationDist = document.getElementById('metric-perturbation-dist').querySelector('.metric-value');
const metricDefenseStatus = document.getElementById('metric-defense-status').querySelector('.metric-value');

const viewOriginal = document.getElementById('view-original');
const viewPerturbation = document.getElementById('view-perturbation');
const viewAdversarial = document.getElementById('view-adversarial');
const viewDefended = document.getElementById('view-defended');
const predsOriginal = document.getElementById('preds-original');
const predsAdversarial = document.getElementById('preds-adversarial');
const predsDefended = document.getElementById('preds-defended');
const noiseDescription = document.getElementById('noise-description');
const noiseFill = document.getElementById('noise-fill');

async function checkBackendStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/status`);
    const data = await response.json();
    if (data.status === 'online') {
      statusDot.className = 'status-dot online';
      statusText.textContent = 'Online';
    }
  } catch {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Offline';
  }
}

function initCanvas() {
  canvas = document.getElementById('mnist-canvas');
  ctx = canvas.getContext('2d');
  clearCanvas();
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseout', stopDrawing);
  
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault(); const t = e.touches[0];
    canvas.dispatchEvent(new MouseEvent('mousedown', { clientX: t.clientX, clientY: t.clientY }));
  });
  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault(); const t = e.touches[0];
    canvas.dispatchEvent(new MouseEvent('mousemove', { clientX: t.clientX, clientY: t.clientY }));
  });
  canvas.addEventListener('touchend', () => canvas.dispatchEvent(new MouseEvent('mouseup', {})));
}

function clearCanvas() {
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function startDrawing(e) { isDrawing = true; draw(e); }
function stopDrawing() { isDrawing = false; ctx.beginPath(); }
function draw(e) {
  if (!isDrawing) return;
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * (canvas.width / r.width);
  const y = (e.clientY - r.top) * (canvas.height / r.height);
  ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.strokeStyle = '#ffffff';
  ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y);
}

function initSliders() {
  epsilonRange.addEventListener('input', () => epsilonVal.textContent = epsilonRange.value);
  alphaRange.addEventListener('input', () => alphaVal.textContent = alphaRange.value);
  iterationsRange.addEventListener('input', () => iterationsVal.textContent = iterationsRange.value);
  
  attackTypeSelect.addEventListener('change', () => {
    const val = attackTypeSelect.value;
    if (val === 'none') { epsilonGroup.classList.add('hidden'); pgdAdvancedFields.classList.add('hidden'); }
    else if (val === 'fgsm' || val === 'patch') { epsilonGroup.classList.remove('hidden'); pgdAdvancedFields.classList.add('hidden'); }
    else if (val === 'pgd') { epsilonGroup.classList.remove('hidden'); pgdAdvancedFields.classList.remove('hidden'); }
  });

  defenseTypeSelect.addEventListener('change', () => {
    const val = defenseTypeSelect.value;
    if (val === 'none') { defenseParamGroup.classList.add('hidden'); }
    else {
      defenseParamGroup.classList.remove('hidden');
      if (val === 'jpeg') {
        defenseParamLabel.textContent = 'JPEG Quality'; defenseParamRange.min = '10'; defenseParamRange.max = '95'; defenseParamRange.step = '5'; defenseParamRange.value = '30';
      } else if (val === 'smoothing') {
        defenseParamLabel.textContent = 'Gaussian Kernel'; defenseParamRange.min = '3'; defenseParamRange.max = '9'; defenseParamRange.step = '2'; defenseParamRange.value = '3';
      } else if (val === 'bit_reduction') {
        defenseParamLabel.textContent = 'Bit Depth'; defenseParamRange.min = '1'; defenseParamRange.max = '6'; defenseParamRange.step = '1'; defenseParamRange.value = '3';
      }
      defenseParamVal.textContent = defenseParamRange.value;
    }
  });
  defenseParamRange.addEventListener('input', () => defenseParamVal.textContent = defenseParamRange.value);
}

function initUploadPlayground() {
  modelMnist.addEventListener('change', () => { activeModel = 'mnist'; mnistContainer.classList.remove('hidden'); imagenetContainer.classList.add('hidden'); });
  modelImagenet.addEventListener('change', () => { activeModel = 'imagenet'; mnistContainer.classList.add('hidden'); imagenetContainer.classList.remove('hidden'); if (!uploadedImageB64) loadSampleImage('corgi'); });
  
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => { if(e.target.files[0]) handleImageFile(e.target.files[0]); });
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); if(e.dataTransfer.files[0]) handleImageFile(e.dataTransfer.files[0]); });

  sampleItems.forEach(item => {
    item.addEventListener('click', () => {
      sampleItems.forEach(i => i.classList.remove('selected'));
      item.classList.add('selected');
      loadSampleImage(item.dataset.sample);
    });
  });
}

function handleImageFile(file) {
  const r = new FileReader();
  r.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas'); c.width = 224; c.height = 224;
      c.getContext('2d').drawImage(img, 0, 0, 224, 224);
      uploadedImageB64 = c.toDataURL('image/png');
      viewOriginal.src = uploadedImageB64;
    };
    img.src = e.target.result;
  };
  r.readAsDataURL(file);
}

function loadSampleImage(name) {
  const c = document.createElement('canvas'); c.width = 224; c.height = 224;
  const t = c.getContext('2d');
  t.fillStyle = '#1e293b'; t.fillRect(0, 0, 224, 224);
  if (name === 'corgi') {
    t.fillStyle = '#d97706'; t.beginPath(); t.arc(112, 120, 50, 0, Math.PI*2); t.fill();
    t.beginPath(); t.moveTo(80, 80); t.lineTo(65, 40); t.lineTo(95, 65); t.fill();
    t.beginPath(); t.moveTo(144, 80); t.lineTo(159, 40); t.lineTo(129, 65); t.fill();
    t.fillStyle = '#000000'; t.beginPath(); t.arc(92, 110, 6, 0, Math.PI*2); t.fill();
    t.beginPath(); t.arc(132, 110, 6, 0, Math.PI*2); t.fill();
    t.beginPath(); t.arc(112, 130, 8, 0, Math.PI*2); t.fill();
  } else if (name === 'zebra') {
    t.fillStyle = '#ffffff'; t.fillRect(0, 0, 224, 224); t.fillStyle = '#000000';
    for (let i = 0; i < 224; i += 30) {
      t.beginPath(); t.moveTo(i, 0); t.lineTo(i+15, 0); t.lineTo(i-10, 224); t.lineTo(i-25, 224); t.fill();
    }
  } else if (name === 'goldfish') {
    t.fillStyle = '#ea580c'; t.beginPath(); t.ellipse(112, 112, 60, 35, 0, 0, Math.PI*2); t.fill();
    t.beginPath(); t.moveTo(52, 112); t.lineTo(15, 80); t.lineTo(30, 112); t.lineTo(15, 144); t.fill();
    t.fillStyle = '#ffffff'; t.beginPath(); t.arc(142, 105, 8, 0, Math.PI*2); t.fill();
    t.fillStyle = '#000000'; t.beginPath(); t.arc(144, 105, 4, 0, Math.PI*2); t.fill();
  } else if (name === 'daisy') {
    t.fillStyle = '#ffffff';
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI)/4; t.beginPath(); t.ellipse(112 + Math.cos(a)*50, 112 + Math.sin(a)*50, 45, 15, a, 0, Math.PI*2); t.fill();
    }
    t.fillStyle = '#eab308'; t.beginPath(); t.arc(112, 112, 30, 0, Math.PI*2); t.fill();
  }
  uploadedImageB64 = c.toDataURL('image/png');
  viewOriginal.src = uploadedImageB64;
}

function renderPredictions(preds, container) {
  container.innerHTML = '';
  if (!preds || preds.length === 0) { container.innerHTML = '<p class="placeholder-text">No predictions</p>'; return; }
  preds.forEach(p => {
    const pct = (p.confidence * 100).toFixed(1);
    const div = document.createElement('div');
    div.className = 'prediction-bar-container';
    div.innerHTML = `<div class="prediction-label" title="${p.label}">${p.label}</div>
      <div class="prediction-progress-bg"><div class="prediction-progress-fill" style="width: ${pct}%;"></div></div>
      <div class="prediction-value">${pct}%</div>`;
    container.appendChild(div);
  });
}

async function runSimulation() {
  const imgPayload = activeModel === 'mnist' ? canvas.toDataURL('image/png') : uploadedImageB64;
  if (!imgPayload) { alert('Draw a digit or select an image.'); return; }

  btnSimulate.textContent = 'Simulating...'; btnSimulate.disabled = true;
  
  const payload = {
    image: imgPayload,
    model_type: activeModel,
    attack_type: attackTypeSelect.value,
    epsilon: parseFloat(epsilonRange.value),
    alpha: parseFloat(alphaRange.value),
    iterations: parseInt(iterationsRange.value),
    defense_type: defenseTypeSelect.value,
    defense_param: parseFloat(defenseParamRange.value)
  };

  try {
    const res = await fetch(`${API_BASE_URL}/api/attack`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    });
    if(!res.ok) throw new Error();
    const data = await res.json();
    
    viewOriginal.src = imgPayload;
    viewAdversarial.src = data.adversarial_image;
    viewPerturbation.src = data.perturbation_image;
    viewDefended.src = data.defended_image;

    renderPredictions(data.original_predictions, predsOriginal);
    renderPredictions(data.adversarial_predictions, predsAdversarial);
    renderPredictions(data.defended_predictions, predsDefended);

    const orig = data.original_predictions[0];
    const adv = data.adversarial_predictions[0];
    const def = data.defended_predictions[0];
    const success = orig.label !== adv.label;

    if (payload.attack_type === 'none') {
      metricAttackSuccess.textContent = 'No Attack'; metricAttackSuccess.className = 'metric-value status-neutral';
    } else if (success) {
      metricAttackSuccess.textContent = 'Successful'; metricAttackSuccess.className = 'metric-value status-danger';
    } else {
      metricAttackSuccess.textContent = 'Resisted'; metricAttackSuccess.className = 'metric-value status-safe';
    }

    metricPerturbationDist.textContent = payload.attack_type === 'none' ? '0.00' : payload.epsilon.toFixed(2);

    if (payload.defense_type === 'none') {
      metricDefenseStatus.textContent = 'N/A'; metricDefenseStatus.className = 'metric-value status-neutral';
    } else {
      if (!success) {
        metricDefenseStatus.textContent = 'Unnecessary'; metricDefenseStatus.className = 'metric-value status-neutral';
      } else if (def.label === orig.label) {
        metricDefenseStatus.textContent = 'Restored'; metricDefenseStatus.className = 'metric-value status-safe';
      } else {
        metricDefenseStatus.textContent = 'Bypassed'; metricDefenseStatus.className = 'metric-value status-danger';
      }
    }
  } catch {
    alert('Communication error with backend.');
  } finally {
    btnSimulate.textContent = 'Run Simulation'; btnSimulate.disabled = false;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  checkBackendStatus(); initCanvas(); initSliders(); initUploadPlayground();
  canvasClearBtn.addEventListener('click', clearCanvas);
  btnSimulate.addEventListener('click', runSimulation);
});
"""

# ==============================================================================
# 6. ROUTE ENDPOINTS
# ==============================================================================

@app.route('/', methods=['GET'])
def serve_index():
    return Response(HTML_CONTENT, mimetype='text/html')

@app.route('/styles.css', methods=['GET'])
def serve_styles():
    return Response(CSS_CONTENT, mimetype='text/css')

@app.route('/app.js', methods=['GET'])
def serve_js():
    return Response(JS_CONTENT, mimetype='application/javascript')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "online",
        "mnist_trained": os.path.exists("backend/data/mnist_toy.pth")
    })

@app.route('/api/attack', methods=['POST'])
def run_simulation():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data"}), 400
            
        model_type = data.get('model_type', 'mnist')
        attack_type = data.get('attack_type', 'fgsm')
        epsilon = float(data.get('epsilon', 0.05))
        iterations = int(data.get('iterations', 10))
        alpha = float(data.get('alpha', 0.01))
        defense_type = data.get('defense_type', 'none')
        defense_param = data.get('defense_param', 3)
        
        is_mnist = (model_type == 'mnist')
        model = mnist_model if is_mnist else imagenet_model
        original_tensor = base64_to_tensor(data['image'], is_mnist)
        
        # Original prediction
        with torch.no_grad():
            orig_logits = forward_wrapper(model, original_tensor, is_mnist)
            orig_preds = get_top_predictions(orig_logits, is_mnist)
        target_label = torch.argmax(orig_logits, dim=1)
        
        # Compute attack
        adversarial_tensor = original_tensor.clone().detach()
        model_fn = lambda x: forward_wrapper(model, x, is_mnist)
        
        if attack_type == 'fgsm':
            # Run using self-contained model sign gradients
            # We track gradient on a detached copy to support wrap structures
            img_var = original_tensor.clone().detach()
            img_var.requires_grad = True
            output = model_fn(img_var)
            loss = F.cross_entropy(output, target_label)
            model.zero_grad()
            loss.backward()
            grad = img_var.grad.data
            adversarial_tensor = torch.clamp(original_tensor + epsilon * grad.sign(), 0.0, 1.0).detach()
        elif attack_type == 'pgd':
            # PGD loop in-line
            adversarial_tensor = original_tensor.clone().detach()
            for _ in range(iterations):
                img_var = adversarial_tensor.clone().detach()
                img_var.requires_grad = True
                output = model_fn(img_var)
                loss = F.cross_entropy(output, target_label)
                model.zero_grad()
                loss.backward()
                grad = img_var.grad.data
                perturbed = adversarial_tensor + alpha * grad.sign()
                eta = torch.clamp(perturbed - original_tensor, -epsilon, epsilon)
                adversarial_tensor = torch.clamp(original_tensor + eta, 0.0, 1.0).detach()
        elif attack_type == 'patch':
            adversarial_tensor = patch_attack(original_tensor, max(0.05, min(0.5, epsilon)))
            
        # Adversarial predictions
        with torch.no_grad():
            adv_logits = model_fn(adversarial_tensor)
            adv_preds = get_top_predictions(adv_logits, is_mnist)
            
        # Scaled Perturbation noise for UI mapping
        perturbation_tensor = adversarial_tensor - original_tensor
        if perturbation_tensor.abs().max() > 0:
            abs_pert = perturbation_tensor.abs()
            scaled_pert = abs_pert / abs_pert.max()
        else:
            scaled_pert = torch.zeros_like(original_tensor)
            
        # Apply defense
        defended_tensor = adversarial_tensor.clone().detach()
        if defense_type == 'jpeg':
            defended_tensor = jpeg_compression_defense(adversarial_tensor, int(defense_param))
        elif defense_type == 'smoothing':
            k_size = int(defense_param)
            if k_size % 2 == 0: k_size += 1
            defended_tensor = spatial_smoothing_defense(adversarial_tensor, k_size, method='gaussian')
        elif defense_type == 'bit_reduction':
            defended_tensor = bit_depth_reduction_defense(adversarial_tensor, int(defense_param))
            
        # Defended predictions
        with torch.no_grad():
            def_logits = model_fn(defended_tensor)
            def_preds = get_top_predictions(def_logits, is_mnist)
            
        # Encode to return base64 data
        return jsonify({
            "original_predictions": orig_preds,
            "adversarial_predictions": adv_preds,
            "defended_predictions": def_preds,
            "adversarial_image": tensor_to_base64(adversarial_tensor),
            "perturbation_image": tensor_to_base64(scaled_pert),
            "defended_image": tensor_to_base64(defended_tensor)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting All-In-One Simulator Server...")
    app.run(host='127.0.0.1', port=5000, debug=True)
