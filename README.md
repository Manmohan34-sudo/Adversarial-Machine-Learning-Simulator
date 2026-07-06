n interactive, self-contained toolkit and visual playground built with **PyTorch** and **Flask** to simulate adversarial machine learning attacks (FGSM, PGD, and Patch) and evaluate pre-processing defense filters in real-time.
This project features both a modular structure for local development and a fully **self-contained, single-file version** (`adversarial_simulator_all_in_one.py`) that serves the backend API and the interactive glassmorphism frontend dashboard from a single command.
---
## 🛡️ Key Features
- **Double Model Support**:
  - **MNIST CNN**: Grayscale CNN trained on-the-fly on synthetic shapes for robust offline operation. Supports an interactive drawing canvas.
  - **ImageNet MobileNetV2**: Small pre-trained vision model to attack uploadable real-world photos (Corgi, Zebra, Goldfish, Daisy).
- **Adversarial Attack Algorithms**:
  - **FGSM (Fast Gradient Sign Method)**: One-step gradient sign ascent.
  - **PGD (Projected Gradient Descent)**: Iterative, projected gradient-based attack (strongest local search).
  - **Adversarial Patch**: Simulated physical sticker occlusion.
- **Defensive Preprocessing**:
  - **JPEG Compression**: Disrupts high-frequency noise by dropping compression coefficients.
  - **Spatial Smoothing**: Standard 2D Gaussian/Box blurs to wash out pixel-level offsets.
  - **Bit-depth Reduction**: Drops color resolution to eliminate sub-threshold gradients.
- **Glassmorphic UI**: Side-by-side comparative dashboard displaying predictions, scaled perturbation heatmaps, and defense results.
---
## 🚀 How to Run the All-In-One File
For the ultimate convenience, you can run the entire project (Frontend + Backend + Models) using a **single Python script**:
### Step 1: Install Dependencies
```bash
pip install torch torchvision flask flask-cors Pillow numpy
```
### Step 2: Run the Simulator
```bash
python adversarial_simulator_all_in_one.py
```
### Step 3: Open in Browser
Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```
---
## 🛠️ Folder Structure (Modular Version)
If you prefer to work with separate modules, the repository contains:
```text
├── backend/
│   ├── app.py             # Flask REST API endpoints
│   ├── models.py          # MNIST and ImageNet model configurations
│   ├── attacks.py         # FGSM, PGD, and Patch math functions
│   └── defenses.py        # JPEG, Gaussian smoothing, and quantization filters
├── frontend/
│   ├── index.html         # Main dashboard layout
│   ├── styles.css         # Glassmorphic layout styling
│   └── app.js             # Canvas drawing, file uploads, AJAX requests
├── requirements.txt       # Dependencies
└── test_attacks.py        # CLI verification harness
```
---
## 📐 Mathematical Formulation
### 1. Fast Gradient Sign Method (FGSM)
FGSM calculates the gradient of the loss with respect to the input image pixels and takes a step in the direction of the sign of the gradient to maximize classification error:
\[ x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(\theta, x, y)) \]
### 2. Projected Gradient Descent (PGD)
PGD iteratively computes gradient steps and projects the result back into the $\epsilon$-bounded $L_\infty$ neighborhood around the original image:
\[ x^{t+1} = \text{Clip}_{x, \epsilon} \{ x^t + \alpha \cdot \text{sign}(\nabla_x L(\theta, x^t, y)) \} \]
---
## 📜 License
This project is open-source and licensed under the MIT License.
