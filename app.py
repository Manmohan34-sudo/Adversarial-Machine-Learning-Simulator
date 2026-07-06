import os
import base64
import io
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

from models import get_mnist_model, get_imagenet_model, get_imagenet_classes
from attacks import fgsm_attack, pgd_attack, patch_attack
from defenses import jpeg_compression_defense, spatial_smoothing_defense, bit_depth_reduction_defense

# Get absolute path to the frontend directory
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
# Enable CORS so frontend can communicate with the backend
CORS(app)

# Load models and classes on startup
print("Initializing models...")
mnist_model = get_mnist_model()
imagenet_model = get_imagenet_model()
imagenet_classes = get_imagenet_classes()

# Constants for ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def forward_wrapper(model, x, is_mnist):
    """
    Wraps the forward pass. Applies ImageNet normalization if needed.
    x is a tensor in range [0, 1]
    """
    if is_mnist:
        return model(x)
    else:
        # Normalize for ImageNet
        x_norm = (x - IMAGENET_MEAN) / IMAGENET_STD
        return model(x_norm)

def base64_to_tensor(b64_string, is_mnist):
    """
    Converts a base64 encoded image string to a PyTorch tensor.
    Resizes image to 28x28 for MNIST and 224x224 for ImageNet.
    """
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
        
    image_bytes = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_bytes))
    
    if is_mnist:
        # Convert to grayscale and resize to 28x28
        image = image.convert("L").resize((28, 28))
        tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 28, 28]
    else:
        # Convert to RGB and resize to 224x224
        image = image.convert("RGB").resize((224, 224))
        tensor = torch.tensor(np.transpose(np.array(image), (2, 0, 1)), dtype=torch.float32) / 255.0
        tensor = tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]
        
    return tensor

def tensor_to_base64(tensor):
    """
    Converts a PyTorch tensor in range [0, 1] to a base64 string.
    """
    # Remove batch dimension if present
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
        
    c, h, w = tensor.shape
    numpy_arr = tensor.cpu().numpy()
    
    if c == 1:
        # Grayscale
        numpy_arr = (np.clip(numpy_arr[0], 0.0, 1.0) * 255.0).astype(np.uint8)
        image = Image.fromarray(numpy_arr, mode="L")
    else:
        # RGB
        numpy_arr = (np.clip(np.transpose(numpy_arr, (1, 2, 0)), 0.0, 1.0) * 255.0).astype(np.uint8)
        image = Image.fromarray(numpy_arr, mode="RGB")
        
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64_bytes = base64.b64encode(buffer.getvalue())
    return "data:image/png;base64," + b64_bytes.decode("utf-8")

def get_top_predictions(logits, is_mnist, top_k=5):
    """
    Extracts top K predictions with label name and confidence value.
    """
    probabilities = F.softmax(logits, dim=1).squeeze(0)
    top_probs, top_indices = torch.topk(probabilities, top_k if not is_mnist else 10)
    
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        idx_str = str(idx.item())
        label = idx_str if is_mnist else imagenet_classes.get(idx_str, f"Index {idx_str}")
        # Clean up labels (e.g. "corgi, Pembroke Welsh Corgi" -> "corgi")
        label = label.split(",")[0].strip()
        predictions.append({
            "label": label,
            "confidence": float(prob.item())
        })
    return predictions

@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')

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
            return jsonify({"error": "No image data provided"}), 400
            
        model_type = data.get('model_type', 'mnist')
        attack_type = data.get('attack_type', 'fgsm')
        epsilon = float(data.get('epsilon', 0.05))
        iterations = int(data.get('iterations', 10))
        alpha = float(data.get('alpha', 0.01))
        
        defense_type = data.get('defense_type', 'none')
        defense_param = data.get('defense_param', 3)
        
        is_mnist = (model_type == 'mnist')
        model = mnist_model if is_mnist else imagenet_model
        
        # 1. Convert input to tensor
        original_tensor = base64_to_tensor(data['image'], is_mnist)
        
        # 2. Get predictions for original image
        with torch.no_grad():
            orig_logits = forward_wrapper(model, original_tensor, is_mnist)
            orig_preds = get_top_predictions(orig_logits, is_mnist)
            
        # Target label for gradient calculation (the model's own predicted class)
        target_label = torch.argmax(orig_logits, dim=1)
        
        # 3. Generate Adversarial Image
        adversarial_tensor = original_tensor.clone().detach()
        if attack_type == 'fgsm':
            # FGSM needs gradient computation
            # Enable gradient tracking on a fresh copy of the image
            input_img = original_tensor.clone().detach()
            input_img.requires_grad = True
            
            output = forward_wrapper(model, input_img, is_mnist)
            loss = F.cross_entropy(output, target_label)
            
            model.zero_grad()
            loss.backward()
            
            grad = input_img.grad.data
            # Let's write the step manually using our normalized forward
            adversarial_tensor = original_tensor + epsilon * grad.sign()
            adversarial_tensor = torch.clamp(adversarial_tensor, 0.0, 1.0).detach()
            
        elif attack_type == 'pgd':
            # PGD iteratively calculates grads
            adversarial_tensor = original_tensor.clone().detach()
            for _ in range(iterations):
                input_img = adversarial_tensor.clone().detach()
                input_img.requires_grad = True
                
                output = forward_wrapper(model, input_img, is_mnist)
                loss = F.cross_entropy(output, target_label)
                
                model.zero_grad()
                loss.backward()
                
                grad = input_img.grad.data
                # Take gradient sign ascent step
                perturbed = adversarial_tensor + alpha * grad.sign()
                # Project back into epsilon ball
                eta = torch.clamp(perturbed - original_tensor, -epsilon, epsilon)
                adversarial_tensor = torch.clamp(original_tensor + eta, 0.0, 1.0).detach()
                
        elif attack_type == 'patch':
            # Patch size pct (epsilon acts as the size ratio here)
            patch_size = max(0.05, min(0.5, epsilon))
            adversarial_tensor = patch_attack(original_tensor, patch_size)
            
        # Get predictions for adversarial image
        with torch.no_grad():
            adv_logits = forward_wrapper(model, adversarial_tensor, is_mnist)
            adv_preds = get_top_predictions(adv_logits, is_mnist)
            
        # Calculate Perturbation (scaled for visibility)
        perturbation_tensor = adversarial_tensor - original_tensor
        # Scale L_inf or absolute values to make the noise visible
        if perturbation_tensor.abs().max() > 0:
            # Shift perturbation to center around 0.5 (gray = no change, bright/dark = change)
            # Or just show the absolute difference normalized
            abs_pert = perturbation_tensor.abs()
            max_val = abs_pert.max()
            scaled_pert = abs_pert / max_val if max_val > 0 else abs_pert
        else:
            scaled_pert = torch.zeros_like(original_tensor)
            
        # 4. Apply Defense preprocessing
        defended_tensor = adversarial_tensor.clone().detach()
        if defense_type == 'jpeg':
            # parameter is quality (e.g. 10 to 100, mapped from parameter slider)
            quality = int(defense_param)
            defended_tensor = jpeg_compression_defense(adversarial_tensor, quality)
        elif defense_type == 'smoothing':
            # parameter is kernel size (3, 5, 7)
            k_size = int(defense_param)
            if k_size % 2 == 0:
                k_size += 1
            defended_tensor = spatial_smoothing_defense(adversarial_tensor, k_size, method='gaussian')
        elif defense_type == 'bit_reduction':
            # parameter is bit depth (1 to 8)
            bits = int(defense_param)
            defended_tensor = bit_depth_reduction_defense(adversarial_tensor, bits)
            
        # Get predictions for defended image
        with torch.no_grad():
            def_logits = forward_wrapper(model, defended_tensor, is_mnist)
            def_preds = get_top_predictions(def_logits, is_mnist)
            
        # 5. Convert all outputs to base64
        adv_b64 = tensor_to_base64(adversarial_tensor)
        pert_b64 = tensor_to_base64(scaled_pert)
        def_b64 = tensor_to_base64(defended_tensor)
        
        return jsonify({
            "original_predictions": orig_preds,
            "adversarial_predictions": adv_preds,
            "defended_predictions": def_preds,
            "adversarial_image": adv_b64,
            "perturbation_image": pert_b64,
            "defended_image": def_b64
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
