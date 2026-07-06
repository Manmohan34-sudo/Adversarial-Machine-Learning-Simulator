import os
import sys
import torch
import numpy as np

# Adjust path to import backend modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models import get_mnist_model
from attacks import fgsm_attack, pgd_attack, patch_attack
from defenses import jpeg_compression_defense, spatial_smoothing_defense, bit_depth_reduction_defense

def run_tests():
    print("==================================================")
    print("      Adversarial Attack Simulator Test Suite      ")
    print("==================================================")
    
    # 1. Test Model Loading & Initializing
    print("\n[1/5] Initializing custom MNIST network...")
    try:
        model = get_mnist_model("backend/data/mnist_toy_test.pth")
        print("✓ MNIST model initialized successfully.")
    except Exception as e:
        print(f"✗ Failed to initialize MNIST model: {e}")
        return
        
    # 2. Generate a Mock Input (Grayscale Digit image tensor: 1x1x28x28)
    # We will generate a clear vertical stroke (representing a digit '1')
    print("\n[2/5] Creating synthetic digit 1 image...")
    image = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    image[0, 0, 4:24, 14] = 1.0  # Vertical line
    
    # Run original inference
    with torch.no_grad():
        orig_output = model(image)
        orig_pred = torch.argmax(orig_output, dim=1).item()
        orig_conf = torch.softmax(orig_output, dim=1)[0, orig_pred].item()
    print(f"✓ Original Prediction: Class {orig_pred} (Confidence: {orig_conf:.2%})")
    
    # 3. Run Adversarial Attacks
    print("\n[3/5] Simulating Attacks...")
    target_label = torch.tensor([orig_pred], dtype=torch.long)
    
    # Test FGSM
    epsilon = 0.15
    perturbed_fgsm = fgsm_attack(model, image, target_label, epsilon)
    fgsm_output = model(perturbed_fgsm)
    fgsm_pred = torch.argmax(fgsm_output, dim=1).item()
    fgsm_conf = torch.softmax(fgsm_output, dim=1)[0, fgsm_pred].item()
    print(f"✓ FGSM Attack (epsilon={epsilon}):")
    print(f"  - Perturbed Image Range: [{perturbed_fgsm.min().item():.2f}, {perturbed_fgsm.max().item():.2f}]")
    print(f"  - Adversarial Prediction: Class {fgsm_pred} (Confidence: {fgsm_conf:.2%})")
    print(f"  - Status: {'SUCCESSFUL FOOLING' if fgsm_pred != orig_pred else 'FAILED TO FOOL'}")
    
    # Test PGD
    perturbed_pgd = pgd_attack(model, image, target_label, epsilon=0.15, alpha=0.03, iterations=15)
    pgd_output = model(perturbed_pgd)
    pgd_pred = torch.argmax(pgd_output, dim=1).item()
    pgd_conf = torch.softmax(pgd_output, dim=1)[0, pgd_pred].item()
    print(f"✓ PGD Attack (epsilon=0.15, iterations=15):")
    print(f"  - Perturbed Image Range: [{perturbed_pgd.min().item():.2f}, {perturbed_pgd.max().item():.2f}]")
    print(f"  - Adversarial Prediction: Class {pgd_pred} (Confidence: {pgd_conf:.2%})")
    print(f"  - Status: {'SUCCESSFUL FOOLING' if pgd_pred != orig_pred else 'FAILED TO FOOL'}")
    
    # Test Patch
    perturbed_patch = patch_attack(image, patch_size_pct=0.20)
    patch_output = model(perturbed_patch)
    patch_pred = torch.argmax(patch_output, dim=1).item()
    patch_conf = torch.softmax(patch_output, dim=1)[0, patch_pred].item()
    print(f"✓ Patch Attack (size=20%):")
    print(f"  - Adversarial Prediction: Class {patch_pred} (Confidence: {patch_conf:.2%})")
    
    # 4. Test Defenses on FGSM Perturbed Image
    print("\n[4/5] Testing Defenses on FGSM Attack...")
    
    # JPEG Defense
    jpeg_defended = jpeg_compression_defense(perturbed_fgsm, quality=30)
    jpeg_output = model(jpeg_defended)
    jpeg_pred = torch.argmax(jpeg_output, dim=1).item()
    jpeg_conf = torch.softmax(jpeg_output, dim=1)[0, jpeg_pred].item()
    print(f"✓ JPEG Compression Defense (Quality=30):")
    print(f"  - Defended Prediction: Class {jpeg_pred} (Confidence: {jpeg_conf:.2%})")
    print(f"  - Status: {'DEFENDED SUCCESSFULLY' if jpeg_pred == orig_pred else 'DEFENSE BYPASSED'}")
    
    # Spatial Smoothing Defense
    smooth_defended = spatial_smoothing_defense(perturbed_fgsm, kernel_size=3)
    smooth_output = model(smooth_defended)
    smooth_pred = torch.argmax(smooth_output, dim=1).item()
    smooth_conf = torch.softmax(smooth_output, dim=1)[0, smooth_pred].item()
    print(f"✓ Spatial Smoothing Defense (Gaussian 3x3):")
    print(f"  - Defended Prediction: Class {smooth_pred} (Confidence: {smooth_conf:.2%})")
    print(f"  - Status: {'DEFENDED SUCCESSFULLY' if smooth_pred == orig_pred else 'DEFENSE BYPASSED'}")
    
    # Bit Reduction Defense
    bit_defended = bit_depth_reduction_defense(perturbed_fgsm, bits=2)
    bit_output = model(bit_defended)
    bit_pred = torch.argmax(bit_output, dim=1).item()
    bit_conf = torch.softmax(bit_output, dim=1)[0, bit_pred].item()
    print(f"✓ Bit Depth Reduction Defense (2-bits):")
    print(f"  - Defended Prediction: Class {bit_pred} (Confidence: {bit_conf:.2%})")
    print(f"  - Status: {'DEFENDED SUCCESSFULLY' if bit_pred == orig_pred else 'DEFENSE BYPASSED'}")
    
    print("\n[5/5] Verification completed.")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
