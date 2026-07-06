import torch
import torch.nn as nn

def fgsm_attack(model, image, label, epsilon):
    """
    Computes Fast Gradient Sign Method adversarial perturbation.
    """
    # Clone the input image and enable gradient tracking
    perturbed_image = image.clone().detach()
    perturbed_image.requires_grad = True
    
    # Forward pass through the model
    output = model(perturbed_image)
    loss = nn.CrossEntropyLoss()(output, label)
    
    # Zero all existing gradients
    model.zero_grad()
    
    # Backward pass to calculate gradients
    loss.backward()
    
    # Collect gradients of the input image
    data_grad = perturbed_image.grad.data
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = perturbed_image + epsilon * data_grad.sign()
    
    # Adding clipping to maintain the range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0.0, 1.0)
    
    # Return the perturbed image
    return perturbed_image.detach()

def pgd_attack(model, image, label, epsilon, alpha=0.01, iterations=10):
    """
    Computes Projected Gradient Descent (iterative FGSM with L_infinity projection).
    """
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()
    
    # Step-by-step optimization
    for _ in range(iterations):
        perturbed_image.requires_grad = True
        
        # Forward pass
        output = model(perturbed_image)
        loss = nn.CrossEntropyLoss()(output, label)
        
        # Zero gradients
        model.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Take a step in the direction of the gradient sign
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        
        # Projection step: project back into L_infinity epsilon-ball of the original image
        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(original_image + eta, min=0.0, max=1.0).detach()
        
    return perturbed_image

def patch_attack(image, patch_size_pct=0.25):
    """
    Simulates a physical adversarial patch by setting a square region to a solid checkerboard or gray color.
    Doesn't require gradients, making it a great contrast to gradient-based attacks.
    """
    perturbed_image = image.clone().detach()
    
    # Retrieve dimensions (channels, height, width)
    if len(image.shape) == 4:
        # Batch dimension present: [1, C, H, W]
        c, h, w = image.shape[1], image.shape[2], image.shape[3]
        patch_h = int(h * patch_size_pct)
        patch_w = int(w * patch_size_pct)
        
        # Center the patch
        y_start = (h - patch_h) // 2
        x_start = (w - patch_w) // 2
        
        # Draw a custom bright pattern (e.g., magenta patch or checkerboard)
        # Channels: 0 is Red (or grayscale if MNIST), 1 is Green, 2 is Blue
        if c == 1:
            perturbed_image[0, 0, y_start:y_start+patch_h, x_start:x_start+patch_w] = 1.0
        else:
            perturbed_image[0, 0, y_start:y_start+patch_h, x_start:x_start+patch_w] = 1.0  # R
            perturbed_image[0, 1, y_start:y_start+patch_h, x_start:x_start+patch_w] = 0.0  # G
            perturbed_image[0, 2, y_start:y_start+patch_h, x_start:x_start+patch_w] = 1.0  # B
            
    else:
        # Shape: [C, H, W]
        c, h, w = image.shape[0], image.shape[1], image.shape[2]
        patch_h = int(h * patch_size_pct)
        patch_w = int(w * patch_size_pct)
        
        y_start = (h - patch_h) // 2
        x_start = (w - patch_w) // 2
        
        if c == 1:
            perturbed_image[0, y_start:y_start+patch_h, x_start:x_start+patch_w] = 1.0
        else:
            perturbed_image[0, y_start:y_start+patch_h, x_start:x_start+patch_w] = 1.0
            perturbed_image[1, y_start:y_start+patch_h, x_start:x_start+patch_w] = 0.0
            perturbed_image[2, y_start:y_start+patch_h, x_start:x_start+patch_w] = 1.0
            
    return perturbed_image
