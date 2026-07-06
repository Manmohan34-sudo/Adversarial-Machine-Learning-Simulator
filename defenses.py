import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io

def jpeg_compression_defense(image_tensor, quality=30):
    """
    Applies JPEG compression to the image tensor as a defense.
    quality: JPEG quality parameter (1-100), lower means stronger compression (more denoising).
    """
    # Expects tensor of shape [C, H, W] or [1, C, H, W] in [0, 1]
    is_batched = len(image_tensor.shape) == 4
    tensor = image_tensor[0] if is_batched else image_tensor
    
    # Convert PyTorch tensor to PIL Image
    c, h, w = tensor.shape
    numpy_img = tensor.cpu().numpy()
    
    # Handle single channel (grayscale) vs color
    if c == 1:
        numpy_img = (numpy_img[0] * 255).astype(np.uint8)
        pil_img = Image.fromarray(numpy_img, mode='L')
    else:
        numpy_img = (np.transpose(numpy_img, (1, 2, 0)) * 255).astype(np.uint8)
        pil_img = Image.fromarray(numpy_img, mode='RGB')
        
    # Compress in-memory
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    
    # Reload PIL Image
    compressed_pil = Image.open(buffer)
    compressed_numpy = np.array(compressed_pil).astype(np.float32) / 255.0
    
    # Convert back to PyTorch tensor
    if c == 1:
        # Grayscale
        compressed_tensor = torch.tensor(compressed_numpy).unsqueeze(0)
    else:
        # RGB
        compressed_tensor = torch.tensor(np.transpose(compressed_numpy, (2, 0, 1)))
        
    if is_batched:
        compressed_tensor = compressed_tensor.unsqueeze(0)
        
    return compressed_tensor

def spatial_smoothing_defense(image_tensor, kernel_size=3, method='gaussian'):
    """
    Applies a spatial smoothing filter (Gaussian or Mean blur) to diffuse noise.
    """
    is_batched = len(image_tensor.shape) == 4
    tensor = image_tensor if is_batched else image_tensor.unsqueeze(0)
    c = tensor.shape[1]
    
    if method == 'gaussian':
        # Create a simple 2D Gaussian Kernel
        if kernel_size == 3:
            kernel = torch.tensor([[1., 2., 1.],
                                   [2., 4., 2.],
                                   [1., 2., 1.]], dtype=torch.float32)
        elif kernel_size == 5:
            kernel = torch.tensor([[1.,  4.,  7.,  4., 1.],
                                   [4., 16., 26., 16., 4.],
                                   [7., 26., 41., 26., 7.],
                                   [4., 16., 26., 16., 4.],
                                   [1.,  4.,  7.,  4., 1.]], dtype=torch.float32)
        else:
            kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32)
            
        kernel = kernel / kernel.sum()
    else:
        # Mean/Box filter
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32)
        kernel = kernel / (kernel_size * kernel_size)
        
    # Replicate kernel weight across channels
    # weight shape: [out_channels, in_channels/groups, height, width]
    weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
    
    # Pad to maintain spatial dimensions
    padding = kernel_size // 2
    
    # Apply Conv2D with grouping to process channels independently
    smoothed = F.conv2d(tensor, weight, padding=padding, groups=c)
    
    if not is_batched:
        smoothed = smoothed.squeeze(0)
        
    return torch.clamp(smoothed, 0.0, 1.0)

def bit_depth_reduction_defense(image_tensor, bits=3):
    """
    Reduces the bit depth of the image (quantizes color values).
    Eliminates subtle adversarial pixel modifications.
    """
    levels = 2 ** bits - 1
    quantized = torch.round(image_tensor * levels) / levels
    return torch.clamp(quantized, 0.0, 1.0)
