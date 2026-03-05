"""
Metrics for Hyperspectral Image Super-Resolution
Bao gồm: PSNR, SSIM, SAM, ERGAS - Các metrics quan trọng cho khóa luận
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


def calculate_psnr(img1, img2, data_range=1.0):
    """Execute `calculate_psnr`.

    Args:
        img1: Input parameter `img1`.
        img2: Input parameter `img2`.
        data_range: Input parameter `data_range`.

    Returns:
        Any: Output produced by this function.
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, data_range=1.0):
    """Execute `calculate_ssim`.

    Args:
        img1: Input parameter `img1`.
        img2: Input parameter `img2`.
        window_size: Input parameter `window_size`.
        data_range: Input parameter `data_range`.

    Returns:
        Any: Output produced by this function.
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / (2*sigma**2)) 
                          for x in range(window_size)])
    gauss = gauss / gauss.sum()
    
    # Create 2D Gaussian kernel
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(img1.size(1), 1, window_size, window_size).contiguous()
    window = window.to(img1.device)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_sam(img1, img2, eps=1e-8):
    """Execute `calculate_sam`.

    Args:
        img1: Input parameter `img1`.
        img2: Input parameter `img2`.
        eps: Input parameter `eps`.

    Returns:
        Any: Output produced by this function.
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # Reshape to [B, C, H*W]
    B, C, H, W = img1.shape
    img1_flat = img1.reshape(B, C, -1)
    img2_flat = img2.reshape(B, C, -1)
    
    # Calculate dot product
    dot_product = torch.sum(img1_flat * img2_flat, dim=1)
    
    # Calculate norms
    norm1 = torch.sqrt(torch.sum(img1_flat ** 2, dim=1))
    norm2 = torch.sqrt(torch.sum(img2_flat ** 2, dim=1))
    
    # Calculate cosine similarity
    cos_theta = dot_product / (norm1 * norm2 + eps)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # Calculate angle in radians, then convert to degrees
    sam_rad = torch.acos(cos_theta)
    sam_deg = torch.mean(sam_rad) * 180.0 / math.pi
    
    return sam_deg.item()


def calculate_ergas(img1, img2, scale=4):
    """Execute `calculate_ergas`.

    Args:
        img1: Input parameter `img1`.
        img2: Input parameter `img2`.
        scale: Input parameter `scale`.

    Returns:
        Any: Output produced by this function.
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    B, C, H, W = img1.shape
    
    sum_squared_relative_error = 0.0
    
    for i in range(C):
        # Get channel i
        ref_band = img1[:, i, :, :]
        pred_band = img2[:, i, :, :]
        
        # Calculate MSE for this band
        mse = torch.mean((ref_band - pred_band) ** 2)
        
        # Calculate mean of reference band
        mean_ref = torch.mean(ref_band)
        
        # Relative squared error
        if mean_ref > 0:
            sum_squared_relative_error += mse / (mean_ref ** 2)
    
    # Calculate ERGAS
    ergas = 100.0 / scale * math.sqrt(sum_squared_relative_error / C)
    
    return float(ergas)


class MetricsCalculator:
    """
    Class để tính tất cả metrics một lần
    Tiện cho evaluation
    """
    
    def __init__(self, data_range=1.0):
        """Initialize the `MetricsCalculator` instance.

        Args:
            data_range: Input parameter `data_range`.

        Returns:
            None: This method initializes state and returns no value.
        """
        self.data_range = data_range
    
    def calculate_all(self, pred, target, scale=4):
        """Execute `calculate_all`.

        Args:
            pred: Input parameter `pred`.
            target: Input parameter `target`.
            scale: Input parameter `scale`.

        Returns:
            Any: Output produced by this function.
        """
        metrics = {}
        
        with torch.no_grad():
            metrics['PSNR'] = calculate_psnr(pred, target, self.data_range)
            metrics['SSIM'] = calculate_ssim(pred, target, data_range=self.data_range)
            metrics['SAM'] = calculate_sam(pred, target)
            metrics['ERGAS'] = calculate_ergas(target, pred, scale)
        
        return metrics
    
    def format_metrics(self, metrics):
        """Execute `format_metrics`.

        Args:
            metrics: Input parameter `metrics`.

        Returns:
            Any: Output produced by this function.
        """
        return (f"PSNR: {metrics['PSNR']:.2f} dB | "
                f"SSIM: {metrics['SSIM']:.4f} | "
                f"SAM: {metrics['SAM']:.4f}° | "
                f"ERGAS: {metrics['ERGAS']:.4f}")


# Test code
if __name__ == '__main__':
    print("Testing Metrics...")
    print("=" * 60)
    
    # Create test images
    B, C, H, W = 2, 31, 128, 128
    
    # Perfect reconstruction (should give best metrics)
    img1 = torch.rand(B, C, H, W)
    img2 = img1.clone()
    
    print("\nTest 1: Perfect reconstruction")
    print("-" * 60)
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    sam = calculate_sam(img1, img2)
    ergas = calculate_ergas(img1, img2, scale=4)
    
    print(f"PSNR: {psnr:.2f} dB (should be inf)")
    print(f"SSIM: {ssim:.4f} (should be 1.0)")
    print(f"SAM: {sam:.4f}° (should be ~0)")
    print(f"ERGAS: {ergas:.4f} (should be ~0)")
    
    # Slightly different images
    img2 = img1 + torch.randn_like(img1) * 0.01
    
    print("\nTest 2: Noisy reconstruction")
    print("-" * 60)
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    sam = calculate_sam(img1, img2)
    ergas = calculate_ergas(img1, img2, scale=4)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"SAM: {sam:.4f}°")
    print(f"ERGAS: {ergas:.4f}")
    
    # Test MetricsCalculator
    print("\nTest 3: MetricsCalculator")
    print("-" * 60)
    calculator = MetricsCalculator(data_range=1.0)
    metrics = calculator.calculate_all(img2, img1, scale=4)
    print(calculator.format_metrics(metrics))
    
    print("\n" + "=" * 60)
    print("✅ All metric tests completed!")