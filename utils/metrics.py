"""
Metrics for Hyperspectral Image Super-Resolution — đo chất lượng ảnh SR.

Bao gồm 4 metrics chuẩn cho HSI-SR:
  - PSNR  : Peak Signal-to-Noise Ratio (dB) — higher is better
  - SSIM  : Structural Similarity Index — higher is better, max=1
  - SAM   : Spectral Angle Mapper (degrees) — lower is better, 0° = perfect
  - ERGAS : Erreur Relative Globale Adimensionnelle de Synthèse — lower is better

QUAN TRỌNG:
  - Tất cả hàm expect tensor [B,C,H,W] hoặc [C,H,W] — tự động unsqueeze batch
  - data_range mặc định = 1.0 (ảnh đã normalize về [0,1])
  - SAM và ERGAS là lower-is-better; PSNR và SSIM là higher-is-better
  - MetricsCalculator tính cả 4 metrics một lúc — dùng trong evaluate.py/train.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


def calculate_psnr(img1, img2, data_range=1.0):
    """Tính PSNR (Peak Signal-to-Noise Ratio) giữa hai ảnh — higher is better.

    Args:
        img1: Tensor [B,C,H,W] hoặc [C,H,W], đã normalize về [0,1].
        img2: Tensor cùng shape với img1.
        data_range: Giá trị max của data (mặc định 1.0).

    Returns:
        float: PSNR tính bằng dB; inf nếu hai ảnh giống hệt nhau (MSE=0).
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
    """Tính SSIM (Structural Similarity Index) — higher is better, max=1.

    Dùng Gaussian window 11×11 (chuẩn SSIM paper); tính trên tất cả channels rồi
    lấy trung bình. Chạy trong float32 để tránh AMP precision issues.

    Args:
        img1: Tensor [B,C,H,W] hoặc [C,H,W].
        img2: Tensor cùng shape với img1.
        window_size: Kích thước Gaussian kernel (mặc định 11).
        data_range: Giá trị max (mặc định 1.0).

    Returns:
        float: SSIM trong [0, 1].
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    # AMP-safe: compute SSIM in float32.
    img1 = img1.float()
    img2 = img2.float()
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.tensor(
        [math.exp(-(x - window_size//2)**2 / (2*sigma**2)) for x in range(window_size)],
        dtype=img1.dtype,
        device=img1.device,
    )
    gauss = gauss / gauss.sum()
    
    # Create 2D Gaussian kernel
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(img1.size(1), 1, window_size, window_size).contiguous()
    
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
    """Tính SAM (Spectral Angle Mapper) trung bình — lower is better, 0° = perfect.

    Đo góc giữa các spectral signatures tại mỗi pixel; phản ánh độ trung thực
    phổ quan trọng hơn PSNR cho hyperspectral images.

    Args:
        img1: Tensor [B,C,H,W] hoặc [C,H,W].
        img2: Tensor cùng shape với img1.
        eps: Epsilon tránh div/0 khi chuẩn hóa (mặc định 1e-8).

    Returns:
        float: Góc SAM trung bình (degrees) trên tất cả pixels và batch.
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
    # Keep acos gradient finite near boundaries.
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # Calculate angle in radians, then convert to degrees
    sam_rad = torch.acos(cos_theta)
    sam_deg = torch.mean(sam_rad) * 180.0 / math.pi
    
    return sam_deg.item()


def calculate_ergas(img1, img2, scale=4):
    """Tính ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse) — lower is better.

    Metric tổng hợp đo lỗi tương đối trên toàn bộ spectral bands, có scale
    factor để normalize theo upscale ratio. img1 = GT, img2 = predicted
    (thứ tự quan trọng cho mean_ref calculation).

    Args:
        img1: HR reference tensor [B,C,H,W] hoặc [C,H,W].
        img2: SR predicted tensor (cùng shape với img1).
        scale: Upscale factor để chuẩn hóa (mặc định 4).

    Returns:
        float: ERGAS — lower is better; 0 = perfect reconstruction.
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
        
        # Relative squared error — dùng eps để tránh div/0 và skip band
        # Hard skip (mean_ref<=0) gây bias vì bỏ qua band đó khỏi average
        eps_ref = 1e-8
        sum_squared_relative_error += mse / (mean_ref ** 2 + eps_ref)
    
    # Calculate ERGAS
    ergas = 100.0 / scale * math.sqrt(sum_squared_relative_error / C)
    
    return float(ergas)


class MetricsCalculator:
    """Calculator tổng hợp — tính cả 4 metrics (PSNR/SSIM/SAM/ERGAS) một lần.

    Dùng trong evaluate.py và train.py validate() để tránh lặp code.
    """

    def __init__(self, data_range=1.0):
        """Khởi tạo calculator với data_range (mặc định 1.0 cho ảnh đã normalize)."""
        self.data_range = data_range
    
    def calculate_all(self, pred, target, scale=4):
        """Tính đồng thời PSNR, SSIM, SAM, ERGAS không có gradient.

        Args:
            pred: SR predicted tensor [B,C,H,W].
            target: HR reference tensor [B,C,H,W].
            scale: Upscale factor dùng cho ERGAS (mặc định 4).

        Returns:
            dict: Keys 'PSNR', 'SSIM', 'SAM', 'ERGAS' (float).
        """
        metrics = {}

        with torch.no_grad():
            pred = pred.float()
            target = target.float()
            metrics['PSNR'] = calculate_psnr(pred, target, self.data_range)
            metrics['SSIM'] = calculate_ssim(pred, target, data_range=self.data_range)
            metrics['SAM'] = calculate_sam(pred, target)
            metrics['ERGAS'] = calculate_ergas(target, pred, scale)

        return metrics
    
    def format_metrics(self, metrics):
        """Format dict metrics thành string một dòng để log/print.

        Args:
            metrics: Dict từ calculate_all() với keys PSNR/SSIM/SAM/ERGAS.

        Returns:
            str: Dạng "PSNR: 34.21 dB | SSIM: 0.9234 | SAM: 2.8123° | ERGAS: 2.1045".
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
