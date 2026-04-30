"""
Visualization utilities cho Hyperspectral SR — vẽ biểu đồ cho báo cáo và paper.

Cung cấp 5 loại visualization:
  - Spectral curves   : so sánh phổ LR/SR/GT tại một pixel cụ thể
  - RGB comparison    : so sánh ảnh RGB (band 25/15/5) LR/SR/GT side-by-side
  - Attention maps    : hiển thị spatial và spectral attention weights
  - Training curves   : vẽ loss + PSNR/SSIM/SAM theo epoch
  - Metrics bar chart : so sánh nhiều model trên cùng trục

QUAN TRỌNG:
  - Tất cả hàm nhận numpy array [C,H,W] hoặc torch tensor — tự convert
  - save_path=None → hiển thị interactive; save_path có path → lưu PNG (300 dpi)
  - plot_spectral_curves giả định upscale=4 khi tính LR pixel coords
  - Dùng bởi evaluate.py, plot_training_log.py, và các script trong visualize/
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_spectral_curves(lr, sr, hr, pixel_coords, save_path=None):
    """Vẽ spectral signatures tại một pixel — so sánh LR/SR/GT.

    Trục X = wavelength (nm, ước tính 400-700nm cho 31 bands CAVE).
    LR pixel coords được tính bằng cách chia 4 (giả định upscale=4).

    Args:
        lr: Array/tensor [C, H, W] — low-resolution image.
        sr: Array/tensor [C, H, W] — super-resolved image.
        hr: Array/tensor [C, H, W] — ground truth high-resolution.
        pixel_coords: Tuple (y, x) trong không gian HR.
        save_path: Đường dẫn PNG; None thì show interactive.

    Returns:
        None
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(lr):
        lr = lr.cpu().numpy()
    if torch.is_tensor(sr):
        sr = sr.cpu().numpy()
    if torch.is_tensor(hr):
        hr = hr.cpu().numpy()
    
    y, x = pixel_coords
    
    # Extract spectral signatures
    lr_spectrum = lr[:, y // 4, x // 4]  # Assuming 4x upscale
    sr_spectrum = sr[:, y, x]
    hr_spectrum = hr[:, y, x]
    
    # Wavelengths (approximate for CAVE dataset)
    # CAVE: 400-700nm, 31 bands
    wavelengths = np.linspace(400, 700, len(hr_spectrum))
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(wavelengths, hr_spectrum, 'g-', linewidth=2, label='Ground Truth (HR)', alpha=0.8)
    plt.plot(wavelengths, sr_spectrum, 'r--', linewidth=2, label='Super-Resolved (SR)', alpha=0.8)
    plt.plot(wavelengths, lr_spectrum, 'b:', linewidth=2, label='Low Resolution (LR)', alpha=0.6)
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Reflectance', fontsize=12)
    plt.title(f'Spectral Signature at Pixel ({y}, {x})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectral curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_rgb_comparison(lr, sr, hr, save_path=None):
    """Vẽ false-color RGB so sánh LR (bicubic up) / SR / GT side-by-side.

    Dùng bands 25/15/5 làm R/G/B (approximate red/green/blue cho CAVE 31 bands).
    LR được upsample bằng scipy zoom order=1 để cùng kích thước với SR/HR.

    Args:
        lr: Array/tensor [C, H, W] (C >= 26) — low-resolution image.
        sr: Array/tensor [C, H, W] — super-resolved image.
        hr: Array/tensor [C, H, W] — ground truth high-resolution.
        save_path: Đường dẫn PNG; None thì show interactive.

    Returns:
        None
    """
    # Convert to numpy
    if torch.is_tensor(lr):
        lr = lr.cpu().numpy()
    if torch.is_tensor(sr):
        sr = sr.cpu().numpy()
    if torch.is_tensor(hr):
        hr = hr.cpu().numpy()
    
    # Extract RGB bands
    def to_rgb(img):
        """Trích bands 25/15/5 thành RGB [H,W,3], clip về [0,1]."""
        r = img[25, :, :]
        g = img[15, :, :]
        b = img[5, :, :]
        rgb = np.stack([r, g, b], axis=2)
        return np.clip(rgb, 0, 1)
    
    lr_rgb = to_rgb(lr)
    sr_rgb = to_rgb(sr)
    hr_rgb = to_rgb(hr)
    
    # Upsample LR for visualization
    from scipy.ndimage import zoom
    scale = sr_rgb.shape[0] // lr_rgb.shape[0]
    lr_rgb_upsampled = zoom(lr_rgb, (scale, scale, 1), order=1)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(lr_rgb_upsampled)
    axes[0].set_title('Low Resolution (Bicubic)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(sr_rgb)
    axes[1].set_title('Super-Resolved (Ours)', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(hr_rgb)
    axes[2].set_title('Ground Truth', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved RGB comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_attention_maps(attention_maps, save_path=None):
    """Visualize spatial attention (heatmap) và spectral attention (bar chart).

    Args:
        attention_maps: Dict với keys 'spatial' và/hoặc 'spectral':
            - 'spatial': tensor [B,1,H,W] — spatial importance map.
            - 'spectral': tensor [B,C,1,1] — per-band weight.
        save_path: Đường dẫn PNG; None thì show interactive.

    Returns:
        None
    """
    spatial_att = attention_maps.get('spatial', None)
    spectral_att = attention_maps.get('spectral', None)
    
    if spatial_att is None and spectral_att is None:
        print("No attention maps provided")
        return
    
    # Plot
    n_plots = sum([spatial_att is not None, spectral_att is not None])
    fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 6))
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Spatial attention
    if spatial_att is not None:
        if torch.is_tensor(spatial_att):
            spatial_att = spatial_att.cpu().numpy()
        
        # Take first sample and channel if batched
        if spatial_att.ndim == 4:
            spatial_att = spatial_att[0, 0, :, :]
        
        im = axes[plot_idx].imshow(spatial_att, cmap='jet')
        axes[plot_idx].set_title('Spatial Attention Map', fontsize=14)
        axes[plot_idx].axis('off')
        plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
        plot_idx += 1
    
    # Spectral attention
    if spectral_att is not None:
        if torch.is_tensor(spectral_att):
            spectral_att = spectral_att.cpu().numpy()
        
        # Spectral attention is typically [B, C, 1, 1] or [C]
        if spectral_att.ndim == 4:
            spectral_att = spectral_att[0, :, 0, 0]
        elif spectral_att.ndim == 2:
            spectral_att = spectral_att[0, :]
        
        bands = np.arange(len(spectral_att))
        axes[plot_idx].bar(bands, spectral_att, color='steelblue', alpha=0.7)
        axes[plot_idx].set_xlabel('Spectral Band', fontsize=12)
        axes[plot_idx].set_ylabel('Attention Weight', fontsize=12)
        axes[plot_idx].set_title('Spectral Attention Weights', fontsize=14)
        axes[plot_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention maps to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(train_losses, val_metrics, save_path=None):
    """Plot training loss and validation metrics with correct epoch x-axis.

    Args:
        train_losses: List of per-epoch training loss values.
        val_metrics:  List of dicts with keys 'epoch', 'PSNR', 'SSIM', 'SAM'.
                      The 'epoch' field is used as the x-axis so validation
                      curves align with the training loss plot even when
                      validate_every > 1.
        save_path:    PNG output path. Shows interactively if None.

    Returns:
        None
    """
    # Training x-axis: actual epoch numbers (1-indexed)
    train_epochs = list(range(1, len(train_losses) + 1))

    # Validation x-axis: use the real epoch number stored in each entry
    # Falls back to sequential index if 'epoch' key is missing (old logs)
    val_epochs = [m.get('epoch', (i + 1)) for i, m in enumerate(val_metrics)]

    # Extract val series — only plot metrics present in the data
    val_psnr = [(e, m['PSNR']) for e, m in zip(val_epochs, val_metrics) if 'PSNR' in m]
    val_ssim = [(e, m['SSIM']) for e, m in zip(val_epochs, val_metrics) if 'SSIM' in m]
    val_sam  = [(e, m['SAM'])  for e, m in zip(val_epochs, val_metrics) if 'SAM'  in m]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ── Training loss ──────────────────────────────────────────────────
    axes[0, 0].plot(train_epochs, train_losses, 'b-', linewidth=1.2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14)
    axes[0, 0].set_xlim(1, max(train_epochs))
    axes[0, 0].grid(True, alpha=0.3)

    # ── Validation PSNR ───────────────────────────────────────────────
    if val_psnr:
        xe, ye = zip(*val_psnr)
        axes[0, 1].plot(xe, ye, 'g-', linewidth=1.5, marker='o', markersize=4)
        axes[0, 1].set_xlim(1, max(train_epochs))
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0, 1].set_title('Validation PSNR', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # ── Validation SSIM ───────────────────────────────────────────────
    if val_ssim:
        xe, ye = zip(*val_ssim)
        axes[1, 0].plot(xe, ye, 'r-', linewidth=1.5, marker='s', markersize=4)
        axes[1, 0].set_xlim(1, max(train_epochs))
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('SSIM', fontsize=12)
    axes[1, 0].set_title('Validation SSIM', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # ── Validation SAM ────────────────────────────────────────────────
    if val_sam:
        xe, ye = zip(*val_sam)
        axes[1, 1].plot(xe, ye, color='orange', linewidth=1.5,
                        marker='^', markersize=4)
        axes[1, 1].set_xlim(1, max(train_epochs))
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('SAM (degrees)', fontsize=12)
    axes[1, 1].set_title('Validation SAM', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_metrics_comparison(models_metrics, save_path=None):
    """Vẽ bar chart so sánh PSNR/SSIM/SAM/ERGAS giữa nhiều model.

    Args:
        models_metrics: Dict mapping model_name → {'PSNR': float, 'SSIM': float, 'SAM': float, 'ERGAS': float}.
        save_path: Đường dẫn PNG; None thì show interactive.

    Returns:
        None
    """
    model_names = list(models_metrics.keys())
    metrics_names = ['PSNR', 'SSIM', 'SAM', 'ERGAS']
    
    # Prepare data
    data = {metric: [models_metrics[model][metric] for model in model_names]
            for metric in metrics_names}
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    x = np.arange(len(model_names))
    width = 0.6
    
    for i, metric in enumerate(metrics_names):
        values = data[metric]
        
        bars = axes[i].bar(x, values, width, color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{values[j]:.2f}',
                        ha='center', va='bottom', fontsize=10)
        
        axes[i].set_xlabel('Model', fontsize=12)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].set_title(f'{metric} Comparison', fontsize=14)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


# Test code
if __name__ == '__main__':
    print("Testing visualization utilities...")
    print("="*70)
    
    # Create dummy data
    lr = torch.rand(31, 32, 32)
    sr = torch.rand(31, 128, 128)
    hr = torch.rand(31, 128, 128)
    
    # Test spectral curves
    print("\n1. Testing spectral curves plot...")
    plot_spectral_curves(lr, sr, hr, pixel_coords=(64, 64), 
                        save_path='./test_spectral.png')
    
    # Test RGB comparison
    print("\n2. Testing RGB comparison plot...")
    plot_rgb_comparison(lr, sr, hr, save_path='./test_rgb.png')
    
    # Test metrics comparison
    print("\n3. Testing metrics comparison plot...")
    models_metrics = {
        'ESSA (Baseline)': {'PSNR': 32.5, 'SSIM': 0.92, 'SAM': 3.2, 'ERGAS': 2.5},
        'ESSA-SSAM (Ours)': {'PSNR': 34.2, 'SSIM': 0.95, 'SAM': 2.8, 'ERGAS': 2.1},
    }
    plot_metrics_comparison(models_metrics, save_path='./test_comparison.png')
    
    print("\n" + "="*70)
    print("✅ Visualization tests completed!")
    print("Check generated PNG files in current directory")