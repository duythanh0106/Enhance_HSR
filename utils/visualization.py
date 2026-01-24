"""
Visualization utilities cho Hyperspectral SR
Dùng cho báo cáo khóa luận và paper
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns


def plot_spectral_curves(lr, sr, hr, pixel_coords, save_path=None):
    """
    Plot spectral signatures của một pixel
    So sánh LR, SR (predicted), và HR (ground truth)
    
    Args:
        lr: [C, H, W] - Low resolution
        sr: [C, H, W] - Super-resolved
        hr: [C, H, W] - High resolution ground truth
        pixel_coords: (y, x) - Pixel location to plot
        save_path: Path to save figure
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
    """
    Plot RGB comparison của LR, SR, HR
    Dùng bands 25 (R), 15 (G), 5 (B) cho CAVE
    
    Args:
        lr: [C, H, W]
        sr: [C, H, W]
        hr: [C, H, W]
        save_path: Path to save figure
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
    """
    Visualize attention maps từ SSAM module
    
    Args:
        attention_maps: Dictionary chứa spatial và spectral attention
        save_path: Path to save figure
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
    """
    Plot training curves: loss và metrics over epochs
    
    Args:
        train_losses: List of training losses
        val_metrics: List of dicts containing validation metrics
        save_path: Path to save figure
    """
    epochs = range(1, len(train_losses) + 1)
    val_epochs = range(1, len(val_metrics) + 1)
    
    # Extract metrics
    val_psnr = [m['PSNR'] for m in val_metrics]
    val_ssim = [m['SSIM'] for m in val_metrics]
    val_sam = [m['SAM'] for m in val_metrics]
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation PSNR
    axes[0, 1].plot(val_epochs, val_psnr, 'g-', linewidth=2, marker='o')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0, 1].set_title('Validation PSNR', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Validation SSIM
    axes[1, 0].plot(val_epochs, val_ssim, 'r-', linewidth=2, marker='s')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('SSIM', fontsize=12)
    axes[1, 0].set_title('Validation SSIM', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation SAM
    axes[1, 1].plot(val_epochs, val_sam, 'orange', linewidth=2, marker='^')
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
    """
    Plot bar chart so sánh metrics của nhiều models
    Dùng cho ablation study và comparison với baselines
    
    Args:
        models_metrics: Dict {model_name: {metric: value}}
        save_path: Path to save figure
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