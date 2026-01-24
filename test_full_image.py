"""
Full-Image Test Script for Hyperspectral Super-Resolution
Paper-style evaluation (no patches, no augmentation, fixed test set)

Usage:
    python test.py --checkpoint ./checkpoints/xxx/best.pth --data_root ./data/CAVE
    python test.py --checkpoint best.pth --config spectrans --save_images
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

from config import Config, ConfigBaseline, ConfigProposed, ConfigSpecTrans
from data.dataset import HyperspectralTestDataset
from models.essa_original import ESSA
from models.essa_improved import ESSA_SSAM
from models.essa_ssam_spectrans import ESSA_SSAM_SpecTrans
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam, calculate_ergas


@torch.no_grad()
def test_full_image(model, dataloader, scale, device, crop_border=True, save_dir=None):
    """
    Test model trên full images (không crop patches)
    
    Args:
        model: Model đã load weights
        dataloader: DataLoader cho test set
        scale: Upscale factor
        device: cuda hoặc cpu
        crop_border: Có crop border hay không (theo paper style)
        save_dir: Nếu không None, save reconstructed images
    
    Returns:
        dict: Metrics averaged over test set
    """
    model.eval()

    psnr_list, ssim_list, sam_list, ergas_list = [], [], [], []
    results_per_image = []

    print("\n" + "="*70)
    print("Testing on Full Images (Paper-style)")
    print("="*70)

    for idx, (lr, hr, filepath) in enumerate(tqdm(dataloader, desc='Testing')):
        lr = lr.to(device)   # (1, C, H, W)
        hr = hr.to(device)
        
        # Forward pass
        sr = model(lr)

        # Crop border (theo paper ESSA)
        if crop_border and scale > 1:
            sr = sr[:, :, scale:-scale, scale:-scale]
            hr = hr[:, :, scale:-scale, scale:-scale]

        # Clamp to [0, 1]
        sr = sr.clamp(0.0, 1.0)
        hr = hr.clamp(0.0, 1.0)

        # Convert to numpy
        sr_np = sr.squeeze(0).cpu().numpy()  # (C, H, W)
        hr_np = hr.squeeze(0).cpu().numpy()

        # Calculate metrics
        psnr = calculate_psnr(sr_np, hr_np, data_range=1.0)
        ssim = calculate_ssim(sr_np, hr_np, data_range=1.0)
        sam = calculate_sam(sr_np, hr_np)
        ergas = calculate_ergas(hr_np, sr_np, scale=scale)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        sam_list.append(sam)
        ergas_list.append(ergas)
        
        # Store per-image results
        image_name = os.path.basename(filepath[0])
        results_per_image.append({
            'image': image_name,
            'PSNR': float(psnr),
            'SSIM': float(ssim),
            'SAM': float(sam),
            'ERGAS': float(ergas)
        })
        
        # Print first few results
        if idx < 5:
            print(f"\n{image_name}:")
            print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, SAM: {sam:.3f}°, ERGAS: {ergas:.3f}")
        
        # Save reconstructed image if requested
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save as .npy
            save_path = os.path.join(save_dir, image_name.replace('.mat', '_SR.npy'))
            np.save(save_path, sr_np)
            
            # Save RGB visualization
            if sr_np.shape[0] >= 31:
                import matplotlib.pyplot as plt
                
                r = sr_np[25, :, :]
                g = sr_np[15, :, :]
                b = sr_np[5, :, :]
                
                rgb = np.stack([r, g, b], axis=2)
                rgb = np.clip(rgb, 0, 1)
                
                rgb_path = os.path.join(save_dir, image_name.replace('.mat', '_SR_RGB.png'))
                plt.imsave(rgb_path, rgb)

    # Calculate average metrics
    avg_metrics = {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "SAM": float(np.mean(sam_list)),
        "ERGAS": float(np.mean(ergas_list)),
        "num_images": len(psnr_list)
    }
    
    return avg_metrics, results_per_image


def build_model(config_or_checkpoint):
    """
    Build model từ config hoặc checkpoint
    
    Args:
        config_or_checkpoint: Config object hoặc checkpoint dict
    
    Returns:
        model: PyTorch model
    """
    # Extract config
    if isinstance(config_or_checkpoint, dict):
        config = config_or_checkpoint
        model_name = config.get('model_name', 'ESSA_SSAM')
        num_bands = config.get('num_spectral_bands', 31)
        feature_dim = config.get('feature_dim', 128)
        upscale = config.get('upscale_factor', 4)
    else:
        model_name = config_or_checkpoint.model_name
        num_bands = config_or_checkpoint.num_spectral_bands
        feature_dim = config_or_checkpoint.feature_dim
        upscale = config_or_checkpoint.upscale_factor
    
    # Build model
    if model_name == "ESSA_Original" or model_name == "ESSA":
        model = ESSA(
            inch=num_bands,
            dim=feature_dim,
            upscale=upscale
        )
    elif model_name == "ESSA_SSAM":
        fusion_mode = config.get('fusion_mode', 'sequential') if isinstance(config_or_checkpoint, dict) else config_or_checkpoint.fusion_mode
        model = ESSA_SSAM(
            inch=num_bands,
            dim=feature_dim,
            upscale=upscale,
            fusion_mode=fusion_mode
        )
    elif model_name == "ESSA_SSAM_SpecTrans":
        fusion_mode = config.get('fusion_mode', 'sequential') if isinstance(config_or_checkpoint, dict) else config_or_checkpoint.fusion_mode
        use_spectrans = config.get('use_spectrans', True) if isinstance(config_or_checkpoint, dict) else config_or_checkpoint.use_spectrans
        spectrans_depth = config.get('spectrans_depth', 2) if isinstance(config_or_checkpoint, dict) else config_or_checkpoint.spectrans_depth
        model = ESSA_SSAM_SpecTrans(
            inch=num_bands,
            dim=feature_dim,
            upscale=upscale,
            fusion_mode=fusion_mode,
            use_spectrans=use_spectrans,
            spectrans_depth=spectrans_depth
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Test Hyperspectral SR Model (Full Image)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (best.pth)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--config', type=str, default=None,
                       choices=['baseline', 'proposed', 'spectrans'],
                       help='Config preset (optional, will use checkpoint config if not provided)')
    parser.add_argument('--dataset_type', type=str, default='CAVE',
                       choices=['CAVE', 'Harvard'],
                       help='Dataset type')
    parser.add_argument('--crop_border', action='store_true', default=True,
                       help='Crop border pixels (paper-style evaluation)')
    parser.add_argument('--save_images', action='store_true',
                       help='Save reconstructed images')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get config from checkpoint or use provided config
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("Using config from checkpoint")
    elif args.config:
        if args.config == 'baseline':
            config_obj = ConfigBaseline()
        elif args.config == 'proposed':
            config_obj = ConfigProposed()
        elif args.config == 'spectrans':
            config_obj = ConfigSpecTrans()
        config = config_obj.__dict__
        print(f"Using config preset: {args.config}")
    else:
        raise ValueError("No config found in checkpoint and no --config provided")
    
    # Override data_root
    if isinstance(config, dict):
        upscale = config.get('upscale_factor', 4)
        num_bands = config.get('num_spectral_bands', 31)
    else:
        upscale = config['upscale_factor']
        num_bands = config['num_spectral_bands']
    
    # Print test info
    print("\n" + "="*70)
    print("TEST CONFIGURATION")
    print("="*70)
    print(f"Model: {config.get('model_name') if isinstance(config, dict) else config['model_name']}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Data root: {args.data_root}")
    print(f"Upscale: x{upscale}")
    print(f"Crop border: {args.crop_border}")
    print(f"Save images: {args.save_images}")
    print("="*70)
    
    # Build model
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    print("✅ Model loaded successfully")
    
    # Build test dataset (FULL IMAGE, FIXED TEST SPLIT)
    test_dataset = HyperspectralTestDataset(
        data_root=args.data_root,
        dataset_type=args.dataset_type,
        split='test',  # ⭐ FIXED test split - NEVER seen during training
        upscale=upscale,
        num_bands=num_bands
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # IMPORTANT: Full image only
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\nTest set: {len(test_dataset)} images")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'test_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    save_dir = os.path.join(output_dir, 'images') if args.save_images else None
    
    # Run test
    avg_metrics, per_image_results = test_full_image(
        model=model,
        dataloader=test_loader,
        scale=upscale,
        device=device,
        crop_border=args.crop_border,
        save_dir=save_dir
    )
    
    # Print results
    print("\n" + "="*70)
    print("📊 FULL-IMAGE TEST RESULTS (Paper-style)")
    print("="*70)
    print(f"Number of test images: {avg_metrics['num_images']}")
    print(f"\nAverage Metrics:")
    print(f"  PSNR  : {avg_metrics['PSNR']:.2f} dB")
    print(f"  SSIM  : {avg_metrics['SSIM']:.4f}")
    print(f"  SAM   : {avg_metrics['SAM']:.3f}°")
    print(f"  ERGAS : {avg_metrics['ERGAS']:.3f}")
    print("="*70)
    
    # Save results to JSON
    results = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset_type,
        'data_root': args.data_root,
        'crop_border': args.crop_border,
        'timestamp': timestamp,
        'average_metrics': avg_metrics,
        'per_image_results': per_image_results
    }
    
    json_path = os.path.join(output_dir, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {json_path}")
    if args.save_images:
        print(f"✅ Images saved to: {save_dir}")
    
    # Save summary table
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FULL-IMAGE TEST RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.dataset_type}\n")
        f.write(f"Test images: {avg_metrics['num_images']}\n\n")
        f.write(f"PSNR  : {avg_metrics['PSNR']:.2f} dB\n")
        f.write(f"SSIM  : {avg_metrics['SSIM']:.4f}\n")
        f.write(f"SAM   : {avg_metrics['SAM']:.3f}°\n")
        f.write(f"ERGAS : {avg_metrics['ERGAS']:.3f}\n")
        f.write("="*70 + "\n\n")
        f.write("Per-image results:\n")
        f.write("-"*70 + "\n")
        for result in per_image_results:
            f.write(f"{result['image']:<30} PSNR: {result['PSNR']:.2f}  SSIM: {result['SSIM']:.4f}  SAM: {result['SAM']:.3f}\n")
    
    print(f"✅ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()