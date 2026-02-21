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
import time

from config import ConfigBaseline, ConfigProposed, ConfigSpecTrans, infer_dataset_name
from data.dataset import HyperspectralTestDataset
from models.factory import build_model_from_config, load_state_dict_compat
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam, calculate_ergas
from utils.device import resolve_device


def format_duration(seconds):
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def remove_dir_if_effectively_empty(path):
    if not path:
        return
    if not os.path.isdir(path):
        return
    entries = [e for e in os.listdir(path) if e != '.DS_Store']
    if not entries:
        ds_store = os.path.join(path, '.DS_Store')
        if os.path.exists(ds_store):
            os.remove(ds_store)
        os.rmdir(path)


@torch.no_grad()
def forward_chop(model, x, scale, patch_size=64, overlap=16):
    """
    Sliding window inference để tránh OOM do Spectral Transformer
    Attention có độ phức tạp O((H×W)^2) nên không thể chạy full ảnh lớn.

    Args:
        model: SR model
        x: LR image tensor (1, C, H, W)
        scale: Upscale factor
        patch_size: kích thước patch LR
        overlap: vùng overlap giữa patch để tránh seam

    Returns:
        sr: SR full image (1, C, H*scale, W*scale)
    """

    b, c, h, w = x.size()
    stride = patch_size - overlap

    h_idx = list(range(0, h - patch_size, stride)) + [h - patch_size]
    w_idx = list(range(0, w - patch_size, stride)) + [w - patch_size]

    sr = torch.zeros(b, c, h * scale, w * scale, device=x.device)
    weight = torch.zeros_like(sr)

    for i in h_idx:
        for j in w_idx:
            patch = x[:, :, i:i+patch_size, j:j+patch_size]
            sr_patch = model(patch)

            h_start = i * scale
            w_start = j * scale
            h_end = h_start + patch_size * scale
            w_end = w_start + patch_size * scale

            sr[:, :, h_start:h_end, w_start:w_end] += sr_patch
            weight[:, :, h_start:h_end, w_start:w_end] += 1

    sr /= weight
    return sr


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

    test_start = time.time()
    pbar = tqdm(dataloader, desc='Testing')
    for idx, (lr, hr, filepath) in enumerate(pbar):

        start_time = time.time()   # ⬅ bắt đầu đo thời gian

        lr = lr.to(device)   # (1, C, H, W)
        hr = hr.to(device)
        
        # --- Sliding Window Inference ---
        with torch.inference_mode():
            sr = forward_chop(model, lr, scale, patch_size=32, overlap=8)

        elapsed = time.time() - start_time   # ⬅ kết thúc đo

        # Crop border (theo paper ESSA)
        if crop_border and scale > 1:
            sr = sr[:, :, scale:-scale, scale:-scale]
            hr = hr[:, :, scale:-scale, scale:-scale]

        # Clamp to [0, 1]   
        sr = sr.clamp(0.0, 1.0)
        hr = hr.clamp(0.0, 1.0)

        # Keep as tensor for metrics
        sr_tensor = sr.squeeze(0).cpu()
        hr_tensor = hr.squeeze(0).cpu()

        # Calculate metrics (using torch tensors)
        psnr = calculate_psnr(sr_tensor, hr_tensor, data_range=1.0)
        ssim = calculate_ssim(sr_tensor, hr_tensor, data_range=1.0)
        sam = calculate_sam(sr_tensor, hr_tensor)
        ergas = calculate_ergas(hr_tensor, sr_tensor, scale=scale)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        sam_list.append(sam)
        ergas_list.append(ergas)

        # Convert to numpy ONLY for saving
        sr_np = sr_tensor.numpy()
        hr_np = hr_tensor.numpy()
        
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
            pbar.write(f"\n{image_name}:")
            pbar.write(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, SAM: {sam:.3f}°, ERGAS: {ergas:.3f}")
            pbar.write(f"  Time: {format_duration(elapsed)} ({elapsed:.2f} seconds)")
        
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

    total_test_time = time.time() - test_start

    # Calculate average metrics
    avg_metrics = {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "SAM": float(np.mean(sam_list)),
        "ERGAS": float(np.mean(ergas_list)),
        "num_images": len(psnr_list),
        "total_inference_time_sec": float(total_test_time),
        "avg_time_per_image_sec": float(total_test_time / max(1, len(psnr_list)))
    }
    
    return avg_metrics, results_per_image


def main():
    script_start = time.time()

    parser = argparse.ArgumentParser(description='Test Hyperspectral SR Model (Full Image)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (best.pth)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--config', type=str, default=None,
                       choices=['baseline', 'proposed', 'spectrans'],
                       help='Config preset (optional, will use checkpoint config if not provided)')
    parser.add_argument(
        '--crop_border',
        dest='crop_border',
        action='store_true',
        help='Crop border pixels (paper-style evaluation)'
    )
    parser.add_argument(
        '--no_crop_border',
        dest='crop_border',
        action='store_false',
        help='Do not crop border pixels'
    )
    parser.set_defaults(crop_border=True)
    parser.add_argument('--save_images', action='store_true',
                       help='Save reconstructed images')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    dataset_name = infer_dataset_name(args.data_root)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
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

    device_pref = config.get('device', 'auto') if isinstance(config, dict) else 'auto'
    device = resolve_device(device_pref)
    print(f"Using device: {device}")
    
    if isinstance(config, dict):
        upscale = config.get('upscale_factor', 4)
        split_seed = config.get('split_seed', 42)
        train_ratio = config.get('train_ratio', 0.8)
        val_ratio = config.get('val_ratio', 0.1)
        test_ratio = config.get('test_ratio', 0.1)
        regenerate_split = config.get('regenerate_split', False)
    else:
        upscale = config['upscale_factor']
        split_seed = 42
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        regenerate_split = False
    
    # Print test info
    print("\n" + "="*70)
    print("TEST CONFIGURATION")
    print("="*70)
    print(f"Model: {config.get('model_name') if isinstance(config, dict) else config['model_name']}")
    print(f"Dataset: {dataset_name}")
    print(f"Data root: {args.data_root}")
    print(f"Upscale: x{upscale}")
    print(f"Crop border: {args.crop_border}")
    print(f"Save images: {args.save_images}")
    print("="*70)
    
    # Build test dataset (FULL IMAGE, FIXED TEST SPLIT)
    test_dataset = HyperspectralTestDataset(
        data_root=args.data_root,
        split='test',  # ⭐ FIXED test split - NEVER seen during training
        upscale=upscale,
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        force_regenerate_split=regenerate_split
    )

    detected_num_bands = test_dataset.num_bands
    config_num_bands = config.get('num_spectral_bands') if isinstance(config, dict) else None
    if config_num_bands != detected_num_bands:
        print(
            f"Updating num_spectral_bands: "
            f"{config_num_bands} -> {detected_num_bands} (auto-detected)"
        )
        config['num_spectral_bands'] = detected_num_bands

    #SAFEGUARD
    if len(test_dataset) == 0:
        raise ValueError(
            f"No test images found!\n"
            f"Data root: {args.data_root}\n"
            f"Dataset: {dataset_name}\n"
            f"Expected split folder: 'test'"
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    print(f"\nTest set: {len(test_dataset)} images")

    # Build model after auto-detecting num_bands from dataset
    model = build_model_from_config(config, num_bands_override=detected_num_bands)
    _, converted_keys = load_state_dict_compat(
        model, checkpoint['model_state_dict'], strict=True
    )
    if converted_keys:
        print(f"Converted {len(converted_keys)} legacy weight tensors for compatibility.")
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    print("✅ Model loaded successfully")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'test_{timestamp}')
    
    save_dir = os.path.join(output_dir, 'images') if args.save_images else None
    try:
        # Run test
        avg_metrics, per_image_results = test_full_image(
            model=model,
            dataloader=test_loader,
            scale=upscale,
            device=device,
            crop_border=args.crop_border,
            save_dir=save_dir
        )

        total_runtime_sec = time.time() - script_start
        
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
        print(f"  Inference Total Time : {format_duration(avg_metrics['total_inference_time_sec'])} ({avg_metrics['total_inference_time_sec']:.2f} seconds)")
        print(f"  Avg Time / Image     : {format_duration(avg_metrics['avg_time_per_image_sec'])} ({avg_metrics['avg_time_per_image_sec']:.2f} seconds)")
        print(f"  Total Runtime        : {format_duration(total_runtime_sec)} ({total_runtime_sec:.2f} seconds)")
        print("="*70)

        # Save results to JSON
        results = {
            'checkpoint': args.checkpoint,
            'dataset': dataset_name,
            'data_root': args.data_root,
            'crop_border': args.crop_border,
            'timestamp': timestamp,
            'total_runtime_sec': float(total_runtime_sec),
            'average_metrics': avg_metrics,
            'per_image_results': per_image_results
        }

        os.makedirs(output_dir, exist_ok=True)
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
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Test images: {avg_metrics['num_images']}\n\n")
            f.write(f"PSNR  : {avg_metrics['PSNR']:.2f} dB\n")
            f.write(f"SSIM  : {avg_metrics['SSIM']:.4f}\n")
            f.write(f"SAM   : {avg_metrics['SAM']:.3f}°\n")
            f.write(f"ERGAS : {avg_metrics['ERGAS']:.3f}\n")
            f.write(f"Inference Total Time : {format_duration(avg_metrics['total_inference_time_sec'])} ({avg_metrics['total_inference_time_sec']:.2f} seconds)\n")
            f.write(f"Avg Time / Image     : {format_duration(avg_metrics['avg_time_per_image_sec'])} ({avg_metrics['avg_time_per_image_sec']:.2f} seconds)\n")
            f.write(f"Total Runtime        : {format_duration(total_runtime_sec)} ({total_runtime_sec:.2f} seconds)\n")
            f.write("="*70 + "\n\n")
            f.write("Per-image results:\n")
            f.write("-"*70 + "\n")
            for result in per_image_results:
                f.write(f"{result['image']:<30} PSNR: {result['PSNR']:.2f}  SSIM: {result['SSIM']:.4f}  SAM: {result['SAM']:.3f}\n")

        print(f"✅ Summary saved to: {summary_path}")
    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")
        remove_dir_if_effectively_empty(save_dir)
        remove_dir_if_effectively_empty(output_dir)
    except Exception:
        remove_dir_if_effectively_empty(save_dir)
        remove_dir_if_effectively_empty(output_dir)
        raise


if __name__ == "__main__":
    main()
