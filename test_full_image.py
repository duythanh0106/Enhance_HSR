"""
Full-Image Test Script for Hyperspectral Super-Resolution
Paper-style evaluation (no patches, no augmentation, fixed test set)

Usage:
    python test.py --checkpoint ./checkpoints/xxx/best.pth --data_root ./data/CAVE
    python test.py --checkpoint best.pth --config spectrans --save_images
"""

import os
import argparse
import re
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime
import time

from config import CONFIG_PRESET_CHOICES, build_config, infer_dataset_name
from data.dataset import HyperspectralTestDataset, build_split_kwargs, load_dataset_with_fallback
from models.factory import build_model_from_config, load_state_dict_compat
from utils.inference import forward_chop
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam, calculate_ergas
from utils.device import resolve_device
from utils.time_utils import format_duration


def _choose_rgb_band_indices(num_bands):
    """Select representative band indices for RGB visualization."""
    if num_bands >= 31:
        return 25, 15, 5
    if num_bands >= 3:
        r = int(round((num_bands - 1) * 0.8))
        g = int(round((num_bands - 1) * 0.5))
        b = int(round((num_bands - 1) * 0.2))
        return r, g, b
    return 0, 0, 0


def _save_rgb_png(cube_chw, output_path):
    """Save CHW hyperspectral cube as RGB PNG using representative bands."""
    import matplotlib.pyplot as plt

    r_idx, g_idx, b_idx = _choose_rgb_band_indices(int(cube_chw.shape[0]))
    rgb = np.stack(
        [
            cube_chw[r_idx, :, :],
            cube_chw[g_idx, :, :],
            cube_chw[b_idx, :, :],
        ],
        axis=2,
    )
    # Stretch to [0, 1] for display — data may be in a small range (e.g. global_fixed /65535)
    lo, hi = rgb.min(), rgb.max()
    if hi > lo:
        rgb = (rgb - lo) / (hi - lo)
    else:
        rgb = np.zeros_like(rgb)
    plt.imsave(output_path, np.clip(rgb, 0.0, 1.0))


def _save_band_pngs(cube_chw, band_dir, file_prefix):
    """Save each spectral band as grayscale PNG for quick visual inspection."""
    import matplotlib.pyplot as plt

    os.makedirs(band_dir, exist_ok=True)
    num_bands = int(cube_chw.shape[0])
    pad = max(2, len(str(num_bands)))
    # Use global min/max across all bands for consistent brightness across PNGs
    global_lo = float(cube_chw.min())
    global_hi = float(cube_chw.max())
    for band_idx in range(num_bands):
        band = cube_chw[band_idx, :, :]
        band_path = os.path.join(
            band_dir,
            f"{file_prefix}_band_{band_idx + 1:0{pad}d}.png",
        )
        plt.imsave(band_path, band, cmap="gray", vmin=global_lo, vmax=global_hi)


def _build_safe_sample_names(sample_id, index):
    """Convert dataset sample identifiers into filesystem-safe names."""
    raw_name = str(sample_id)
    normalized = os.path.normpath(raw_name)
    base_name = os.path.basename(normalized.rstrip(os.sep)) or raw_name
    stem = os.path.splitext(base_name)[0] if os.path.splitext(base_name)[0] else base_name
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not safe_stem:
        safe_stem = f"sample_{index:04d}"
    return raw_name, safe_stem


def remove_dir_if_effectively_empty(path):
    """Xóa thư mục path nếu rỗng (bỏ qua .DS_Store); no-op nếu không tồn tại."""
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


def test_full_image(
    model,
    dataloader,
    scale,
    device,
    crop_border=True,
    save_dir=None,
    chop_patch_size=32,
    chop_overlap=8,
    save_band_png=False
):
    """Đánh giá paper-style trên toàn bộ ảnh test (không patch, không augmentation).

    Args:
        model: SR model đã load weights.
        dataloader: DataLoader trả về (lr, hr, filepath) tuples, batch_size=1.
        scale: Upscale factor (ví dụ: 4).
        device: torch.device để chạy inference.
        crop_border: Nếu True, crop scale pixels trên mỗi cạnh trước khi tính metrics.
        save_dir: Thư mục lưu SR/HR/LR images; None để không lưu.
        chop_patch_size: Patch size LR cho sliding-window inference (mặc định 32).
        chop_overlap: Overlap giữa các patches (mặc định 8).
        save_band_png: Nếu True, lưu thêm từng band dạng grayscale PNG.

    Returns:
        tuple: (avg_metrics, results_per_image) — dict metrics trung bình và list
               dict metrics từng ảnh.
    """
    model.eval()

    psnr_list, ssim_list, sam_list, ergas_list, inference_time_list = [], [], [], [], []
    results_per_image = []

    print("\n" + "="*70)
    print("Testing on Full Images (Paper-style)")
    print("="*70)

    test_start = time.time()
    pbar = tqdm(dataloader, desc='Testing')
    for idx, (lr, hr, filepath) in enumerate(pbar):

        # Lấy LR numpy để save ảnh TRƯỚC khi move lên device
        # (tránh vấn đề khi MPS fallback sang CPU giữa chừng)
        lr_np_cpu = lr.squeeze(0).numpy()

        start_time = time.time()   # ⬅ bắt đầu đo thời gian inference

        lr = lr.to(device)   # (1, C, H, W)
        hr = hr.to(device)
        
        # --- Sliding Window Inference ---
        with torch.inference_mode():
            sr = forward_chop(
                model,
                lr,
                scale,
                patch_size=chop_patch_size,
                overlap=chop_overlap
            )

        elapsed = time.time() - start_time   # ⬅ kết thúc đo inference (không bao gồm save)

        # Safety alignment: keep common region if shapes are not identical.
        if sr.shape[-2:] != hr.shape[-2:]:
            min_h = min(sr.shape[-2], hr.shape[-2])
            min_w = min(sr.shape[-1], hr.shape[-1])
            sr = sr[:, :, :min_h, :min_w]
            hr = hr[:, :, :min_h, :min_w]
            pbar.write(
                f"⚠️ Shape mismatch resolved by cropping to common size: "
                f"{min_h}x{min_w}"
            )

        # Store per-image results — phải gán trước crop_border warning dùng image_name
        image_name, safe_image_stem = _build_safe_sample_names(filepath[0], idx)

        # Crop border (theo paper ESSA)
        if crop_border and scale > 1:
            # Crop scale pixels trên mỗi cạnh (paper-style, tránh boundary artifacts)
            # Dùng explicit index thay vì -scale để tránh off-by-one khi scale lớn
            h, w = hr.shape[-2], hr.shape[-1]
            if h > 2 * scale and w > 2 * scale:
                sr = sr[:, :, scale:h - scale, scale:w - scale]
                hr = hr[:, :, scale:h - scale, scale:w - scale]
            else:
                pbar.write(
                    f"⚠️ Skipping crop_border for '{image_name}': "
                    f"image too small ({h}x{w}) for scale={scale} "
                    f"(cần > {2 * scale}x{2 * scale})"
                )

        # Clamp to [0, 1]   
        sr = sr.clamp(0.0, 1.0)
        hr = hr.clamp(0.0, 1.0)

        # Keep as tensor for metrics
        sr_tensor = sr.squeeze(0).cpu()
        hr_tensor = hr.squeeze(0).cpu()

        psnr = calculate_psnr(sr_tensor, hr_tensor, data_range=1.0)
        ssim = calculate_ssim(sr_tensor, hr_tensor, data_range=1.0)
        sam = calculate_sam(sr_tensor, hr_tensor)
        ergas = calculate_ergas(hr_tensor, sr_tensor, scale=scale)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        sam_list.append(sam)
        ergas_list.append(ergas)
        inference_time_list.append(elapsed)

        # Convert to numpy ONLY for saving
        sr_np = sr_tensor.numpy()
        hr_np = hr_tensor.numpy()
        
        # Store per-image results
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
            
            image_save_dir = os.path.join(save_dir, safe_image_stem)
            os.makedirs(image_save_dir, exist_ok=True)

            # Save as .npy
            save_path = os.path.join(image_save_dir, f"{safe_image_stem}_SR.npy")
            np.save(save_path, sr_np)

            # Save RGB visualizations for SR/HR/LR.
            _save_rgb_png(sr_np, os.path.join(image_save_dir, f"{safe_image_stem}_SR_RGB.png"))
            _save_rgb_png(hr_np, os.path.join(image_save_dir, f"{safe_image_stem}_HR_RGB.png"))
            _save_rgb_png(
                lr_np_cpu,
                os.path.join(image_save_dir, f"{safe_image_stem}_LR_RGB.png"),
            )

            # Optional: save all band images as grayscale PNGs.
            if save_band_png:
                _save_band_pngs(
                    cube_chw=hr_np,
                    band_dir=os.path.join(image_save_dir, "bands_hr"),
                    file_prefix=f"{safe_image_stem}_HR",
                )
                _save_band_pngs(
                    cube_chw=sr_np,
                    band_dir=os.path.join(image_save_dir, "bands_sr"),
                    file_prefix=f"{safe_image_stem}_SR",
                )

    total_test_time = time.time() - test_start
    total_inference_time = float(sum(inference_time_list))

    # Calculate average metrics
    avg_metrics = {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "SAM": float(np.mean(sam_list)),
        "ERGAS": float(np.mean(ergas_list)),
        "num_images": len(psnr_list),
        # Wall-clock time: bao gồm cả save images, dataloader, v.v.
        "total_test_time_sec": float(total_test_time),
        "avg_time_per_image_sec": float(total_test_time / max(1, len(psnr_list))),
        # Inference-only time: chỉ tính forward pass (không bao gồm save/load)
        "total_inference_time_sec": total_inference_time,
        "avg_inference_time_per_image_sec": float(total_inference_time / max(1, len(psnr_list))),
    }
    
    return avg_metrics, results_per_image


def main():
    """Parse CLI args, load checkpoint, build test dataset và chạy full-image evaluation."""
    script_start = time.time()

    parser = argparse.ArgumentParser(description='Test Hyperspectral SR Model (Full Image)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (best.pth)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--config', type=str, default=None,
                       help=(
                           'Config preset (optional, dùng checkpoint config nếu không set). '
                           f'Built-in: {", ".join(CONFIG_PRESET_CHOICES)}. '
                           'Dataset presets: cave, harvard, chikusei, pavia (+ _x2/_x4).'
                       ))
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Override device for testing (default: use checkpoint config)'
    )
    parser.add_argument(
        '--chop_patch_size',
        type=int,
        default=64,
        help=(
            'LR patch size for sliding-window inference. '
            'Larger = fewer tiles, less averaging artifact, faster. '
            'For CAVE (LR ~128×128) dùng --chop_patch_size 128 để 1 forward pass. '
            'Giảm xuống nếu OOM.'
        )
    )
    parser.add_argument(
        '--chop_overlap',
        type=int,
        default=16,
        help='Overlap size between LR patches (higher = smoother seams, slower)'
    )
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
    parser.add_argument(
        '--save_band_png',
        action='store_true',
        help='When --save_images is enabled, also save every HR/SR band as grayscale PNG'
    )
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Output directory for results')
    parser.add_argument(
        '--split_seed',
        type=int,
        default=None,
        help='Override split seed used to read/generate split.json'
    )
    parser.add_argument(
        '--regenerate_split',
        action='store_true',
        help='Force regenerate split.json before loading test split'
    )
    parser.add_argument(
        '--normalization_mode',
        type=str,
        default=None,
        choices=['per_image_minmax', 'global_fixed'],
        help='Override normalization mode from checkpoint config'
    )

    args = parser.parse_args()
    if args.chop_patch_size <= 0:
        raise ValueError(f"--chop_patch_size must be > 0, got {args.chop_patch_size}")
    if args.chop_overlap < 0:
        raise ValueError(f"--chop_overlap must be >= 0, got {args.chop_overlap}")
    if args.chop_overlap >= args.chop_patch_size:
        raise ValueError(
            f"--chop_overlap must be < --chop_patch_size "
            f"(got overlap={args.chop_overlap}, patch_size={args.chop_patch_size})"
        )

    dataset_name = infer_dataset_name(args.data_root)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    # weights_only=False vì checkpoint chứa config Python object
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Get config from checkpoint or use provided config
    checkpoint_num_bands = None
    checkpoint_data_root = None
    if 'config' in checkpoint:
        config = checkpoint['config']
        if not isinstance(config, dict):
            config = {
                k: v for k, v in vars(config).items()
                if not k.startswith('_') and not callable(v)
            }
        checkpoint_num_bands = config.get('num_spectral_bands')
        checkpoint_data_root = config.get('data_root')
        print("Using config from checkpoint")
    elif args.config:
        config_obj = build_config(args.config)
        config = config_obj.__dict__
        print(f"Using config preset: {args.config}")
    else:
        raise ValueError("No config found in checkpoint and no --config provided")

    device_pref = args.device if args.device is not None else config.get('device', 'auto')
    device = resolve_device(device_pref)
    print(f"Using device: {device}")
    
    upscale = config.get('upscale_factor', 4)
    effective_split_seed = int(args.split_seed) if args.split_seed is not None else int(config.get('split_seed', 42))
    effective_regenerate_split = bool(config.get('regenerate_split', False)) or bool(args.regenerate_split)
    effective_normalization_mode = args.normalization_mode if args.normalization_mode is not None else str(config.get('normalization_mode', 'global_fixed'))
    effective_normalization_scale = float(config.get('normalization_scale', 65535.0))
    config['split_seed'] = effective_split_seed
    config['regenerate_split'] = effective_regenerate_split
    config['normalization_mode'] = effective_normalization_mode
    config['normalization_scale'] = effective_normalization_scale

    split_kwargs = build_split_kwargs(
        upscale=upscale,
        split_seed=effective_split_seed,
        train_ratio=config.get('train_ratio', 0.8),
        val_ratio=config.get('val_ratio', 0.1),
        test_ratio=config.get('test_ratio', 0.1),
        force_regenerate_split=effective_regenerate_split,
    )
    
    # Print test info
    print("\n" + "="*70)
    print("TEST CONFIGURATION")
    print("="*70)
    print(f"Model: {config.get('model_name')}")
    print(f"Dataset: {dataset_name}")
    print(f"Data root: {args.data_root}")
    print(f"Upscale: x{upscale}")
    print(f"Chop patch size: {args.chop_patch_size}")
    print(f"Chop overlap: {args.chop_overlap}")
    print(f"Crop border: {args.crop_border}")
    print(f"Save images: {args.save_images}")
    print(f"Save band PNG: {args.save_band_png}")
    split_seed_note = " (CLI override)" if args.split_seed is not None else ""
    regenerate_note = " (CLI override)" if args.regenerate_split else ""
    print(f"Split seed: {effective_split_seed}{split_seed_note}")
    print(f"Regenerate split: {effective_regenerate_split}{regenerate_note}")
    norm_note = " (CLI override)" if args.normalization_mode is not None else ""
    print(
        f"Normalization: {effective_normalization_mode}{norm_note} "
        f"(scale={effective_normalization_scale})"
    )
    print("="*70)
    
    test_dataset, _ = load_dataset_with_fallback(
        dataset_cls=HyperspectralTestDataset,
        primary_split='test',
        fallback_split=None,
        data_root=args.data_root,
        log_fn=print,
        normalization_mode=effective_normalization_mode,
        normalization_scale=effective_normalization_scale,
        **split_kwargs,
    )

    detected_num_bands = test_dataset.num_bands
    if checkpoint_num_bands is not None and int(checkpoint_num_bands) != int(detected_num_bands):
        raise ValueError(
            "Checkpoint spectral bands do not match test dataset bands.\n"
            f"  Checkpoint bands: {checkpoint_num_bands}\n"
            f"  Test dataset bands: {detected_num_bands}\n"
            f"  Checkpoint data_root: {checkpoint_data_root}\n"
            f"  Requested data_root: {args.data_root}\n"
            "Use a checkpoint trained on the same spectral band count as the test dataset."
        )

    config_num_bands = config.get('num_spectral_bands')
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

    # Load model weights
    # Lưu ý: nếu train dùng use_ema=True, best.pth lưu model weights TRƯỚC KHI apply EMA
    # (EMA chỉ được apply trong validate()). Checkpoint 'model_state_dict' là EMA-applied
    # weights vì validate() gọi ema.apply_shadow() rồi save. Không cần xử lý thêm.
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
    
    # Create output directory (same naming style as checkpoints)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = str(config.get('model_name', 'model'))
    dataset_tag = infer_dataset_name(args.data_root)
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._") or "model"
    safe_dataset = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_tag).strip("._") or "dataset"
    test_experiment_name = f"{safe_model}_{safe_dataset}_x{upscale}_{timestamp}"
    output_dir = os.path.join(args.output_dir, test_experiment_name)
    
    save_dir = os.path.join(output_dir, 'images') if args.save_images else None
    print(f"Test experiment: {test_experiment_name}")
    try:
        # Run test
        try:
            avg_metrics, per_image_results = test_full_image(
                model=model,
                dataloader=test_loader,
                scale=upscale,
                device=device,
                crop_border=args.crop_border,
                save_dir=save_dir,
                chop_patch_size=args.chop_patch_size,
                chop_overlap=args.chop_overlap,
                save_band_png=args.save_band_png,
            )
        except RuntimeError as runtime_err:
            if device.type != "mps":
                raise
            print("\n⚠️ MPS runtime error detected during inference. Falling back to CPU and retrying...")
            print(f"   Error: {str(runtime_err).splitlines()[0]}")
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            device = torch.device("cpu")
            model = model.to(device)
            avg_metrics, per_image_results = test_full_image(
                model=model,
                dataloader=test_loader,
                scale=upscale,
                device=device,
                crop_border=args.crop_border,
                save_dir=save_dir,
                chop_patch_size=args.chop_patch_size,
                chop_overlap=args.chop_overlap,
                save_band_png=args.save_band_png,
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
        print(f"  Test Wall-clock Time  : {format_duration(avg_metrics['total_test_time_sec'])} ({avg_metrics['total_test_time_sec']:.2f} seconds) (includes load/save)")
        print(f"  Avg Time / Image      : {format_duration(avg_metrics['avg_time_per_image_sec'])} ({avg_metrics['avg_time_per_image_sec']:.2f} seconds) (wall-clock)")
        print(f"  Total Inference Time  : {format_duration(avg_metrics['total_inference_time_sec'])} ({avg_metrics['total_inference_time_sec']:.2f} seconds) (forward pass only)")
        print(f"  Avg Inference / Image : {format_duration(avg_metrics['avg_inference_time_per_image_sec'])} ({avg_metrics['avg_inference_time_per_image_sec']:.2f} seconds)")
        print(f"  Total Runtime         : {format_duration(total_runtime_sec)} ({total_runtime_sec:.2f} seconds)")
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
        if os.path.exists(json_path):
            print(f"⚠️  Overwriting existing results: {json_path}")
        tmp_path = json_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        os.replace(tmp_path, json_path)

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
            f.write(f"Test Wall-clock Time  : {format_duration(avg_metrics['total_test_time_sec'])} ({avg_metrics['total_test_time_sec']:.2f} seconds) (includes load/save)\n")
            f.write(f"Avg Time / Image      : {format_duration(avg_metrics['avg_time_per_image_sec'])} ({avg_metrics['avg_time_per_image_sec']:.2f} seconds) (wall-clock)\n")
            f.write(f"Total Inference Time  : {format_duration(avg_metrics['total_inference_time_sec'])} ({avg_metrics['total_inference_time_sec']:.2f} seconds) (forward pass only)\n")
            f.write(f"Avg Inference / Image : {format_duration(avg_metrics['avg_inference_time_per_image_sec'])} ({avg_metrics['avg_inference_time_per_image_sec']:.2f} seconds)\n")
            f.write(f"Total Runtime         : {format_duration(total_runtime_sec)} ({total_runtime_sec:.2f} seconds)\n")
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
