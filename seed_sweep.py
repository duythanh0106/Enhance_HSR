"""
Seed sweep utility for HSI-SR.

Purpose:
- Keep split fixed via split_seed.
- Sweep training randomness seed.
- Rank seeds by validation score (patch-val or full-image val).

Example:
python3 seed_sweep.py \
  --config universal_best \
  --data_root ./data/Harvard \
  --seeds 7,11,19,23,29 \
  --epochs 100 \
  --selection_mode full_image_val
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from config import (
    CONFIG_PRESET_CHOICES,
    build_config,
)
from data.dataset import HyperspectralTestDataset, build_split_kwargs, load_dataset_with_fallback
from models.factory import build_model_from_config, load_state_dict_compat
from train import Trainer
from utils.device import resolve_device
from utils.inference import forward_chop
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam, calculate_ergas
from utils.scoring import compute_selection_score_from_config
from utils.time_utils import format_duration


def parse_seeds(text: str):
    """Parse chuỗi seeds dạng "7,11,19" thành list int.

    Args:
        text: Chuỗi seeds phân cách bởi dấu phẩy.

    Returns:
        list[int]: List các seed đã parse; raise ValueError nếu rỗng.
    """
    seeds = []
    for token in (text or "").split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("No seeds parsed from --seeds")
    return seeds


@torch.no_grad()
def evaluate_full_image_split(model, dataloader, scale, device, crop_border, chop_patch_size, chop_overlap):
    """Đánh giá model trên toàn bộ ảnh trong dataloader, trả về metrics trung bình.

    Args:
        model: SR model đã load weights.
        dataloader: DataLoader trả về (lr, hr, filepath) tuples.
        scale: Upscale factor.
        device: torch.device để chạy inference.
        crop_border: Nếu True, crop scale pixels trên mỗi cạnh trước metrics.
        chop_patch_size: Patch size LR cho sliding-window inference.
        chop_overlap: Overlap giữa các patches.

    Returns:
        dict: PSNR, SSIM, SAM, ERGAS trung bình, num_images và inference_time_sec.
    """
    model.eval()
    psnr_list = []
    ssim_list = []
    sam_list = []
    ergas_list = []

    start = time.time()
    for lr, hr, _ in dataloader:
        lr = lr.to(device)
        hr = hr.to(device)

        sr = forward_chop(
            model,
            lr,
            scale=scale,
            patch_size=chop_patch_size,
            overlap=chop_overlap
        )

        if sr.shape[-2:] != hr.shape[-2:]:
            min_h = min(sr.shape[-2], hr.shape[-2])
            min_w = min(sr.shape[-1], hr.shape[-1])
            sr = sr[:, :, :min_h, :min_w]
            hr = hr[:, :, :min_h, :min_w]

        if crop_border and scale > 1:
            sr = sr[:, :, scale:-scale, scale:-scale]
            hr = hr[:, :, scale:-scale, scale:-scale]

        sr = sr.clamp(0.0, 1.0).squeeze(0).cpu()
        hr = hr.clamp(0.0, 1.0).squeeze(0).cpu()

        psnr_list.append(calculate_psnr(sr, hr, data_range=1.0))
        ssim_list.append(calculate_ssim(sr, hr, data_range=1.0))
        sam_list.append(calculate_sam(sr, hr))
        ergas_list.append(calculate_ergas(hr, sr, scale=scale))

    elapsed = time.time() - start
    n = max(1, len(psnr_list))
    return {
        "PSNR": float(sum(psnr_list) / n),
        "SSIM": float(sum(ssim_list) / n),
        "SAM": float(sum(sam_list) / n),
        "ERGAS": float(sum(ergas_list) / n),
        "num_images": int(len(psnr_list)),
        "inference_time_sec": float(elapsed),
    }


def load_val_dataset(config_dict, data_root, split_name):
    """Load HyperspectralTestDataset cho split_name dựa trên thông số trong config_dict.

    Args:
        config_dict: Dict config (từ checkpoint hoặc build_config).
        data_root: Đường dẫn tới thư mục dữ liệu.
        split_name: Tên split cần load ('val' hoặc 'test').

    Returns:
        HyperspectralTestDataset: Dataset đã load.
    """
    split_kwargs = build_split_kwargs(
        upscale=config_dict.get("upscale_factor", 4),
        split_seed=config_dict.get("split_seed", 42),
        train_ratio=config_dict.get("train_ratio", 0.8),
        val_ratio=config_dict.get("val_ratio", 0.1),
        test_ratio=config_dict.get("test_ratio", 0.1),
        force_regenerate_split=config_dict.get("regenerate_split", False),
    )
    ds, _ = load_dataset_with_fallback(
        dataset_cls=HyperspectralTestDataset,
        primary_split=split_name,
        fallback_split="train",
        data_root=data_root,
        log_fn=print,
        **split_kwargs,
    )
    return ds


def evaluate_checkpoint_on_full_image_val(checkpoint_path, data_root, split_name, args):
    """Load checkpoint và chạy full-image evaluation trên split_name.

    Args:
        checkpoint_path: Đường dẫn file .pth checkpoint.
        data_root: Đường dẫn tới thư mục dữ liệu.
        split_name: Tên split ('val' hoặc 'test').
        args: Parsed argparse Namespace (chứa eval_device, no_crop_border, v.v.).

    Returns:
        tuple: (metrics_dict, config_dict) — metrics và config của checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_dict = checkpoint.get("config", {})
    device = resolve_device(args.eval_device or config_dict.get("device", "auto"))
    scale = int(config_dict.get("upscale_factor", 4))

    dataset = load_val_dataset(config_dict, data_root, split_name)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_model_from_config(config_dict, num_bands_override=dataset.num_bands).to(device)
    _, converted = load_state_dict_compat(model, checkpoint["model_state_dict"], strict=True)
    if converted:
        print(f"Converted {len(converted)} weight tensors for compatibility.")

    try:
        metrics = evaluate_full_image_split(
            model=model,
            dataloader=dataloader,
            scale=scale,
            device=device,
            crop_border=not args.no_crop_border,
            chop_patch_size=args.chop_patch_size,
            chop_overlap=args.chop_overlap,
        )
        metrics["eval_device"] = str(device)
        return metrics, config_dict
    except RuntimeError as runtime_err:
        if device.type != "mps":
            raise
        print("⚠️ MPS runtime error in full-image validation. Retrying on CPU.")
        print(f"   Error: {str(runtime_err).splitlines()[0]}")
        cpu_device = torch.device("cpu")
        model = model.to(cpu_device)
        metrics = evaluate_full_image_split(
            model=model,
            dataloader=dataloader,
            scale=scale,
            device=cpu_device,
            crop_border=not args.no_crop_border,
            chop_patch_size=args.chop_patch_size,
            chop_overlap=args.chop_overlap,
        )
        metrics["eval_device"] = "cpu (fallback)"
        return metrics, config_dict


def run_one_seed(args, seed):
    """Train và evaluate một seed — trả về dict kết quả để xếp hạng.

    Args:
        args: Parsed argparse Namespace với cấu hình sweep.
        seed: Training seed cần chạy.

    Returns:
        dict: Kết quả gồm seed, checkpoint path, rank_score và các metrics.
    """
    seed_start = time.time()
    cfg = build_config(args.config)
    if args.data_root:
        cfg.data_root = args.data_root
        if hasattr(cfg, "apply_dataset_profile"):
            cfg.apply_dataset_profile()

    cfg.seed = int(seed)
    cfg.split_seed = int(args.split_seed)
    cfg.regenerate_split = bool(args.regenerate_split)
    cfg.best_selection_metric = args.best_selection_metric

    if args.epochs > 0:
        cfg.num_epochs = int(args.epochs)
    if args.train_virtual_samples > 0:
        cfg.train_virtual_samples_per_epoch = int(args.train_virtual_samples)
    if args.val_virtual_samples > 0:
        cfg.val_virtual_samples_per_epoch = int(args.val_virtual_samples)
    cfg.num_workers = int(args.num_workers)

    cfg.timestamp = f"{args.sweep_tag}_seed{seed}"
    cfg.refresh_output_paths()

    print("\n" + "=" * 70)
    print(f"Seed {seed} | data_root={cfg.data_root}")
    print(f"split_seed={cfg.split_seed}, epochs={cfg.num_epochs}, mode={args.selection_mode}")
    print("=" * 70)

    trainer = Trainer(cfg)
    trainer.train()

    best_ckpt = os.path.join(cfg.checkpoint_dir, "best.pth")
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

    ckpt = torch.load(best_ckpt, map_location="cpu")
    patch_metrics = ckpt.get("metrics", {})
    patch_score = ckpt.get("selection_score")
    if patch_score is None and patch_metrics:
        patch_score = compute_selection_score_from_config(patch_metrics, ckpt.get("config", {}))

    result = {
        "seed": int(seed),
        "checkpoint": best_ckpt,
        "experiment": cfg.experiment_name,
        "selection_mode": args.selection_mode,
        "split_seed": int(cfg.split_seed),
        "train_best_psnr_patch": float(ckpt.get("best_psnr", trainer.best_psnr)),
        "train_best_score_patch": (None if patch_score is None else float(patch_score)),
    }

    if args.selection_mode == "full_image_val":
        full_metrics, cfg_dict = evaluate_checkpoint_on_full_image_val(
            checkpoint_path=best_ckpt,
            data_root=cfg.data_root,
            split_name=args.val_split_name,
            args=args,
        )
        full_score = compute_selection_score_from_config(full_metrics, cfg_dict)
        result["rank_score"] = float(full_score)
        result["val_full_image"] = full_metrics
    else:
        if patch_score is None:
            patch_score = float("-inf")
        result["rank_score"] = float(patch_score)
        result["val_patch"] = patch_metrics

    result["seed_runtime_sec"] = float(time.time() - seed_start)
    return result


def save_results(output_dir, results, total_runtime_sec, args):
    """Lưu kết quả sweep ra JSON, CSV, meta JSON và summary TXT.

    Args:
        output_dir: Thư mục đầu ra.
        results: List dict kết quả từng seed (đã sắp xếp theo rank_score).
        total_runtime_sec: Tổng thời gian sweep (giây).
        args: Parsed argparse Namespace (dùng để ghi metadata).

    Returns:
        tuple: (json_path, meta_path, csv_path, txt_path) — đường dẫn các file đã lưu.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "seed_sweep_results.json")
    meta_path = os.path.join(output_dir, "seed_sweep_meta.json")
    csv_path = os.path.join(output_dir, "seed_sweep_results.csv")
    txt_path = os.path.join(output_dir, "summary.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selection_mode": args.selection_mode,
        "best_selection_metric": args.best_selection_metric,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "seeds": parse_seeds(args.seeds),
        "total_runtime_sec": float(total_runtime_sec),
        "total_runtime_hms": format_duration(total_runtime_sec),
        "avg_runtime_per_seed_sec": (
            float(total_runtime_sec) / max(1, len(results))
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "seed",
            "rank_score",
            "train_best_psnr_patch",
            "train_best_score_patch",
            "seed_runtime_sec",
            "checkpoint",
        ])
        for i, row in enumerate(results, start=1):
            writer.writerow([
                i,
                row["seed"],
                row.get("rank_score"),
                row.get("train_best_psnr_patch"),
                row.get("train_best_score_patch"),
                row.get("seed_runtime_sec"),
                row.get("checkpoint"),
            ])

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("SEED SWEEP SUMMARY\n")
        f.write("=" * 70 + "\n")
        for i, row in enumerate(results, start=1):
            f.write(
                f"#{i} seed={row['seed']} "
                f"rank_score={row.get('rank_score', float('nan')):.6f} "
                f"best_psnr_patch={row.get('train_best_psnr_patch', float('nan')):.4f} "
                f"seed_time={format_duration(row.get('seed_runtime_sec', 0.0))} "
                f"({row.get('seed_runtime_sec', 0.0):.2f}s)\n"
            )
        f.write("=" * 70 + "\n")
        if results:
            top = results[0]
            f.write(f"Best seed: {top['seed']}\n")
            f.write(f"Best checkpoint: {top['checkpoint']}\n")
            f.write(f"Rank score: {top['rank_score']:.6f}\n")
        f.write(f"Total sweep time: {format_duration(total_runtime_sec)} ({total_runtime_sec:.2f}s)\n")
        f.write(
            f"Avg time/seed: {format_duration(total_runtime_sec / max(1, len(results)))} "
            f"({(total_runtime_sec / max(1, len(results))):.2f}s)\n"
        )

    return json_path, meta_path, csv_path, txt_path


def main():
    """Parse CLI args, chạy seed sweep tuần tự và lưu kết quả xếp hạng."""
    parser = argparse.ArgumentParser(description="Sweep seeds for HSI-SR training")
    parser.add_argument(
        "--config",
        type=str,
        default="universal_best",
        choices=CONFIG_PRESET_CHOICES,
        help="Config preset",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root")
    parser.add_argument("--seeds", type=str, default="7,11,19,23,29", help="Comma-separated seeds")
    parser.add_argument("--split_seed", type=int, default=42, help="Fixed split seed")
    parser.add_argument("--epochs", type=int, default=0, help="Override num_epochs (0 = keep config)")
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="full_image_val",
        choices=["patch_val", "full_image_val"],
        help="Ranking mode for seed selection",
    )
    parser.add_argument(
        "--best_selection_metric",
        type=str,
        default="psnr",
        choices=["psnr", "composite"],
        help="Metric mode for checkpoint selection during each run",
    )
    parser.add_argument("--val_split_name", type=str, default="val", help="Split name for full-image ranking")
    parser.add_argument("--no_crop_border", action="store_true", help="Disable border crop in full-image eval")
    parser.add_argument("--chop_patch_size", type=int, default=32, help="Patch size for full-image eval")
    parser.add_argument("--chop_overlap", type=int, default=8, help="Patch overlap for full-image eval")
    parser.add_argument("--eval_device", type=str, default=None, help="Override eval device: auto/cuda/mps/cpu")
    parser.add_argument("--train_virtual_samples", type=int, default=0, help="Override train virtual samples")
    parser.add_argument("--val_virtual_samples", type=int, default=0, help="Override val virtual samples")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for training runs")
    parser.add_argument("--regenerate_split", action="store_true", help="Regenerate split before each run")
    parser.add_argument("--output_dir", type=str, default="./seed_sweep_results", help="Output directory")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.sweep_tag = f"sweep_{timestamp}"

    sweep_start = time.time()
    all_results = []
    for seed in seeds:
        one = run_one_seed(args, seed)
        all_results.append(one)
        print(
            f"✅ Seed {seed} done in {format_duration(one.get('seed_runtime_sec', 0.0))} "
            f"({one.get('seed_runtime_sec', 0.0):.2f}s)"
        )

    # Primary sort by rank_score, secondary by patch PSNR
    all_results.sort(
        key=lambda x: (
            float(x.get("rank_score", float("-inf"))),
            float(x.get("train_best_psnr_patch", float("-inf"))),
        ),
        reverse=True,
    )

    output_dir = os.path.join(args.output_dir, args.sweep_tag)
    total_runtime_sec = time.time() - sweep_start
    json_path, meta_path, csv_path, txt_path = save_results(
        output_dir,
        all_results,
        total_runtime_sec=total_runtime_sec,
        args=args,
    )

    print("\n" + "=" * 70)
    print("Seed sweep completed")
    for i, row in enumerate(all_results, start=1):
        print(
            f"#{i} seed={row['seed']} "
            f"rank_score={row.get('rank_score', float('nan')):.6f} "
            f"best_psnr_patch={row.get('train_best_psnr_patch', float('nan')):.4f}"
        )
    if all_results:
        print(f"Best seed: {all_results[0]['seed']}")
        print(f"Best checkpoint: {all_results[0]['checkpoint']}")
    print(
        f"Total sweep time: {format_duration(total_runtime_sec)} "
        f"({total_runtime_sec:.2f} seconds)"
    )
    print(f"Saved: {json_path}")
    print(f"Saved: {meta_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
