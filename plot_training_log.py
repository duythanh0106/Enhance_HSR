"""
Plot training curves from a training.log file.

Usage:
    python plot_training_log.py --log ./logs/<experiment>/training.log --out ./logs/<experiment>/training_curves.png
"""

import argparse
import os
import re

from utils.visualization import plot_training_curves


EPOCH_RE = re.compile(r"^Epoch\s+(\d+)/")
TRAIN_LOSS_RE = re.compile(r"^\s*Train Loss\s*:\s*([0-9.eE+-]+)")
VAL_PSNR_RE = re.compile(r"^\s*Val PSNR\s*:\s*([0-9.eE+-]+)")
VAL_SSIM_RE = re.compile(r"^\s*Val SSIM\s*:\s*([0-9.eE+-]+)")
VAL_SAM_RE = re.compile(r"^\s*Val SAM\s*:\s*([0-9.eE+-]+)")


def parse_training_log(log_path):
    """Parse training.log into lists for plotting."""
    epochs = []
    train_losses = []
    val_psnr = []
    val_ssim = []
    val_sam = []

    current_epoch = None
    current = {
        "train_loss": None,
        "val_psnr": None,
        "val_ssim": None,
        "val_sam": None,
    }

    def flush():
        if current_epoch is None:
            return
        if current["train_loss"] is None:
            return
        epochs.append(current_epoch)
        train_losses.append(current["train_loss"])
        if current["val_psnr"] is not None:
            val_psnr.append((current_epoch, current["val_psnr"]))
        if current["val_ssim"] is not None:
            val_ssim.append((current_epoch, current["val_ssim"]))
        if current["val_sam"] is not None:
            val_sam.append((current_epoch, current["val_sam"]))

    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            epoch_match = EPOCH_RE.match(line)
            if epoch_match and line.endswith("Summary"):
                flush()
                current_epoch = int(epoch_match.group(1))
                current = {
                    "train_loss": None,
                    "val_psnr": None,
                    "val_ssim": None,
                    "val_sam": None,
                }
                continue

            match = TRAIN_LOSS_RE.match(line)
            if match:
                current["train_loss"] = float(match.group(1))
                continue

            match = VAL_PSNR_RE.match(line)
            if match:
                current["val_psnr"] = float(match.group(1))
                continue

            match = VAL_SSIM_RE.match(line)
            if match:
                current["val_ssim"] = float(match.group(1))
                continue

            match = VAL_SAM_RE.match(line)
            if match:
                current["val_sam"] = float(match.group(1))
                continue

    flush()

    val_metrics = []
    max_len = max(len(val_psnr), len(val_ssim), len(val_sam), 0)
    for idx in range(max_len):
        epoch = None
        entry = {}
        if idx < len(val_psnr):
            epoch, value = val_psnr[idx]
            entry["PSNR"] = value
        if idx < len(val_ssim):
            epoch = epoch or val_ssim[idx][0]
            entry["SSIM"] = val_ssim[idx][1]
        if idx < len(val_sam):
            epoch = epoch or val_sam[idx][0]
            entry["SAM"] = val_sam[idx][1]
        if entry:
            entry["epoch"] = epoch if epoch is not None else (idx + 1)
            val_metrics.append(entry)

    return epochs, train_losses, val_metrics


def main():
    """Parse CLI args, đọc training.log và lưu training curves PNG."""
    parser = argparse.ArgumentParser(description="Plot training curves from training.log")
    parser.add_argument("--log", type=str, required=True, help="Path to training.log")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (optional)")
    args = parser.parse_args()

    log_path = args.log
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    out_path = args.out
    if out_path is None:
        out_dir = os.path.dirname(os.path.abspath(log_path))
        out_path = os.path.join(out_dir, "training_curves.png")

    epochs, train_losses, val_metrics = parse_training_log(log_path)
    if not epochs:
        raise ValueError("No epochs found in log.")

    plot_training_curves(train_losses, val_metrics, save_path=out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
