"""
01_spectral_signature.py
========================
So sánh đường phổ tại các pixel đại diện:
  LR (downsample GT) | ESSA gốc | Đề xuất | Ground Truth

Cách chạy:
  python 01_spectral_signature.py --dataset CAVE --scale 2 --scene 0
  python 01_spectral_signature.py --dataset CAVE --scale 2 --scene real_and_fake_apples_ms
  python 01_spectral_signature.py --all
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_scene_names, load_gt, load_sr, load_lr_spectral,
    pick_pixels, pixel_sam, WAVELENGTHS,
)

STYLE = {
    "GT":       {"color": "#2C2C2A", "lw": 2.2, "ls": "-",  "zorder": 5,
                 "label": "Ground Truth"},
    "Proposed": {"color": "#185FA5", "lw": 2.0, "ls": "-",  "zorder": 4,
                 "label": "Đề xuất (ESSA-SSAM-SpecTrans)"},
    "Baseline": {"color": "#D85A30", "lw": 1.6, "ls": "--", "zorder": 3,
                 "label": "ESSA gốc"},
    "LR":       {"color": "#888780", "lw": 1.4, "ls": ":",  "zorder": 2,
                 "label": "LR (bilinear)"},
}


def plot_spectral(dataset, scale, scene, n_pixels=4,
                  manual_pixels=None, output_dir="figures"):

    waves = WAVELENGTHS[dataset]

    proposed = load_sr(dataset, scale, scene, which="proposed")
    baseline = load_sr(dataset, scale, scene, which="baseline")
    _, H, W  = proposed.shape
    gt       = load_gt(dataset, scene, target_hw=(H, W))
    lr       = load_lr_spectral(dataset, scale, scene, target_hw=(H, W))

    if manual_pixels:
        pixels = [(r, c, f"Pixel ({r},{c})") for r, c in manual_pixels]
    else:
        pixels = pick_pixels(gt, n=n_pixels)

    n_cols = min(n_pixels, 4)
    n_rows = (len(pixels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 3.8 * n_rows),
                             squeeze=False)

    for i, (r, c, label) in enumerate(pixels):
        ax = axes[i // n_cols][i % n_cols]

        sigs = {
            "GT":       gt[:, r, c],
            "Proposed": proposed[:, r, c],
            "Baseline": baseline[:, r, c],
            "LR":       lr[:, r, c],
        }

        for key, sig in sigs.items():
            s = STYLE[key]
            ax.plot(waves, sig, color=s["color"], lw=s["lw"],
                    ls=s["ls"], zorder=s["zorder"], label=s["label"])

        sam_p = pixel_sam(proposed, gt, r, c)
        sam_b = pixel_sam(baseline, gt, r, c)

        ax.set_title(
            f"{label}  (row={r}, col={c})\n"
            f"SAM đề xuất: {sam_p:.3f}°  |  ESSA gốc: {sam_b:.3f}°",
            fontsize=9.5, pad=5,
        )
        ax.set_xlabel("Bước sóng (nm)", fontsize=9)
        ax.set_ylabel("Cường độ phản xạ", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.set_xlim(waves[0], waves[-1])
        ax.set_ylim(-0.02, 1.05)

        if i == 0:
            ax.legend(fontsize=8, loc="upper right",
                      framealpha=0.88, edgecolor="#cccccc", fancybox=False)

    for j in range(len(pixels), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle(
        f"Chữ ký phổ — {dataset} x{scale}  |  {scene}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    stem = f"spectral_{dataset}_x{scale}_{scene}"
    fig.savefig(f"{output_dir}/{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_dir}/{stem}.png", dpi=200, bbox_inches="tight")
    print(f"[OK] {output_dir}/{stem}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CAVE",
                        choices=["CAVE", "Harvard", "Chikusei", "Pavia"])
    parser.add_argument("--scale",   type=int, default=2, choices=[2, 4])
    parser.add_argument("--scene",   default="0",
                        help="Tên scene hoặc index (0, 1, 2, ...)")
    parser.add_argument("--n_pixels", type=int, default=4)
    parser.add_argument("--pixels",  default=None,
                        help="Pixel thu cong: 'r1,c1;r2,c2;...'")
    parser.add_argument("--output",  default="figures")
    parser.add_argument("--all",     action="store_true")
    args = parser.parse_args()

    manual = None
    if args.pixels:
        manual = [tuple(int(x) for x in p.split(","))
                  for p in args.pixels.split(";")]

    datasets = ["CAVE", "Harvard", "Chikusei", "Pavia"] if args.all \
               else [args.dataset]
    scales   = [2, 4] if args.all else [args.scale]

    for ds in datasets:
        for sc in scales:
            try:
                scenes = get_scene_names(ds, sc)
            except Exception as e:
                print(f"[SKIP] {ds} x{sc}: {e}")
                continue

            if args.all:
                scene_list = scenes[:1]
            elif args.scene.isdigit():
                scene_list = [scenes[int(args.scene)]]
            else:
                scene_list = [args.scene]

            for scene in scene_list:
                try:
                    plot_spectral(ds, sc, scene,
                                  n_pixels=args.n_pixels,
                                  manual_pixels=manual,
                                  output_dir=args.output)
                except Exception as e:
                    print(f"[SKIP] {ds} x{sc} {scene}: {e}")


if __name__ == "__main__":
    main()
