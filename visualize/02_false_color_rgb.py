"""
02_false_color_rgb.py
=====================
Render 3 band thành RGB, so sánh LR / ESSA gốc / Đề xuất / GT side-by-side.
GT load từ 31 PNG band. LR render từ _LR.png. SR load từ .npy.

Cách chạy:
  python 02_false_color_rgb.py --dataset CAVE --scale 2 --scene 0
  python 02_false_color_rgb.py --dataset CAVE --scale 2 --bands 20,10,2
  python 02_false_color_rgb.py --all
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from utils import (
    get_scene_names, load_gt, load_sr,
    to_rgb, psnr, mean_sam,
    WAVELENGTHS, DEFAULT_RGB_BANDS, RESULTS_ROOT, RESULT_FOLDERS,
)


def load_lr_png_rgb(dataset, scale, scene) -> np.ndarray:
    """Load _LR.png trực tiếp làm ảnh RGB preview (H, W, 3)."""
    proposed_folder, _ = RESULT_FOLDERS[(dataset, scale)]
    lr_path = (RESULTS_ROOT / proposed_folder / "images"
               / scene / f"{scene}_LR_RGB.png")
    img = Image.open(lr_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr   # (H, W, 3) — dùng để hiển thị, không cần band


def load_hr_png_rgb(dataset, scale, scene) -> np.ndarray:
    """Load _HR.png làm ảnh RGB preview của GT."""
    proposed_folder, _ = RESULT_FOLDERS[(dataset, scale)]
    hr_path = (RESULTS_ROOT / proposed_folder / "images"
               / scene / f"{scene}_HR_RGB.png")
    img = Image.open(hr_path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def add_zoom_inset(ax, img, region, loc="lower right"):
    r0, r1, c0, c1 = region
    patch = img[r0:r1, c0:c1]

    sz = 0.30
    if loc == "lower right":
        inset_ax = ax.inset_axes([1 - sz - 0.01, 0.01, sz, sz])
    else:
        inset_ax = ax.inset_axes([0.01, 0.01, sz, sz])

    inset_ax.imshow(patch, interpolation="nearest")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for sp in inset_ax.spines.values():
        sp.set_edgecolor("#EF9F27")
        sp.set_linewidth(1.8)

    H, W = img.shape[:2]
    rect = mpatches.Rectangle(
        (c0, r0), c1 - c0, r1 - r0,
        lw=1.8, edgecolor="#EF9F27", facecolor="none",
    )
    ax.add_patch(rect)


def plot_false_color(dataset, scale, scene,
                     bands=None, output_dir="figures",
                     show_zoom=True):

    if bands is None:
        bands = DEFAULT_RGB_BANDS[dataset]

    waves = WAVELENGTHS[dataset]

    # Load dữ liệu
    proposed = load_sr(dataset, scale, scene, which="proposed")
    baseline = load_sr(dataset, scale, scene, which="baseline")
    _, H, W  = proposed.shape
    gt_hsi   = load_gt(dataset, scene, target_hw=(H, W))   # (C, H, W) spectral

    # Render RGB từ spectral
    gt_rgb   = to_rgb(gt_hsi,   bands)
    prop_rgb = to_rgb(proposed, bands)
    base_rgb = to_rgb(baseline, bands)

    # LR dùng PNG preview (đã resize trong inference pipeline)
    try:
        lr_rgb = load_lr_png_rgb(dataset, scale, scene)
        # Resize LR về cùng kích thước HR nếu khác
        H, W = gt_rgb.shape[:2]
        if lr_rgb.shape[:2] != (H, W):
            from PIL import Image as PILImage
            lr_pil = PILImage.fromarray((lr_rgb * 255).astype(np.uint8))
            lr_pil = lr_pil.resize((W, H), PILImage.BILINEAR)
            lr_rgb = np.array(lr_pil, dtype=np.float32) / 255.0
    except FileNotFoundError:
        # fallback: downsample GT
        from scipy.ndimage import zoom
        lr_small = zoom(gt_hsi, (1, 1/scale, 1/scale), order=1)
        lr_up    = zoom(lr_small, (1, scale, scale), order=1)
        lr_rgb   = to_rgb(lr_up, bands)

    # Metric (dùng spectral)
    metrics = {
        "LR":       None,   # không có LR spectral đầy đủ
        "Baseline": {"psnr": psnr(baseline, gt_hsi), "sam": mean_sam(baseline, gt_hsi)},
        "Proposed": {"psnr": psnr(proposed, gt_hsi), "sam": mean_sam(proposed, gt_hsi)},
    }

    # Vùng zoom — 1/4 ảnh, góc trên-trái
    H, W = gt_rgb.shape[:2]
    r0, r1 = H // 4, H // 4 + max(H // 5, 40)
    c0, c1 = W // 4, W // 4 + max(W // 5, 40)
    zoom_region = (r0, r1, c0, c1)

    panels = [
        ("LR (bilinear)",                  lr_rgb,   "LR"),
        ("ESSA gốc",                       base_rgb, "Baseline"),
        ("Đề xuất\n(ESSA-SSAM-SpecTrans)", prop_rgb, "Proposed"),
        ("Ground Truth",                   gt_rgb,   None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("white")

    wl = [waves[b] for b in bands]
    band_info = f"Bands — R: {wl[0]:.0f} nm  G: {wl[1]:.0f} nm  B: {wl[2]:.0f} nm"

    for i, (title, img, mk) in enumerate(panels):
        ax = axes[i]
        ax.imshow(img, interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])

        if mk == "Proposed" and metrics[mk]:
            m = metrics[mk]
            subtitle = f"PSNR: {m['psnr']:.2f} dB  |  SAM: {m['sam']:.3f}°"
            color = "#185FA5"
            fw = "bold"
        elif mk == "Baseline" and metrics[mk]:
            m = metrics[mk]
            subtitle = f"PSNR: {m['psnr']:.2f} dB  |  SAM: {m['sam']:.3f}°"
            color = "#444441"
            fw = "normal"
        elif mk is None:
            subtitle = "Tham chiếu"
            color = "#2C2C2A"
            fw = "normal"
        else:
            subtitle = ""
            color = "#888780"
            fw = "normal"

        ax.set_title(f"{title}\n{subtitle}", fontsize=10,
                     color=color, fontweight=fw, pad=5)

        if show_zoom:
            loc = "lower right" if i < 2 else "lower left"
            add_zoom_inset(ax, img, zoom_region, loc=loc)

        if mk == "Proposed":
            for sp in ax.spines.values():
                sp.set_edgecolor("#185FA5")
                sp.set_linewidth(2.5)

    fig.suptitle(
        f"So sánh ảnh False Color — {dataset} x{scale}  |  {scene}\n{band_info}",
        fontsize=11, fontweight="bold", y=1.03,
    )
    plt.tight_layout(w_pad=0.3)

    os.makedirs(output_dir, exist_ok=True)
    stem = f"falsecolor_{dataset}_x{scale}_{scene}"
    fig.savefig(f"{output_dir}/{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_dir}/{stem}.png", dpi=200, bbox_inches="tight")
    print(f"[OK] {output_dir}/{stem}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CAVE",
                        choices=["CAVE", "Harvard", "Chikusei", "Pavia"])
    parser.add_argument("--scale",   type=int, default=2, choices=[2, 4])
    parser.add_argument("--scene",   default="0")
    parser.add_argument("--bands",   default=None,
                        help="R,G,B band index 0-indexed. Vi du: 20,10,2")
    parser.add_argument("--output",  default="figures")
    parser.add_argument("--no_zoom", action="store_true")
    parser.add_argument("--all",     action="store_true")
    args = parser.parse_args()

    bands = None
    if args.bands:
        parts = [int(x) for x in args.bands.split(",")]
        bands = tuple(parts[:3])

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
                    plot_false_color(ds, sc, scene,
                                     bands=bands,
                                     output_dir=args.output,
                                     show_zoom=not args.no_zoom)
                except Exception as e:
                    print(f"[SKIP] {ds} x{sc} {scene}: {e}")


if __name__ == "__main__":
    main()
