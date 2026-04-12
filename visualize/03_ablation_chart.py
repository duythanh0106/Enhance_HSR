"""
03_ablation_chart.py
====================
So sánh đóng góp từng thành phần:
  ESSA gốc → ESSA + SSAM → ESSA + SSAM + SpecTrans (đề xuất)

Điền số liệu thực vào ABLATION_DATA bên dưới rồi chạy.

Cách chạy:
  python 03_ablation_chart.py --dataset CAVE --scale 2
  python 03_ablation_chart.py --multi --metric PSNR --scale 2
  python 03_ablation_chart.py --all
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ══════════════════════════════════════════════════════════════════════════════
# ĐIỀN SỐ LIỆU THỰC VÀO ĐÂY
# Các dòng có comment "← điền" cần bạn chạy thực nghiệm ablation rồi điền vào.
# Số liệu ESSA gốc và Đề xuất đã có từ bảng kết quả của bạn.
# ══════════════════════════════════════════════════════════════════════════════
ABLATION_DATA = {
    "CAVE": {
        2: {
            "ESSA gốc":                {"PSNR": 39.32, "SAM": 2.432, "SSIM": 0.9772, "ERGAS": 4.507},
            "ESSA + SSAM + SpecTrans": {"PSNR": 39.01, "SAM": 2.637, "SSIM": 0.9791, "ERGAS": 4.722},
        },
        4: {
            "ESSA gốc":                {"PSNR": 34.65, "SAM": 3.248, "SSIM": 0.9442, "ERGAS": 3.839},
            "ESSA + SSAM + SpecTrans": {"PSNR": 34.68, "SAM": 3.299, "SSIM": 0.9465, "ERGAS": 3.846},
        },
    },
    "Harvard": {
        2: {
            "ESSA gốc":                {"PSNR": 46.14, "SAM": 2.031, "SSIM": 0.9856, "ERGAS": 4.061}, # ← điền
            "ESSA + SSAM + SpecTrans": {"PSNR": 46.50, "SAM": 2.004, "SSIM": 0.9860, "ERGAS": 3.917},
        },
        4: {
            "ESSA gốc":                {"PSNR": 40.97, "SAM": 2.482, "SSIM": 0.9585, "ERGAS": 3.257}, # ← điền
            "ESSA + SSAM + SpecTrans": {"PSNR": 41.11, "SAM": 2.477, "SSIM": 0.9588, "ERGAS": 3.194},
        },
    },
    "Chikusei": {
        2: {
            "ESSA gốc":                {"PSNR": 39.35, "SAM": 1.217, "SSIM": 0.9764, "ERGAS": 4.663},  # ← điền
            "ESSA + SSAM + SpecTrans": {"PSNR": 39.52, "SAM": 1.214, "SSIM": 0.9772, "ERGAS": 4.594},
        },
        4: {
            "ESSA gốc":                {"PSNR": 32.50, "SAM": 2.311, "SSIM": 0.8949, "ERGAS": 4.974},  # ← điền
            "ESSA + SSAM + SpecTrans": {"PSNR": 32.61, "SAM": 2.304, "SSIM": 0.8980, "ERGAS": 4.930},
        },
    },
    "Pavia": {
        2: {
            "ESSA gốc":                {"PSNR": 36.02, "SAM": 3.416, "SSIM": 0.9559, "ERGAS": 5.799}, # ← điền
            "ESSA + SSAM + SpecTrans": {"PSNR": 36.25, "SAM": 3.371, "SSIM": 0.9575, "ERGAS": 5.663},
        },
        4: {
            "ESSA gốc":                {"PSNR": 29.66, "SAM": 5.103, "SSIM": 0.8229, "ERGAS": 5.740},  # ← điền
            "ESSA + SSAM + SpecTrans": {"PSNR": 29.80, "SAM": 5.058, "SSIM": 0.8279, "ERGAS": 5.663},
        },
    },
}

VARIANT_COLORS = {
    "ESSA gốc":                "#888780",
    "ESSA + SSAM + SpecTrans": "#185FA5",
}

METRICS_CFG = {
    "PSNR":  ("PSNR (dB) ↑",  "up",   2),
    "SAM":   ("SAM (°) ↓",    "down", 3),
    "SSIM":  ("SSIM ↑",       "up",   4),
    "ERGAS": ("ERGAS ↓",      "down", 3),
}


def plot_ablation_single(dataset, scale, metrics=None, output_dir="figures"):
    if metrics is None:
        metrics = ["PSNR", "SAM", "SSIM", "ERGAS"]

    data = ABLATION_DATA.get(dataset, {}).get(scale, {})
    if not data:
        print(f"[SKIP] Không có dữ liệu: {dataset} x{scale}")
        return

    variants   = list(data.keys())
    n_metrics  = len(metrics)
    n_variants = len(variants)

    fig, axes = plt.subplots(1, n_metrics,
                             figsize=(4.5 * n_metrics, 4.8),
                             sharey=False)
    if n_metrics == 1:
        axes = [axes]

    bar_w = 0.55
    x     = np.arange(n_variants)

    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx]
        label, direction, dec = METRICS_CFG[metric]
        vals   = [data[v][metric] for v in variants]
        colors = [VARIANT_COLORS.get(v, "#4C9BE8") for v in variants]

        bars = ax.bar(x, vals, width=bar_w, color=colors,
                      edgecolor="white", linewidth=0.8, zorder=3)

        best = int(np.argmax(vals) if direction == "up" else np.argmin(vals))
        bars[best].set_edgecolor("#185FA5")
        bars[best].set_linewidth(2.2)

        y_range = max(vals) - min(vals) if max(vals) != min(vals) else max(vals) * 0.05
        offset  = y_range * 0.015

        for i, (bar, val) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{val:.{dec}f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if i == best else "normal",
                    color=VARIANT_COLORS.get(variants[i], "#333"))

        # Delta vs baseline
        base_val = vals[0]
        for i in range(1, n_variants):
            delta = vals[i] - base_val
            sign  = "+" if delta >= 0 else ""
            good  = (direction == "up" and delta > 0) or \
                    (direction == "down" and delta < 0)
            ax.text(x[i], min(vals) - y_range * 0.35,
                    f"({sign}{delta:.{dec}f})",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#185FA5" if good else "#D85A30",
                    style="italic")

        ax.set_ylim(min(vals) - y_range * 0.9,
                    max(vals) + y_range * 0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [v.replace("ESSA + SSAM + SpecTrans", "ESSA +\nSSAM +\nSpecTrans")
              .replace("ESSA + SSAM", "ESSA +\nSSAM")
             for v in variants],
            fontsize=8.5,
        )
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(label.split("(")[0].strip(), fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.2, lw=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(color=VARIANT_COLORS.get(v, "#888"), label=v)
               for v in variants]
    fig.legend(handles=patches, loc="upper center",
               bbox_to_anchor=(0.5, 1.07), ncol=n_variants,
               fontsize=9, framealpha=0.9,
               edgecolor="#cccccc", fancybox=False)

    fig.suptitle(f"Ablation Study — {dataset} x{scale}",
                 fontsize=13, fontweight="bold", y=1.12)
    plt.tight_layout(w_pad=1.0)

    os.makedirs(output_dir, exist_ok=True)
    stem = f"ablation_{dataset}_x{scale}"
    fig.savefig(f"{output_dir}/{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_dir}/{stem}.png", dpi=200, bbox_inches="tight")
    print(f"[OK] {output_dir}/{stem}.pdf")
    plt.close()


def plot_ablation_multi(scale, metric="PSNR", output_dir="figures"):
    """Biểu đồ tổng hợp: 1 metric, tất cả dataset, 3 variant."""
    datasets = [ds for ds in ABLATION_DATA if scale in ABLATION_DATA[ds]]
    variants = list(next(iter(next(iter(ABLATION_DATA.values())).values())).keys())
    label, direction, dec = METRICS_CFG[metric]

    n_ds  = len(datasets)
    n_var = len(variants)
    bw    = 0.72 / n_var
    x     = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(3.5 * n_ds, 5.2))

    for v_idx, variant in enumerate(variants):
        vals   = [ABLATION_DATA[ds][scale][variant][metric] for ds in datasets]
        offset = (v_idx - n_var / 2 + 0.5) * bw
        color  = VARIANT_COLORS.get(variant, "#888")
        bars   = ax.bar(x + offset, vals, width=bw * 0.9,
                        color=color, label=variant,
                        edgecolor="white", linewidth=0.7, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(vals) * 0.004),
                    f"{val:.{dec}f}",
                    ha="center", va="bottom", fontsize=7, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_title(f"Ablation — {label}  |  x{scale}",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(label, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    loc = "upper right" if direction == "up" else "upper left"
    ax.legend(fontsize=9, loc=loc, framealpha=0.9,
              edgecolor="#cccccc", fancybox=False)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    stem = f"ablation_all_{metric}_x{scale}"
    fig.savefig(f"{output_dir}/{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_dir}/{stem}.png", dpi=200, bbox_inches="tight")
    print(f"[OK] {output_dir}/{stem}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CAVE",
                        choices=list(ABLATION_DATA.keys()) + ["all"])
    parser.add_argument("--scale",   type=int, default=2, choices=[2, 4])
    parser.add_argument("--metrics", default="PSNR,SAM,SSIM,ERGAS")
    parser.add_argument("--multi",   action="store_true",
                        help="Bieu do tong hop tat ca dataset")
    parser.add_argument("--metric",  default="PSNR",
                        help="Metric dung cho --multi")
    parser.add_argument("--output",  default="figures")
    parser.add_argument("--all",     action="store_true",
                        help="Chay tat ca")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",")]

    if args.all:
        for ds in ABLATION_DATA:
            for sc in [2, 4]:
                plot_ablation_single(ds, sc, metrics, args.output)
        for sc in [2, 4]:
            for m in metrics:
                plot_ablation_multi(sc, m, args.output)
    elif args.multi:
        plot_ablation_multi(args.scale, args.metric, args.output)
    elif args.dataset == "all":
        for ds in ABLATION_DATA:
            plot_ablation_single(ds, args.scale, metrics, args.output)
    else:
        plot_ablation_single(args.dataset, args.scale, metrics, args.output)


if __name__ == "__main__":
    main()
