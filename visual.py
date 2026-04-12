"""
Script tạo figure định tính cho Chương 5.3 khóa luận
Style theo bài báo ESSAformer (Zhang et al., 2023):
  - Figure 1: So sánh ảnh RGB full + zoom patch
  - Figure 2: Spectral profile tại pixel đại diện
"""

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image

# ── CẤU HÌNH ─────────────────────────────────────────────────
ESSA_DIR     = Path("/Users/thanh.pd/Enhance_HSR2/test_results/harvard_baseline_x2")
PROPOSED_DIR = Path("/Users/thanh.pd/Enhance_HSR2/test_results/best_harvard_x2")
SCENE        = "imgb3"
DATASET      = "Harvard"
SCALE        = "x2"
OUT_DIR      = Path("/Users/thanh.pd/Enhance_HSR2/figures")
MAT_DIR      = Path("/Users/thanh.pd/Enhance_HSR2/dataset/HARVARD/HARVARD")

# Vùng zoom — chỉnh theo nội dung ảnh
ZOOM_X, ZOOM_Y, ZOOM_W, ZOOM_H = 300, 200, 200, 200

# Pixel để vẽ spectral profile — None = tự chọn
PIXELS = None   # hoặc [(r1,c1), (r2,c2)]

OUT_DIR.mkdir(exist_ok=True, parents=True)
DPI = 200

# ── Helpers ───────────────────────────────────────────────────
def safe_name():
    d = Path(str(DATASET)).name.replace(' ','_').replace('/','_').replace('.','')
    s = str(SCALE).replace('×','x').replace('/','_').replace('.','')
    return d, s

def load_rgb(folder, scene, suffix):
    p = folder / "images" / scene
    files = list(p.glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy *{suffix} trong {p}")
    img = Image.open(files[0]).convert('RGB')
    arr = np.array(img)
    if arr.mean() < 80:
        arr = arr.astype(np.float32)
        arr = np.power(arr / 255.0, 0.4) * 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def load_sr_npy(folder, scene):
    """Load SR tensor từ bands_sr/ PNG hoặc _SR.npy"""
    p_bands = folder / "images" / scene / "bands_sr"
    band_files = sorted(p_bands.glob("*.png"))
    if band_files:
        bands = [np.array(Image.open(f).convert('L')).astype(np.float32)
                 for f in band_files]
        arr = np.stack(bands, axis=-1)
        if arr.max() > 1.0:
            arr = arr / arr.max()
        return arr
    # fallback _SR.npy
    p = folder / "images" / scene
    files = list(p.glob("*_SR.npy"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy SR data trong {p}")
    arr = np.load(files[0])
    if arr.ndim == 3 and arr.shape[0] < arr.shape[2]:
        arr = arr.transpose(1, 2, 0)
    return arr

def load_hr(folder, scene):
    """Load HR GT từ bands_hr/ PNG hoặc .mat"""
    p_bands = folder / "images" / scene / "bands_hr"
    band_files = sorted(p_bands.glob("*.png"))
    if band_files:
        bands = [np.array(Image.open(f).convert('L')).astype(np.float32)
                 for f in band_files]
        arr = np.stack(bands, axis=-1)
        if arr.max() > 1.0:
            arr = arr / arr.max()
        return arr
    # fallback .mat
    mat_path = MAT_DIR / f"{scene}.mat"
    if mat_path.exists():
        mat = sio.loadmat(str(mat_path))
        keys = [k for k in mat.keys() if not k.startswith('_')]
        arr = max([mat[k] for k in keys],
                  key=lambda x: x.size if hasattr(x,'size') else 0)
        arr = arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] < arr.shape[2]:
            arr = arr.transpose(1, 2, 0)
        if arr.max() > 1.0:
            arr = arr / arr.max()
        return arr
    raise FileNotFoundError(f"Không tìm thấy HR data cho scene {scene}")

# ── Figure 1: RGB full + zoom (style ESSAformer Figure 3/5) ──
def fig_visual_comparison():
    print("Tạo Figure so sánh ảnh RGB...")

    lr       = load_rgb(ESSA_DIR, SCENE, "_LR_RGB.png")
    hr       = load_rgb(ESSA_DIR, SCENE, "_HR_RGB.png")
    sr_essa  = load_rgb(ESSA_DIR, SCENE, "_SR_RGB.png")
    sr_prop  = load_rgb(PROPOSED_DIR, SCENE, "_SR_RGB.png")

    H_img, W_img = hr.shape[:2]
    print(f"  Ảnh HR size: {W_img}x{H_img}")

    # Tính zoom region hợp lệ
    zoom_w = min(ZOOM_W, W_img // 3)
    zoom_h = min(ZOOM_H, H_img)
    zx = min(max(0, ZOOM_X), W_img - zoom_w)
    zy = 0  # luôn bắt đầu từ top
    print(f"  Zoom region: x={zx}, y={zy}, w={zoom_w}, h={zoom_h}")

    COLS   = ['ESSA gốc', 'Đề xuất\n(ESSA-SSAM-SpecTrans)', 'Ground Truth']
    COLORS = ['#2563EB', '#16A34A', '#DC2626']
    imgs_full = [sr_essa, sr_prop, hr]

    # Kiểm tra có thể zoom không
    can_zoom = zoom_w > 5 and zoom_h > 5
    n_rows = 2 if can_zoom else 1

    fig, axes = plt.subplots(n_rows, 3,
                              figsize=(14, 5*n_rows),
                              gridspec_kw={'hspace': 0.06, 'wspace': 0.03})
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # → (1, 3)
    fig.patch.set_facecolor('white')

    for j, (img_f, col, color) in enumerate(zip(imgs_full, COLS, COLORS)):
        # Hàng 0: ảnh full
        axes[0, j].imshow(img_f)
        if can_zoom:
            from matplotlib.patches import Rectangle as Rect
            rect = Rect((zx, zy), zoom_w, zoom_h,
                        linewidth=2.5, edgecolor='yellow',
                        facecolor='none', zorder=5)
            axes[0, j].add_patch(rect)
        axes[0, j].set_title(col, fontsize=12, fontweight='bold',
                              color=color, pad=6)
        axes[0, j].axis('off')

        # Hàng 1: zoom patch
        if can_zoom:
            img_z = img_f[zy:zy+zoom_h, zx:zx+zoom_w]
            if img_z.size > 0:
                axes[1, j].imshow(img_z, interpolation='nearest')
                for spine in axes[1, j].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)
            axes[1, j].axis('off')

    axes[0, 0].set_ylabel('Ảnh tái tạo', fontsize=11,
                           fontweight='bold', labelpad=8)
    if can_zoom:
        axes[1, 0].set_ylabel('Chi tiết\nvùng zoom', fontsize=11,
                               fontweight='bold', labelpad=8)

    d, s = safe_name()
    fig.suptitle(f'So sánh ảnh tái tạo — {d} {s} (false-color RGB)',
                 fontsize=13, fontweight='bold', color='#0F172A', y=0.99)

    plt.tight_layout()
    out = OUT_DIR / f"fig_visual_{d}_{s}.png"
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → Saved: {out}")

# ── Figure 2: Spectral Profile (style ESSAformer Figure 4/8-13) ──
def fig_spectral():
    print("Tạo Figure spectral profile...")

    sr_essa = load_sr_npy(ESSA_DIR, SCENE)
    sr_prop = load_sr_npy(PROPOSED_DIR, SCENE)
    try:
        gt = load_hr(ESSA_DIR, SCENE)
    except FileNotFoundError:
        gt = load_hr(PROPOSED_DIR, SCENE)

    # Ensure (H, W, C)
    def to_hwc(arr):
        if arr.ndim == 3 and arr.shape[0] < arr.shape[2]:
            arr = arr.transpose(1, 2, 0)
        return arr
    gt = to_hwc(gt); sr_essa = to_hwc(sr_essa); sr_prop = to_hwc(sr_prop)
    print(f"  GT: {gt.shape}, ESSA: {sr_essa.shape}, Prop: {sr_prop.shape}")

    H, W, C = gt.shape
    band_idx = np.arange(C)

    # Chọn pixel
    pixels = PIXELS
    if pixels is None:
        var_map = gt.var(axis=2)
        flat = np.argsort(var_map.ravel())[::-1]
        p1 = np.unravel_index(flat[0], (H, W))
        p2 = np.unravel_index(flat[len(flat)//4], (H, W))
        pixels = [p1, p2]
        print(f"  Auto pixels: {pixels}")

    n = len(pixels)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 4.5))
    if n == 1: axes = [axes]
    fig.patch.set_facecolor('white')

    def sam_deg(a, b):
        cos = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)
        return np.degrees(np.arccos(np.clip(cos,-1,1)))

    for ax, (r, c) in zip(axes, pixels):
        gt_s   = gt[r, c, :]
        essa_s = sr_essa[r, c, :]
        prop_s = sr_prop[r, c, :]

        ax.plot(band_idx, gt_s,   color='#DC2626', lw=2.2,
                label='Ground Truth', zorder=4)
        ax.plot(band_idx, essa_s, color='#2563EB', lw=1.6,
                linestyle='--', label='ESSA gốc', zorder=3)
        ax.plot(band_idx, prop_s, color='#16A34A', lw=1.6,
                linestyle='-.', label='Đề xuất', zorder=3)

        ax.set_title(f'Pixel ({r}, {c})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Chỉ số kênh phổ', fontsize=10)
        ax.set_ylabel('Giá trị phản xạ', fontsize=10)
        ax.legend(fontsize=9.5, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle=':')
        ax.set_facecolor('#FAFAFA')

        s_essa = sam_deg(essa_s, gt_s)
        s_prop = sam_deg(prop_s, gt_s)
        ax.text(0.02, 0.02,
                f'SAM ESSA gốc: {s_essa:.3f}°\nSAM Đề xuất: {s_prop:.3f}°',
                transform=ax.transAxes, fontsize=9, va='bottom',
                bbox=dict(boxstyle='round,pad=0.35', fc='white',
                          ec='#CBD5E1', alpha=0.95))

    d, s = safe_name()
    fig.suptitle(
        f'Chữ ký phổ tại các pixel đại diện — {d} {s}',
        fontsize=13, fontweight='bold', color='#0F172A'
    )
    plt.tight_layout()
    out = OUT_DIR / f"fig_spectral_{d}_{s}.png"
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → Saved: {out}")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Dataset: {DATASET} {SCALE}")
    print(f"Scene:   {SCENE}")
    print()
    fig_visual_comparison()
    fig_spectral()
    print(f"\nXong! Figure lưu tại: {OUT_DIR}/")