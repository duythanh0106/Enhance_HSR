"""
utils.py — Hàm dùng chung cho cả 3 script viz
"""

import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom


# ── Cấu hình đường dẫn gốc — chỉnh 2 dòng này cho đúng máy bạn ──────────────
DATASET_ROOT   = Path("/Users/thanh.pd/Enhance_HSR2/dataset")
RESULTS_ROOT   = Path("/Users/thanh.pd/Enhance_HSR2/test_results")

# Tên folder kết quả theo dataset + scale
RESULT_FOLDERS = {
    # (proposed_folder, baseline_folder)
    ("CAVE",     2): ("best_cave_x2"),
    ("CAVE",     4): ("best_cave_x4_256dims"),
    ("Harvard",  2): ("best_harvard_x2"),
    ("Harvard",  4): ("best_harvard_x4"),
    ("Chikusei", 2): ("best2_chikusei_x2"),
    ("Chikusei", 4): ("best2_chikusei_x4"),
    ("Pavia",    2): ("best2_pavia_x2"),
    ("Pavia",    4): ("best2_pavia_x4"),
}

# Bước sóng (nm) theo dataset
WAVELENGTHS = {
    "CAVE":     np.linspace(400, 700, 31),
    "Harvard":  np.linspace(420, 720, 31),
    "Chikusei": np.linspace(363, 1018, 128),
    "Pavia":    np.linspace(430, 860, 102),
}

# Band RGB mặc định (R, G, B) — 0-indexed
DEFAULT_RGB_BANDS = {
    "CAVE":     (20, 10, 2),
    "Harvard":  (20, 10, 2),
    "Chikusei": (70, 40, 10),
    "Pavia":    (60, 35, 10),
}


def get_scene_names(dataset: str, scale: int) -> list[str]:
    """Lấy danh sách tên cảnh từ folder proposed."""
    proposed_folder, _ = RESULT_FOLDERS[(dataset, scale)]
    images_dir = RESULTS_ROOT / proposed_folder / "images"
    scenes = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    if not scenes:
        raise FileNotFoundError(f"Không tìm thấy cảnh nào trong {images_dir}")
    return scenes


def load_gt(dataset: str, scene_name: str,
            target_hw: tuple = None) -> np.ndarray:
    """Load GT từ N band PNG trong dataset folder, hỗ trợ 16-bit; tùy chọn center-crop.

    Args:
        dataset: Tên dataset ('CAVE', 'Harvard', v.v.) — dùng làm thư mục con.
        scene_name: Tên cảnh cần load.
        target_hw: Nếu (H, W) được truyền, center-crop GT về kích thước này.

    Returns:
        np.ndarray: Float32 array (C, H, W) trong khoảng [0, 1].
    """
    scene_dir = DATASET_ROOT / dataset / scene_name / scene_name
    if not scene_dir.exists():
        scene_dir = DATASET_ROOT / dataset / scene_name

    pngs = sorted(scene_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"Không tìm thấy PNG trong {scene_dir}")

    bands = []
    for p in pngs:
        img = Image.open(p)
        arr = np.array(img, dtype=np.float32)
        # 16-bit PNG: max ~65535
        if arr.max() > 255:
            arr = arr / 65535.0
        else:
            arr = arr / 255.0
        bands.append(arr)

    gt = np.stack(bands, axis=0)   # (C, H_orig, W_orig)

    # Center-crop về đúng kích thước SR nếu cần
    if target_hw is not None:
        th, tw = target_hw
        _, gh, gw = gt.shape
        r0 = (gh - th) // 2
        c0 = (gw - tw) // 2
        gt = gt[:, r0:r0+th, c0:c0+tw]

    return gt


def load_lr_png(dataset: str, scale: int, scene_name: str) -> np.ndarray:
    """Load LR RGB preview (_LR.png) từ test_results; trả về (H, W, 3) float32 [0, 1]."""
    proposed_folder, _ = RESULT_FOLDERS[(dataset, scale)]
    lr_path = (RESULTS_ROOT / proposed_folder / "images"
               / scene_name / f"{scene_name}_LR.png")
    
    img = Image.open(lr_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
    # LR.png chỉ là RGB preview — dùng để hiển thị, không phải spectral
    return arr


def load_sr(dataset: str, scale: int, scene_name: str,
            which: str = "proposed") -> np.ndarray:
    """Load SR tensor từ _SR.npy; trả về (C, H, W) float32 [0, 1].

    Args:
        dataset: Tên dataset ('CAVE', 'Harvard', v.v.).
        scale: Upscale factor (2 hoặc 4).
        scene_name: Tên cảnh.
        which: 'proposed' hoặc 'baseline' — xác định folder kết quả.

    Returns:
        np.ndarray: Float32 array (C, H, W) trong khoảng [0, 1].
    """
    proposed_folder, baseline_folder = RESULT_FOLDERS[(dataset, scale)]
    folder = proposed_folder if which == "proposed" else baseline_folder
    npy_path = (RESULTS_ROOT / folder / "images"
                / scene_name / f"{scene_name}_SR.npy")
    
    arr = np.load(str(npy_path)).astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / arr.max()
    return arr   # (C, H, W)


def load_lr_spectral(dataset: str, scale: int, scene_name: str,
                     target_hw: tuple = None) -> np.ndarray:
    """
    Load LR spectral bằng cách downsample GT rồi upsample bilinear.
    Shape: (C, H, W) — cùng kích thước HR.
    """
    gt = load_gt(dataset, scene_name, target_hw=target_hw)
    factor   = 1.0 / scale
    lr_small = zoom(gt, (1, factor, factor), order=1)
    lr_up    = zoom(lr_small, (1, scale, scale), order=1)
    return lr_up


def to_rgb(hsi: np.ndarray, bands: tuple) -> np.ndarray:
    """Trích 3 band, percentile stretch, trả về (H, W, 3) float32 [0,1]."""
    img = np.stack([hsi[b] for b in bands], axis=-1)
    lo  = np.percentile(img, 2)
    hi  = np.percentile(img, 98)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Tính PSNR (dB) giữa pred và gt; trả về 100.0 nếu MSE gần bằng 0."""
    mse = np.mean((pred - gt) ** 2)
    return 100.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def mean_sam(pred: np.ndarray, gt: np.ndarray) -> float:
    """Tính SAM trung bình (độ) trên toàn ảnh giữa pred và gt (C, H, W)."""
    dot  = np.sum(pred * gt, axis=0)
    norm = np.linalg.norm(pred, axis=0) * np.linalg.norm(gt, axis=0) + 1e-8
    return float(np.degrees(np.arccos(np.clip(dot / norm, -1, 1))).mean())


def pixel_sam(pred: np.ndarray, gt: np.ndarray, r: int, c: int) -> float:
    """Tính SAM (độ) tại pixel (r, c) giữa pred và gt (C, H, W)."""
    p = pred[:, r, c]
    g = gt[:, r, c]
    cos = np.dot(p, g) / (np.linalg.norm(p) * np.linalg.norm(g) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def pick_pixels(gt: np.ndarray, n: int = 4) -> list[tuple]:
    """Chọn n pixel đại diện theo phân vị độ sáng trung bình."""
    C, H, W = gt.shape
    mean_map = gt.mean(axis=0).flatten()
    percentiles = np.linspace(10, 90, n)
    labels = ["vùng tối", "vùng tối-vừa", "vùng sáng-vừa", "vùng sáng"]
    result = []
    for i, p in enumerate(percentiles):
        val = np.percentile(mean_map, p)
        idx = int(np.argmin(np.abs(mean_map - val)))
        r, c = divmod(idx, W)
        result.append((r, c, labels[i % len(labels)]))
    return result
