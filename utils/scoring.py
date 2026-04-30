"""
Selection-score utilities — tính composite score để chọn best checkpoint.

So sánh và xếp hạng model dựa trên nhiều metrics (PSNR, SSIM, SAM, ERGAS).
Dùng bởi train.py và seed_sweep.py để quyết định checkpoint nào là "best".

QUAN TRỌNG:
  - SAM và ERGAS là lower-is-better → được đảo nghịch trước khi tổng hợp
  - Score cuối thuộc [0, 1]; score cao hơn = model tốt hơn
  - Default weights: PSNR=0.45, SSIM=0.25, SAM=0.20, ERGAS=0.10
  - refs là giá trị tham chiếu để chuẩn hóa (PSNR ref=50, SAM ref=10°, ERGAS ref=20)
"""


def clamp01(value):
    """Kẹp giá trị float về [0, 1] — dùng nội bộ khi chuẩn hóa metrics."""
    return max(0.0, min(1.0, float(value)))


def _cfg_get(config_like, key, default):
    """Đọc key từ dict hoặc object config — unified accessor nội bộ.

    Args:
        config_like: Dict hoặc object có attribute.
        key: Tên field cần đọc.
        default: Fallback nếu key không tồn tại.

    Returns:
        Giá trị config[key] hoặc getattr(config_like, key, default).
    """
    if isinstance(config_like, dict):
        return config_like.get(key, default)
    return getattr(config_like, key, default)


def compute_selection_score(metrics, selection_metric="psnr", weights=None, refs=None):
    """Tính composite score từ dict metrics để chọn checkpoint tốt nhất.

    Args:
        metrics: Dict với keys 'PSNR', 'SSIM', 'SAM', 'ERGAS' (float).
        selection_metric: 'psnr' (dùng PSNR trực tiếp) hoặc 'composite'.
        weights: Dict trọng số mỗi metric — mặc định xem module docstring.
        refs: Dict giá trị tham chiếu để normalize metrics.

    Returns:
        float: Score trong [0, 1] — score cao hơn = model tốt hơn.
    """
    mode = str(selection_metric or "psnr").lower()
    if mode == "psnr":
        return float(metrics["PSNR"])

    weights = weights or {}
    refs = refs or {}
    w_psnr = float(weights.get("psnr", 0.45))
    w_ssim = float(weights.get("ssim", 0.25))
    w_sam = float(weights.get("sam", 0.20))
    w_ergas = float(weights.get("ergas", 0.10))

    psnr_ref = float(refs.get("psnr", 50.0))
    sam_ref = float(refs.get("sam", 10.0))
    ergas_ref = float(refs.get("ergas", 20.0))

    psnr_score = clamp01(float(metrics["PSNR"]) / max(psnr_ref, 1e-6))
    ssim_score = clamp01(float(metrics["SSIM"]))
    sam_score = clamp01(1.0 - float(metrics["SAM"]) / max(sam_ref, 1e-6))
    ergas_score = clamp01(1.0 - float(metrics["ERGAS"]) / max(ergas_ref, 1e-6))

    return float(
        w_psnr * psnr_score
        + w_ssim * ssim_score
        + w_sam * sam_score
        + w_ergas * ergas_score
    )


def compute_selection_score_from_config(metrics, config_like):
    """Đọc weights/refs từ config rồi gọi compute_selection_score.

    Args:
        metrics: Dict PSNR/SSIM/SAM/ERGAS.
        config_like: Config object hoặc dict có best_selection_metric/weights/refs.

    Returns:
        float: Score (higher is better).
    """
    selection_metric = _cfg_get(config_like, "best_selection_metric", "psnr")
    weights = _cfg_get(config_like, "best_score_weights", {}) or {}
    refs = _cfg_get(config_like, "best_score_refs", {}) or {}
    return compute_selection_score(
        metrics=metrics,
        selection_metric=selection_metric,
        weights=weights,
        refs=refs,
    )
