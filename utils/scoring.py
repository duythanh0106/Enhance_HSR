"""Selection-score utilities for checkpoint/model ranking."""


def clamp01(value):
    """Execute `clamp01`.

    Args:
        value: Input parameter `value`.

    Returns:
        Any: Output produced by this function.
    """
    return max(0.0, min(1.0, float(value)))


def _cfg_get(config_like, key, default):
    """Internal helper for `cfg_get` operations.

    Args:
        config_like: Input parameter `config_like`.
        key: Input parameter `key`.
        default: Input parameter `default`.

    Returns:
        Any: Output produced by this function.
    """
    if isinstance(config_like, dict):
        return config_like.get(key, default)
    return getattr(config_like, key, default)


def compute_selection_score(metrics, selection_metric="psnr", weights=None, refs=None):
    """Execute `compute_selection_score`.

    Args:
        metrics: Input parameter `metrics`.
        selection_metric: Input parameter `selection_metric`.
        weights: Input parameter `weights`.
        refs: Input parameter `refs`.

    Returns:
        Any: Output produced by this function.
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
    """Execute `compute_selection_score_from_config`.

    Args:
        metrics: Input parameter `metrics`.
        config_like: Input parameter `config_like`.

    Returns:
        Any: Output produced by this function.
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
