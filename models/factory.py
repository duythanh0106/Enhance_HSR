"""Model factory utilities."""

from .essa_improved import ESSA_SSAM
from .essa_original import ESSA
from .essa_ssam_spectrans import ESSA_SSAM_SpecTrans


def build_model_by_name(
    model_name,
    num_bands,
    feature_dim=128,
    upscale=4,
    fusion_mode="sequential",
    use_spectrans=True,
    spectrans_depth=2,
):
    """Instantiate a model from explicit arguments."""
    if model_name in {"ESSA_Original", "ESSA"}:
        return ESSA(inch=num_bands, dim=feature_dim, upscale=upscale)

    if model_name == "ESSA_SSAM":
        return ESSA_SSAM(
            inch=num_bands,
            dim=feature_dim,
            upscale=upscale,
            fusion_mode=fusion_mode,
        )

    if model_name == "ESSA_SSAM_SpecTrans":
        return ESSA_SSAM_SpecTrans(
            inch=num_bands,
            dim=feature_dim,
            upscale=upscale,
            fusion_mode=fusion_mode,
            use_spectrans=use_spectrans,
            spectrans_depth=spectrans_depth,
        )

    raise ValueError(f"Unknown model: {model_name}")


def build_model_from_config(config, num_bands_override=None):
    """
    Build model from config dict/object.

    Args:
        config: dict-like or object with attributes.
        num_bands_override: detected number of spectral bands to enforce.
    """
    if isinstance(config, dict):
        model_name = config.get("model_name", "ESSA_SSAM")
        num_bands = (
            num_bands_override
            if num_bands_override is not None
            else config.get("num_spectral_bands", 31)
        )
        feature_dim = config.get("feature_dim", 128)
        upscale = config.get("upscale_factor", 4)
        fusion_mode = config.get("fusion_mode", "sequential")
        use_spectrans = config.get("use_spectrans", True)
        spectrans_depth = config.get("spectrans_depth", 2)
    else:
        model_name = getattr(config, "model_name", "ESSA_SSAM")
        num_bands = (
            num_bands_override
            if num_bands_override is not None
            else getattr(config, "num_spectral_bands", 31)
        )
        feature_dim = getattr(config, "feature_dim", 128)
        upscale = getattr(config, "upscale_factor", 4)
        fusion_mode = getattr(config, "fusion_mode", "sequential")
        use_spectrans = getattr(config, "use_spectrans", True)
        spectrans_depth = getattr(config, "spectrans_depth", 2)

    return build_model_by_name(
        model_name=model_name,
        num_bands=num_bands,
        feature_dim=feature_dim,
        upscale=upscale,
        fusion_mode=fusion_mode,
        use_spectrans=use_spectrans,
        spectrans_depth=spectrans_depth,
    )
