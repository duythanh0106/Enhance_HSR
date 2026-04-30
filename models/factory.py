"""Model factory utilities."""

import torch

from .essa_improved import ESSA_SSAM
from .essa_original import ESSA
from .essa_ssam_spectrans import ESSA_SSAM_SpecTrans


# Registry: model_name → builder callable(params_dict) → nn.Module
# To add a new model: import it and add an entry here, no other file needs changing.
_MODEL_REGISTRY = {
    'ESSA':             lambda p: ESSA(inch=p['num_bands'], dim=p['feature_dim'], upscale=p['upscale']),
    'ESSA_Original':    lambda p: ESSA(inch=p['num_bands'], dim=p['feature_dim'], upscale=p['upscale']),
    'ESSA_SSAM':        lambda p: ESSA_SSAM(inch=p['num_bands'], dim=p['feature_dim'], upscale=p['upscale'], fusion_mode=p['fusion_mode']),
    'ESSA_SSAM_SpecTrans': lambda p: ESSA_SSAM_SpecTrans(
        inch=p['num_bands'], dim=p['feature_dim'], upscale=p['upscale'],
        fusion_mode=p['fusion_mode'], use_spectrans=p['use_spectrans'],
        spectrans_depth=p['spectrans_depth'],
    ),
}


def _cfg(config, key, default=None):
    """Read a key from either a dict or an object config."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def build_model_by_name(
    model_name,
    num_bands,
    feature_dim=128,
    upscale=4,
    fusion_mode="sequential",
    use_spectrans=True,
    spectrans_depth=2,
):
    builder = _MODEL_REGISTRY.get(model_name)
    if builder is None:
        known = sorted(_MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{model_name}'. Known: {known}")
    return builder({
        'num_bands': num_bands,
        'feature_dim': feature_dim,
        'upscale': upscale,
        'fusion_mode': fusion_mode,
        'use_spectrans': use_spectrans,
        'spectrans_depth': spectrans_depth,
    })


def build_model_from_config(config, num_bands_override=None):
    num_bands = num_bands_override if num_bands_override is not None else _cfg(config, 'num_spectral_bands', 31)
    return build_model_by_name(
        model_name=_cfg(config, 'model_name', 'ESSA_SSAM'),
        num_bands=num_bands,
        feature_dim=_cfg(config, 'feature_dim', 128),
        upscale=_cfg(config, 'upscale_factor', 4),
        fusion_mode=_cfg(config, 'fusion_mode', 'sequential'),
        use_spectrans=_cfg(config, 'use_spectrans', True),
        spectrans_depth=_cfg(config, 'spectrans_depth', 2),
    )


def _adapt_state_dict_for_model(model, state_dict):
    target = model.state_dict()
    adapted = {}
    converted_keys = []

    for key, value in state_dict.items():
        if key not in target:
            adapted[key] = value
            continue

        target_value = target[key]
        if value.shape == target_value.shape:
            adapted[key] = value
            continue

        if value.ndim == 2 and target_value.ndim == 3 and target_value.shape[-1] == 1:
            if value.shape == target_value.shape[:2]:
                adapted[key] = value.unsqueeze(-1)
                converted_keys.append(key)
                continue

        if value.ndim == 3 and value.shape[-1] == 1 and target_value.ndim == 2:
            if value.shape[:2] == target_value.shape:
                adapted[key] = value.squeeze(-1)
                converted_keys.append(key)
                continue

        adapted[key] = value

    return adapted, converted_keys


def load_state_dict_compat(model, state_dict, strict=True):
    adapted_state_dict, converted_keys = _adapt_state_dict_for_model(model, state_dict)
    load_result = model.load_state_dict(adapted_state_dict, strict=strict)
    return load_result, converted_keys
