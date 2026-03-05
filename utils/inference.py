"""Inference helpers."""

import torch


@torch.no_grad()
def forward_chop(model, x, scale, patch_size=64, overlap=16):
    """Execute `forward_chop`.

    Args:
        model: Input parameter `model`.
        x: Input parameter `x`.
        scale: Input parameter `scale`.
        patch_size: Input parameter `patch_size`.
        overlap: Input parameter `overlap`.

    Returns:
        Any: Output produced by this function.
    """
    b, c, h, w = x.size()
    patch_size = max(1, int(patch_size))
    patch_size = min(patch_size, h, w)
    overlap = max(0, int(overlap))
    overlap = min(overlap, patch_size - 1)
    stride = patch_size - overlap

    h_idx = list(range(0, h - patch_size, stride)) + [h - patch_size]
    w_idx = list(range(0, w - patch_size, stride)) + [w - patch_size]

    sr = torch.zeros(b, c, h * scale, w * scale, device=x.device)
    weight = torch.zeros_like(sr)

    for i in h_idx:
        for j in w_idx:
            patch = x[:, :, i:i + patch_size, j:j + patch_size]
            sr_patch = model(patch)

            h_start = i * scale
            w_start = j * scale
            h_end = h_start + patch_size * scale
            w_end = w_start + patch_size * scale

            sr[:, :, h_start:h_end, w_start:w_end] += sr_patch
            weight[:, :, h_start:h_end, w_start:w_end] += 1

    sr /= weight
    return sr
