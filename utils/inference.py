"""
Inference helpers — sliding-window inference cho ảnh lớn.

forward_chop chia LR input lớn thành patches có overlap, chạy model trên từng
patch, rồi tổng hợp kết quả bằng weighted averaging để tránh block artifacts.

QUAN TRỌNG:
  - Cần thiết vì model không thể xử lý toàn ảnh lớn do GPU memory
  - Vùng chồng lấp (overlap) được average thay vì chọn một bên → mượt hơn
  - @torch.no_grad() — không cần gradient khi inference, tiết kiệm VRAM
  - Dùng bởi test_full_image.py và evaluate.py
"""

import torch


@torch.no_grad()
def forward_chop(model, x, scale, patch_size=64, overlap=16):
    """Sliding-window SR inference — xử lý ảnh lớn theo từng patch có overlap.

    Args:
        model: SR model nhận [B,C,H,W] và trả về [B,C,H*scale,W*scale].
        x: LR input tensor [B, C, H, W].
        scale: Upscale factor (ví dụ: 4).
        patch_size: Kích thước patch LR tính bằng pixel (mặc định 64).
        overlap: Số pixel overlap giữa các patches liền kề (mặc định 16).

    Returns:
        torch.Tensor: SR output [B, C, H*scale, W*scale] — full image đã super-resolve.
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
