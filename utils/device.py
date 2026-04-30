"""
Device selection utilities — chọn compute device cho training và inference.

Thứ tự ưu tiên: explicit request → CUDA → MPS → CPU.
Dùng chung bởi train.py, test_full_image.py, evaluate.py và các script khác.

QUAN TRỌNG:
  - 'auto' luôn chọn thiết bị mạnh nhất available trên máy hiện tại
  - MPS là Apple Silicon GPU (M1/M2/M3); không hỗ trợ tất cả ops của PyTorch
  - Nếu MPS gặp lỗi at runtime, các script thường fallback sang CPU tự động
"""

import torch


def resolve_device(preferred='auto'):
    """Chọn torch.device phù hợp nhất với hardware hiện tại.

    Args:
        preferred: Thiết bị ưu tiên — 'auto', 'cuda', 'mps', hoặc 'cpu' (case-insensitive).

    Returns:
        torch.device: Thiết bị tốt nhất available theo thứ tự ưu tiên.
    """
    preferred = (preferred or 'auto').lower()

    if preferred == 'cpu':
        return torch.device('cpu')

    if preferred == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')

    if preferred == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')

    if torch.cuda.is_available():
        return torch.device('cuda')

    if torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')
