"""
MPS Compatibility & Speed Check
Chạy script này TRƯỚC KHI train để phát hiện sớm vấn đề.

Usage:
    python check_mps.py                    # kiểm tra tất cả
    python check_mps.py --dataset cave     # kiểm tra config cụ thể
    python check_mps.py --dataset chikusei --bands 128
"""

import argparse
import time
import sys

import torch
import torch.nn as nn


# ─── Helpers ──────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def ok(msg):   print(f"  ✓  {msg}")
def warn(msg): print(f"  ⚠  {msg}")
def fail(msg): print(f"  ✗  {msg}")


# ─── Checks ───────────────────────────────────────────────────────────────────

def check_mps_available():
    """Kiểm tra MPS available và built; trả về torch.device('mps') hoặc None."""
    section("1. MPS availability")
    if not torch.backends.mps.is_available():
        fail("MPS not available — bạn đang chạy trên máy không có Apple Silicon?")
        fail("Nếu là Intel Mac thì không có MPS, dùng CPU.")
        return None
    if not torch.backends.mps.is_built():
        fail("PyTorch không được build với MPS support.")
        fail("Chạy: pip install --upgrade torch torchvision")
        return None
    ok("MPS available và built")
    device = torch.device("mps")
    return device


def check_fallback_env():
    """Kiểm tra biến môi trường PYTORCH_ENABLE_MPS_FALLBACK; trả về True nếu được set."""
    section("2. MPS fallback environment")
    import os
    val = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    if val == "1":
        ok(f"PYTORCH_ENABLE_MPS_FALLBACK=1 (fallback về CPU cho unsupported ops)")
        warn("Một số op sẽ chạy trên CPU — training sẽ chậm hơn nhưng không crash")
    else:
        warn("PYTORCH_ENABLE_MPS_FALLBACK không được set")
        warn("Nếu gặp 'NotImplementedError' khi train, chạy:")
        warn("  export PYTORCH_ENABLE_MPS_FALLBACK=1")
        warn("Sau đó chạy lại train.py")
    return val == "1"


def check_ops(device):
    """Chạy thử các ops quan trọng (LayerNorm, PixelShuffle, attention, v.v.); trả về True nếu tất cả pass."""
    section("3. Kiểm tra ops quan trọng cho model")
    results = {}

    ops_to_test = {
        "LayerNorm":        lambda: nn.LayerNorm(31).to(device)(torch.randn(2, 64*64, 31, device=device)),
        "AdaptiveAvgPool2d":lambda: nn.AdaptiveAvgPool2d(1)(torch.randn(2, 31, 64, 64).to(device)),
        "AdaptiveMaxPool2d":lambda: nn.AdaptiveMaxPool2d(1)(torch.randn(2, 31, 64, 64).to(device)),
        "PixelShuffle(2)":  lambda: nn.PixelShuffle(2)(torch.randn(2, 4, 64, 64).to(device)),
        "PixelUnshuffle(2)":lambda: nn.PixelUnshuffle(2)(torch.randn(2, 1, 128, 128).to(device)),
        "Softmax (attn)":   lambda: torch.softmax(torch.randn(2, 4, 31, 31).to(device), dim=-1),
        "bmm / matmul":     lambda: torch.matmul(
                                torch.randn(2, 4, 31, 16).to(device),
                                torch.randn(2, 4, 16, 31).to(device)),
        "Conv2d 1x1":       lambda: nn.Conv2d(31, 31, 1).to(device)(torch.randn(2, 31, 64, 64).to(device)),
        "Conv2d 3x3":       lambda: nn.Conv2d(31, 31, 3, 1, 1).to(device)(torch.randn(2, 31, 64, 64).to(device)),
        "Conv1d (SpecTrans)":lambda: nn.Conv1d(31, 31*3, 1).to(device)(torch.randn(2, 31, 64*64).to(device)),
        "Dropout2d":        lambda: nn.Dropout2d(0.05)(torch.randn(2, 31, 64, 64).to(device)),
        "LeakyReLU":        lambda: nn.LeakyReLU(0.2)(torch.randn(2, 31, 64, 64).to(device)),
        "GELU":             lambda: nn.GELU()(torch.randn(2, 64*64, 31).to(device)),
    }

    all_ok = True
    for name, fn in ops_to_test.items():
        try:
            fn()
            ok(name)
            results[name] = True
        except Exception as e:
            fail(f"{name}: {e}")
            results[name] = False
            all_ok = False

    return all_ok


def check_backward(device):
    """Kiểm tra backward pass cơ bản với conv proxy model; trả về True nếu thành công."""
    section("4. Backward pass (gradient flow)")
    try:
        x = torch.randn(1, 31, 64, 64, device=device, requires_grad=True)
        model = nn.Sequential(
            nn.Conv2d(31, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 31, 3, 1, 1),
        ).to(device)
        out = model(x)
        loss = out.mean()
        loss.backward()
        ok("Backward pass thành công")
        return True
    except Exception as e:
        fail(f"Backward pass thất bại: {e}")
        return False


def check_speed(device, bands=31, feature_dim=128, patch_size=64, n_iters=10):
    """Benchmark tốc độ forward+backward với conv proxy; in ước tính thời gian/epoch."""
    section(f"5. Speed benchmark (bands={bands}, dim={feature_dim}, patch={patch_size})")

    try:
        # Simulate một forward pass giống model thực
        model = nn.Sequential(
            nn.Conv2d(bands, feature_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim, feature_dim * 2, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim, bands, 3, 1, 1),
        ).to(device)

        x = torch.randn(1, bands, patch_size, patch_size, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            out = model(x)
            loss = (out - x).abs().mean()
            loss.backward()
            optimizer.step()

        # Benchmark
        torch.mps.synchronize() if hasattr(torch, 'mps') else None
        t0 = time.perf_counter()
        for _ in range(n_iters):
            optimizer.zero_grad()
            out = model(x)
            loss = (out - x).abs().mean()
            loss.backward()
            optimizer.step()
        torch.mps.synchronize() if hasattr(torch, 'mps') else None
        elapsed = time.perf_counter() - t0

        sec_per_iter = elapsed / n_iters
        ok(f"Conv-only proxy: {sec_per_iter*1000:.0f}ms/iter")

        # Rough estimate: full model ~5–8x slower than conv proxy
        est_full = sec_per_iter * 6
        ok(f"Full model estimate: ~{est_full*1000:.0f}ms/iter (×6 heuristic)")

        # Epoch time estimate — dùng số samples thực tế
        dataset_samples = {
            'CAVE':             26,   # 32 scenes × 0.8 train ratio ≈ 26
            'Harvard':          40,   # 50 scenes × 0.8
            'Chikusei virtual': 400,  # virtual_samples_per_epoch
            'Pavia virtual':    500,  # virtual_samples_per_epoch
        }
        for dataset, n_samples in dataset_samples.items():
            epoch_sec = est_full * n_samples
            print(f"       {dataset:22}: ~{epoch_sec/60:.1f} min/epoch  (~{epoch_sec*260/3600:.1f}h total)")

        return sec_per_iter

    except Exception as e:
        fail(f"Speed benchmark thất bại: {e}")
        return None


def check_memory(device, bands=31, feature_dim=128, patch_size=64):
    """Đo mức dùng MPS memory với conv proxy; trả về True nếu không OOM."""
    section(f"6. Memory usage (bands={bands}, dim={feature_dim}, patch={patch_size})")
    try:
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
            before = torch.mps.current_allocated_memory()
        else:
            before = 0

        model = nn.Sequential(
            nn.Conv2d(bands, feature_dim, 3, 1, 1),
            nn.Conv2d(feature_dim, feature_dim * 2, 3, 1, 1),
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.Conv2d(feature_dim, bands, 3, 1, 1),
        ).to(device)

        x = torch.randn(1, bands, patch_size, patch_size, device=device)
        out = model(x)
        loss = out.mean()
        loss.backward()

        if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
            after = torch.mps.current_allocated_memory()
            used_mb = (after - before) / 1024**2
            ok(f"Allocated (proxy model): {used_mb:.0f} MB")
            ok("Full model sẽ lớn hơn ~3–5x tùy depth")
        else:
            ok("Memory tracking không available — model chạy OK")

        # Cleanup
        del model, x, out, loss
        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            fail(f"OOM! Giảm feature_dim hoặc patch_size")
        else:
            fail(str(e))
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Chạy toàn bộ MPS compatibility checks và in tóm tắt."""
    parser = argparse.ArgumentParser(description='MPS compatibility check cho HSI-SR training')
    parser.add_argument('--dataset', default='cave',
                        choices=['cave', 'harvard', 'chikusei', 'pavia'],
                        help='Dataset để estimate speed (ảnh hưởng n_samples ước tính)')
    parser.add_argument('--bands', type=int, default=None,
                        help='Override số bands (mặc định theo dataset)')
    parser.add_argument('--dim', type=int, default=128,
                        help='feature_dim (mặc định 128 cho CAVE/Harvard, dùng 64 cho Chikusei/Pavia trên MPS)')
    parser.add_argument('--patch', type=int, default=64, help='patch_size')
    args = parser.parse_args()

    # Default bands theo dataset
    default_bands = {'cave': 31, 'harvard': 31, 'chikusei': 128, 'pavia': 102}
    bands = args.bands or default_bands[args.dataset]

    print(f"\n{'='*60}")
    print(f"  MPS Check — dataset={args.dataset}, bands={bands}, dim={args.dim}, patch={args.patch}")
    print(f"{'='*60}")

    device = check_mps_available()
    if device is None:
        sys.exit(1)

    fallback_ok = check_fallback_env()
    ops_ok      = check_ops(device)
    bwd_ok      = check_backward(device)
    sec         = check_speed(device, bands, args.dim, args.patch)
    mem_ok      = check_memory(device, bands, args.dim, args.patch)

    section("Tóm tắt")
    all_good = ops_ok and bwd_ok and mem_ok
    if all_good:
        ok("Tất cả checks passed — sẵn sàng train!")
        if not fallback_ok:
            warn("Khuyến nghị set: export PYTORCH_ENABLE_MPS_FALLBACK=1")
    else:
        fail("Có vấn đề cần fix trước khi train dài.")
        if not ops_ok:
            warn("→ Set PYTORCH_ENABLE_MPS_FALLBACK=1 để fallback về CPU cho unsupported ops")
        if not mem_ok:
            warn("→ Giảm feature_dim (128→64) hoặc patch_size (64→48)")


if __name__ == '__main__':
    main()