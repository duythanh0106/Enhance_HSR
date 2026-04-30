"""
Spectral Transformer — học long-range dependencies giữa các spectral bands.

Đây là contribution bổ sung trong ESSA-SSAM-SpecTrans: transformer chỉ hoạt
động trên spectral dimension thay vì spatial, giảm complexity đáng kể:
  O(C² × HW) thay vì O((HW)² × C²) của full spatial transformer.

Gồm 4 thành phần theo thứ tự từ nhỏ đến lớn:
  SpectralMultiHeadAttention  — attention trên bands cho mỗi pixel
  SpectralFeedForward         — MLP trên spectral dim
  SpectralTransformerBlock    — LayerNorm + Attention + FFN với residuals
  SpectralTransformer         — stack N blocks + final LayerNorm
  SpectralTransformerWithConv — hybrid: conv spatial + SpectralTransformer

QUAN TRỌNG:
  - Input/Output của SpectralTransformer: [B, C, H, W] — shape bảo toàn
  - Reshape nội bộ: [B,C,H,W] → [B,HW,C] → xử lý → [B,C,H,W]
  - num_bands phải chia hết cho num_heads (kiểm tra trong __init__)
  - Dùng bởi essa_ssam_spectrans.py khi use_spectrans=True
"""

import torch
import torch.nn as nn
import math


class SpectralMultiHeadAttention(nn.Module):
    """Multi-head attention trên spectral dimension — mỗi head xử lý head_dim bands.

    Khác với spatial transformer: attention được tính giữa các bands (C tokens),
    không phải giữa các pixels (HW tokens). Dùng Conv1d để project Q/K/V.
    Input/Output shape (sau reshape nội bộ): [B, H*W, C].
    """

    def __init__(self, num_bands=31, num_heads=4, dropout=0.1):
        """Khởi tạo spectral multi-head attention.

        Args:
            num_bands: Số spectral bands C — phải chia hết cho num_heads.
            num_heads: Số attention heads; head_dim = num_bands // num_heads.
            dropout: Dropout rate sau attention và projection (mặc định 0.1).
        """
        super().__init__()

        assert num_bands % num_heads == 0, (
            f"num_bands ({num_bands}) must be divisible by num_heads ({num_heads})"
        )

        self.num_bands = num_bands
        self.num_heads = num_heads
        self.head_dim = num_bands // num_heads

        # Learnable projections on spectral channels for each spatial position.
        # Input in attention is reshaped to [B, C, N], N = H*W.
        self.qkv = nn.Conv1d(num_bands, num_bands * 3, kernel_size=1)
        self.proj = nn.Conv1d(num_bands, num_bands, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Input [B,HW,C] → Output [B,HW,C] — spectral multi-head attention."""
        B, N, C = x.shape

        # [B, N, C] -> [B, C, N] so attention is computed across spectral tokens.
        x = x.transpose(1, 2).contiguous()

        # Generate Q, K, V in spectral space: [B, 3C, N].
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Split heads: [B, C, N] -> [B, heads, head_dim, N].
        q = q.reshape(B, self.num_heads, self.head_dim, N)
        k = k.reshape(B, self.num_heads, self.head_dim, N)
        v = v.reshape(B, self.num_heads, self.head_dim, N)

        # Attention over spectral dimension per head: [B, heads, head_dim, head_dim].
        # Scale by head_dim (the key dimension), not N (spatial tokens).
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply spectral attention and merge heads back to [B, C, N].
        x = torch.matmul(attn, v).reshape(B, C, N)

        # Output projection then restore [B, N, C].
        x = self.proj(x)
        x = self.dropout(x)
        return x.transpose(1, 2).contiguous()


class SpectralFeedForward(nn.Module):
    """FFN (2-layer MLP) cho spectral transformer block — expand rồi project về num_bands."""

    def __init__(self, num_bands=31, mlp_ratio=4.0, dropout=0.1):
        """Khởi tạo spectral feed-forward network.

        Args:
            num_bands: Số spectral bands C (kích thước input và output).
            mlp_ratio: Tỉ lệ mở rộng hidden dim — hidden = num_bands * mlp_ratio.
            dropout: Dropout rate sau activation (mặc định 0.1).
        """
        super().__init__()

        hidden_dim = int(num_bands * mlp_ratio)

        self.fc1 = nn.Linear(num_bands, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_bands)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Input [B,HW,C] → Output [B,HW,C] — MLP trên spectral dim với GELU."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SpectralTransformerBlock(nn.Module):
    """Một block transformer spectral: LN → Attention → residual → LN → FFN → residual.

    Chuẩn Pre-LN transformer block nhưng attention trên bands thay vì pixels.
    Input/Output: [B, H*W, C].
    """

    def __init__(self, num_bands=31, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        """Khởi tạo một spectral transformer block.

        Args:
            num_bands: Số spectral bands C.
            num_heads: Số attention heads — xem SpectralMultiHeadAttention.
            mlp_ratio: Hidden dim multiplier — xem SpectralFeedForward.
            dropout: Dropout rate (mặc định 0.1).
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(num_bands)
        self.attn = SpectralMultiHeadAttention(num_bands, num_heads, dropout)

        self.norm2 = nn.LayerNorm(num_bands)
        self.ffn = SpectralFeedForward(num_bands, mlp_ratio, dropout)

    def forward(self, x):
        """Input [B,HW,C] → Output [B,HW,C] — attention + FFN với double residual."""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SpectralTransformer(nn.Module):
    """Stack của N SpectralTransformerBlock + final LayerNorm.

    Học long-range spectral correlation với complexity O(C² × HW).
    Input/Output: [B, C, H, W] — reshape nội bộ sang [B,HW,C] và back.
    """

    def __init__(self, num_bands=31, depth=2, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        """Khởi tạo stack N transformer blocks.

        Args:
            num_bands: Số spectral bands C.
            depth: Số SpectralTransformerBlock xếp chồng.
            num_heads: Số attention heads.
            mlp_ratio: Hidden dim multiplier cho FFN.
            dropout: Dropout rate (mặc định 0.1).
        """
        super().__init__()

        self.num_bands = num_bands
        self.depth = depth

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            SpectralTransformerBlock(
                num_bands=num_bands,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(num_bands)

    def forward(self, x):
        """Input [B,C,H,W] → Output [B,C,H,W] — spectral transformer với N blocks."""
        B, C, H, W = x.shape

        # Reshape: [B, C, H, W] -> [B, HW, C]
        # Mỗi pixel có spectral signature length C
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Reshape back: [B, HW, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


class SpectralTransformerWithConv(nn.Module):
    """Hybrid: conv spatial features + SpectralTransformer + conv, với residual.

    Conv học spatial structure; SpecTrans học spectral correlations; kết hợp
    cả hai qua residual connection về input.
    Input/Output: [B, num_bands, H, W].
    """

    def __init__(self, num_bands=31, depth=2, num_heads=4):
        """Khởi tạo hybrid conv + spectral transformer module.

        Args:
            num_bands: Số spectral bands C.
            depth: Số transformer blocks.
            num_heads: Số attention heads.
        """
        super().__init__()

        # Conv cho spatial features
        self.conv_before = nn.Conv2d(num_bands, num_bands, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # Spectral Transformer
        self.spectral_transformer = SpectralTransformer(
            num_bands=num_bands,
            depth=depth,
            num_heads=num_heads
        )

        # Conv sau transformer
        self.conv_after = nn.Conv2d(num_bands, num_bands, 3, 1, 1)

    def forward(self, x):
        """Input [B,C,H,W] → Output [B,C,H,W] — conv + spectral transformer + residual."""
        identity = x

        # Spatial features (Conv)
        x = self.conv_before(x)
        x = self.relu(x)

        # Spectral dependencies (Transformer)
        x = self.spectral_transformer(x)

        # Final conv
        x = self.conv_after(x)

        # Residual
        x = x + identity

        return x


# Test code
if __name__ == '__main__':
    print("Testing Spectral Transformer Components...")
    print("=" * 70)

    # Create test input
    B, C, H, W = 2, 31, 64, 64
    x = torch.randn(B, C, H, W)

    # Test 1: Spectral Multi-Head Attention
    print("\n1. Testing SpectralMultiHeadAttention...")
    x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
    attn = SpectralMultiHeadAttention(num_bands=31, num_heads=1)
    out = attn(x_flat)
    print(f"   Input: {x_flat.shape} -> Output: {out.shape}")
    assert out.shape == x_flat.shape
    print("   ✓ Pass")

    # Test 2: Spectral Transformer Block
    print("\n2. Testing SpectralTransformerBlock...")
    block = SpectralTransformerBlock(num_bands=31, num_heads=1)
    out = block(x_flat)
    print(f"   Input: {x_flat.shape} -> Output: {out.shape}")
    assert out.shape == x_flat.shape
    print("   ✓ Pass")

    # Test 3: Spectral Transformer (full)
    print("\n3. Testing SpectralTransformer...")
    spectrans = SpectralTransformer(num_bands=31, depth=2, num_heads=1)
    out = spectrans(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == x.shape
    print("   ✓ Pass")

    # Count parameters
    num_params = sum(p.numel() for p in spectrans.parameters())
    print(f"   Parameters: {num_params:,}")

    # Test 4: Spectral Transformer with Conv
    print("\n4. Testing SpectralTransformerWithConv...")
    spectrans_conv = SpectralTransformerWithConv(num_bands=31, depth=2, num_heads=1)
    out = spectrans_conv(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == x.shape
    print("   ✓ Pass")

    num_params = sum(p.numel() for p in spectrans_conv.parameters())
    print(f"   Parameters: {num_params:,}")

    # Complexity analysis
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print(f"   Input size: [{B}, {C}, {H}, {W}]")
    print(f"   Spectral Transformer: O(C² × HW) = O({C**2} × {H*W}) = O({C**2 * H*W:,})")
    print(f"   Full Transformer:     O((HW)² × C²) = O({(H*W)**2} × {C**2})")
    print(f"   Speedup: {((H*W)**2 * C**2) / (C**2 * H*W):.0f}x faster")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
