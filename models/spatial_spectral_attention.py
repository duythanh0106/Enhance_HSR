"""
Spatial-Spectral Attention Module (SSAM) — module attention đề xuất chính.

Kết hợp hai nhánh attention bổ trợ nhau:
  - SpectralAttention : học correlation giữa bands (global avg+max pool → MLP → sigmoid)
  - SpatialAttention  : học vị trí quan trọng (channel pool → conv → sigmoid)

Ba chế độ fusion trong SpatialSpectralAttention:
  'sequential' — spectral trước, spatial sau (mặc định, hiệu quả nhất thực nghiệm)
  'parallel'   — áp dụng song song rồi cộng output
  'adaptive'   — weighted sum với learnable alpha/beta

QUAN TRỌNG:
  - Input/Output của tất cả modules: [B, C, H, W] — shape được bảo toàn
  - SSAMBlock (conv + SSAM + conv) là building block cho ESSA-SSAM và ESSA-SSAM-SpecTrans
  - use_residual=False trong SSAMBlock vì caller tự quản lý skip connection
  - Dùng bởi essa_improved.py và essa_ssam_spectrans.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAttention(nn.Module):
    """Học tầm quan trọng của từng spectral band qua global pooling + MLP.

    Avg-pool và max-pool trên spatial dims → MLP → sigmoid → channel-wise scale.
    Input/Output: [B, C, H, W] — shape bảo toàn, chỉ scale các channels.
    """

    def __init__(self, num_channels, reduction=4):
        """Khởi tạo spectral attention với global pooling + MLP.

        Args:
            num_channels: Số spectral bands C.
            reduction: Tỉ lệ nén hidden dim của MLP — hidden = C // reduction.
        """
        super(SpectralAttention, self).__init__()
        
        # Global pooling để tổng hợp thông tin spatial
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP để học spectral relationships
        self.mlp = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Input [B,C,H,W] → Output [B,C,H,W] — x nhân với spectral attention weights."""
        # Global pooling theo spatial dimensions
        avg_out = self.mlp(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.mlp(self.max_pool(x))  # [B, C, 1, 1]
        
        # Combine average và max pooling
        spectral_att = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
        
        # Apply attention weights
        out = x * spectral_att
        
        return out


class SpatialAttention(nn.Module):
    """Học vị trí spatial quan trọng qua channel pooling + conv.

    Avg-pool và max-pool trên channel dim → concat → conv → sigmoid → spatial scale.
    Input/Output: [B, C, H, W] — shape bảo toàn.
    """

    def __init__(self, kernel_size=7):
        """Khởi tạo spatial attention với channel pooling + conv.

        Args:
            kernel_size: Kích thước conv kernel — 3 hoặc 7; dùng 7 cho receptive field rộng hơn.
        """
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # Convolution để học spatial relationships
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Input [B,C,H,W] → Output [B,C,H,W] — x nhân với spatial attention map."""
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate along channel dimension
        pooled = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        # Learn spatial attention
        spatial_att = self.sigmoid(self.conv(pooled))  # [B, 1, H, W]
        
        # Apply attention weights
        out = x * spatial_att
        
        return out


class SpatialSpectralAttention(nn.Module):
    """Module attention đề xuất chính — kết hợp SpectralAttention + SpatialAttention.

    Hỗ trợ 3 fusion modes (xem module docstring).
    use_residual=True thêm skip connection x + out; False để caller quản lý residual.
    """

    def __init__(self, num_channels, reduction=4, kernel_size=7, fusion_mode='sequential', use_residual=True):
        """Khởi tạo SSAM — kết hợp SpectralAttention và SpatialAttention.

        Args:
            num_channels: Số channels C (spectral bands hoặc features).
            reduction: Tỉ lệ nén hidden dim của spectral MLP (C // reduction).
            kernel_size: Kernel size cho spatial conv — 3 hoặc 7.
            fusion_mode: Chiến lược kết hợp — 'sequential', 'parallel', hoặc 'adaptive'.
            use_residual: True để cộng skip connection x vào output (standalone use).
        """
        super(SpatialSpectralAttention, self).__init__()

        self.fusion_mode = fusion_mode
        self.use_residual = use_residual
        
        # Spectral attention branch
        self.spectral_attention = SpectralAttention(num_channels, reduction)
        
        # Spatial attention branch
        self.spatial_attention = SpatialAttention(kernel_size)
        
        # Adaptive fusion (nếu dùng adaptive mode)
        if fusion_mode == 'adaptive':
            self.alpha = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        """Input [B,C,H,W] → Output [B,C,H,W] — kết hợp spectral và spatial attention."""
        if self.fusion_mode == 'sequential':
            # Sequential: Spectral first, then Spatial
            out = self.spectral_attention(x)
            out = self.spatial_attention(out)
            
        elif self.fusion_mode == 'parallel':
            # Parallel: Apply both, then add
            spectral_out = self.spectral_attention(x)
            spatial_out = self.spatial_attention(x)
            out = spectral_out + spatial_out
            
        elif self.fusion_mode == 'adaptive':
            # Adaptive: Learnable weights
            spectral_out = self.spectral_attention(x)
            spatial_out = self.spatial_attention(x)
            out = self.alpha * spectral_out + self.beta * spatial_out
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        if self.use_residual:
            out = out + x

        return out


class SSAMBlock(nn.Module):
    """Building block: conv3×3 → SSAM (no internal residual) → conv3×3 → + skip.

    Được thiết kế để thay thế ESSAttn trong ESSA gốc; caller tự quản lý residual.
    """

    def __init__(self, num_channels, reduction=4, kernel_size=7, fusion_mode='sequential'):
        """Khởi tạo SSAMBlock — conv + SSAM + conv với skip.

        Args:
            num_channels: Số channels C.
            reduction: Tỉ lệ nén spectral MLP — truyền vào SpatialSpectralAttention.
            kernel_size: Spatial conv kernel size — truyền vào SpatialSpectralAttention.
            fusion_mode: SSAM fusion strategy — truyền vào SpatialSpectralAttention.
        """
        super(SSAMBlock, self).__init__()
        
        # Pre-processing convolution
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        # SSAM module — residual handled by this block, not internally
        self.ssam = SpatialSpectralAttention(
            num_channels,
            reduction=reduction,
            kernel_size=kernel_size,
            fusion_mode=fusion_mode,
            use_residual=False,
        )
        
        # Post-processing convolution
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        
        # Layer normalization
        self.norm = nn.LayerNorm(num_channels)
    
    def forward(self, x):
        """Input [B,C,H,W] → Output [B,C,H,W] — conv + SSAM + conv với skip."""
        identity = x

        # Convolution
        out = self.conv1(x)
        out = self.relu1(out)
        
        # SSAM attention
        out = self.ssam(out)
        
        # Convolution
        out = self.conv2(out)
        
        # Residual connection
        out = out + identity
        
        return out


# Test code
if __name__ == '__main__':
    # Test Spectral Attention
    print("Testing Spectral Attention...")
    x = torch.randn(2, 31, 64, 64)  # [B, C, H, W]
    spectral_att = SpectralAttention(num_channels=31)
    out = spectral_att(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    
    # Test Spatial Attention
    print("\nTesting Spatial Attention...")
    spatial_att = SpatialAttention(kernel_size=7)
    out = spatial_att(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    
    # Test SSAM
    print("\nTesting SSAM (Sequential mode)...")
    ssam = SpatialSpectralAttention(num_channels=31, fusion_mode='sequential')
    out = ssam(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    
    # Test SSAM Block
    print("\nTesting SSAM Block...")
    block = SSAMBlock(num_channels=31)
    out = block(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    
    # Count parameters
    num_params = sum(p.numel() for p in ssam.parameters())
    print(f"\nSSAM parameters: {num_params:,}")
    
    print("\n✅ All tests passed!")