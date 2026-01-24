"""
Spatial-Spectral Attention Module (SSAM)
Đề xuất cho khóa luận: Cải tiến ESSA với attention riêng biệt cho spatial và spectral
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAttention(nn.Module):
    """
    Spectral Attention: Học correlation giữa các spectral bands
    Input: [B, C, H, W] với C = số bands (31 cho CAVE/Harvard)
    Output: [B, C, 1, 1] - attention weights cho mỗi band
    """
    
    def __init__(self, num_channels, reduction=4):
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
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] với spectral attention applied
        """
        # Global pooling theo spatial dimensions
        avg_out = self.mlp(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.mlp(self.max_pool(x))  # [B, C, 1, 1]
        
        # Combine average và max pooling
        spectral_att = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
        
        # Apply attention weights
        out = x * spectral_att
        
        return out


class SpatialAttention(nn.Module):
    """
    Spatial Attention: Học vị trí spatial quan trọng
    Input: [B, C, H, W]
    Output: [B, 1, H, W] - attention map cho spatial locations
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # Convolution để học spatial relationships
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] với spatial attention applied
        """
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
    """
    Spatial-Spectral Attention Module (SSAM)
    Kết hợp cả Spectral Attention và Spatial Attention
    
    ĐÂY LÀ MODULE ĐỀ XUẤT CHÍNH CỦA KHÓA LUẬN!
    """
    
    def __init__(self, num_channels, reduction=4, kernel_size=7, fusion_mode='sequential'):
        """
        Args:
            num_channels: Số channels (31 cho hyperspectral)
            reduction: Reduction ratio cho spectral attention
            kernel_size: Kernel size cho spatial attention (3 hoặc 7)
            fusion_mode: Cách kết hợp 2 attention branches
                - 'sequential': Spectral -> Spatial (mặc định)
                - 'parallel': Parallel fusion
                - 'adaptive': Learnable fusion weights
        """
        super(SpatialSpectralAttention, self).__init__()
        
        self.fusion_mode = fusion_mode
        
        # Spectral attention branch
        self.spectral_attention = SpectralAttention(num_channels, reduction)
        
        # Spatial attention branch
        self.spatial_attention = SpatialAttention(kernel_size)
        
        # Adaptive fusion (nếu dùng adaptive mode)
        if fusion_mode == 'adaptive':
            self.alpha = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] với spatial-spectral attention applied
        """
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
        
        # Residual connection
        out = out + x
        
        return out


class SSAMBlock(nn.Module):
    """
    SSAM Block: Convolution + SSAM + Convolution
    Đây là building block để thay thế ESSAttn trong ESSA gốc
    """
    
    def __init__(self, num_channels, reduction=4, kernel_size=7, fusion_mode='sequential'):
        super(SSAMBlock, self).__init__()
        
        # Pre-processing convolution
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        # SSAM module
        self.ssam = SpatialSpectralAttention(
            num_channels, 
            reduction=reduction,
            kernel_size=kernel_size,
            fusion_mode=fusion_mode
        )
        
        # Post-processing convolution
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        
        # Layer normalization
        self.norm = nn.LayerNorm(num_channels)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
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