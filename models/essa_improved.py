"""
ESSA-SSAM: ESSA with Spatial-Spectral Attention Module
ĐÂY LÀ MODEL ĐỀ XUẤT CHÍNH CỦA KHÓA LUẬN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatial_spectral_attention import SpatialSpectralAttention, SSAMBlock


class PatchEmbed(nn.Module):
    """Chuyển từ image space sang patch embedding"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, HW, C]
        return x


class PatchUnEmbed(nn.Module):
    """Chuyển từ patch embedding về image space"""
    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


class ConvdownSSAM(nn.Module):
    """
    Convdown block với SSAM thay vì ESSAttn
    Đây là cải tiến chính!
    """
    def __init__(self, dim, fusion_mode='sequential'):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        
        # Convolution layers
        self.convd = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim * 2, dim, 1, 1, 0)
        )

        # SSAM thay vì ESSAttn
        self.attn = SpatialSpectralAttention(
            num_channels=dim,
            reduction=4,
            kernel_size=7,
            fusion_mode=fusion_mode
        )
        
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])
        
        # Patch embedding -> SSAM attention -> Patch unembed
        x_embed = self.patch_embed(x)
        x_embed = self.norm(x_embed)
        
        # Convert back to image for SSAM
        x_img = self.patch_unembed(x_embed, x_size)
        x_img = self.attn(x_img)
        
        # Back to embedding
        x_embed = self.patch_embed(x_img)
        x = self.drop(self.patch_unembed(x_embed, x_size))
        
        # Concatenate with shortcut and process
        x = torch.cat((x, shortcut), dim=1)
        x = self.convd(x)
        x = x + shortcut
        
        return x


class ConvupSSAM(nn.Module):
    """
    Convup block với SSAM thay vì ESSAttn
    """
    def __init__(self, dim, fusion_mode='sequential'):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        
        # Convolution layers
        self.convu = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim * 2, dim, 1, 1, 0)
        )
        
        # SSAM thay vì ESSAttn
        self.attn = SpatialSpectralAttention(
            num_channels=dim,
            reduction=4,
            kernel_size=7,
            fusion_mode=fusion_mode
        )
        
        self.drop = nn.Dropout2d(0.2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])
        
        # Patch embedding -> SSAM attention -> Patch unembed
        x_embed = self.patch_embed(x)
        x_embed = self.norm(x_embed)
        
        # Convert back to image for SSAM
        x_img = self.patch_unembed(x_embed, x_size)
        x_img = self.attn(x_img)
        
        # Back to embedding
        x_embed = self.patch_embed(x_img)
        x = self.drop(self.patch_unembed(x_embed, x_size))
        
        # Concatenate with shortcut and process
        x = torch.cat((x, shortcut), dim=1)
        x = self.convu(x)
        x = x + shortcut
        
        return x


class Downsample(nn.Sequential):
    """Downsample layer using PixelUnshuffle"""
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1))
                m.append(nn.PixelUnshuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, num_feat // 9, 3, 1, 1))
            m.append(nn.PixelUnshuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Downsample, self).__init__(*m)


class Upsample(nn.Sequential):
    """Upsample layer using PixelShuffle"""
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class BlockupSSAM(nn.Module):
    """
    Blockup với SSAM modules
    """
    def __init__(self, dim, upscale, fusion_mode='sequential'):
        super(BlockupSSAM, self).__init__()
        self.convup = ConvupSSAM(dim, fusion_mode=fusion_mode)
        self.convdown = ConvdownSSAM(dim, fusion_mode=fusion_mode)
        self.convupsample = Upsample(scale=upscale, num_feat=dim)
        self.convdownsample = Downsample(scale=upscale, num_feat=dim)

    def forward(self, x):
        # Progressive refinement with skip connections
        xup = self.convupsample(x)
        x1 = self.convup(xup)
        
        xdown = self.convdownsample(x1) + x
        x2 = self.convdown(xdown)
        
        xup = self.convupsample(x2) + x1
        x3 = self.convup(xup)
        
        xdown = self.convdownsample(x3) + x2
        x4 = self.convdown(xdown)
        
        xup = self.convupsample(x4) + x3
        x5 = self.convup(xup)
        
        return x5


class ESSA_SSAM(nn.Module):
    """
    ESSA with Spatial-Spectral Attention Module
    
    ĐÂY LÀ MODEL ĐỀ XUẤT CHÍNH!
    
    Cải tiến so với ESSA gốc:
    - Thay ESSAttn bằng SpatialSpectralAttention
    - Tách biệt spatial và spectral attention
    - Hỗ trợ 3 fusion modes: sequential, parallel, adaptive
    """
    
    def __init__(self, inch=31, dim=128, upscale=4, fusion_mode='sequential'):
        """
        Args:
            inch: Số kênh input (31 cho CAVE/Harvard)
            dim: Số feature channels
            upscale: Upscale factor (2, 4, 8)
            fusion_mode: Cách fusion spatial-spectral ('sequential', 'parallel', 'adaptive')
        """
        super(ESSA_SSAM, self).__init__()
        
        self.inch = inch
        self.dim = dim
        self.upscale = upscale
        self.fusion_mode = fusion_mode
        
        # First convolution
        self.conv_first = nn.Conv2d(inch, dim, 3, 1, 1)
        
        # Main processing block with SSAM
        self.blockup = BlockupSSAM(dim=dim, upscale=upscale, fusion_mode=fusion_mode)
        
        # Last convolution
        self.conv_last = nn.Conv2d(dim, inch, 3, 1, 1)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - Low resolution hyperspectral image
        Returns:
            out: [B, C, H*upscale, W*upscale] - High resolution output
        """
        x = self.conv_first(x)
        x = self.blockup(x)
        x = self.conv_last(x)
        return x
    
    def get_model_info(self):
        """Trả về thông tin model để report trong khóa luận"""
        num_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'ESSA-SSAM',
            'input_channels': self.inch,
            'feature_dim': self.dim,
            'upscale_factor': self.upscale,
            'fusion_mode': self.fusion_mode,
            'total_parameters': f'{num_params:,}',
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# Test code
if __name__ == '__main__':
    print("=" * 60)
    print("Testing ESSA-SSAM Model")
    print("=" * 60)
    
    # Test với different fusion modes
    fusion_modes = ['sequential', 'parallel', 'adaptive']
    
    for mode in fusion_modes:
        print(f"\n{'='*60}")
        print(f"Testing fusion mode: {mode}")
        print(f"{'='*60}")
        
        # Create model
        model = ESSA_SSAM(inch=31, dim=128, upscale=4, fusion_mode=mode)
        
        # Print model info
        info = model.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Test forward pass
        x = torch.randn(1, 31, 64, 64)
        print(f"\nInput shape: {x.shape}")
        
        with torch.no_grad():
            out = model(x)
        
        print(f"Output shape: {out.shape}")
        assert out.shape == (1, 31, 256, 256), "Output shape mismatch!"
        
        print(f"✅ Test passed for {mode} mode!")
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)