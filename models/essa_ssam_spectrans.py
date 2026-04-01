"""
ESSA-SSAM-SpecTrans: ESSA + Spatial-Spectral Attention + Spectral Transformer

FINAL PROPOSED MODEL CHO KHÓA LUẬN! 🚀⭐

Architecture:
    CNN (local) + SSAM (attention) + Spectral Transformer (global spectral)
"""

import math
import torch
import torch.nn as nn

try:
    from .spatial_spectral_attention import SpatialSpectralAttention
    from .spectral_transformer import SpectralTransformer
except ImportError:
    # Support direct script execution: `python models/essa_ssam_spectrans.py`
    from spatial_spectral_attention import SpatialSpectralAttention
    from spectral_transformer import SpectralTransformer


class PatchEmbed(nn.Module):
    """Chuyển từ image space sang patch embedding"""
    def __init__(self):
        """Initialize the `PatchEmbed` instance.

        Args:
            None.

        Returns:
            None: This method initializes state and returns no value.
        """
        super().__init__()

    def forward(self, x):
        """Run the forward computation for this module.

        Args:
            x: Input parameter `x`.

        Returns:
            Any: Output produced by this function.
        """
        x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, HW, C]
        return x


class PatchUnEmbed(nn.Module):
    """Reshape từ sequence [B, HW, C] về image space [B, C, H, W]."""

    def __init__(self):
        super().__init__()

    def forward(self, x, x_size):
        """Run the forward computation for this module.

        Args:
            x: Tensor [B, HW, C].
            x_size: Tuple (H, W) of the spatial dimensions to restore.

        Returns:
            Tensor [B, C, H, W].
        """
        B, HW, C = x.shape
        # Use C from the tensor — do NOT use a stored embed_dim, which breaks
        # when the channel count differs from the value set at __init__ time.
        x = x.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        return x


def _make_num_heads(dim):
    """Chọn num_heads tối ưu cho SpectralTransformer.

    Ưu tiên head_dim trong khoảng [16, 64] — sweet spot cho attention quality.
    Fallback về heads lớn nhất chia hết dim nếu không có lựa chọn tốt hơn.
    """
    # Ưu tiên: head_dim trong [16, 64]
    for h in (8, 4, 2, 1):
        if dim % h == 0 and 16 <= dim // h <= 64:
            return h
    # Fallback: heads lớn nhất chia hết
    for h in (8, 4, 2, 1):
        if dim % h == 0:
            return h
    return 1


class _ConvBlockSpecTrans(nn.Module):
    """Base block: LayerNorm → SSAM → SpecTrans → Conv(2x→1x) + skip connection.

    Subclasses chỉ khác tên attribute của conv sequential (để backward-compat
    với checkpoints cũ). Override `_conv_attr` để đặt tên.
    """

    _conv_attr = 'conv'  # tên attribute lưu nn.Sequential

    def __init__(self, dim, fusion_mode='sequential', use_spectrans=True,
                 spectrans_depth=2, dropout=0.05):
        """Khởi tạo block.

        Args:
            dim: Số channels đầu vào / đầu ra.
            fusion_mode: Chiến lược fusion SSAM ('sequential', 'parallel', 'adaptive').
            use_spectrans: Có dùng SpectralTransformer hay không.
            spectrans_depth: Số block trong SpectralTransformer.
            dropout: Dropout rate trong conv sequential (mặc định 0.05 thay vì 0.2
                     — rate 0.2 quá cao cho SR task và làm giảm PSNR).
        """
        super().__init__()

        self.use_spectrans = use_spectrans
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()

        # Conv sequential: [dim*2 → dim*2 → dim]
        setattr(self, self._conv_attr, nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(dim * 2, dim, 1, 1, 0),
        ))

        # SSAM — residual handled here, not inside the module
        self.ssam = SpatialSpectralAttention(
            num_channels=dim,
            reduction=4,
            kernel_size=7,
            fusion_mode=fusion_mode,
            use_residual=True,  # standalone use: keep internal residual
        )

        if use_spectrans:
            self.spectral_transformer = SpectralTransformer(
                num_bands=dim,
                depth=spectrans_depth,
                num_heads=_make_num_heads(dim),
            )

        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        """Forward pass: norm → SSAM → SpecTrans → conv + skip."""
        shortcut = x
        x_size = (x.shape[2], x.shape[3])

        # Normalize in sequence space, then restore to image space for SSAM
        x_embed = self.norm(self.patch_embed(x))
        x_img = self.ssam(self.patch_unembed(x_embed, x_size))

        if self.use_spectrans:
            x_img = self.spectral_transformer(x_img)

        # Light dropout before skip-cat
        x_out = self.drop(self.patch_unembed(self.patch_embed(x_img), x_size))

        # Skip-concat then project back to dim channels
        x_out = torch.cat((x_out, shortcut), dim=1)
        x_out = getattr(self, self._conv_attr)(x_out)
        return x_out + shortcut


class ConvdownSpecTrans(_ConvBlockSpecTrans):
    """Convdown block: SSAM + Spectral Transformer + Conv.

    Tên attribute conv là 'convd' để tương thích với checkpoints cũ.
    """
    _conv_attr = 'convd'


class ConvupSpecTrans(_ConvBlockSpecTrans):
    """Convup block: SSAM + Spectral Transformer + Conv.

    Tên attribute conv là 'convu' để tương thích với checkpoints cũ.
    """
    _conv_attr = 'convu'


class Downsample(nn.Sequential):
    """Downsample layer using PixelUnshuffle"""
    def __init__(self, scale, num_feat):
        """Initialize the `Downsample` instance.

        Args:
            scale: Input parameter `scale`.
            num_feat: Input parameter `num_feat`.

        Returns:
            None: This method initializes state and returns no value.
        """
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
        """Initialize the `Upsample` instance.

        Args:
            scale: Input parameter `scale`.
            num_feat: Input parameter `num_feat`.

        Returns:
            None: This method initializes state and returns no value.
        """
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


class BlockupSpecTrans(nn.Module):
    """
    Blockup với SSAM + Spectral Transformer modules
    """
    def __init__(self, dim, upscale, fusion_mode='sequential', use_spectrans=True, spectrans_depth=2):
        """Initialize the `BlockupSpecTrans` instance.

        Args:
            dim: Input parameter `dim`.
            upscale: Input parameter `upscale`.
            fusion_mode: Input parameter `fusion_mode`.
            use_spectrans: Input parameter `use_spectrans`.
            spectrans_depth: Input parameter `spectrans_depth`.

        Returns:
            None: This method initializes state and returns no value.
        """
        super(BlockupSpecTrans, self).__init__()
        
        self.convup = ConvupSpecTrans(dim, fusion_mode, use_spectrans, spectrans_depth)
        self.convdown = ConvdownSpecTrans(dim, fusion_mode, use_spectrans, spectrans_depth)
        self.convupsample = Upsample(scale=upscale, num_feat=dim)
        self.convdownsample = Downsample(scale=upscale, num_feat=dim)

    def forward(self, x):
        # Progressive refinement with skip connections
        """Run the forward computation for this module.

        Args:
            x: Input parameter `x`.

        Returns:
            Any: Output produced by this function.
        """
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


class ESSA_SSAM_SpecTrans(nn.Module):
    """
    ESSA with Spatial-Spectral Attention + Spectral Transformer
    
    ĐÂY LÀ MODEL ĐỀ XUẤT CUỐI CÙNG CHO KHÓA LUẬN! 🚀⭐
    
    Contributions:
    1. Spatial-Spectral Attention (SSAM) - Decoupled spatial/spectral processing
    2. Spectral Transformer - Long-range spectral dependencies
    3. Efficient design - O(C²·HW) complexity
    
    Improvements over ESSA:
    - Better spectral fidelity (lower SAM)
    - Better structural similarity (higher SSIM)
    - Better overall quality (higher PSNR)
    """
    
    def __init__(self, dataset=None, inch=None, dim=128, upscale=4, fusion_mode='sequential', 
                 use_spectrans=True, spectrans_depth=2):
        """Initialize the `ESSA_SSAM_SpecTrans` instance.

        Args:
            dataset: Input parameter `dataset`.
            inch: Input parameter `inch`.
            dim: Input parameter `dim`.
            upscale: Input parameter `upscale`.
            fusion_mode: Input parameter `fusion_mode`.
            use_spectrans: Input parameter `use_spectrans`.
            spectrans_depth: Input parameter `spectrans_depth`.

        Returns:
            None: This method initializes state and returns no value.
        """
        super(ESSA_SSAM_SpecTrans, self).__init__()
        
        # -------- AUTO DETECT NUMBER OF BANDS --------
        if dataset is not None:
            inch = dataset.num_bands

        if inch is None:
            raise ValueError("You must provide either dataset or inch.")
        
        self.inch = inch
        self.dim = dim
        self.upscale = upscale
        self.fusion_mode = fusion_mode
        self.use_spectrans = use_spectrans
        self.spectrans_depth = spectrans_depth
        
        # First convolution
        self.conv_first = nn.Conv2d(inch, dim, 3, 1, 1)
        
        # Main processing block with SSAM + Spectral Transformer
        self.blockup = BlockupSpecTrans(
            dim=dim, 
            upscale=upscale, 
            fusion_mode=fusion_mode,
            use_spectrans=use_spectrans,
            spectrans_depth=spectrans_depth
        )
        
        # Last convolution
        self.conv_last = nn.Conv2d(dim, inch, 3, 1, 1)

    def forward(self, x):
        """Run the forward computation for this module.

        Args:
            x: Input parameter `x`.

        Returns:
            Any: Output produced by this function.
        """
        x = self.conv_first(x)
        x = self.blockup(x)
        x = self.conv_last(x)
        return x
    
    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        """Execute `from_dataset`.

        Args:
            dataset: Input parameter `dataset`.
            **kwargs: Input parameter `**kwargs`.

        Returns:
            Any: Output produced by this function.
        """
        return cls(dataset=dataset, **kwargs)
    
    def get_model_info(self):
        """Execute `get_model_info`.

        Args:
            None.

        Returns:
            Any: Output produced by this function.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'ESSA-SSAM-SpecTrans',
            'input_channels': self.inch,
            'feature_dim': self.dim,
            'upscale_factor': self.upscale,
            'fusion_mode': self.fusion_mode,
            'use_spectrans': self.use_spectrans,
            'spectrans_depth': self.spectrans_depth,
            'total_parameters': f'{num_params:,}',
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }



# Test code
if __name__ == '__main__':
    print("=" * 70)
    print("Testing ESSA-SSAM-SpecTrans Model")
    print("=" * 70)
    
    # Test với different configurations
    configs = [
        {'use_spectrans': False, 'name': 'ESSA-SSAM (without SpecTrans)'},
        {'use_spectrans': True, 'spectrans_depth': 1, 'name': 'ESSA-SSAM-SpecTrans (depth=1)'},
        {'use_spectrans': True, 'spectrans_depth': 2, 'name': 'ESSA-SSAM-SpecTrans (depth=2)'},
    ]
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"{'='*70}")
        
        # Create model
        model = ESSA_SSAM_SpecTrans(
            inch=31, 
            dim=128, 
            upscale=4, 
            fusion_mode='sequential',
            use_spectrans=config.get('use_spectrans', True),
            spectrans_depth=config.get('spectrans_depth', 2)
        )
        
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
        
        print(f"✅ Test passed for {config['name']}!")
    
    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)
    
    # Parameter comparison
    print("\n" + "="*70)
    print("Parameter Comparison:")
    print("="*70)
    
    model_no_trans = ESSA_SSAM_SpecTrans(inch=31, dim=128, upscale=4, use_spectrans=False)
    model_with_trans = ESSA_SSAM_SpecTrans(inch=31, dim=128, upscale=4, use_spectrans=True, spectrans_depth=2)
    
    params_no_trans = sum(p.numel() for p in model_no_trans.parameters())
    params_with_trans = sum(p.numel() for p in model_with_trans.parameters())
    
    print(f"ESSA-SSAM (no SpecTrans): {params_no_trans:,} parameters")
    print(f"ESSA-SSAM-SpecTrans: {params_with_trans:,} parameters")
    print(f"Additional parameters: {params_with_trans - params_no_trans:,} (+{((params_with_trans - params_no_trans) / params_no_trans * 100):.1f}%)")
    
    print("\n🚀 ESSA-SSAM-SpecTrans is ready for training!")
