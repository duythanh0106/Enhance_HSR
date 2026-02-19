"""
ESSA-SSAM: ESSA with Spatial-Spectral Attention Module
VERSION: Auto-detect spectral bands from dataset
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatial_spectral_attention import SpatialSpectralAttention


# ==========================================================
# Patch Embedding
# ==========================================================

class PatchEmbed(nn.Module):
    def forward(self, x):
        return x.flatten(2).transpose(1, 2)


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        return x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])


# ==========================================================
# SSAM Blocks
# ==========================================================

class ConvdownSSAM(nn.Module):
    def __init__(self, dim, fusion_mode='sequential'):
        super().__init__()

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

        self.attn = SpatialSpectralAttention(
            num_channels=dim,
            reduction=4,
            kernel_size=7,
            fusion_mode=fusion_mode
        )

        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout2d(0.2)

        self.convd = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim * 2, dim, 1, 1, 0)
        )

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])

        x_embed = self.patch_embed(x)
        x_embed = self.norm(x_embed)

        x_img = self.patch_unembed(x_embed, x_size)
        x_img = self.attn(x_img)

        x_embed = self.patch_embed(x_img)
        x = self.drop(self.patch_unembed(x_embed, x_size))

        x = torch.cat((x, shortcut), dim=1)
        x = self.convd(x)
        return x + shortcut


class ConvupSSAM(nn.Module):
    def __init__(self, dim, fusion_mode='sequential'):
        super().__init__()

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

        self.attn = SpatialSpectralAttention(
            num_channels=dim,
            reduction=4,
            kernel_size=7,
            fusion_mode=fusion_mode
        )

        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout2d(0.2)

        self.convu = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim * 2, dim, 1, 1, 0)
        )

    def forward(self, x):
        shortcut = x
        x_size = (x.shape[2], x.shape[3])

        x_embed = self.patch_embed(x)
        x_embed = self.norm(x_embed)

        x_img = self.patch_unembed(x_embed, x_size)
        x_img = self.attn(x_img)

        x_embed = self.patch_embed(x_img)
        x = self.drop(self.patch_unembed(x_embed, x_size))

        x = torch.cat((x, shortcut), dim=1)
        x = self.convu(x)
        return x + shortcut


# ==========================================================
# Upsample / Downsample
# ==========================================================

class Downsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1))
                m.append(nn.PixelUnshuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, num_feat // 9, 3, 1, 1))
            m.append(nn.PixelUnshuffle(3))
        else:
            raise ValueError("Supported scales: 2^n or 3.")
        super().__init__(*m)


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError("Supported scales: 2^n or 3.")
        super().__init__(*m)


# ==========================================================
# Main Model
# ==========================================================

class ESSA_SSAM(nn.Module):
    """
    ESSA-SSAM (Auto Band Detection Compatible)
    """

    def __init__(self, dataset=None, inch=None,
                 dim=128, upscale=4,
                 fusion_mode='sequential'):
        super().__init__()

        # -------- AUTO DETECT --------
        if dataset is not None:
            inch = dataset.num_bands

        if inch is None:
            raise ValueError("You must provide either dataset or inch.")

        self.inch = inch
        self.dim = dim
        self.upscale = upscale
        self.fusion_mode = fusion_mode

        self.conv_first = nn.Conv2d(inch, dim, 3, 1, 1)
        self.blockup = BlockupSSAM(dim, upscale, fusion_mode)
        self.conv_last = nn.Conv2d(dim, inch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.blockup(x)
        x = self.conv_last(x)
        return x

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        return cls(dataset=dataset, **kwargs)

    def get_model_info(self):
        num_params = sum(p.numel() for p in self.parameters())
        return {
            "model_name": "ESSA-SSAM",
            "input_channels": self.inch,
            "feature_dim": self.dim,
            "upscale_factor": self.upscale,
            "fusion_mode": self.fusion_mode,
            "total_parameters": f"{num_params:,}"
        }


# ==========================================================
# BlockupSSAM (unchanged)
# ==========================================================

class BlockupSSAM(nn.Module):
    def __init__(self, dim, upscale, fusion_mode='sequential'):
        super().__init__()
        self.convup = ConvupSSAM(dim, fusion_mode)
        self.convdown = ConvdownSSAM(dim, fusion_mode)
        self.convupsample = Upsample(upscale, dim)
        self.convdownsample = Downsample(upscale, dim)

    def forward(self, x):
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
