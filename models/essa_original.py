"""
ESSA Original (Baseline) — model baseline không có SSAM hay SpecTrans.

Đây là model gốc ESSA (Efficient Spectral Super-resolution Architecture) dùng
ESSAttn — cơ chế attention riêng dựa trên power normalization thay vì softmax.

Kiến trúc: conv_first → blockup (5 bước progressive) → conv_last
Khác với ESSA-SSAM: dùng ESSAttn thay vì SpatialSpectralAttention.

QUAN TRỌNG:
  - Dùng làm baseline để so sánh với ESSA-SSAM và ESSA-SSAM-SpecTrans
  - inch phải chỉ định qua dataset= hoặc inch=
  - Supported upscale: 2^n hoặc 3 (PixelShuffle/PixelUnshuffle)
  - Factory build thông qua models/factory.py với key 'ESSA' hoặc 'ESSA_Original'
"""

import math
import torch
import torch.nn as nn
import numpy as np

np.random.seed(0)


class Convdown(nn.Module):
    """Conv block với ESSAttn — dùng ở nhánh feature extraction (LR space).

    Pipeline: PatchEmbed → ESSAttn → PatchUnEmbed → concat(x, shortcut) → convd → + shortcut
    """

    def __init__(self, dim):
        """Khởi tạo với dim feature channels."""
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.convd = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim, 1, 1, 0))

        self.attn = ESSAttn(dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        """Input [B,dim,H,W] → Output [B,dim,H,W] — LR-space feature refinement."""
        shortcut = x
        x_size = (x.shape[2], x.shape[3])
        x_embed = self.patch_embed(x)
        x_embed = self.attn(self.norm(x_embed))
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = torch.cat((x, shortcut), dim=1)
        x = self.convd(x)
        x = x + shortcut
        return x


class Convup(nn.Module):
    """Conv block với ESSAttn — dùng ở nhánh HR space (sau upsample).

    Cấu trúc giống Convdown nhưng dùng convu thay convd.
    """

    def __init__(self, dim):
        """Khởi tạo với dim feature channels."""
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.convu = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.2),
                                   nn.Conv2d(dim * 2, dim, 1, 1, 0))
        self.drop = nn.Dropout2d(0.2)
        self.norm = nn.LayerNorm(dim)
        self.attn = ESSAttn(dim)

    def forward(self, x):
        """Input [B,dim,H,W] → Output [B,dim,H,W] — HR-space feature refinement."""
        shortcut = x
        x_size = (x.shape[2], x.shape[3])
        x_embed = self.patch_embed(x)
        x_embed = self.attn(self.norm(x_embed))
        x = self.drop(self.patch_unembed(x_embed, x_size))
        x = torch.cat((x, shortcut), dim=1)
        x = self.convu(x)
        x = x + shortcut
        return x


class blockup(nn.Module):
    """5-bước progressive refinement: LR→HR→LR→HR→LR→HR với accumulated skip connections."""

    def __init__(self, dim, upscale):
        """Khởi tạo 5-step progressive refinement block.

        Args:
            dim: Số feature channels.
            upscale: SR scale factor (2^n hoặc 3).
        """
        super(blockup, self).__init__()
        self.convup = Convup(dim)
        self.convdown = Convdown(dim)
        self.convupsample = Upsample(scale=upscale, num_feat=dim)
        self.convdownsample = Downsample(scale=upscale, num_feat=dim)

    def forward(self, x):
        """5 bước progressive: Input [B,dim,H,W] → Output [B,dim,H*upscale,W*upscale]."""
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


class PatchEmbed(nn.Module):
    """Flatten spatial dims và transpose: [B,C,H,W] → [B,H*W,C]."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """[B,C,H,W] → [B,H*W,C] — chuẩn bị cho attention/norm trên channel dim."""
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchUnEmbed(nn.Module):
    """Đảo ngược PatchEmbed: [B,H*W,C] → [B,embed_dim,H,W]."""

    def __init__(self, in_chans=3, embed_dim=96):
        """Khởi tạo với embed_dim channels — dùng khi reshape về image space."""
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        """[B,H*W,C] → [B,embed_dim,H,W] — x_size là tuple (H, W)."""
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


class ESSAttn(nn.Module):
    """Efficient Spectral-Spatial Attention dùng power normalization thay softmax.

    Q và K được zero-centered rồi chuẩn hóa bằng L2-power, attention map
    tính bằng q²·(k²ᵀ·v)/√N — hiệu quả hơn scaled dot-product trên HSI.
    Input/Output: [B, N, C] (sequence of spatial tokens).
    """

    def __init__(self, dim):
        """Khởi tạo với feature channel width dim."""
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        """Input [B,N,C] → Output [B,N,C] — spectral-spatial attention với power norm."""
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)
        attn = t1 + t2
        attn = self.ln(attn)
        return attn


class Downsample(nn.Sequential):
    """PixelUnshuffle downsampling: [B,num_feat,H,W] → [B,num_feat,H/scale,W/scale].

    Chỉ hỗ trợ scale là 2^n hoặc 3.
    """

    def __init__(self, scale, num_feat):
        """Xây dựng chuỗi PixelUnshuffle layers.

        Args:
            scale: Downscale factor — phải là 2^n hoặc 3.
            num_feat: Số feature channels đầu vào.
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
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Downsample, self).__init__(*m)


class Upsample(nn.Sequential):
    """PixelShuffle upsampling: [B,num_feat,H,W] → [B,num_feat,H*scale,W*scale].

    Chỉ hỗ trợ scale là 2^n hoặc 3.
    """

    def __init__(self, scale, num_feat):
        """Xây dựng chuỗi PixelShuffle layers.

        Args:
            scale: Upscale factor — phải là 2^n hoặc 3.
            num_feat: Số feature channels đầu vào.
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
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class ESSA(nn.Module):
    """ESSA baseline — model gốc không có SSAM hay SpecTrans.

    Kiến trúc: conv_first (inch→dim) → blockup (ESSAttn) → conv_last (dim→inch)
    """

    def __init__(self, dataset=None, inch=None, dim=128, upscale=4):
        """Khởi tạo ESSA baseline.

        Args:
            dataset: HyperspectralDataset — tự đọc num_bands. Hoặc truyền inch trực tiếp.
            inch: Số spectral bands (bắt buộc nếu dataset=None).
            dim: Feature channel width (mặc định 128).
            upscale: SR scale factor (mặc định 4).
        """
        super(ESSA, self).__init__()

        # -------- AUTO DETECT --------
        if dataset is not None:
            inch = dataset.num_bands

        if inch is None:
            raise ValueError("You must provide either dataset or inch.")

        self.inch = inch
        self.dim = dim
        self.upscale = upscale

        self.conv_first = nn.Conv2d(inch, dim, 3, 1, 1)
        self.blockup = blockup(dim=dim, upscale=upscale)
        self.conv_last = nn.Conv2d(dim, inch, 3, 1, 1)

    def forward(self, x):
        """SR inference — Input [B,inch,H,W] → Output [B,inch,H*upscale,W*upscale]."""
        x = self.conv_first(x)
        x = self.blockup(x)
        x = self.conv_last(x)
        return x

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        """Tạo model từ dataset object — tự đọc num_bands."""
        return cls(dataset=dataset, **kwargs)

    def get_model_info(self):
        """Trả về dict thông tin model: name, channels, dim, upscale, params."""
        num_params = sum(p.numel() for p in self.parameters())
        return {
            "model_name": "ESSA (Baseline)",
            "input_channels": self.inch,
            "feature_dim": self.dim,
            "upscale_factor": self.upscale,
            "total_parameters": f"{num_params:,}"
        }
