"""
ESSA Original Model (Baseline)
Code gốc từ file bạn cung cấp
"""

import math
import torch
import torch.nn as nn
import numpy as np

np.random.seed(0)


class Convdown(nn.Module):
    def __init__(self, dim):
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
    def __init__(self, dim):
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
    def __init__(self, dim, upscale):
        super(blockup, self).__init__()
        self.convup = Convup(dim)
        self.convdown = Convdown(dim)
        self.convupsample = Upsample(scale=upscale, num_feat=dim)
        self.convdownsample = Downsample(scale=upscale, num_feat=dim)

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


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


class ESSAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
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
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Downsample, self).__init__(*m)


class Upsample(nn.Sequential):
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
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class ESSA(nn.Module):
    """
    ESSA Original Model (Baseline)
    Auto-compatible with HyperspectralDataset
    """

    def __init__(self, dataset=None, inch=None, dim=128, upscale=4):
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
            "model_name": "ESSA (Baseline)",
            "input_channels": self.inch,
            "feature_dim": self.dim,
            "upscale_factor": self.upscale,
            "total_parameters": f"{num_params:,}"
        }
