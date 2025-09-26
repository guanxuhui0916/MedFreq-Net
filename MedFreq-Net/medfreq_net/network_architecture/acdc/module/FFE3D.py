import numpy as np
import torch
import torch.nn as nn


class DynamicFrequencyFilter(nn.Module):

    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // ratio, 1),
            nn.ReLU(),
            nn.Conv3d(in_channels // ratio, in_channels * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _, _ = x.shape
        weights = self.channel_attention(x)
        return weights.view(B, 3, C, 1, 1, 1)


class FFE3D(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.low_cutoff = nn.Parameter(torch.tensor(0.2))
        self.high_cutoff = nn.Parameter(torch.tensor(0.7))
        self.register_buffer('eps', torch.tensor(1e-6))

        self.dynamic_filter = DynamicFrequencyFilter(in_channels)

        self.low_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(4, in_channels)
        )
        self.mid_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(4, in_channels)
        )
        self.high_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(4, in_channels)
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv3d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def get_frequency_mask(self, shape, device):
        H, W, D = shape
        z = torch.linspace(-1, 1, H, device=device)
        y = torch.linspace(-1, 1, W, device=device)
        x = torch.linspace(-1, 1, D, device=device)

        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
        distance = torch.sqrt(X ** 2 + Y ** 2 + Z ** 2) / np.sqrt(3)

        low_mask = torch.sigmoid((self.low_cutoff - distance) / self.eps)
        mid_mask = torch.sigmoid((distance - self.low_cutoff) / self.eps) * \
                   torch.sigmoid((self.high_cutoff - distance) / self.eps)
        high_mask = torch.sigmoid((distance - self.high_cutoff) / self.eps)

        return low_mask, mid_mask, high_mask

    def forward(self, x):
        B, C, H, W, D = x.shape

        channel_weights = self.dynamic_filter(x)

        fft_feat = torch.fft.fftshift(torch.fft.fftn(x, dim=(-3, -2, -1)))
        low_mask, mid_mask, high_mask = self.get_frequency_mask((H, W, D), x.device)

        low_fft = fft_feat * low_mask
        mid_fft = fft_feat * mid_mask
        high_fft = fft_feat * high_mask

        low_feat = torch.fft.ifftn(torch.fft.ifftshift(low_fft), dim=(-3, -2, -1)).real
        mid_feat = torch.fft.ifftn(torch.fft.ifftshift(mid_fft), dim=(-3, -2, -1)).real
        high_feat = torch.fft.ifftn(torch.fft.ifftshift(high_fft), dim=(-3, -2, -1)).real

        low_feat = self.low_conv(low_feat) + x
        mid_feat = self.mid_conv(mid_feat) + x
        high_feat = self.high_conv(high_feat) + x

        fused_feat = (channel_weights[:, 0] * low_feat +
                      channel_weights[:, 1] * mid_feat +
                      channel_weights[:, 2] * high_feat)
        spatial_weight = self.spatial_attn(fused_feat)
        return fused_feat * spatial_weight


if __name__ == '__main__':

    x = torch.randn(1, 16, 16, 16, 16)
    model = FFE3D(16)

    y = model(x)
