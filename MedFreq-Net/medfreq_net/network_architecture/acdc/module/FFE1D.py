import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicFrequencyFilter(nn.Module):

    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // ratio, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // ratio, in_channels * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, N = x.shape
        weights = self.channel_attention(x)
        return weights.view(B, 3, C, 1)


class FFE1D(nn.Module):

    def __init__(self, in_channels, target_shape=(16, 16)):
        super().__init__()
        self.in_channels = in_channels
        self.target_H, self.target_W = target_shape
        self.n_pixels = self.target_H * self.target_W

        self.low_cutoff = nn.Parameter(torch.tensor(0.2))
        self.high_cutoff = nn.Parameter(torch.tensor(0.7))
        self.register_buffer('eps', torch.tensor(1e-6))

        self.dynamic_filter = DynamicFrequencyFilter(in_channels)

        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(5, in_channels)
        )
        self.mid_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(5, in_channels)
        )
        self.high_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(5, in_channels)
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def reshape_to_square(self, x):
        B, C, N = x.shape
        if N < self.n_pixels:
            x_padded = F.pad(x, (0, self.n_pixels - N))
        else:
            x_padded = x[:, :, :self.n_pixels]
        return x_padded.view(B, C, self.target_H, self.target_W)

    def get_frequency_mask(self, device):
        Y, X = torch.meshgrid(
            torch.linspace(-1, 1, self.target_H, device=device),
            torch.linspace(-1, 1, self.target_W, device=device),
            indexing='ij'
        )
        distance = torch.sqrt(X ** 2 + Y ** 2) / np.sqrt(2)

        low_mask = torch.sigmoid((self.low_cutoff - distance) / self.eps)
        mid_mask = torch.sigmoid((distance - self.low_cutoff) / self.eps) * \
                   torch.sigmoid((self.high_cutoff - distance) / self.eps)
        high_mask = torch.sigmoid((distance - self.high_cutoff) / self.eps)

        return low_mask, mid_mask, high_mask

    def forward(self, x):
        B, C, N = x.shape

        x_2d = self.reshape_to_square(x)

        channel_weights = self.dynamic_filter(x).view(B, 3, C, 1, 1)  # [B,3,C,1,1]

        fft_feat = torch.fft.fftshift(torch.fft.fft2(x_2d))

        low_mask, mid_mask, high_mask = self.get_frequency_mask(x.device)

        low_fft = fft_feat * low_mask
        mid_fft = fft_feat * mid_mask
        high_fft = fft_feat * high_mask

        low_feat = torch.fft.ifft2(torch.fft.ifftshift(low_fft)).real
        mid_feat = torch.fft.ifft2(torch.fft.ifftshift(mid_fft)).real
        high_feat = torch.fft.ifft2(torch.fft.ifftshift(high_fft)).real

        low_feat = self.low_conv(low_feat) + x_2d
        mid_feat = self.mid_conv(mid_feat) + x_2d
        high_feat = self.high_conv(high_feat) + x_2d

        fused_feat = (channel_weights[:, 0] * low_feat +
                      channel_weights[:, 1] * mid_feat +
                      channel_weights[:, 2] * high_feat)
        spatial_weight = self.spatial_attn(fused_feat)
        output = (fused_feat * spatial_weight).view(B, C, self.n_pixels)
        return output[:, :, :N]


if __name__ == '__main__':
    x = torch.randn(1, 50, 256)
    model = FFE1D(50, target_shape=(16, 16))

    y = model(x)
