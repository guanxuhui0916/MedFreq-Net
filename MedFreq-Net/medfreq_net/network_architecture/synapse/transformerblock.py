from typing import Tuple
from medfreq_net.network_architecture.dynunet_block import UnetResBlock
import torch
import torch.nn as nn
import torch.nn.functional as F


class MWF3d(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv3x3 = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.conv5x5 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv7x7 = nn.Conv3d(dim, dim, 7, padding=3, groups=dim)

        self.fusion = nn.Sequential(
            nn.Conv3d(3 * dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv3d(dim // 4, 3, 1),
            nn.Softmax(dim=1)
        )

        self.conv_out = nn.Conv3d(dim, dim, 1)

        self._init_weights()

    def _init_weights(self):
        for conv in [self.conv3x3, self.conv5x5, self.conv7x7]:
            nn.init.dirac_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        identity = x.clone()

        attn3 = self.conv3x3(x)
        attn5 = self.conv5x5(x)
        attn7 = self.conv7x7(x)

        combined = torch.cat([attn3, attn5, attn7], dim=1)
        weights = self.fusion(combined)  # [B,3,H,W,D]

        attn = (weights[:, 0:1] * attn3 +
                weights[:, 1:2] * attn5 +
                weights[:, 2:3] * attn7)

        attn = self.conv_out(attn)

        return identity * attn


class MAP3d(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.proj_1 = nn.Sequential(
            nn.Conv3d(d_model, d_model, 1),
            nn.GroupNorm(8, d_model),
            nn.GELU()
        )

        self.spatial_gating_unit = MWF3d(d_model)

        self.pos_enc = nn.Parameter(torch.randn(1, d_model, 8, 8, 8) * 0.02)
        self.pos_norm = nn.LayerNorm(d_model)
        self.proj_2 = nn.Sequential(
            nn.Conv3d(d_model, d_model, 1),
            nn.Dropout3d(0.1)
        )

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0, 2, 1).view(B, C, H, W, D)
        shortcut = x.clone()

        x = self.proj_1(x)

        x = self.spatial_gating_unit(x)

        pos_enc = F.interpolate(self.pos_enc, size=(H, W, D),
                                mode='trilinear', align_corners=False)
        pos_enc = self.pos_norm(pos_enc.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = x + pos_enc.expand(B, -1, -1, -1, -1)

        x = self.proj_2(x)

        x = x + shortcut

        x = x.view(B, C, H * W * D).permute(0, 2, 1)
        return x

    def flops(self, H, W, D):
        flops = 0
        flops += H * W * D * self.d_model * self.d_model  # proj_1
        flops += H * W * D * self.d_model * self.d_model  # proj_2
        flops += H * W * D * 3 ** 3 * self.d_model
        flops += H * W * D * 5 ** 3 * self.d_model
        flops += H * W * D * 7 ** 3 * self.d_model
        flops += H * W * D * 3 * self.d_model * (self.d_model // 4)
        flops += H * W * D * (self.d_model // 4) * 3
        flops += H * W * D * self.d_model * self.d_model

        return flops

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            spatial_shape: Tuple[int, int, int],
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.map_block = MAP3d(d_model=hidden_size)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.map_block(self.norm(x), B, C, H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


