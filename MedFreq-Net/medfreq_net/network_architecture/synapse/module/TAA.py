import torch
from torch import nn
from torch.nn import Softmax
import math


class TAA3D(nn.Module):
    def __init__(self, in_dim, q_k_dim, patch_ini, num_heads=8, axis='D'):
        """
        Parameters
        ----------
        in_dim : int
            channel of input tensor
        q_k_dim : int
            channel of Q, K vector
        num_heads : int
            number of attention heads
        axis : str
            attention axis, can be 'D', 'H', or 'W'
        """
        super(TAA3D, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.num_heads = num_heads
        self.head_dim = q_k_dim // num_heads
        self.axis = axis
        D, H, W = patch_ini[0], patch_ini[1], patch_ini[2]

        assert q_k_dim % num_heads == 0, "q_k_dim must be divisible by num_heads"

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        if self.axis == 'D':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, D, 1, 1))
        elif self.axis == 'H':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, H, 1))
        elif self.axis == 'W':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, 1, W))
        else:
            raise ValueError("Axis must be one of 'D', 'H', or 'W'.")

        nn.init.xavier_uniform_(self.pos_embed)
        self.softmax = Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, processed):
        B, C, H, W, D = x.size()

        Q = self.query_conv(x) + self.pos_embed
        K = self.key_conv(processed) + self.pos_embed
        V = self.value_conv(processed)

        Q = Q.view(B, self.num_heads, self.head_dim, H, W, D)
        K = K.view(B, self.num_heads, self.head_dim, H, W, D)
        V = V.view(B, self.num_heads, self.in_dim // self.num_heads, H, W, D)

        scale = math.sqrt(self.head_dim)

        if self.axis == 'D':
            Q = Q.permute(0, 1, 4, 5, 3, 2).contiguous()  # (B, num_heads, H, W, D, head_dim)
            Q = Q.view(B * self.num_heads * H * W, D, self.head_dim)

            K = K.permute(0, 1, 4, 5, 2, 3).contiguous()  # (B, num_heads, H, W, head_dim, D)
            K = K.view(B * self.num_heads * H * W, self.head_dim, D)

            V = V.permute(0, 1, 4, 5, 3, 2).contiguous()  # (B, num_heads, H, W, D, head_dim)
            V = V.view(B * self.num_heads * H * W, D, self.in_dim // self.num_heads)

            attn = torch.bmm(Q, K) / scale
            attn = self.softmax(attn)

            out = torch.bmm(attn, V)
            out = out.view(B, self.num_heads, H, W, D, self.in_dim // self.num_heads)
            out = out.permute(0, 1, 5, 4, 2, 3).contiguous()

        elif self.axis == 'H':
            Q = Q.permute(0, 1, 3, 5, 4, 2).contiguous()  # (B, num_heads, D, W, H, head_dim)
            Q = Q.view(B * self.num_heads * D * W, H, self.head_dim)

            K = K.permute(0, 1, 3, 5, 2, 4).contiguous()  # (B, num_heads, D, W, head_dim, H)
            K = K.view(B * self.num_heads * D * W, self.head_dim, H)

            V = V.permute(0, 1, 3, 5, 4, 2).contiguous()  # (B, num_heads, D, W, H, head_dim)
            V = V.view(B * self.num_heads * D * W, H, self.in_dim // self.num_heads)

            attn = torch.bmm(Q, K) / scale
            attn = self.softmax(attn)

            out = torch.bmm(attn, V)
            out = out.view(B, self.num_heads, D, W, H, self.in_dim // self.num_heads)
            out = out.permute(0, 1, 5, 2, 4, 3).contiguous()

        else:  # self.axis == 'W'
            Q = Q.permute(0, 1, 3, 4, 5, 2).contiguous()  # (B, num_heads, D, H, W, head_dim)
            Q = Q.view(B * self.num_heads * D * H, W, self.head_dim)

            K = K.permute(0, 1, 3, 4, 2, 5).contiguous()  # (B, num_heads, D, H, head_dim, W)
            K = K.view(B * self.num_heads * D * H, self.head_dim, W)

            V = V.permute(0, 1, 3, 4, 5, 2).contiguous()  # (B, num_heads, D, H, W, head_dim)
            V = V.view(B * self.num_heads * D * H, W, self.in_dim // self.num_heads)

            attn = torch.bmm(Q, K) / scale
            attn = self.softmax(attn)

            out = torch.bmm(attn, V)
            out = out.view(B, self.num_heads, D, H, W, self.in_dim // self.num_heads)
            out = out.permute(0, 1, 5, 2, 3, 4).contiguous()

        out = out.view(B, C, H, W, D)

        gamma = torch.sigmoid(self.gamma)
        out = gamma * out + (1 - gamma) * x
        return out


if __name__ == '__main__':

    x = torch.randn(1, 16, 2, 5, 5)
    model = TAA3D(16, 16, (2, 5, 5))

    y = model(x, x)
