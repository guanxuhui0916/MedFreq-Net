import torch
import torch.nn as nn
import torch.fft
from medfreq_net.network_architecture.synapse.module.TAA import TAA3D

class FrequencyDomainProcessor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        x_amp = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        x_amp = self.conv(x_amp)

        x_recon = torch.fft.ifftn(x_amp * torch.exp(1j * x_phase), dim=(-3, -2, -1))
        return torch.real(x_recon)


class MFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1_1 = nn.Conv3d(256, 32, kernel_size=1)
        self.conv1x1_2 = nn.Conv3d(128, 32, kernel_size=1)
        self.conv1x1_3 = nn.Conv3d(64, 32, kernel_size=1)

        self.freq_processor = FrequencyDomainProcessor(32)
        self.TAA1 = TAA3D(256,256,(4,4,4),axis='H')
        self.TAA2 = TAA3D(128,128,(8,8,8),axis='W')
        self.TAA3 = TAA3D(64,64,(16,16,16),axis='D')
    def forward(self, feat1, feat2, feat3):
        feat1 = self.TAA1(feat1, feat1)
        feat2 = self.TAA2(feat2, feat2)
        feat3 = self.TAA3(feat3, feat3)
        out1 = self.conv1x1_1(feat1)
        out1 = nn.functional.interpolate(out1, size=(32, 32, 32), mode='trilinear')

        out2 = self.conv1x1_2(feat2)
        out2 = nn.functional.interpolate(out2, size=(32, 32, 32), mode='trilinear')

        out3 = self.conv1x1_3(feat3)
        out3 = nn.functional.interpolate(out3, size=(32, 32, 32), mode='trilinear')

        freq1 = self.freq_processor(out1)
        freq2 = self.freq_processor(out2)
        freq3 = self.freq_processor(out3)

        fused1 = out1 + freq1
        fused2 = out2 + freq2
        fused3 = out3 + freq3

        final_output = fused1 + fused2 + fused3
        return final_output

if __name__ == '__main__':
    feat1 = torch.randn(1, 256, 4, 4, 4)
    feat2 = torch.randn(1, 128, 8, 8, 8)
    feat3 = torch.randn(1, 64, 16, 16, 16)

    model = MFF()
    output = model(feat1, feat2, feat3)
    print(output.shape)  # 输出：torch.Size([1, 16, 16, 160, 160])