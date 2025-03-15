import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict


class MultiOutputUnet3D(nn.Module):
    """
    3D U-Net architecture supporting multiple output heads (e.g., segmentation, flow).
    Adapted from Li, X. et al. Real-time denoising enables high-sensitivity fluorescence time-lapse imaging
    beyond the shot-noise limit. Nat Biotechnol 41, 282–292 (2023).
    """
    def __init__(
        self,
        in_channels: int = 1,
        output_heads: Dict[str, dict] = None,
        n_filter: int = 16,
        use_interpolation: bool = True
    ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels (e.g. 1 for grayscale).
        output_heads : Dict[str, dict]
            Dictionary defining output heads, e.g.
            {
                'seg':  {'channels': 1, 'activation': 'sigmoid'},
                'flow': {'channels': 2, 'activation': None}
            }
        n_filter : int
            Base number of convolutional filters.
        use_interpolation : bool
            Whether to use interpolation instead of transposed convolutions.
        """
        super().__init__()
        self.output_heads = output_heads or {
            'default': {'channels': 1, 'activation': 'sigmoid'}
        }
        self.use_interpolation = use_interpolation

        # Encoding path
        self.encode1 = self.conv3d(in_channels, n_filter // 2)
        self.encode2 = self.conv3d(n_filter // 2, n_filter)
        if not use_interpolation:
            self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encode3 = self.conv3d(n_filter, n_filter)
        self.encode4 = self.conv3d(n_filter, 2 * n_filter)
        if not use_interpolation:
            self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encode5 = self.conv3d(2 * n_filter, 2 * n_filter)
        self.encode6 = self.conv3d(2 * n_filter, 4 * n_filter)
        if not use_interpolation:
            self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Middle
        self.middle_conv1 = self.conv3d(4 * n_filter, 4 * n_filter)
        self.middle_conv2 = self.conv3d(4 * n_filter, 8 * n_filter)

        # Decoding path
        if not use_interpolation:
            self.up1 = nn.ConvTranspose3d(8 * n_filter, 8 * n_filter, kernel_size=2, stride=2)
            self.up2 = nn.ConvTranspose3d(4 * n_filter, 4 * n_filter, kernel_size=2, stride=2)
            self.up3 = nn.ConvTranspose3d(2 * n_filter, 2 * n_filter, kernel_size=2, stride=2)
        else:
            self.up1_conv = self.conv3d(8 * n_filter, 8 * n_filter)
            self.up2_conv = self.conv3d(4 * n_filter, 4 * n_filter)
            self.up3_conv = self.conv3d(2 * n_filter, 2 * n_filter)

        self.decode1 = self.conv3d(12 * n_filter, 4 * n_filter)
        self.decode2 = self.conv3d(4 * n_filter, 4 * n_filter)
        self.decode3 = self.conv3d(6 * n_filter, 2 * n_filter)
        self.decode4 = self.conv3d(2 * n_filter, 2 * n_filter)
        self.decode5 = self.conv3d(3 * n_filter, n_filter)
        self.decode6 = self.conv3d(n_filter, n_filter // 2)

        # Output heads
        self.output_layers = nn.ModuleDict()
        for name, cfg in self.output_heads.items():
            self.output_layers[name] = nn.Conv3d(n_filter // 2, cfg['channels'], kernel_size=1)

    def conv3d(self, in_channels, out_channels, kernel_size=3, dropout=0.0, dilation=1):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=dilation, dilation=dilation),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout3d(dropout),
        ]
        return nn.Sequential(*layers)

    def concat(self, x1, x2):
        return torch.cat((x1, x2), dim=1)

    def apply_activation(self, x, activation):
        if activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'tanh':
            return torch.tanh(x)
        elif activation == 'relu':
            return F.relu(x)
        return x

    def forward(self, x):
        # Encoding
        e1 = self.encode1(x)
        e2 = self.encode2(e1)

        if self.use_interpolation:
            m1 = F.interpolate(e2, scale_factor=0.5, mode='nearest')
        else:
            m1 = self.maxpool1(e2)

        e3 = self.encode3(m1)
        e4 = self.encode4(e3)

        if self.use_interpolation:
            m2 = F.interpolate(e4, scale_factor=0.5, mode='nearest')
        else:
            m2 = self.maxpool2(e4)

        e5 = self.encode5(m2)
        e6 = self.encode6(e5)

        if self.use_interpolation:
            m3 = F.interpolate(e6, scale_factor=0.5, mode='nearest')
        else:
            m3 = self.maxpool3(e6)

        # Middle
        mid1 = self.middle_conv1(m3)
        mid2 = self.middle_conv2(mid1)

        # Decoding
        if self.use_interpolation:
            u1 = F.interpolate(mid2, scale_factor=2, mode='nearest')
            u1 = self.up1_conv(u1)
        else:
            u1 = self.up1(mid2)
        c1 = self.concat(u1, e6)
        d1 = self.decode1(c1)
        d2 = self.decode2(d1)

        if self.use_interpolation:
            u2 = F.interpolate(d2, scale_factor=2, mode='nearest')
            u2 = self.up2_conv(u2)
        else:
            u2 = self.up2(d2)
        c2 = self.concat(u2, e4)
        d3 = self.decode3(c2)
        d4 = self.decode4(d3)

        if self.use_interpolation:
            u3 = F.interpolate(d4, scale_factor=2, mode='nearest')
            u3 = self.up3_conv(u3)
        else:
            u3 = self.up3(d4)
        c3 = self.concat(u3, e2)
        d5 = self.decode5(c3)
        d6 = self.decode6(d5)

        # Output heads
        outputs = {}
        for name, cfg in self.output_heads.items():
            logits = self.output_layers[name](d6)
            outputs[name] = self.apply_activation(logits, cfg.get('activation'))

        return outputs
