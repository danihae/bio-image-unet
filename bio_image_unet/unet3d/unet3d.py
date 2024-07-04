import torch
from torch import nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    """
    Neural network for time-consistent segmentation or volume segmentation,
    adapted from Li, X. et al. Real-time denoising enables high-sensitivity fluorescence time-lapse imaging
    beyond the shot-noise limit. Nat Biotechnol 41, 282–292 (2023).


    Parameters
    ----------
    n_filter : int
        Number of convolutional filters (commonly 16, 32, or 64)
    """
    def __init__(self, in_channels=1, out_channels=1, n_filter=16, use_interpolation=False):

        super().__init__()
        self.use_interpolation = use_interpolation

        # encode
        self.encode1 = self.conv3D(in_channels=in_channels, out_channels=n_filter // 2)
        self.encode2 = self.conv3D(n_filter // 2, n_filter)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encode3 = self.conv3D(n_filter, n_filter)
        self.encode4 = self.conv3D(n_filter, 2 * n_filter)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encode5 = self.conv3D(2 * n_filter, 2 * n_filter)
        self.encode6 = self.conv3D(2 * n_filter, 4 * n_filter)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # middle
        self.middle_conv1 = self.conv3D(4 * n_filter, 4 * n_filter)
        self.middle_conv2 = self.conv3D(4 * n_filter, 8 * n_filter)

        # decode
        if not use_interpolation:
            self.up1 = nn.ConvTranspose3d(8 * n_filter, 8 * n_filter, kernel_size=2, stride=2)
            self.up2 = nn.ConvTranspose3d(4 * n_filter, 4 * n_filter, kernel_size=2, stride=2)
            self.up3 = nn.ConvTranspose3d(2 * n_filter, 2 * n_filter, kernel_size=2, stride=2)

        self.decode1 = self.conv3D(12 * n_filter, 4 * n_filter)
        self.decode2 = self.conv3D(4 * n_filter, 4 * n_filter)
        self.decode3 = self.conv3D(6 * n_filter, 2 * n_filter)
        self.decode4 = self.conv3D(2 * n_filter, 2 * n_filter)
        self.decode5 = self.conv3D(3 * n_filter, n_filter)
        self.decode6 = self.conv3D(n_filter, n_filter // 2)
        self.final = nn.Conv3d(n_filter // 2, out_channels=out_channels, kernel_size=1, padding=0)

    def conv3D(self, in_channels, out_channels, kernel_size=3, dropout=0., dilation=1):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # Using LeakyReLU
            nn.Dropout3d(dropout)]
        return nn.Sequential(*layers)

    def concat(self, x1, x2):
        return torch.cat((x1, x2), 1)

    def forward(self, x):
        e1 = self.encode1(x)
        e2 = self.encode2(e1)
        m1 = self.maxpool1(e2)
        e3 = self.encode3(m1)
        e4 = self.encode4(e3)
        m2 = self.maxpool2(e4)
        e5 = self.encode5(m2)
        e6 = self.encode6(e5)
        m3 = self.maxpool3(e6)

        mid1 = self.middle_conv1(m3)
        mid2 = self.middle_conv2(mid1)

        if self.use_interpolation:
            u1 = F.interpolate(mid2, scale_factor=2, mode='trilinear', align_corners=False)
        else:
            u1 = self.up1(mid2)
        c1 = self.concat(u1, e6)
        d1 = self.decode1(c1)
        d2 = self.decode2(d1)
        if self.use_interpolation:
            u2 = F.interpolate(d2, scale_factor=2, mode='trilinear', align_corners=False)
        else:
            u2 = self.up2(d2)
        c2 = self.concat(u2, e4)
        d3 = self.decode3(c2)
        d4 = self.decode4(d3)
        if self.use_interpolation:
            u3 = F.interpolate(d4, scale_factor=2, mode='trilinear', align_corners=False)
        else:
            u3 = self.up3(d4)
        c3 = self.concat(u3, e2)
        d5 = self.decode5(c3)
        d6 = self.decode6(d5)
        logits = self.final(d6)
        return torch.sigmoid(logits), logits
