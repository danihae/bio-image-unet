import torch
from torch import nn


class Unet(nn.Module):
    """
    Neural network for semantic image segmentation U-Net (PyTorch),
    Reference:  Falk, T. et al. U-Net: deep learning for cell counting, detection, and morphometry. Nat Methods 16,
    67–70 (2019).

    Parameters
    ----------
    n_filter : int
        Number of convolutional filters (commonly 16, 32, or 64)
    """
    def __init__(self, in_channels=1, out_channels=1, n_filter=32, dilation=1):

        super().__init__()
        # encode
        self.encode1 = self.conv(in_channels=in_channels, out_channels=n_filter, dilation=dilation)
        self.encode2 = self.conv(n_filter, n_filter, dilation=dilation)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode3 = self.conv(n_filter, 2 * n_filter, dilation=dilation)
        self.encode4 = self.conv(2 * n_filter, 2 * n_filter, dilation=dilation)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode5 = self.conv(2 * n_filter, 4 * n_filter, dilation=dilation)
        self.encode6 = self.conv(4 * n_filter, 4 * n_filter, dilation=dilation)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode7 = self.conv(4 * n_filter, 8 * n_filter, dilation=dilation)
        self.encode8 = self.conv(8 * n_filter, 8 * n_filter, dilation=dilation)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # middle
        self.middle_conv1 = self.conv(8 * n_filter, 16 * n_filter, dilation=dilation)
        self.middle_conv2 = self.conv(16 * n_filter, 16 * n_filter, dilation=dilation)

        # decode
        self.up1 = nn.ConvTranspose2d(16 * n_filter, 8 * n_filter, kernel_size=2, stride=2)
        self.decode1 = self.conv(16 * n_filter, 8 * n_filter)
        self.decode2 = self.conv(8 * n_filter, 8 * n_filter)
        self.up2 = nn.ConvTranspose2d(8 * n_filter, 4 * n_filter, kernel_size=2, stride=2)
        self.decode3 = self.conv(8 * n_filter, 4 * n_filter)
        self.decode4 = self.conv(4 * n_filter, 4 * n_filter)
        self.up3 = nn.ConvTranspose2d(4 * n_filter, 2 * n_filter, kernel_size=2, stride=2)
        self.decode5 = self.conv(4 * n_filter, 2 * n_filter)
        self.decode6 = self.conv(2 * n_filter, 2 * n_filter)
        self.up4 = nn.ConvTranspose2d(2 * n_filter, 1 * n_filter, kernel_size=2, stride=2)
        self.decode7 = self.conv(2 * n_filter, 1 * n_filter)
        self.decode8 = self.conv(1 * n_filter, 1 * n_filter)
        self.final = nn.Sequential(
            nn.Conv2d(n_filter, out_channels=out_channels, kernel_size=1, padding=0),
        )

    def conv(self, in_channels, out_channels, kernel_size=3, dropout=0., dilation=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # Using LeakyReLU
            nn.Dropout2d(dropout)]
        return nn.Sequential(*layers)

    def concat(self, x1, x2):
        if x1.shape == x2.shape:
            return torch.cat((x1, x2), 1)
        else:
            print(x1.shape, x2.shape)
            raise ValueError('concatenation failed: wrong dimensions')

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
        e7 = self.encode7(m3)
        e8 = self.encode8(e7)
        m4 = self.maxpool4(e8)

        mid1 = self.middle_conv1(m4)
        mid2 = self.middle_conv2(mid1)

        u1 = self.up1(mid2)
        c1 = self.concat(u1, e8)
        d1 = self.decode1(c1)
        d2 = self.decode2(d1)
        u2 = self.up2(d2)
        c2 = self.concat(u2, e6)
        d3 = self.decode3(c2)
        d4 = self.decode4(d3)
        u3 = self.up3(d4)
        c3 = self.concat(u3, e4)
        d5 = self.decode5(c3)
        d6 = self.decode6(d5)
        u4 = self.up4(d6)
        c4 = self.concat(u4, e2)
        d7 = self.decode7(c4)
        d8 = self.decode8(d7)
        logits = self.final(d8)
        return torch.sigmoid(logits), logits
