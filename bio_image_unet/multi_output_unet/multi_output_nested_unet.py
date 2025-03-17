from typing import Dict, Tuple, Union

import torch
from torch import nn


class FirstVGGBlock(nn.Module):
    """VGG block with Instance Normalization for the first layer"""

    def __init__(self, in_channels, middle_channels, out_channels, dropout=0.):
        super().__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(middle_channels)  # Instance Norm instead of Batch Norm
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels)  # Instance Norm instead of Batch Norm
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=0., dilation=1):
        super().__init__()
        padding = dilation  # ensures same spatial dimensions
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out


class MultiOutputNestedUNet(nn.Module):
    def __init__(self, in_channels=1, output_heads: Dict[str, dict] = None, n_filter: int = 32,
                 deep_supervision: bool = False, dilation: Union[bool, Tuple[int, int, int, int, int]] = False,
                 train_mode: bool = True):
        super().__init__()
        self.output_heads = output_heads or {'default': {'channels': 1, 'activation': 'sigmoid'}}
        self.deep_supervision = deep_supervision
        self.train_mode = train_mode
        self.dilation = dilation
        if self.dilation is False:
            self.dilation = (1, 1, 1, 1, 1)

        nb_filter = [n_filter, n_filter * 2, n_filter * 4, n_filter * 8, n_filter * 16]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0], dilation=self.dilation[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], dilation=self.dilation[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], dilation=self.dilation[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], dilation=self.dilation[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], dilation=self.dilation[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.output_layers = nn.ModuleDict()
            for name, config in self.output_heads.items():
                self.output_layers[f"{name}_1"] = nn.Conv2d(nb_filter[0], config['channels'], kernel_size=1)
                self.output_layers[f"{name}_2"] = nn.Conv2d(nb_filter[0], config['channels'], kernel_size=1)
                self.output_layers[f"{name}_3"] = nn.Conv2d(nb_filter[0], config['channels'], kernel_size=1)
                self.output_layers[f"{name}_4"] = nn.Conv2d(nb_filter[0], config['channels'], kernel_size=1)
        else:
            self.output_layers = nn.ModuleDict()
            for name, config in self.output_heads.items():
                self.output_layers[name] = nn.Conv2d(nb_filter[0], config['channels'], kernel_size=1)

    def apply_activation(self, x, activation):
        if activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'tanh':
            return torch.tanh(x)
        elif activation == 'relu':
            return torch.relu(x)
        return x

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        outputs = {}
        if self.deep_supervision:
            for name, config in self.output_heads.items():
                if self.train_mode:
                    outputs[f"{name}_1"] = self.apply_activation(self.output_layers[f"{name}_1"](x0_1),
                                                                 config.get('activation'))
                    outputs[f"{name}_2"] = self.apply_activation(self.output_layers[f"{name}_2"](x0_2),
                                                                 config.get('activation'))
                    outputs[f"{name}_3"] = self.apply_activation(self.output_layers[f"{name}_3"](x0_3),
                                                                 config.get('activation'))
                    outputs[f"{name}_4"] = self.apply_activation(self.output_layers[f"{name}_4"](x0_4),
                                                                 config.get('activation'))
                    outputs[name] = outputs[f"{name}_4"]
                else:
                    outputs[name] = self.apply_activation(self.output_layers[f"{name}_4"](x0_4),
                                                          config.get('activation'))
        else:
            for name, config in self.output_heads.items():
                outputs[name] = self.apply_activation(self.output_layers[name](x0_4), config.get('activation'))

        return outputs


class MultiOutputNestedUNet_3Levels(nn.Module):
    def __init__(self, in_channels=1, output_heads: Dict[str, dict] = None, n_filter=32, deep_supervision=False,
                 dilation=False, train_mode=True, **kwargs):
        super().__init__()
        self.output_heads = output_heads or {'default': {'channels': 1, 'activation': 'sigmoid'}}
        self.deep_supervision = deep_supervision
        self.train_mode = train_mode
        self.dilation = dilation if dilation is not False else (1, 1, 1, 1)

        # Reduced from five to four entries to remove the fourth pooling level
        nb_filter = [n_filter, n_filter * 2, n_filter * 4, n_filter * 8]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0], self.dilation[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], self.dilation[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], self.dilation[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], self.dilation[3])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.output_layers = nn.ModuleDict()
            for name, config in self.output_heads.items():
                self.output_layers[f"{name}_1"] = nn.Conv2d(nb_filter[0], config['channels'], kernel_size=1)
                self.output_layers[f"{name}_2"] = nn.Conv2d(nb_filter[0], config['channels'], kernel_size=1)
                self.output_layers[f"{name}_3"] = nn.Conv2d(nb_filter[0], config['channels'], kernel_size=1)
        else:
            self.output_layers = nn.ModuleDict()
            for name, config in self.output_heads.items():
                self.output_layers[name] = nn.Conv2d(nb_filter[0], config['channels'], kernel_size=1)

    def apply_activation(self, x, activation):
        if activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'tanh':
            return torch.tanh(x)
        elif activation == 'relu':
            return torch.relu(x)
        return x

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        outputs = {}
        if self.deep_supervision:
            for name, config in self.output_heads.items():
                if self.train_mode:
                    outputs[f"{name}_1"] = self.apply_activation(self.output_layers[f"{name}_1"](x0_1),
                                                                 config.get('activation'))
                    outputs[f"{name}_2"] = self.apply_activation(self.output_layers[f"{name}_2"](x0_2),
                                                                 config.get('activation'))
                    outputs[f"{name}_3"] = self.apply_activation(self.output_layers[f"{name}_3"](x0_3),
                                                                 config.get('activation'))
                    outputs[name] = outputs[f"{name}_3"]
                else:
                    outputs[name] = self.apply_activation(self.output_layers[f"{name}_3"](x0_3),
                                                          config.get('activation'))
        else:
            for name, config in self.output_heads.items():
                outputs[name] = self.apply_activation(self.output_layers[name](x0_3), config.get('activation'))

        return outputs
