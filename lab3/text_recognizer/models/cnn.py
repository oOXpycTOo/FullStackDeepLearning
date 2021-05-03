from typing import Any, Dict, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
IMAGE_SIZE = 28
DEFAULT_BACKBONE = 'resnet'
DEFAULT_BLOCK = 'residual'
DEFAULT_POOL_TYPE = 'max_pool'
SE_REDUCTION_RATE = 4
N_BLOCKS = 2


class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
    """

    def __init__(self, input_channels: int, output_channels: int, **kwargs: nn.Module) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C_in, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C_out, H, W)
        """
        c = self.conv(x)
        r = self.relu(c)
        return r


class ResidualBlock(nn.Module):
    """
    A ResNet-like block, with 3x3 conv and padding of size 1, followed by ReLU and residual connection.
    """
    def __init__(self, input_channels: int, output_channels: int, **kwargs: nn.Module) -> None:
        super().__init__()
        downsample = kwargs.get('downsample', False)
        res_stride = 1
        if downsample:
            res_stride = 2
        self.conv1 = nn.Conv2d(input_channels,
                                output_channels,
                                kernel_size=3,
                                stride=res_stride,
                                padding=1,
                                bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=output_channels) 
        self.conv2 = nn.Conv2d(output_channels,
                                output_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=output_channels)
        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(input_channels,
                                        output_channels,
                                        kernel_size=1,
                                        stride=res_stride,
                                        bias=False)
            self.bn_downsample = nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C_in, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C_out, H, W)
        """
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
            identity = self.bn_downsample(identity)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + identity)
        return out


class SqueezeExcitationBlock(nn.Module):
    """
    A SE block with ResNet block as backbone, with 3x3 conv and padding of size 1, followed by ReLU and residual connection.
    """
    def __init__(self, input_channels: int, output_channels: int, **kwargs: nn.Module) -> None:
        super().__init__()
        downsample = kwargs.get('downsample', False)
        reduction_rate = kwargs.get('reduction_rate', SE_REDUCTION_RATE)
        res_stride = 1
        if downsample:
            res_stride = 2
        self.conv1 = nn.Conv2d(input_channels,
                                output_channels,
                                kernel_size=3,
                                stride=res_stride,
                                padding=1,
                                bias=False)
        self.conv2 = nn.Conv2d(output_channels,
                                output_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=output_channels)
        self.bn2 = nn.BatchNorm2d(num_features=output_channels)
        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(input_channels,
                                        output_channels,
                                        kernel_size=1,
                                        stride=res_stride,
                                        bias=False)
            self.bn_downsample = nn.BatchNorm2d(num_features=output_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_se_1 = nn.Linear(output_channels, output_channels // reduction_rate)
        self.fc_se_2 = nn.Linear(output_channels // reduction_rate, output_channels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C_in, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C_out, H, W)
        """
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
            identity = self.bn_downsample(identity)
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        weights = self.avg_pool(residual).permute(0, 2, 3, 1)
        weights = self.fc_se_1(weights)
        weights = self.relu(weights)
        weights = self.fc_se_2(weights)
        weights = self.sigmoid(weights).permute(0, 3, 1, 2)

        out = residual * weights
        out = self.relu(out + identity)
        return out


class ResNetBackbone(nn.Module):
    DOWNSAMPLE_EVERY = 2
    BLOCKS = {'residual': ResidualBlock, 'squeeze_excitation': SqueezeExcitationBlock}
    def __init__(self, input_size: int,
                conv_dim: int,
                n_blocks: int,
                fc_dim: int,
                block_type: str) -> None:
        super().__init__()
        self.__block_type = block_type
        self.__conv_dim = conv_dim
        conv_output_dim = conv_dim * 2 ** ((n_blocks - 2) // self.DOWNSAMPLE_EVERY)
        self.init_conv = nn.Conv2d(input_size,
                                    conv_dim,
                                    kernel_size=7,
                                    padding=3,
                                    stride=2,
                                    bias=False)
        self.init_bn = nn.BatchNorm2d(conv_dim)
        self.init_pool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.blocks = self._get_blocks(n_blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(conv_output_dim, fc_dim)
        self.relu = nn.ReLU()

    def _get_blocks(self, n_blocks: int) -> torch.nn.ModuleList:
        blocks = []
        input_dim = self.__conv_dim
        output_dim = self.__conv_dim
        block_builder = self.BLOCKS[self.__block_type]
        for i in range(n_blocks):
            if i >= 2 and (i - 2) % self.DOWNSAMPLE_EVERY == 0:
                downsample = True
                output_dim *= 2
            else:
                downsample = False
            blocks.append(block_builder(input_dim, output_dim, downsample=downsample))
            input_dim = output_dim
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.relu(x)
        x = self.init_pool(x)
        x = self.blocks(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.relu(x)
        return x

class StupidSimpleBackbone(nn.Module):
    def __init__(self, input_size: int,
                conv_dim: int,
                n_blocks: int,
                fc_dim: int,
                block_type: str) -> None:
        super().__init__()
        self.conv_1_1 = ConvBlock(input_channels=input_size, output_channels=conv_dim,
                                kernel_size=5, padding=2)
        self.conv_1_2 = ConvBlock(input_channels=conv_dim, output_channels=conv_dim,
                                kernel_size=5, padding=2)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2_1 = ConvBlock(input_channels=conv_dim, output_channels=conv_dim // 2,
                                kernel_size=3, padding=1)
        self.conv_2_2 = ConvBlock(input_channels=conv_dim // 2, output_channels=conv_dim // 2,
                                kernel_size=3, padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(conv_dim // 2 * 49, fc_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.max_pool_1(x)
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.max_pool_2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        return x


class CNN(nn.Module):
    """Simple CNN for recognizing characters in a square image."""
    BACKBONES = {'lenet': StupidSimpleBackbone, 'resnet': ResNetBackbone}

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}


        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])

        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)

        backbone = self.args.get("backbone", DEFAULT_BACKBONE)
        n_blocks = self.args.get("n_blocks", N_BLOCKS)
        block_type = self.args.get("block_type", DEFAULT_BLOCK)
        self.backbone = self.BACKBONES[backbone](input_dims[0], conv_dim, n_blocks, fc_dim, block_type)

        use_dropout = self.args.get("use_dropout")
        if use_dropout:
            self.dropout = nn.Dropout(0.25)
        else:
            self.dropout = nn.Identity()

        self.out_fc = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.out_fc(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
        parser.add_argument("--block_type", type=str, default=DEFAULT_BLOCK)
        parser.add_argument("--n_blocks", type=int, default=N_BLOCKS)
        parser.add_argument("--use_dropout", action='store_true')
        return parser
