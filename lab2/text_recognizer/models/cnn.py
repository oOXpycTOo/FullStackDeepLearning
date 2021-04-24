from typing import Any, Dict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
IMAGE_SIZE = 28
DEFAULT_CONV_BLOCK = 'ConvBlock'
DEFAULT_POOL_TYPE = 'ConvPool'


class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
    """

    def __init__(self, input_channels: int, output_channels: int, **kwargs: nn.Module) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
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
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=output_channels)
        self.bn2 = nn.BatchNorm2d(num_features=output_channels)
        downsample = kwargs.get('downsample', False)
        if downsample:
            self.downsample = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        else:
            self.downsample = None

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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + identity)
        return out


class CNN(nn.Module):
    """Simple CNN for recognizing characters in a square image."""
    CONV_BLOCKS = {'ConvBlock': ConvBlock, 'ResidualBlock': ResidualBlock}

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])

        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)

        conv_block = self.args.get("conv_block", DEFAULT_CONV_BLOCK)
        instantiator = self.CONV_BLOCKS[conv_block]

        self.conv1 = instantiator(input_dims[0], conv_dim, downsample=True)
        self.conv2 = instantiator(conv_dim, conv_dim, downsample=False)
        self.dropout = nn.Dropout(0.25)

        pool_type = self.args.get("pool_type", DEFAULT_POOL_TYPE)
        self.pooling = self._get_pooling(pool_type, conv_dim)

        # Because our 3x3 convs have padding size 1, they leave the input size unchanged.
        # The 2x2 max-pool divides the input size by 2. Flattening squares it.
        conv_output_size = IMAGE_SIZE // 2
        fc_input_dim = int(conv_output_size * conv_output_size * conv_dim)
        self.fc1 = nn.Linear(fc_input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
        
    def _get_pooling(self, pool_type: str, dim: int):
        if pool_type == 'MaxPool':
            return nn.MaxPool2d(2)
        elif pool_type == 'ConvPool':
            return nn.Conv2d(dim, dim, kernel_size=2, stride=2)

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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.poolingl(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--conv_block", type=str, default=DEFAULT_CONV_BLOCK)
        parser.add_argument("--pool_type", type=str, default=DEFAULT_POOL_TYPE)
        return parser
