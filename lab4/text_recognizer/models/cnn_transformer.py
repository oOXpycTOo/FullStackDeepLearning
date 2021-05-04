from typing import Any, Dict, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 3, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.query_map = self._conv2d(kernel_size, hidden_dim, hidden_dim * n_heads)
        self.key_map = self._conv2d(kernel_size, hidden_dim, hidden_dim * n_heads)
        self.value_map = self._conv2d(kernel_size, hidden_dim, hidden_dim * n_heads)
        self.softmax = nn.Softmax(dim=3)
        self.hidden_dim = hidden_dim
        self.unify = self._conv2d(kernel_size, hidden_dim * n_heads, hidden_dim)

    def _conv2d(self,
                kernel_size,
                input_channels,
                output_channels):
        return nn.Conv2d(input_channels,
                         output_channels,
                         kernel_size=kernel_size,
                         padding=1,
                         bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x tensor of shape (B, N_TILES, N_TILES, HIDDEN_DIM, H, W)
        Returns:
            A tensor of shape (B, S, C)
        """
        bs, n_tiles_h, n_tiles_w, hidden, height, width = x.size()
        x = x.view(-1, hidden, height, width)
        n_heads = self.n_heads
        out_shape = [-1, n_tiles_h * n_tiles_w, n_heads, hidden, height, width]
        query = self.query_map(x).view(*out_shape)
        key = self.key_map(x).view(*out_shape)
        value = self.value_map(x).view(*out_shape)  # Output: (B, N_TILES**2, N_HEADS, HIDDEN, HEIGHT, WEIGHT)
                                                            #  i, l         , j      , c     , m     , n

        query /= (height * width) ** (1 / 4)
        key /= (height * width) ** (1 / 4)

        weights = torch.einsum('ijklmn,icklmn->ikjc', query, key)  # Output: (B, N_HEADS, N_TILES**2, N_TILES**2)
        weights = self.softmax(weights)

        output = torch.einsum('ijkl,iljcmn->ikjcmn', weights, value)
        output = output.contiguous().view(bs * n_tiles_h * n_tiles_w, n_heads * hidden, height, width)
        output = self.unify(output)
        output = output.view(bs, n_tiles_h, n_tiles_w, hidden, height, width)

        return output


class PositionalEncoding2D(nn.Module):
    def __init__(self, height: int, width: int, hidden_dim: int):
        super().__init__()
        assert hidden_dim % 2 == 0, 'Hidden dim should be prime.'
        t_h = torch.arange(height)
        t_w = torch.arange(width)
        t_h, t_w = torch.meshgrid(t_h, t_w)
        t_h, t_w = t_h.flatten(), t_w.flatten()
        k = torch.arange(hidden_dim)
        self.pos = torch.zeros(height * width, hidden_dim)
        functions = [torch.sin, torch.cos]
        for i in range(4):
            f1 = functions[i%2]
            f2 = functions[i//2]
            self.pos[:, i::4] = f1(t_h.reshape(-1, 1) * self._w_k(k[i::4])) + \
                                f2(t_w.reshape(-1, 1) * self._w_k(k[i::4]))
        self.pos = self.pos.view(height, width, hidden_dim)

    def _w_k(self, k):
        d = k.shape[0]
        return 1 / (2 ** (2 * k / d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        pos_expanded = self.pos[None, :, :, :, None, None]
        return pos_expanded


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, height: int, width: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.self_attention = CNNSelfAttention(hidden_dim, 3, n_heads)
        self.mlp = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1),
                                  nn.ReLU(),
                                  nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1))
        self.layer_norm_1 = nn.LayerNorm((hidden_dim, height, width))
        self.layer_norm_2 = nn.LayerNorm((hidden_dim, height, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n_t_h, n_t_w, hidden, h, w = x.size()
        out = self.self_attention(x)  # Output shape: (B, N_TILES, N_TILES, HIDDEN, HEIGHT, WIDTH)
        out = out.view(-1, hidden, h, w)
        identity = x.view(-1, hidden, h, w)
        identity = self.layer_norm_1(out + identity)
        out = self.mlp(identity)
        out = self.layer_norm_2(out + identity)
        return out.view(b, n_t_h, n_t_w, hidden, h, w)


class CNNTransformer(nn.Module):
    def __init__(self, height: int = 512, width: int = 512,
                 hidden_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 2,
                 n_tiles: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.height = height // n_tiles
        self.width = width // n_tiles
        self.n_tiles = n_tiles
        self.layers = nn.Sequential(*[TransformerLayer(hidden_dim, self.height, self.width, n_heads)
                                        for _ in range(n_layers)])
        self.positional_embedding = PositionalEncoding2D(self.n_tiles, self.n_tiles, hidden_dim)
        self.semantic_embedding = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=1),
            nn.ReLU())
        self.output_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1))

    def convert_into_tiles(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unfold(2, self.height, self.width).unfold(3, self.height, self.width)  # convert into tiles
        x = x.contiguous()
        x = x.permute(0, 2, 3, 1, 4, 5)  # Output: (B, N_TILES, N_TILES, C, H, W)
        return x.contiguous()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: torch.Tensor, (B, 3, H, W), B - batch_size, H - image height, W - image width.
        """
        sem_emb = self.semantic_embedding(x)
        sem_emb = self.convert_into_tiles(sem_emb)
        pos_emb = self.positional_embedding(sem_emb)
        out = self.layers(pos_emb + sem_emb)  # (B, N_TILES, N_TILES, HIDDEN, H, W)
        out = out.view(-1, self.n_tiles * self.n_tiles, self.hidden_dim * self.height * self.width)
        out = out.permute(0, 2, 1)
        out = out.contiguous()
        output_size = (self.height * self.n_tiles, self.width * self.n_tiles)
        kernel_size = (self.height, self.width)
        stride = (self.height, self.width)
        out = F.fold(out, output_size=output_size, kernel_size=kernel_size, stride=stride)
        img = self.output_layer(out)
        return img
