from typing import Any, Dict, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.query_map = nn.Linear(hidden_dim, hidden_dim * n_heads)
        self.key_map = nn.Linear(hidden_dim, hidden_dim * n_heads)
        self.value_map = nn.Linear(hidden_dim, hidden_dim * n_heads)
        self.softmax = nn.Softmax(dim=2)
        self.hidden_dim = hidden_dim
        self.unify = nn.Linear(hidden_dim * n_heads, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x tensor of shape (B, S, C)
        Returns:
            A tensor of shape (B, S, C)
        """
        b, s, c = x.size()
        n_heads = self.n_heads
        query = self.query_map(x).view(b, s, n_heads, c)  # (B, S, C) -> (B, S, N_HEADS, C)
        key = self.key_map(x).view(b, s, n_heads, c)  # (B, S, C) -> (B, S, N_HEADS, C)
        value = self.value_map(x).view(b, s, n_heads, c)  # (B, S, C) -> (B, S, N_HEADS, C)

        query = query.transpose(1, 2).contiguous().view(b * n_heads, s, c)
        key = key.transpose(1, 2).contiguous().view(b * n_heads, s, c)
        value = value.transpose(1, 2).contiguous().view(b * n_heads, s, c)

        query /= c ** (1 / 4)
        key /= c ** (1 / 4)

        weights = torch.bmm(query, key.transpose(1, 2))  # computes w_ij as q_i by k_j in a batch manner
                                                         # (B * N_HEADS, S, S)
        weights = self.softmax(weights) # softmax along channel dim
        
        output = torch.bmm(weights, value).view(b, n_heads, s, c)
        output = output.transpose(1, 2).contiguous().view(b, s, c * n_heads)
        output = self.unify(output)
        return output


class PositionalEncoding1D(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int):
        super().__init__()
        assert hidden_dim % 2 == 0, 'Hidden dim should be prime.'
        t = torch.arange(seq_len)
        k = torch.arange(hidden_dim)
        self.pos = torch.zeros(seq_len, hidden_dim)
        self.pos[:, ::2] = torch.sin(t.reshape(-1, 1) * self._w_k(k[::2]))
        self.pos[:, 1::2] = torch.cos(t.reshape(-1, 1) * self._w_k(k[1::2]))
    
    def _w_k(self, k):
        d = k.shape[0]
        return 1 / (1e4 ** (2 * k / d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        b, s, c = t.size()
        pos_expanded = self.pos[None].expand(b, s, c)
        return pos_expanded


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

    def _w_k(self, k):
        d = k.shape[0]
        return 1 / (1e4 ** (2 * k / d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        b, s, c = t.size()
        pos_expanded = self.pos[None].expand(b, s, c)
        return pos_expanded


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.self_attention = SelfAttention(hidden_dim, n_heads)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim * 4, hidden_dim))
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.self_attention(x)
        identity = self.layer_norm_1(out + identity)
        out = self.mlp(identity)
        out = self.layer_norm_2(out + identity)
        return out



class Transformer(nn.Module):
    def __init__(self, num_tokens: int,
                seq_len: int,
                hidden_dim: int = 256,
                n_heads: int = 8,
                n_layers: int = 2) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.layers = nn.Sequential(*[TransformerLayer(hidden_dim, n_heads) for _ in range(n_layers)])
        self.positional_embedding = PositionalEncoding1D(seq_len, hidden_dim)
        self.semantic_embedding = nn.Embedding(num_tokens, hidden_dim)
        self.top_probs = nn.Linear(hidden_dim, num_tokens)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: torch.Tensor, (B, S), B - batch_size, S - sequence length.
        """
        sem_emb = self.semantic_embedding(x)
        pos_emb = self.positional_embedding(sem_emb)
        out = self.layers(pos_emb + sem_emb)
        return self.top_probs(out)


class ImageTransformerEncoder(nn.Module):
    def __init__(self, height: int = 512, width: int = 512,
                 hidden_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 2,
                 n_tiles: int = 32) -> None:
        super().__init__()
        self.height = height // n_tiles
        self.width = width // n_tiles
        self.n_tiles = n_tiles
        self.layers = nn.Sequential(*[TransformerLayer(hidden_dim, n_heads) for _ in range(n_layers)])
        self.positional_embedding = nn.Embedding(n_tiles * n_tiles, hidden_dim)
        self.semantic_embedding = nn.Linear(self.height * self.width * 3, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 3 * self.height * self.width)

    def convert_into_tiles(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unfold(2, self.height, self.width).unfold(3, self.height, self.width)  # convert into tiles
        x = x.contiguous()
        x = x.view(-1, 3, self.n_tiles * self.n_tiles, self.height * self.width)  # flatten the tiles
        x = x.permute(0, 2, 1, 3)  # (B, 3, S, C) -> (B, S, 3, C)
        x = x.contiguous()
        return x.view(-1, self.n_tiles * self.n_tiles, 3 * self.height * self.width)  # (B, S, 3 * C)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: torch.Tensor, (B, 3, H, W), B - batch_size, H - image height, W - image width.
        """
        tiles = self.convert_into_tiles(x)
        sem_emb = self.semantic_embedding(tiles)
        t = torch.arange(self.n_tiles)
        t_x, t_y = torch.meshgrid(t, t)
        t = t_x.flatten() + t_y.flatten()
        t = t[None].expand(x.size(0), self.n_tiles * self.n_tiles)
        pos_emb = self.positional_embedding(t)
        out = self.layers(pos_emb + sem_emb)
        out = self.output_layer(out)
        tiles = out.view(-1, self.n_tiles * self.n_tiles, 3 * self.height * self.width).permute(0, 2, 1)
        tiles = tiles.contiguous()
        output_size = (self.height * self.n_tiles, self.width * self.n_tiles)
        kernel_size = (self.height, self.width)
        stride = (self.height, self.width)
        img = F.fold(tiles, output_size=output_size, kernel_size=kernel_size, stride=stride)
        return img
