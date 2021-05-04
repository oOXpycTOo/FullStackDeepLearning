from typing import Any, Dict, Tuple, List
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


class TrigPosEncoding2D(nn.Module):
    def __init__(self, hidden: int, *args, **kwargs):
        super().__init__()
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.size()
        i = torch.arange(h, device=x.device)
        j = torch.arange(w, device=x.device)
        i, j = torch.meshgrid(i, j)
        i, j = i.reshape(-1, 1), j.reshape(-1, 1)
        w_k = 1 / (1e4 ** (torch.arange(self.hidden, device=x.device) / self.hidden))
        pos = torch.sin(w_k * i) + torch.cos(w_k * j)  # (H*W, C)
        return pos[None]  # (1, H*W, C)


EMBEDDINGS = {'trig': TrigPosEncoding2D}


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
    def __init__(self, height: int = 512,
                 width: int = 512,
                 hidden_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 2,
                 n_tiles: int = 32,
                 embedding: str = 'trig') -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.layers = nn.Sequential(*self._get_transformer_layers(hidden_dim, n_heads, n_layers))
        self.positional_embedding = TrigPosEncoding2D(hidden_dim)
        self.semantic_embedding = nn.Linear(3, hidden_dim)

    def _get_transformer_layers(self, hidden_dim: int, n_heads: int, n_layers: int) -> List[nn.Module]:
        layers = []
        for _ in range(n_layers):
            layers.append(TransformerLayer(hidden_dim, n_heads))
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: torch.Tensor, (B, 3, H, W), B - batch_size, H - image height, W - image width.
        """
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        pos_emb = self.positional_embedding(x)  # (1, h*w, hidden)
        x = x.view(b, h * w, c)  # (b, h*w, c)
        sem_emb = self.semantic_embedding(x)  # (b, h*w, hidden)
        out = self.layers(pos_emb + sem_emb)
        res = self.output_layer(out)
        return out.permute(0, 2, 1).contiguous(), res


class ImageBiTransformer(nn.Module):
    def __init__(self, height: int = 512, width: int = 512,
                 hidden_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 2,
                 n_tiles: int = 32) -> None:
        super().__init__()
        self.height = height // n_tiles
        self.width = width // n_tiles
        self.n_tiles = n_tiles
        self.hidden_dim = hidden_dim
        self.mini_transformer = ImageTransformerEncoder(self.height,
                                                        self.width,
                                                        hidden_dim // n_tiles,
                                                        n_heads // 2,
                                                        n_layers)
        self.layers = nn.Sequential(*self._get_transformer_layers(hidden_dim, n_heads, n_layers))
        self.positional_embedding = TrigPosEncoding2D(hidden_dim)
        self.semantic_embedding = nn.Linear(self.height * self.width, hidden_dim)
        self.output_layer = nn.Sequential(nn.Upsample((self.height * self.n_tiles, self.width * self.n_tiles),
                                                        mode='bilinear',
                                                        align_corners=True),
                                          nn.Conv2d(hidden_dim,
                                                    hidden_dim,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=False),
                                          nn.ReLU(),
                                          nn.Conv2d(hidden_dim,
                                                    3,
                                                    kernel_size=1,
                                                    bias=False))

    def _get_transformer_layers(self, hidden_dim: int, n_heads: int, n_layers: int) -> List[nn.Module]:
        layers = []
        for _ in range(n_layers):
            layers.append(TransformerLayer(hidden_dim, n_heads))
        return layers

    def image_to_tiles(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unfold(2, self.height, self.width).unfold(3, self.height, self.width)  # convert into tiles
        x = x.contiguous()
        return x
    
    def tiles_to_image(self, x: torch.tensor) -> torch.Tensor:
        x = x.view(-1, self.n_tiles * self.n_tiles, self.hidden_dim * self.height * self.width)
        x = x.permute(0, 2, 1)
        x = x.contiguous()
        output_size = (self.height * self.n_tiles, self.width * self.n_tiles)
        kernel_size = (self.height, self.width)
        stride = (self.height, self.width)
        return F.fold(x, output_size=output_size, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: torch.Tensor, (B, 3, H, W), B - batch_size, H - image height, W - image width.
        """
        tiles = self.image_to_tiles(x)  # (B, 3, NT, NT, TH, TW)
        bs, c, nt, nt, h, w = tiles.size()
        tiles = tiles.permute(0, 2, 3, 1, 4, 5)
        tiles = tiles.contiguous()
        tiles = tiles.view(bs * nt * nt, c, h, w)  # (B * NT ** 2, 3, H, W)
        tiles, encoded = self.mini_transformer(tiles)  # (B*NT**2, TH*TW)
        tiles = tiles.view(bs, nt, nt, self.hidden_dim, self.height, self.width)  # (B, NT, NT, TH*TW)
        img = self.tiles_to_image(tiles)  # (B, HIDDEN, H, W)
        pos_emb = self.positional_embedding(encoded)  # (1, NT, HIDDEN)
        encoded = encoded.view(bs, nt*nt, -1)
        sem_emb = self.semantic_embedding(encoded)  # (B, NT*NT, HIDDEN)
        out = self.layers(pos_emb + sem_emb)  # (B, NT*NT, HIDDEN)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(bs, -1, nt, nt)
        out = F.interpolate(out,
                            size=(self.height * self.n_tiles, self.width * self.n_tiles),
                            mode='bilinear')
        out = out + img
        out = self.output_layer(out)
        return out
