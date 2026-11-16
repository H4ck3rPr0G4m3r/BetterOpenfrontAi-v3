"""Map-agnostic encoders (spec §11): minimap CNN, shared window CNN, player
transformer with edge-biased attention, entity MLPs, globals, fusion."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(*sizes: int, last_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if last_act or i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.n1 = nn.GroupNorm(8, c)
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(F.gelu(self.n1(x)))
        y = self.c2(F.gelu(self.n2(y)))
        return x + y


def coords(n: int, h: int, w: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij")).unsqueeze(0).expand(n, 2, h, w)


class MinimapEncoder(nn.Module):
    """[N, P, 128, 128] -> summary [N, width], coarse keys [N, ptr, 32, 32], fine keys [N, fine, 128, 128]."""

    def __init__(self, planes: int, channels: tuple, blocks: int, width: int, ptr_dim: int, fine_dim: int,
                 stem_stride: int = 4) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.stem = nn.Conv2d(planes + 2, c1, stem_stride, stride=stem_stride)  # 128 / stride
        self.s1 = nn.Sequential(*[ResBlock(c1) for _ in range(blocks)])
        self.down2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.s2 = nn.Sequential(*[ResBlock(c2) for _ in range(blocks)])
        self.down3 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.s3 = nn.Sequential(*[ResBlock(c3) for _ in range(blocks)])
        self.up = nn.ConvTranspose2d(c3, c2, 2, stride=2)  # back to the keys' resolution
        self.keys = nn.Conv2d(c2, ptr_dim, 1)
        self.fine = nn.Conv2d(planes + 2, fine_dim, 3, padding=1)
        self.summary = nn.Linear(2 * c3, width)

    def forward(self, planes: torch.Tensor):
        n, _, h, w = planes.shape
        x = torch.cat([planes, coords(n, h, w, planes.device, planes.dtype)], 1)
        a = self.s1(self.stem(x))
        b = self.s2(self.down2(a))
        c = self.s3(self.down3(b))
        keys = self.keys(b + self.up(c))
        pooled = torch.cat([c.mean((2, 3)), c.amax((2, 3))], 1)
        return self.summary(pooled), keys, self.fine(x)


class WindowEncoder(nn.Module):
    """[N, W, P, 32, 32] + meta [N, W, M] -> tokens [N, W, width], cell keys [N, W, ptr, 32, 32].

    The CNN runs at 32 / stem_stride; cell keys are its features upsampled back to 32x32 plus a
    1x1 projection of the input planes, so the near pointer keeps cell resolution cheaply."""

    def __init__(self, planes: int, channels: tuple, blocks: int, width: int, ptr_dim: int, meta: int,
                 stem_stride: int = 2) -> None:
        super().__init__()
        c1, c2 = channels
        self.stride = stem_stride
        self.stem = nn.Conv2d(planes + 2, c1, 3, stride=stem_stride, padding=1)
        self.s1 = nn.Sequential(*[ResBlock(c1) for _ in range(blocks)])
        self.down = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.s2 = nn.Sequential(*[ResBlock(c2) for _ in range(blocks)])
        self.keys = nn.Conv2d(c1, ptr_dim, 1)
        self.fine = nn.Conv2d(planes + 2, ptr_dim, 1)
        self.token = nn.Linear(2 * c2, width)
        self.meta = mlp(meta, width, width, last_act=False)

    def forward(self, planes: torch.Tensor, meta: torch.Tensor):
        n, nw, p, h, w = planes.shape
        x = planes.reshape(n * nw, p, h, w)
        x = torch.cat([x, coords(n * nw, h, w, x.device, x.dtype)], 1)
        a = self.s1(self.stem(x))
        b = self.s2(self.down(a))
        tok = self.token(torch.cat([b.mean((2, 3)), b.amax((2, 3))], 1)).view(n, nw, -1)
        keys = self.keys(a)
        if self.stride > 1:
            keys = keys.repeat_interleave(self.stride, 2).repeat_interleave(self.stride, 3)
        keys = (keys + self.fine(x)).view(n, nw, -1, h, w)
        return tok + self.meta(meta), keys


class BiasedAttentionLayer(nn.Module):
    """Pre-norm transformer layer whose attention logits get an additive bias per head."""

    def __init__(self, width: int, heads: int, ff_mult: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.n1 = nn.LayerNorm(width)
        self.n2 = nn.LayerNorm(width)
        self.ff = mlp(width, ff_mult * width, width, last_act=False)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        n, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(n, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.out(y.transpose(1, 2).reshape(n, t, d))
        return x + self.ff(self.n2(x))


class PlayerEncoder(nn.Module):
    """Player tokens [N, 34, PF] with edges [N, 34, 34, EF] -> [N, 34, width]."""

    def __init__(self, pf: int, ef: int, width: int, heads: int, layers: int, ff_mult: int) -> None:
        super().__init__()
        self.embed = mlp(pf, width, width, last_act=False)
        self.slot = nn.Parameter(torch.zeros(34, width))
        self.edge_bias = nn.ModuleList([nn.Linear(ef, heads) for _ in range(layers)])
        self.layers = nn.ModuleList([BiasedAttentionLayer(width, heads, ff_mult) for _ in range(layers)])

    def forward(self, players: torch.Tensor, edges: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        n, t, _ = players.shape
        x = self.embed(players) + self.slot[:t]
        allowed = valid.unsqueeze(1) | torch.eye(t, dtype=torch.bool, device=x.device).unsqueeze(0)  # [N, q, k]
        block = torch.zeros(allowed.shape, dtype=x.dtype, device=x.device).masked_fill(~allowed, -1e4)
        for eb, layer in zip(self.edge_bias, self.layers):
            bias = eb(edges).permute(0, 3, 1, 2) + block.unsqueeze(1)
            x = layer(x, bias)
        return x


class FusionEncoder(nn.Module):
    def __init__(self, width: int, heads: int, layers: int, ff_mult: int, n_kinds: int) -> None:
        super().__init__()
        self.kind = nn.Embedding(n_kinds, width)
        layer = nn.TransformerEncoderLayer(width, heads, ff_mult * width, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens: torch.Tensor, kinds: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = tokens + self.kind(kinds).unsqueeze(0)
        return self.norm(self.tf(x, src_key_padding_mask=~valid))


def sinusoid(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    out = torch.zeros(n, d)
    out[:, 0::2] = torch.sin(pos * div)
    out[:, 1::2] = torch.cos(pos * div)
    return out
