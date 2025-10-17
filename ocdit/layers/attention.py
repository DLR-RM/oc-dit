from typing import Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ModulatedMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.module = Mlp(
            in_features,
            hidden_features,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(in_features, 3 * in_features, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(t).chunk(3, dim=1)
        x = x + gate.unsqueeze(1) * self.module(modulate(self.norm(x), shift, scale))

        return x


class ModulatedModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        hidden_size: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.module = module
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(t).chunk(3, dim=1)

        if mem is None:
            output, weights = self.module(
                modulate(self.norm(x), shift, scale), return_weights=return_weights
            )
            x = x + gate.unsqueeze(1) * output
        else:
            output, weights = self.module(
                modulate(self.norm(x), shift, scale), mem, return_weights=return_weights
            )
            x = x + gate.unsqueeze(1) * output
        return x, weights


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        # self.self_attn = Attention(dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        assert mem is None, "SelfAttention does not support memory"
        q = k = v = x
        x, w = self.self_attn(q, k, v, need_weights=return_weights)
        return self.dropout(x), w


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor:
        q = x
        k = v = mem
        x, w = self.cross_attn(q, k, v, need_weights=return_weights)

        return self.dropout(x), w
