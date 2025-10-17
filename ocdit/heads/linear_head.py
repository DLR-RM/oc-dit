from typing import Optional, Tuple

from einops import rearrange
from torch import nn

from ocdit.layers.attention import modulate


class LinearHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int,
        out_channels: int,
        t_embed_dim: int = 1024,
        act_layer: Optional[nn.Module] = nn.GELU,
        norm_layer: Optional[nn.Module] = nn.LayerNorm,
        num_patches: int = 16,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.norm = norm_layer(in_channels) if norm_layer else nn.Identity()

        layers = []
        in_sizes = [in_channels] + [hidden_size] * (num_layers - 1)
        out_sizes = [hidden_size] * (num_layers)
        for in_s, out_s in zip(in_sizes, out_sizes):
            layers.extend([nn.Linear(in_s, out_s, bias=True), act_layer()])

        layers.append(nn.Linear(hidden_size, out_channels))
        self.linear = nn.Sequential(*layers)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(t_embed_dim, 2 * in_channels, bias=True)
        )

    def forward(self, x, t, size: Tuple[int, int]):
        nc = x.shape[1]
        x = rearrange(x, "b nc t e -> b (nc t) e")

        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)

        h, w = size
        x = rearrange(x, "b (nc h w) c -> b nc c h w", nc=nc, h=h, w=w)  # bs bc z h w

        return x
