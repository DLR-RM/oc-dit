import torch
from einops import rearrange
from torch import nn

from ocdit.layers.attention import (
    CrossAttention,
    ModulatedMlp,
    ModulatedModule,
    SelfAttention,
)


class OCDiTBLock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.cross_attn = ModulatedModule(
            CrossAttention(
                hidden_size,
                num_heads=num_heads,
                dropout=dropout,
            ),
            hidden_size,
        )
        self.self_attn = ModulatedModule(
            SelfAttention(
                hidden_size,
                num_heads=num_heads,
                dropout=dropout,
            ),
            hidden_size,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = ModulatedMlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            dropout=dropout,
        )

    def adaLN_zero(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in [self.cross_attn, self.self_attn, self.mlp]:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        template_tokens: torch.Tensor,
        t: torch.Tensor,
    ):
        bs, nc = x.shape[:2]

        # only attend to the templates of the class
        x = rearrange(x, "bs nc t e -> (bs nc) t e", nc=nc)
        template_tokens = rearrange(template_tokens, "bs nc t e -> (bs nc) t e")
        x, w = self.cross_attn(x, t=t.repeat(nc, 1), mem=template_tokens)
        # self-attention over all class tokens
        x = rearrange(x, "(bs nc) t e -> bs (nc t) e", nc=nc)
        x, _ = self.self_attn(x, t)

        x = self.mlp(x, t)
        x = rearrange(x, "bs (nc t) e -> bs nc t e", nc=nc)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        depth: int,
        mlp_ratio=4.0,
        dropout: float = 0.0,
        return_intermediate_features: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.return_intermediate_features = return_intermediate_features

        self.blocks = nn.ModuleList(
            [
                OCDiTBLock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.initialize_weights()

    def initialize_weights(self):
        for b in self.blocks:
            b.adaLN_zero()

    def forward(self, x, template_tokens, t):
        output = []
        for i, block in enumerate(self.blocks):
            x = block(x, template_tokens, t=t)  # (N, T, D)

            output.append(x)

        if self.return_intermediate_features:
            return output

        return x
