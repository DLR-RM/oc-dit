from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from ocdit.utils import misc


def normal_init(
    module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def make_attn(in_channels: int, attn_type: str = "vanilla"):
    assert attn_type in ["vanilla", "none"], (
        f"Attention type {attn_type} not supported. "
        f"Supported types are ['vanilla', 'none']"
    )
    if attn_type == "vanilla":
        return AttentionBlock(in_channels)
    else:
        return nn.Identity(in_channels)


def make_block(
    in_channels: int,
    out_channels: int,
    num_groups: int = 16,
    block_type: str = "resnet",
):
    assert block_type in ["resnet"], (
        f"Block type {block_type} not supported. Supported types are ['resnet']"
    )
    return ResnetBlock(in_channels, out_channels, num_groups=num_groups)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int, num_heads: int = 8, num_groups: int = 8):
        super(AttentionBlock, self).__init__()
        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6
        )
        self.attention = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.linear = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        out = x
        batch_size, channels, h, w = x.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn = self.attention(in_attn, in_attn, in_attn, need_weights=False)[0]
        out_attn = self.linear(out_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        return out + out_attn


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super(ResnetBlock, self).__init__()

        self.conv_first = nn.Sequential(
            nn.GroupNorm(
                num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
            ),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.conv_second = nn.Sequential(
            nn.GroupNorm(
                num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True
            ),
            nn.SiLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),
        )

        self.residual_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        out = x
        out = self.conv_first(out)
        out = self.conv_second(out)

        x = self.residual_conv(x)

        return x + out


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_channels: List[int] = [256, 256, 128, 64],
        out_channels: int = 256,
        attn_resolutions: List[int] = [0, 1],
        num_groups: int = 16,
        num_resolution_blocks: int = 1,
        downsample_with_conv: bool = True,
        block_type: str = "resnet",
    ):
        super().__init__()

        block_in = num_channels[0]
        # conv in
        self.conv_in = nn.Conv2d(
            in_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # blocks down
        self.up_blocks = nn.ModuleList()
        for i, channels in enumerate(num_channels[1:]):
            block_out = channels
            for _ in range(num_resolution_blocks):
                self.up_blocks.append(
                    make_block(block_in, block_out, num_groups, block_type)
                )
                block_in = block_out

            if i in attn_resolutions:
                self.up_blocks.append(make_attn(block_in))

            self.up_blocks.append(Upsample(block_in, with_conv=downsample_with_conv))

        # conv out
        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=block_in, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(block_in, out_channels, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.up_blocks:
            x = block(x)

        x = self.conv_out(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_channels: List[int] = [32, 64, 128, 256, 512],
        out_channels: int = 256,
        num_groups: int = 16,
        num_resolution_blocks: int = 1,
        downsample_with_conv: bool = True,
        block_type: str = "resnet",
        attn_resolutions: List[int] = [-1],
    ):
        super().__init__()

        block_in = num_channels[0]
        # conv in
        self.conv_in = nn.Conv2d(
            in_channels, block_in, kernel_size=7, stride=1, padding=3
        )

        # blocks down
        self.down_blocks = nn.ModuleList()
        num_resolution = len(num_channels) - 1
        attn_resolutions = [num_resolution + res for res in attn_resolutions]
        for i, channels in enumerate(num_channels[1:]):
            block_out = channels
            for _ in range(num_resolution_blocks):
                self.down_blocks.append(
                    make_block(block_in, block_out, num_groups, block_type)
                )
                block_in = block_out

            if i in attn_resolutions:
                self.down_blocks.append(make_attn(block_in))

            self.down_blocks.append(
                Downsample(block_in, with_conv=downsample_with_conv)
            )

        self.down_blocks.append(make_attn(block_in))

        # conv out
        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=block_in, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)

        x = self.conv_out(x)
        return x


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, chunk_dim: int = 1, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=chunk_dim)

        self.sum_dims = [1, 2, 3] if self.mean.ndim == 4 else [1]

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self) -> torch.Tensor:
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=self.sum_dims,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=self.sum_dims,
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class VAEBase(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def forward(self, x, sample_posterior=True):
        raise NotImplementedError


class ConvVAE(VAEBase):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_blocks: int = 2,
        z_channels: int = 32,
        down_channels: List[int] = [32, 64, 128, 256, 256],
        up_channels: List[int] = [256, 256, 128, 64, 32],
        num_groups: int = 16,
    ):
        super().__init__()
        self._z_channels = z_channels
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=2 * z_channels,
            num_resolution_blocks=num_blocks,
            num_channels=down_channels,
            num_groups=num_groups,
        )
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.decoder = Decoder(
            in_channels=z_channels,
            num_channels=up_channels,
            num_resolution_blocks=num_blocks + 1,
            out_channels=out_channels,
            num_groups=num_groups,
        )

    def z_channels(self) -> int:
        return self._z_channels

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def encode_latents(
        self, x: torch.Tensor, sample_posterior: bool = True
    ) -> torch.Tensor:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if sample_posterior:
            return posterior.sample()
        else:
            return posterior.mode()

    def decode(
        self, z: torch.Tensor, input_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        return_latent: bool = False,
    ):
        posterior = self.encode(x)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode(z)

        if return_latent:
            return dec, posterior, z

        return dec, posterior


class MasksVAE(nn.Module):
    def __init__(
        self,
        loss: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.model = ConvVAE()
        self.loss = loss

    def z_channels(self) -> int:
        return self.model.z_channels()

    def z_resolution(self) -> int:
        return self.model.z_resolution()

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        apply_sigmoid: bool = False,
    ):
        y, posterior, z = self.model(x, sample_posterior, return_latent=True)

        if apply_sigmoid:
            y = torch.sigmoid(y)

        return y, posterior, z

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.model.encode(x)

    def encode_latents(
        self, x: torch.Tensor, sample_posterior: bool = True
    ) -> DiagonalGaussianDistribution:
        return self.model.encode_latents(x, sample_posterior)

    def decode(
        self,
        z: torch.Tensor,
        apply_sigmoid: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        out = self.model.decode(z, input_size=input_size)
        if apply_sigmoid:
            return torch.sigmoid(out)
        return out


class BinaryMaskVAE(nn.Module):
    def __init__(
        self,
        raw_std: float = 1.0,
        final_std: float = 0.5,
    ):
        super().__init__()

        self.model = MasksVAE()
        for param in self.model.parameters():
            param.requires_grad = False

        z_channels = self.z_channels()

        raw_std: List[float] = z_channels * [raw_std]
        final_std: List[float] = z_channels * [final_std]

        assert len(raw_std) == z_channels, (
            "raw_std must have the same length as z_channels"
        )

        self.scale = np.float32(final_std) / np.float32(raw_std)

    def z_channels(self) -> int:
        return self.model.z_channels()

    def z_resolution(self) -> int:
        return self.model.z_resolution()

    def encode_latents(
        self, x: torch.Tensor, sample_posterior: bool = True
    ) -> torch.Tensor:
        z = self.model.encode_latents(x, sample_posterior)

        z = z * misc.const_like(z, self.scale).reshape(1, -1, 1, 1)
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        z = z / misc.const_like(z, self.scale).reshape(1, -1, 1, 1)
        return self.model.decode(z, **kwargs)
