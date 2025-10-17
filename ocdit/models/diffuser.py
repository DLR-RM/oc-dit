from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from ocdit.heads.linear_head import LinearHead
from ocdit.layers.decoder import Decoder
from ocdit.layers.embed import (
    MPFourier,
    QueryEmbedder,
    TemplateEmbedder,
)
from ocdit.utils.misc import mean_flat, truncated_exp_normal_dist


class SigmaEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.emb_fourier = MPFourier(hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
        )
        self.t_embed_dim = hidden_size

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        emb = self.emb_fourier(sigma)
        emb = self.proj(emb)
        return emb


class Precond:
    def __init__(
        self,
        P_mean: float,
        P_std: float,
        sigma_data: float,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        dist: str = "normal",
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.sigma_distribution = dist

    def forward(
        self, bs: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sigma_distribution == "normal":
            rnd_normal = torch.randn([bs, 1, 1, 1, 1], device=device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        elif self.sigma_distribution == "normal_trunc":
            sigma = truncated_exp_normal_dist(
                self.P_mean,
                self.P_std,
                0.002,
                80,
                bs,  # Updated to use bs instead of image_features.shape[0]
                device,
            )
            sigma = sigma.reshape(-1, 1, 1, 1, 1)
        elif self.sigma_distribution == "uniform":
            # interval [0.002, 80]
            r1, r2 = np.log(0.002), np.log(80)
            rnd_uniform = (r1 - r2) * torch.rand([bs, 1, 1, 1, 1], device=device) + r2
            sigma = rnd_uniform.exp()
        else:
            raise ValueError(f"Unknown sigma distribution {self.sigma_distribution}")

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        return sigma, weight


class Diffuser(nn.Module):
    def __init__(
        self,
        image_feature_channels: int,
        latent_channels: int,
        input_dims: Tuple[int, int] = (480, 640),
        template_dims: Tuple[int, int] = (128, 128),
        patch_size: int = 16,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_num_templates: int = 12,
        max_num_classes: int = 8,
        precond: Dict[str, float] = {"P_mean": -0.4, "P_std": 1.2, "sigma_data": 0.5},
        sigma_distribution: str = "uniform",
    ):
        super().__init__()
        self.sigma_embedder = SigmaEmbedder(hidden_size)
        self.latent_channels = latent_channels
        queries_dim = latent_channels + image_feature_channels

        self._build_embed(
            input_dims,
            template_dims,
            image_feature_channels,
            patch_size,
            hidden_size,
            queries_dim,
            max_num_classes,
            max_num_templates,
        )

        self.decoder = Decoder(
            hidden_size,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )

        self.final_layer = LinearHead(
            hidden_size=hidden_size,
            num_layers=2,
            in_channels=hidden_size,
            out_channels=latent_channels,
            t_embed_dim=self.sigma_embedder.t_embed_dim,
        )

        self.precond = Precond(
            P_mean=precond["P_mean"],
            P_std=precond["P_std"],
            sigma_data=precond["sigma_data"],
            dist=sigma_distribution,
        )

        self.logvar_mlp = nn.Sequential(
            MPFourier(128),
            nn.Linear(128, 1),
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        image_features: torch.Tensor,
        templates_features: torch.Tensor,
    ):
        B, c, h, w = image_features.shape

        if x.shape[0] != sigma.shape[0] and sigma.shape[0] == 1:
            sigma = sigma.repeat_interleave(B)

        # embed timestep
        t = self.sigma_embedder(sigma)

        # embed templates and queries
        templates_tokens = self.template_embedder(templates_features)
        queries_tokens = self._embed_queries(x, image_features)

        x = self.decoder(queries_tokens, templates_tokens, t)

        out = self.final_layer(x, t, size=(h, w))  # (B, H * W, out_channels)

        return out

    def forward_loss(
        self,
        image_features: torch.Tensor,
        templates_features: torch.Tensor,
        latents: torch.Tensor,
    ):
        sigma, weight = self.precond.forward(
            image_features.shape[0], image_features.device
        )

        y = latents
        n = torch.randn_like(latents) * sigma

        D_yn, logvar = self.forward_noise(
            y + n, sigma, image_features, templates_features, return_logvar=True
        )
        loss_mask = (y - D_yn) ** 2
        loss = ((weight / logvar.exp()) * loss_mask) + logvar

        return {
            "loss": mean_flat(loss),
            "weight": weight.flatten(),
            "loss_masks": mean_flat(loss_mask),
        }

    def forward_noise(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        image_features: torch.Tensor,
        templates_features: torch.Tensor,
        return_logvar: bool = False,
    ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1, 1)
        sigma_data = self.precond.sigma_data

        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
        c_in = 1 / (sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.flatten().log() / 4

        F_x = self.forward((c_in * x), c_noise, image_features, templates_features)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        if return_logvar:
            logvar = self.logvar_mlp(c_noise).reshape(-1, 1, 1, 1, 1)
            return D_x, logvar

        return D_x

    def sample(
        self,
        image_features: torch.Tensor,
        templates_features: torch.Tensor,
        ensemble_size: int = 1,
        sampling_parameters: Dict = {},
    ):
        num_steps = sampling_parameters.get("num_steps", 18)
        sigma_min, sigma_max = (
            sampling_parameters.get("sigma_min", 0.002),
            sampling_parameters.get("sigma_max", 80),
        )
        rho = sampling_parameters.get("rho", 7)

        S_churn, S_min, S_max = 0, 0, float("inf")

        bs, n_classes = templates_features.shape[:2]
        latent_rows, latent_cols = image_features.shape[2:]
        # prepare ensemble
        image_features = (
            image_features.unsqueeze(1).repeat(1, ensemble_size, 1, 1, 1).flatten(0, 1)
        )
        # template feature bs nc nv c h w
        templates_features = (
            templates_features.unsqueeze(1)
            .repeat(1, ensemble_size, 1, 1, 1, 1, 1)
            .flatten(0, 1)
        )

        noise = torch.randn(
            (
                image_features.shape[0],
                n_classes,
                self.latent_channels,
                latent_rows,
                latent_cols,
            )
        ).to(image_features.device)

        # Time step discretization.
        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=image_features.device
        )

        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

        t_steps = torch.cat(
            [torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0
        # Main sampling loop.
        x_next = noise * t_steps[0]

        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.forward_noise(
                x_hat, t_hat, image_features, templates_features
            ).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.forward_noise(
                    x_next, t_next, image_features, templates_features
                ).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.to(torch.float32)

    def _embed_queries(
        self, x: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        n_classes = x.shape[1]

        queries = torch.cat(
            [
                image_features.unsqueeze(1).repeat(1, n_classes, 1, 1, 1),
                x,
            ],
            dim=2,
        )

        queries_tokens = self.queries_embedder(queries)  # (B, 1, H, W, C)

        return queries_tokens

    def _build_embed(
        self,
        input_dims,
        template_dims,
        image_feature_channels,
        patch_size,
        hidden_size,
        queries_dim,
        max_num_classes,
        max_num_templates,
    ) -> None:
        self.queries_embedder = QueryEmbedder(
            input_dims=input_dims,
            feature_dim=queries_dim,
            patch_size=patch_size,
            max_num_classes=max_num_classes,
            embed_dim=hidden_size,
        )

        self.template_embedder = TemplateEmbedder(
            input_dims=template_dims,
            feature_dim=image_feature_channels,
            max_num_classes=max_num_classes,
            max_num_templates=max_num_templates,
            patch_size=patch_size,
            embed_dim=hidden_size,
        )
