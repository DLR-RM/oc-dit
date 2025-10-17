from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers.patch_embed import PatchEmbed
from torch.nn.init import trunc_normal_


def class_embedding(
    embedding: torch.Tensor,
    max_num_classes: int,
    current_num_classes: int,
):
    start_index = torch.randint(0, max_num_classes - current_num_classes, (1,))
    class_embed = embedding[:, start_index : start_index + current_num_classes]

    return class_embed


def interpolate_embedding(embedding, dim_before, dim_after):
    # 1d embedding
    pe_classes = embedding  # 1 max_num_classes dim
    dim = pe_classes.shape[-1]
    pe_classes = pe_classes.reshape(1, dim, dim_before)
    pe_classes = nn.functional.interpolate(pe_classes, size=dim_after)

    return pe_classes.reshape(1, dim_after, 1, dim)


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


class Im2Patches(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            patches = x.flatten(2)
            patches = patches.permute(0, 2, 1)
            return patches

        py = x.shape[-2] // self.patch_size
        px = x.shape[-1] // self.patch_size
        patches = rearrange(
            x,
            "b c (py yy) (px xx) -> b (py px) (c yy xx)",
            py=py,
            yy=self.patch_size,
            px=px,
            xx=self.patch_size,
        )
        return patches


class LearnedPE(nn.Module):
    def __init__(self, *size: Sequence[int]):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(size))

        self.initialize()

    def initialize(self):
        trunc_normal_(self.pe, std=0.02)

    def __call__(self):
        return self.pe


class QueryEmbedder(nn.Module):
    def __init__(
        self,
        input_dims: Tuple[int, int],
        feature_dim: int,
        patch_size: int,
        max_num_classes: int,
        embed_dim: int,
    ):
        super().__init__()
        self.embedder = PatchEmbed(
            img_size=None,
            in_chans=feature_dim,
            patch_size=1,
            embed_dim=embed_dim,
            flatten=True,
        )
        self.num_rows = input_dims[0] // patch_size
        self.num_cols = input_dims[1] // patch_size
        self.num_patches = self.num_rows * self.num_cols
        self.max_num_classes = max_num_classes

        self.pe_patches = LearnedPE(
            1,
            1,
            self.num_patches,
            embed_dim,
        )
        self.pe_classes = LearnedPE(
            1,
            max_num_classes,
            1,
            embed_dim,
        )

    def _get_patch_embeddings(self, input_dims: Tuple[int, int]):
        if (self.num_rows, self.num_cols) == input_dims:
            return self.pe_patches()
        else:
            pos_embed = rearrange(
                self.pe_patches(),
                "a b (h w) d -> a b d h w",
                h=self.num_rows,
                w=self.num_cols,
            )
            # pos_embed = (
            #     self.pe_patches()
            #     .reshape(1, 1, self.num_rows, self.num_cols, -1)
            #     .permute(0, 1, 4, 2, 3)
            # )

            def window_select(pos_embed):
                if input_dims[0] < pos_embed.shape[-2]:
                    pos_embed = pos_embed[..., : input_dims[0], :]
                if input_dims[1] < pos_embed.shape[-1]:
                    pos_embed = pos_embed[..., :, : input_dims[1]]
                return pos_embed

            pos_embed = window_select(pos_embed)
            pos_embed = rearrange(pos_embed, "a b d h w -> a b (h w) d")

            return pos_embed

    def _get_class_embeddings(self, num_classes: int):
        if num_classes == self.max_num_classes:
            return self.pe_classes()
        elif num_classes > self.max_num_classes:
            class_embed = interpolate_embedding(
                self.pe_classes(), self.max_num_classes, num_classes
            )

            return class_embed

        # num_classes < self.max_num_classes
        if self.random_class_indices is not None and self.training:
            class_embed = class_embedding(
                self.pe_classes(),
                self.max_num_classes,
                num_classes,
            )

            return class_embed

        class_embed = self.pe_classes()[:, :num_classes]
        return class_embed

    def _get_pos_embeddings(self, input_dims: Tuple[int, int], num_classes: int):
        patch_embedding = self._get_patch_embeddings(input_dims)
        class_embedding = self._get_class_embeddings(num_classes)

        assert patch_embedding.ndim == class_embedding.ndim, (
            f"Patch embedding and class embedding should have the same number of dimensions, "
            f"but got {patch_embedding.ndim} and {class_embedding.ndim}"
        )
        return patch_embedding + class_embedding

    def forward(self, x, cls_token: Optional[torch.Tensor] = None):
        input_dims, nc = x.shape[3:], x.shape[1]

        x = rearrange(x, "b nc c h w -> (b nc) c h w")
        x = self.embedder(x)

        if cls_token is not None:
            x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = rearrange(x, "(b nc) t e -> b nc t e", nc=nc)

        x = x + self._get_pos_embeddings(input_dims, nc)

        return x


class TemplateEmbedder(nn.Module):
    def __init__(
        self,
        input_dims: Tuple[int, int],
        feature_dim: int,
        max_num_classes: int,
        max_num_templates: int,
        patch_size: int,
        embed_dim: int,
    ):
        super().__init__()

        self.im_to_patches = Im2Patches(patch_size=1)
        self.embedder = PatchEmbed(
            img_size=None,
            in_chans=feature_dim,
            patch_size=1,
            embed_dim=embed_dim,
            flatten=True,
        )

        self.num_rows = input_dims[0] // patch_size
        self.num_cols = input_dims[1] // patch_size

        self.num_patches = self.num_rows * self.num_cols
        self.max_num_classes = max_num_classes
        self.max_num_templates = max_num_templates

        self.pe_patches = LearnedPE(
            1,
            1,
            1,
            self.num_patches,
            embed_dim,
        )
        self.pe_classes = LearnedPE(
            1,
            max_num_classes,
            1,
            1,
            embed_dim,
        )

        self.pe_views = LearnedPE(
            1,
            1,
            self.max_num_templates,
            1,
            embed_dim,
        )

    def _get_patch_embeddings(self, input_dims: Tuple[int, int]):
        if (self.num_rows, self.num_cols) == input_dims:
            return self.pe_patches()
        else:
            raise NotImplementedError(
                "Currently only support templates of uniform shape"
            )

    def _get_class_embeddings(self, num_classes: int):
        if num_classes == self.max_num_classes:
            return self.pe_classes()
        elif num_classes > self.max_num_classes:
            class_embed = interpolate_embedding(
                self.pe_classes(), self.max_num_classes, num_classes
            )
            return class_embed.reshape(1, num_classes, 1, 1, -1)

        if self.random_class_indices and self.training:
            class_embed = class_embedding(
                self.pe_classes(),
                self.max_num_classes,
                num_classes,
            )

            return class_embed.reshape(1, num_classes, 1, 1, -1)

        return self.pe_classes()[:, :num_classes]

    def _get_template_embeddings(self, num_templates: int):
        if num_templates == self.max_num_templates:
            return self.pe_views()
        elif num_templates > self.max_num_templates:
            template_embed = interpolate_embedding(
                self.pe_views(), self.max_num_templates, num_templates
            )
            return template_embed.reshape(1, 1, num_templates, 1, -1)

        # num_templates < self.max_num_templates
        if self.training:
            start_index = torch.randint(0, self.max_num_templates, (1,))
            end_index = (start_index + num_templates) % self.max_num_templates
            if start_index < end_index:
                embedding = self.pe_views()[:, :, start_index:end_index]
            else:
                embedding = torch.cat(
                    (
                        self.pe_views()[:, :, start_index:],
                        self.pe_views()[:, :, :end_index],
                    ),
                    dim=2,
                )
            return embedding.reshape(1, 1, num_templates, 1, -1)

        return self.pe_views()[:, :, :num_templates]

    def _get_pos_embeddings(
        self, input_dims: Tuple[int, int], num_classes: int, num_templates: int
    ):
        patch_embedding = self._get_patch_embeddings(input_dims)
        class_embedding = self._get_class_embeddings(num_classes)
        template_embeddings = self._get_template_embeddings(num_templates)

        assert patch_embedding.ndim == class_embedding.ndim, (
            f"Patch embedding and class embedding should have the same number of dimensions, "
            f"but got {patch_embedding.ndim} and {class_embedding.ndim}"
        )

        return patch_embedding + class_embedding + template_embeddings

    def forward(self, x):
        input_dims, nc, nv = x.shape[4:], x.shape[1], x.shape[2]

        x = rearrange(x, "b nc nv c h w -> (b nc nv) c h w")
        x = self.embedder(x)
        x = rearrange(x, "(b nc nv) t e -> b nc nv t e", nc=nc, nv=nv)
        x = x + self._get_pos_embeddings(input_dims, nc, nv)
        # x = x + self.pe_patches() + self.pe_classes() + self.pe_views()
        x = rearrange(x, "b nc nv t e -> b nc (nv t) e", nc=nc, nv=nv)
        return x
