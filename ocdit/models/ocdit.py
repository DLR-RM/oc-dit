from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from ocdit.models.diffuser import Diffuser
from ocdit.models.feature_extractor import SpatialDinov2
from ocdit.models.vae import BinaryMaskVAE


class OCDiT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        image_size: Tuple[int, int],
        template_size: Tuple[int, int],
        embed_dim: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
    ):
        super().__init__()

        self.feature_extractor = SpatialDinov2(freeze_weights=True)
        self.vae = BinaryMaskVAE()
        self.diffuser = Diffuser(
            input_dims=image_size,
            template_dims=template_size,
            hidden_size=embed_dim,
            depth=depth,
            num_heads=num_heads,
            image_feature_channels=self.feature_extractor.feature_dim,
            patch_size=self.feature_extractor.patch_size,
            latent_channels=self.vae.z_channels(),
        )
        self.sampling_parameters = {
            "num_steps": 9,
            "sigma_min": 0.002,
            "sigma_max": 80,
            "rho": 9,
        }

    def forward_loss(self, images, templates, masks_gt):
        num_classes = templates.shape[1]

        image_features, templates_features = self.extract_features(images, templates)

        masks = rearrange(masks_gt, "b nc h w -> (b nc) h w")
        masks_latents = self.vae.encode_latents(
            masks.unsqueeze(1), sample_posterior=True
        )

        masks_latents = rearrange(
            masks_latents, "(b nc) z h w -> b nc z h w", nc=num_classes
        )
        loss_dict = self.diffuser.forward_loss(
            image_features,
            templates_features,
            masks_latents,
        )
        return loss_dict

    def generate_segmentations(
        self,
        images: torch.Tensor,
        templates: torch.Tensor,
        ensemble_size: int = 3,
    ):
        image_features, templates_features = self.extract_features(images, templates)
        n_classes = templates_features.shape[1]

        z_pred = self.diffuser.sample(
            image_features,
            templates_features,
            ensemble_size=ensemble_size,
            sampling_parameters=self.sampling_parameters,
        )
        z_pred = rearrange(z_pred, "b nc z h w -> (b nc) z h w")
        z_pred_splits = torch.split(z_pred, 1, dim=0)
        outputs = []
        for z in z_pred_splits:
            m = self.vae.decode(z, apply_sigmoid=True, input_size=images.shape[2:])
            outputs.append(m)

        pred_masks = torch.cat(outputs, dim=0)

        pred_masks = rearrange(
            pred_masks.squeeze(1),
            "(b e nc) h w -> b e nc h w",
            e=ensemble_size,
            nc=n_classes,
        )

        # average over ensemble dimension
        pred_masks = torch.mean(pred_masks, 1)

        return pred_masks

    def extract_features(self, images, templates):
        num_classes = templates.shape[1]

        # forward images
        images = TF.normalize(images, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        image_features = self.feature_extractor(images)

        # templates and images can have different resolutions
        templates = rearrange(templates, "b nc nv c h w -> b (nc nv) c h w")
        templates = TF.normalize(templates, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        template_features = self.feature_extractor(templates)

        template_features = rearrange(
            template_features, "b (nc nv) c h w -> b nc nv c h w", nc=num_classes
        )
        return image_features, template_features
