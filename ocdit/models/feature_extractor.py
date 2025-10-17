import torch
import torch.nn as nn


class SpatialDinov2(nn.Module):
    def __init__(
        self,
        freeze_weights=True,
        model_type="dinov2_vits14",
    ):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_type)
        self.feature_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Spatial dimensions of output will be H // 14, W // 14. If autoresize is True,
        then the output will be resized to the correct dimensions.

        Args:
            x (torch.Tensor): Images (B, C, H, W). Should be ImageNet normalized.
            autoresize (bool): Whether to resize the input to match the num_patch
                dimensions.

        Returns:
            feature_map (torch.tensor): (B, C, h, w)
        """
        *B, c, h, w = x.shape

        x = x.reshape(-1, c, h, w)

        output = self.model.forward_features(x)
        features = output["x_norm_patchtokens"]
        features = features.permute(0, 2, 1)
        features = features.reshape(  # (B, C, H, W)
            -1, self.feature_dim, h // 14, w // 14
        )
        features = features.reshape(*B, self.feature_dim, h // 14, w // 14)
        return features
