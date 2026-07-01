import torch
import timm
from torch import nn


def _create_vector_backbone(model_name: str, pretrained: bool, image_size: int | None = None) -> nn.Module:
    model_kwargs = {}
    if image_size is not None and model_name.startswith("swin_"):
        model_kwargs["img_size"] = image_size
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        **model_kwargs,
    )


def _feature_dim(backbone: nn.Module) -> int:
    if hasattr(backbone, "num_features"):
        return int(backbone.num_features)
    if hasattr(backbone, "head") and hasattr(backbone.head, "in_features"):
        return int(backbone.head.in_features)
    raise ValueError("Could not infer backbone feature dimension.")


class ESANet(nn.Module):
    """EfficientNet local branch + Swin global branch for binary deepfake detection."""

    def __init__(
        self,
        efficient_name: str = "efficientnet_b0",
        swin_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        dropout: float = 0.3,
        image_size: int | None = None,
    ) -> None:
        super().__init__()
        self.local_backbone = _create_vector_backbone(
            efficient_name,
            pretrained=pretrained,
            image_size=image_size,
        )
        self.global_backbone = _create_vector_backbone(
            swin_name,
            pretrained=pretrained,
            image_size=image_size,
        )

        local_dim = _feature_dim(self.local_backbone)
        global_dim = _feature_dim(self.global_backbone)
        fused_dim = local_dim + global_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feature = self.local_backbone(x)
        global_feature = self.global_backbone(x)
        fused = torch.cat([local_feature, global_feature], dim=1)
        return self.classifier(fused)


class CrossAttentionFusion(nn.Module):
    """Fuse local (CNN) and global (Transformer) features with cross-attention."""

    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        proj_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj_local = nn.Linear(local_dim, proj_dim)
        self.proj_global = nn.Linear(global_dim, proj_dim)
        self.norm_local = nn.LayerNorm(proj_dim)
        self.norm_global = nn.LayerNorm(proj_dim)

        self.local_attends_global = nn.MultiheadAttention(
            proj_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.global_attends_local = nn.MultiheadAttention(
            proj_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, 2),
        )
        self.out_dim = proj_dim * 2

    def forward(self, local_feature: torch.Tensor, global_feature: torch.Tensor) -> torch.Tensor:
        local_token = self.norm_local(self.proj_local(local_feature)).unsqueeze(1)
        global_token = self.norm_global(self.proj_global(global_feature)).unsqueeze(1)

        local_ctx, _ = self.local_attends_global(local_token, global_token, global_token)
        global_ctx, _ = self.global_attends_local(global_token, local_token, local_token)
        local_ctx = local_ctx.squeeze(1)
        global_ctx = global_ctx.squeeze(1)

        gate_weights = torch.softmax(self.gate(torch.cat([local_ctx, global_ctx], dim=1)), dim=1)
        local_out = local_ctx * gate_weights[:, 0:1]
        global_out = global_ctx * gate_weights[:, 1:2]
        return torch.cat([local_out, global_out], dim=1)


class ESANetV2(nn.Module):
    """EfficientNet + Swin dual-branch model with cross-attention fusion."""

    def __init__(
        self,
        efficient_name: str = "efficientnet_b0",
        swin_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        dropout: float = 0.3,
        image_size: int | None = None,
        proj_dim: int = 512,
        num_heads: int = 4,
        use_aux_heads: bool = True,
    ) -> None:
        super().__init__()
        self.local_backbone = _create_vector_backbone(
            efficient_name,
            pretrained=pretrained,
            image_size=image_size,
        )
        self.global_backbone = _create_vector_backbone(
            swin_name,
            pretrained=pretrained,
            image_size=image_size,
        )

        local_dim = _feature_dim(self.local_backbone)
        global_dim = _feature_dim(self.global_backbone)

        self.feat_dropout_local = nn.Dropout(dropout * 0.5)
        self.feat_dropout_global = nn.Dropout(dropout * 0.5)
        self.fusion = CrossAttentionFusion(
            local_dim,
            global_dim,
            proj_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout * 0.33,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.out_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, 1),
        )

        self.use_aux_heads = use_aux_heads
        if use_aux_heads:
            self.aux_local = nn.Linear(local_dim, 1)
            self.aux_global = nn.Linear(global_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        local_feature = self.local_backbone(x)
        global_feature = self.global_backbone(x)

        fused = self.fusion(
            self.feat_dropout_local(local_feature),
            self.feat_dropout_global(global_feature),
        )
        main_logit = self.classifier(fused)

        if self.use_aux_heads and self.training:
            return main_logit, self.aux_local(local_feature), self.aux_global(global_feature)

        return main_logit


MAGNet = ESANetV2
ESANetPlus = ESANetV2


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = ESANet(pretrained=False)
    y = model(x)
    print(y.shape)

    model = ESANetV2(pretrained=False)
    y = model(x)
    print(y[0].shape if isinstance(y, tuple) else y.shape)
