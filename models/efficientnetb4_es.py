import timm
from torch import nn

from .modules import ECAAttention, SCConv


class EfficientNetB4_ES(nn.Module):
    """EfficientNet-B4 backbone with ECA attention and SCConv head."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.4) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        feature_channels = self.backbone.num_features

        self.eca = ECAAttention(feature_channels)
        self.scconv = SCConv(feature_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_channels, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.eca(features)
        features = self.scconv(features)
        features = self.pool(features)
        logits = self.classifier(features)
        return logits.squeeze(1)


class TimmBackbone_ES(nn.Module):
    """Generic timm backbone with a dropout classifier head."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        dropout: float = 0.4,
        image_size: int | None = None,
    ) -> None:
        super().__init__()
        model_kwargs = {}
        if image_size is not None and model_name.startswith("swin_"):
            model_kwargs["img_size"] = image_size

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            **model_kwargs,
        )
        self.feature_channels = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_channels, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(1)


_BACKBONE_ALIASES = {
    "efficientnet_b4": "efficientnetb4_es",
    "efficientnetb4": "efficientnetb4_es",
    "efficientnetb4_es": "efficientnetb4_es",
    "resnet50": "resnet50",
    "resnet_50": "resnet50",
    "swin_tiny": "swin_tiny",
    "swin-tiny": "swin_tiny",
    "swin_t": "swin_tiny",
    "swin-t": "swin_tiny",
}

_TIMM_BACKBONES = {
    "resnet50": "resnet50",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


def normalize_backbone_name(backbone: str) -> str:
    key = backbone.strip().lower()
    if key not in _BACKBONE_ALIASES:
        valid_backbones = ", ".join(sorted(_BACKBONE_ALIASES))
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose one of: {valid_backbones}")
    return _BACKBONE_ALIASES[key]


def build_model(
    backbone: str = "efficientnetb4_es",
    pretrained: bool = True,
    dropout: float = 0.4,
    image_size: int | None = None,
) -> nn.Module:
    backbone_name = normalize_backbone_name(backbone)
    if backbone_name == "efficientnetb4_es":
        return EfficientNetB4_ES(pretrained=pretrained, dropout=dropout)
    return TimmBackbone_ES(
        model_name=_TIMM_BACKBONES[backbone_name],
        pretrained=pretrained,
        dropout=dropout,
        image_size=image_size,
    )
