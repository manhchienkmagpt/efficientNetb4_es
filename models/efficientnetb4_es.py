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
