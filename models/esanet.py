import torch
import timm
from torch import nn
from torch.nn import functional as F


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


class MAGNet(nn.Module):
    """ESANet++ with multi-scale local features, frequency features, and gated fusion."""

    def __init__(
        self,
        local_backbone: str = "efficientnet_b4",
        global_backbone: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
        image_size: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        local_kwargs = {"features_only": True, "pretrained": pretrained}
        self.local_backbone = timm.create_model(local_backbone, **local_kwargs)
        local_channels = self.local_backbone.feature_info.channels()
        self.local_adapters = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for channels in local_channels
        )
        self.local_fusion = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.local_pool = nn.AdaptiveAvgPool2d(1)

        self.global_backbone = _create_vector_backbone(
            global_backbone,
            pretrained=pretrained,
            image_size=image_size,
        )
        global_dim = _feature_dim(self.global_backbone)

        self.frequency_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.local_proj = nn.Linear(embed_dim, embed_dim)
        self.global_proj = nn.Linear(global_dim, embed_dim)
        self.freq_proj = nn.Linear(embed_dim, embed_dim)
        self.local_to_global_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.global_to_local_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.gate = nn.Linear(embed_dim * 3, 3)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def extract_fft_features(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray, norm="ortho")
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(fft))
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (magnitude - mean) / std

    def _extract_local_feature(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = self.local_backbone(x)
        target_size = feature_maps[-1].shape[-2:]
        adapted_maps = []
        for feature_map, adapter in zip(feature_maps, self.local_adapters):
            adapted = adapter(feature_map)
            if adapted.shape[-2:] != target_size:
                adapted = F.interpolate(
                    adapted,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            adapted_maps.append(adapted)
        local_map = torch.stack(adapted_maps, dim=0).sum(dim=0)
        local_map = self.local_fusion(local_map)
        return self.local_pool(local_map).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feature = self._extract_local_feature(x)
        global_feature = self.global_backbone(x)
        freq_feature = self.frequency_branch(self.extract_fft_features(x)).flatten(1)

        local_proj = self.local_proj(local_feature)
        global_proj = self.global_proj(global_feature)
        freq_proj = self.freq_proj(freq_feature)

        local_token = local_proj.unsqueeze(1)
        global_token = global_proj.unsqueeze(1)
        local_attended, _ = self.local_to_global_attn(
            query=local_token,
            key=global_token,
            value=global_token,
            need_weights=False,
        )
        global_attended, _ = self.global_to_local_attn(
            query=global_token,
            key=local_token,
            value=local_token,
            need_weights=False,
        )
        local_cross = local_proj + local_attended.squeeze(1)
        global_cross = global_proj + global_attended.squeeze(1)

        gate_input = torch.cat([local_cross, global_cross, freq_proj], dim=1)
        weights = torch.softmax(self.gate(gate_input), dim=1)
        fused = (
            weights[:, 0:1] * local_cross
            + weights[:, 1:2] * global_cross
            + weights[:, 2:3] * freq_proj
        )
        return self.classifier(fused)


ESANetPlus = MAGNet


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = ESANet(pretrained=False)
    y = model(x)
    print(y.shape)

    model = MAGNet(pretrained=False)
    y = model(x)
    print(y.shape)
