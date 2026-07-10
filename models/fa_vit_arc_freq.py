from typing import Tuple

import timm
import torch
from torch import nn

from .favit_freq_lite import CNNLocalExtractor, GAM, LAM, make_fft_image


class FFTAutoEncoderBranch(nn.Module):
    """Encode FFT magnitude images with an autoencoder bottleneck."""

    def __init__(self, in_channels: int = 3, out_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, fft_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(fft_x)
        reconstruction = self.decoder(encoded)
        feature = self.proj(self.pool(encoded))
        return feature, reconstruction


class ArcFaceRGBBranch(nn.Module):
    """Frozen ArcFace embedding branch backed by InsightFace FaceAnalysis."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: tuple[str, ...] | None = None,
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.providers = providers or ("CUDAExecutionProvider", "CPUExecutionProvider")
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.app = None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
        )

    def _ensure_app(self):
        if self.app is None:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(name=self.model_name, providers=list(self.providers))
            self.app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        return self.app

    def _to_bgr_images(self, x: torch.Tensor) -> list:
        images = (x.detach().cpu() * self.std.cpu()) + self.mean.cpu()
        images = images.clamp(0.0, 1.0)
        images = (images.permute(0, 2, 3, 1).numpy() * 255.0).round().astype("uint8")
        return [image[:, :, ::-1].copy() for image in images]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        app = self._ensure_app()
        embeddings = []
        for image in self._to_bgr_images(x):
            faces = app.get(image)
            if faces:
                face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
                embeddings.append(torch.from_numpy(face.embedding).float())
            else:
                embeddings.append(torch.zeros(512, dtype=torch.float32))

        features = torch.stack(embeddings, dim=0).to(device=x.device, dtype=x.dtype)
        features = self.proj(features)
        return nn.functional.normalize(features, dim=1)


class FAViTArcFreq(nn.Module):
    """FA-ViT RGB branch fused with FFT-autoencoder and ArcFace RGB branches."""

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        arcface_model_name: str = "buffalo_l",
        arcface_providers: tuple[str, ...] | None = None,
        arcface_ctx_id: int = 0,
        arcface_det_size: tuple[int, int] = (640, 640),
        pretrained: bool = True,
        num_classes: int = 1,
        freq_in_channels: int = 3,
        freq_dim: int = 128,
        arc_dim: int = 128,
        fusion_dim: int | None = None,
        use_freq: bool = True,
        use_arcface: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("FAViTArcFreq is configured for binary classification with one logit.")

        self.use_freq = use_freq
        self.use_arcface = use_arcface
        self.last_fft_reconstruction = None

        self.vit = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.vit_dim = int(self.vit.num_features)
        self.rgb_norm = None if hasattr(self.vit, "norm") else nn.LayerNorm(self.vit_dim)
        for param in self.vit.parameters():
            param.requires_grad = False

        num_blocks = len(self.vit.blocks)
        required_lam_indices = (0, 3, 6)
        invalid_indices = [idx for idx in required_lam_indices if idx >= num_blocks]
        if invalid_indices:
            raise ValueError(f"LAM block indices out of range for {num_blocks} ViT blocks: {invalid_indices}")

        self.local_cnn = CNNLocalExtractor(self.vit_dim)
        self.gam_blocks = nn.ModuleList(GAM(self.vit_dim) for _ in range(num_blocks))
        self.lam1 = LAM(self.vit_dim, num_heads=8)
        self.lam2 = LAM(self.vit_dim, num_heads=8)
        self.lam3 = LAM(self.vit_dim, num_heads=8)

        self.freq_branch = FFTAutoEncoderBranch(in_channels=freq_in_channels, out_dim=freq_dim)
        self.arcface_branch = ArcFaceRGBBranch(
            model_name=arcface_model_name,
            providers=arcface_providers,
            ctx_id=arcface_ctx_id,
            det_size=arcface_det_size,
            out_dim=arc_dim,
        )

        input_dim = self.vit_dim
        if use_freq:
            input_dim += freq_dim
        if use_arcface:
            input_dim += arc_dim

        if fusion_dim is None:
            fusion_dim = self.vit_dim

        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        return x

    def forward_rgb_features(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.local_cnn(x)
        tokens = self._embed_tokens(x)

        for idx, block in enumerate(self.vit.blocks):
            cls_token = tokens[:, :1]
            patch_tokens = tokens[:, 1:]
            patch_tokens = self.gam_blocks[idx](patch_tokens)

            if idx == 0:
                patch_tokens = self.lam1(patch_tokens, cnn_feat)
            elif idx == 3:
                patch_tokens = self.lam2(patch_tokens, cnn_feat)
            elif idx == 6:
                patch_tokens = self.lam3(patch_tokens, cnn_feat)

            tokens = torch.cat((cls_token, patch_tokens), dim=1)
            tokens = block(tokens)

        norm = self.vit.norm if hasattr(self.vit, "norm") else self.rgb_norm
        tokens = norm(tokens)
        return tokens[:, 0]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = [self.forward_rgb_features(x)]

        if self.use_freq:
            fft_x = make_fft_image(x)
            freq_feature, fft_reconstruction = self.freq_branch(fft_x)
            self.last_fft_reconstruction = fft_reconstruction
            features.append(freq_feature)
        else:
            self.last_fft_reconstruction = None

        if self.use_arcface:
            features.append(self.arcface_branch(x))

        fused_feature = self.fusion(torch.cat(features, dim=1))
        logits = self.classifier(fused_feature)
        return logits, fused_feature
