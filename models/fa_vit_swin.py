from pathlib import Path
from typing import Any, Mapping, Tuple

import timm
import torch
from torch import nn

from .redesigned_favit import RedesignedFAViT


def _checkpoint_state(path: str | Path) -> Mapping[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    checkpoint: Any = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must contain a state dict: {path}")
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break
    if not checkpoint or not all(isinstance(key, str) for key in checkpoint):
        raise ValueError(f"No valid model state dict found in checkpoint: {path}")
    return checkpoint


def _load_branch(module: nn.Module, checkpoint_path: str | Path, prefixes: tuple[str, ...]) -> None:
    """Load a branch from either a raw state dict or a full training checkpoint."""
    source = _checkpoint_state(checkpoint_path)
    target = module.state_dict()
    candidates = []
    for prefix in ("", "module.", *prefixes):
        candidate = {
            key[len(prefix):]: value
            for key, value in source.items()
            if key.startswith(prefix) and key[len(prefix):] in target
        }
        candidates.append(candidate)

    state = max(candidates, key=len)
    missing = [key for key in target if key not in state]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' is incompatible with {type(module).__name__}; "
            f"missing {len(missing)} parameters (first: {preview})."
        )
    module.load_state_dict(state, strict=True)


class FAViTSwin(nn.Module):
    """Fuse frozen, checkpoint-initialized Redesigned FA-ViT and Swin-S features."""

    def __init__(
        self,
        favit_checkpoint: str | Path,
        swin_checkpoint: str | Path,
        num_classes: int = 1,
        dropout: float = 0.3,
        image_size: int | None = None,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("FAViTSwin is configured for binary classification with one logit.")

        self.favit = RedesignedFAViT(pretrained=False, num_classes=1)
        swin_kwargs = {"img_size": image_size} if image_size is not None else {}
        self.swin = timm.create_model(
            "swin_small_patch4_window7_224",
            pretrained=False,
            num_classes=0,
            **swin_kwargs,
        )

        _load_branch(
            self.favit,
            favit_checkpoint,
            ("favit.", "model.favit.", "module.favit.", "module.model.favit."),
        )
        _load_branch(
            self.swin,
            swin_checkpoint,
            (
                "backbone.",
                "swin.",
                "model.backbone.",
                "model.swin.",
                "module.backbone.",
                "module.swin.",
                "module.model.backbone.",
                "module.model.swin.",
            ),
        )

        for branch in (self.favit, self.swin):
            branch.requires_grad_(False)
            branch.eval()

        self.favit_dim = self.favit.embed_dim
        self.swin_dim = int(self.swin.num_features)
        fused_dim = self.favit_dim + self.swin_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, num_classes),
        )

    def train(self, mode: bool = True) -> "FAViTSwin":
        super().train(mode)
        # Frozen branches must keep dropout/stochastic depth and BatchNorm disabled.
        self.favit.eval()
        self.swin.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _, favit_feature = self.favit(x)
            swin_feature = self.swin(x)
        fused_feature = torch.cat((favit_feature, swin_feature), dim=1)
        logits = self.classifier(fused_feature)
        return logits, fused_feature
