import inspect
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FFPP_LABELS: Dict[str, int] = {
    "original": 0,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceShifter": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
}

CELEBDF_LABELS: Dict[str, int] = {
    "real": 0,
    "fake": 1,
}


def _coarse_dropout():
    signature = inspect.signature(A.CoarseDropout)
    if "num_holes_range" in signature.parameters:
        return A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.03, 0.12),
            hole_width_range=(0.03, 0.12),
            fill=0,
            p=0.35,
        )
    return A.CoarseDropout(
        max_holes=8,
        max_height=48,
        max_width=48,
        min_holes=1,
        min_height=12,
        min_width=12,
        fill_value=0,
        p=0.35,
    )


def get_train_transform(image_size: int = 380) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.7,
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
            _coarse_dropout(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_eval_transform(image_size: int = 380) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class DeepfakeFrameDataset(Dataset):
    """Frame-level dataset for FF++ and CelebDF-style folder layouts."""

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        dataset_type: str = "ffpp",
        train_transform: Optional[A.Compose] = None,
        eval_transform: Optional[A.Compose] = None,
        original_upsample_factor: int = 0,
        mode: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.dataset_type = dataset_type.lower()
        self.mode = (mode or split or "test").lower()
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.original_upsample_factor = max(0, int(original_upsample_factor))

        self.split_dir = self._resolve_split_dir()
        self.class_to_label = self._get_class_mapping()
        self.samples: List[Tuple[Path, float, bool]] = self._collect_samples()

        if not self.samples:
            raise RuntimeError(f"No supported images found in {self.split_dir}")

    def _resolve_split_dir(self) -> Path:
        split_dir = self.root_dir / self.split if self.split else self.root_dir
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {split_dir}")
        if not split_dir.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {split_dir}")
        return split_dir

    def _get_class_mapping(self) -> Dict[str, int]:
        if self.dataset_type == "ffpp":
            return FFPP_LABELS
        if self.dataset_type == "celebdf":
            return CELEBDF_LABELS
        raise ValueError("dataset_type must be either 'ffpp' or 'celebdf'")

    def _collect_class_images(self, class_dir: Path) -> List[Path]:
        return sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _collect_samples(self) -> List[Tuple[Path, float, bool]]:
        samples: List[Tuple[Path, float, bool]] = []
        original_samples: List[Tuple[Path, float, bool]] = []
        missing_classes: List[str] = []

        for class_name, label in self.class_to_label.items():
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                missing_classes.append(class_name)
                continue

            class_samples = [(path, float(label), False) for path in self._collect_class_images(class_dir)]
            samples.extend(class_samples)

            if self.dataset_type == "ffpp" and class_name == "original":
                original_samples = class_samples

        if missing_classes:
            print(f"Warning: missing class folders under {self.split_dir}: {', '.join(missing_classes)}")

        if self.dataset_type == "ffpp" and self.mode == "train" and self.original_upsample_factor > 0:
            for _ in range(self.original_upsample_factor):
                samples.extend((path, label, True) for path, label, _ in original_samples)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label, augmented = self.samples[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = self._select_transform(label=label, augmented=augmented)
        if transform is not None:
            image = transform(image=image)["image"]

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor, str(image_path)

    def _select_transform(self, label: float, augmented: bool):
        if self.mode == "train":
            if label == 1.0:
                return self.train_transform or self.eval_transform
            if augmented:
                return self.train_transform or self.eval_transform
            return self.eval_transform
        return self.eval_transform


class GANFrameDataset(Dataset):
    """Frame-level dataset for GAN fake and real image folders."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: Optional[str] = None,
        fake_dir: Optional[str] = None,
        real_dir: Optional[str] = None,
        train_transform: Optional[A.Compose] = None,
        eval_transform: Optional[A.Compose] = None,
        mode: Optional[str] = None,
        label: float = 1.0,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.split = split
        self.fake_dir = Path(fake_dir) if fake_dir is not None else None
        self.real_dir = Path(real_dir) if real_dir is not None else None
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.mode = (mode or split or "train").lower()
        self.label = float(label)

        self.sources = self._resolve_sources()
        self.samples: List[Tuple[Path, float]] = self._collect_samples()

        if not self.samples:
            source_paths = ", ".join(str(source_dir) for source_dir, _ in self.sources)
            raise RuntimeError(f"No supported GAN images found in: {source_paths}")

    def _resolve_image_dir(self, image_dir: Path, description: str) -> Path:
        split_dir = image_dir / self.split if self.split else image_dir
        if not split_dir.exists():
            raise FileNotFoundError(f"{description} folder does not exist: {split_dir}")
        if not split_dir.is_dir():
            raise NotADirectoryError(f"{description} path is not a directory: {split_dir}")
        return split_dir

    def _resolve_sources(self) -> List[Tuple[Path, float]]:
        if self.fake_dir is not None or self.real_dir is not None:
            if self.fake_dir is None or self.real_dir is None:
                raise ValueError("Set both fake_dir and real_dir for GANFrameDataset.")
            return [
                (self._resolve_image_dir(self.fake_dir, "GAN fake"), 1.0),
                (self._resolve_image_dir(self.real_dir, "GAN real"), 0.0),
            ]

        if self.root_dir is None:
            raise ValueError("Set either root_dir or both fake_dir and real_dir for GANFrameDataset.")
        return [(self._resolve_image_dir(self.root_dir, "GAN dataset"), self.label)]

    def _collect_class_images(self, class_dir: Path) -> List[Path]:
        return sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _collect_samples(self) -> List[Tuple[Path, float]]:
        samples: List[Tuple[Path, float]] = []
        for image_dir, label in self.sources:
            samples.extend((path, label) for path in self._collect_class_images(image_dir))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read GAN image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = self.train_transform if self.mode == "train" else self.eval_transform
        transform = transform or self.eval_transform
        if transform is not None:
            image = transform(image=image)["image"]

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor, str(image_path)
