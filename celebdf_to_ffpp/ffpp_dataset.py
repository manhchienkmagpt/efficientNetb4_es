from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset

from datasets.deepfake_dataset import IMAGE_EXTENSIONS


FFPP_TEST_LABELS: Dict[str, int] = {
    "original": 0,
    "deepfakes": 1,
    "face2face": 1,
    "faceshifter": 1,
    "faceswap": 1,
    "neuraltextures": 1,
}


class FFPPTestDataset(Dataset):
    def __init__(self, root_dir: str, split: str, eval_transform) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.eval_transform = eval_transform
        self.split_dir = self._resolve_split_dir()
        self.samples = self._collect_samples()

        if not self.samples:
            raise RuntimeError(f"No supported FF++ images found in {self.split_dir}")

    def _resolve_split_dir(self) -> Path:
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"FF++ split folder does not exist: {split_dir}")
        if not split_dir.is_dir():
            raise NotADirectoryError(f"FF++ split path is not a directory: {split_dir}")
        return split_dir

    def _find_class_dir(self, class_name: str) -> Path:
        exact_path = self.split_dir / class_name
        if exact_path.exists():
            return exact_path

        for child in self.split_dir.iterdir():
            if child.is_dir() and child.name.lower() == class_name:
                return child

        return exact_path

    def _collect_class_images(self, class_dir: Path) -> List[Path]:
        return sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _collect_samples(self) -> List[Tuple[Path, float]]:
        samples: List[Tuple[Path, float]] = []
        for class_name, label in FFPP_TEST_LABELS.items():
            class_dir = self._find_class_dir(class_name)
            if not class_dir.exists():
                continue
            samples.extend((path, float(label)) for path in self._collect_class_images(class_dir))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.eval_transform is not None:
            image = self.eval_transform(image=image)["image"]

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor, str(image_path)
