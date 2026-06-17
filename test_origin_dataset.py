import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import DeepfakeFrameDataset, get_eval_transform
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.metrics import binary_confusion_matrix, compute_binary_metrics, format_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Test a configurable backbone on the origin dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Checkpoint path")
    parser.add_argument("--output-csv", type=str, default="origin_dataset_predictions.csv", help="CSV output path")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(config_device: str) -> torch.device:
    if config_device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(config_device)


def predict(model, loader, device) -> Tuple[list, list, list]:
    model.eval()
    image_paths = []
    labels_all = []
    probs_all = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Test origin dataset"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)

            image_paths.extend(paths)
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())

    return image_paths, labels_all, probs_all


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(str(config.get("device", "cuda")))
    threshold = float(config.get("threshold", 0.5))

    dataset = DeepfakeFrameDataset(
        root_dir=config["data_root"],
        split=config["test_dir"],
        dataset_type="origin",
        eval_transform=get_eval_transform(int(config["image_size"])),
        original_upsample_factor=0,
        mode="test",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
    )

    model = build_model(
        backbone=str(config.get("backbone", "efficientnetb4_es")),
        pretrained=False,
        dropout=float(config.get("dropout", 0.4)),
        image_size=int(config["image_size"]),
    ).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    image_paths, labels, probs = predict(model, loader, device)
    metrics = compute_binary_metrics(labels, probs, threshold=threshold)
    cm = binary_confusion_matrix(labels, probs, threshold=threshold)
    preds = [int(prob >= threshold) for prob in probs]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True) if output_csv.parent != Path(".") else None
    pd.DataFrame(
        {
            "image_path": image_paths,
            "label": labels,
            "probability": probs,
            "prediction": preds,
        }
    ).to_csv(output_csv, index=False)

    print(f"Origin Dataset Test | {format_metrics(metrics)}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"Saved predictions to: {output_csv}")


if __name__ == "__main__":
    main()
