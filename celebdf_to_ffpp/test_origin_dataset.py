import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import DEFAULT_CONFIG_PATH

from datasets import DeepfakeFrameDataset, get_eval_transform
from models import build_model
from train import load_config, resolve_device
from utils.checkpoint import load_checkpoint
from utils.metrics import binary_confusion_matrix, compute_binary_metrics, format_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Test a CelebDF-trained checkpoint on CelebDF test")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/celebdf_origin_predictions.csv",
        help="CSV output path",
    )
    return parser.parse_args()


def predict(model, loader, device) -> Tuple[list, list, list]:
    model.eval()
    image_paths = []
    labels_all = []
    probs_all = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Test CelebDF origin dataset"):
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
        root_dir=config["celebdf_root"],
        split=config.get("celebdf_test_dir", "test"),
        dataset_type="celebdf",
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
        backbone=str(config.get("backbone", "efficientnetb4")),
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
    if not output_csv.is_absolute():
        output_csv = Path(__file__).resolve().parent / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "image_path": image_paths,
            "label": labels,
            "probability": probs,
            "prediction": preds,
        }
    ).to_csv(output_csv, index=False)

    print(f"CelebDF Origin Test | {format_metrics(metrics)}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"Saved predictions to: {output_csv}")


if __name__ == "__main__":
    main()
