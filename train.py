import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import DeepfakeFrameDataset, get_eval_transform, get_train_transform
from models import EfficientNetB4_ES
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_binary_metrics, format_metrics
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNetB4-ES for deepfake detection")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
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


def current_lrs(optimizer) -> list:
    return [group["lr"] for group in optimizer.param_groups]


def lr_reduced(before: list, after: list) -> bool:
    return any(new_lr < old_lr for old_lr, new_lr in zip(before, after))


def build_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    image_size = int(config["image_size"])
    train_transform = get_train_transform(image_size)
    eval_transform = get_eval_transform(image_size)

    train_dataset = DeepfakeFrameDataset(
        root_dir=config["data_root"],
        split=config["train_dir"],
        dataset_type="ffpp",
        train_transform=train_transform,
        eval_transform=eval_transform,
        original_upsample_factor=int(config.get("original_upsample_factor", 0)),
        mode="train",
    )
    val_dataset = DeepfakeFrameDataset(
        root_dir=config["data_root"],
        split=config["val_dir"],
        dataset_type="ffpp",
        train_transform=None,
        eval_transform=eval_transform,
        original_upsample_factor=0,
        mode="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def run_one_epoch(model, loader, criterion, device, optimizer=None, threshold: float = 0.5):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    labels_all = []
    probs_all = []

    progress = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
    for images, labels, _ in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        probs = torch.sigmoid(logits.detach())

        labels_all.extend(labels.detach().cpu().numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    return compute_binary_metrics(labels_all, probs_all, threshold=threshold, loss=avg_loss)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = resolve_device(str(config.get("device", "cuda")))
    threshold = float(config.get("threshold", 0.5))
    save_dir = Path(config.get("save_dir", "checkpoints"))
    checkpoint_path = save_dir / "best_model.pth"

    train_loader, val_loader = build_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = EfficientNetB4_ES(
        pretrained=bool(config.get("pretrained", True)),
        dropout=float(config.get("dropout", 0.4)),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler_patience = int(config.get("lr_scheduler_patience", 2))
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config.get("lr_scheduler_factor", 0.5)),
        # PyTorch reduces after bad_epochs > patience. This makes the config value
        # mean "reduce after this many consecutive non-improving epochs."
        patience=max(0, scheduler_patience - 1),
        min_lr=float(config.get("min_lr", 1e-7)),
    )

    best_auc = -math.inf
    best_accuracy = -math.inf
    epochs_without_accuracy_improvement = 0
    early_stopping_patience = int(config.get("early_stopping_patience", 5))
    total_epochs = int(config.get("epochs", 30))
    start_epoch = 1

    resume_path = args.resume if args.resume is not None else config.get("resume_from")
    if resume_path:
        checkpoint = load_checkpoint(str(resume_path), device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best_auc = float(checkpoint.get("best_auc", best_auc))
        best_accuracy = float(checkpoint.get("best_accuracy", best_accuracy))
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = checkpoint_epoch + 1
        print(f"Resumed from checkpoint: {resume_path} (epoch {checkpoint_epoch})")

        if start_epoch > total_epochs:
            print(
                f"Start epoch {start_epoch} is greater than total epochs {total_epochs}. "
                "Nothing to do."
            )
            return

    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            threshold=threshold,
        )
        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            threshold=threshold,
        )

        val_auc = float(val_metrics["auc"])
        val_accuracy = float(val_metrics["accuracy"])
        auc_improved = math.isfinite(val_auc) and val_auc > best_auc
        accuracy_improved = math.isfinite(val_accuracy) and val_accuracy > best_accuracy
        checkpoint_saved = False

        if auc_improved:
            best_auc = val_auc

        if accuracy_improved:
            best_accuracy = val_accuracy
            epochs_without_accuracy_improvement = 0
            save_checkpoint(
                save_path=str(checkpoint_path),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_auc=best_auc,
                best_accuracy=best_accuracy,
                scheduler=scheduler,
                config=config,
            )
            checkpoint_saved = True
        else:
            epochs_without_accuracy_improvement += 1

        lrs_before = current_lrs(optimizer)
        scheduler_metric = val_accuracy if math.isfinite(val_accuracy) else -math.inf
        scheduler.step(scheduler_metric)
        lrs_after = current_lrs(optimizer)
        was_lr_reduced = lr_reduced(lrs_before, lrs_after)

        print(f"Train | {format_metrics(train_metrics)}")
        print(f"Val   | {format_metrics(val_metrics)}")
        print(f"Current LR: {', '.join(f'{lr:.8g}' for lr in lrs_after)}")
        print(
            f"Best validation accuracy: {best_accuracy:.4f}"
            if math.isfinite(best_accuracy)
            else "Best validation accuracy: n/a"
        )
        print(f"Epochs without accuracy improvement: {epochs_without_accuracy_improvement}")
        print(f"Checkpoint saved: {'yes' if checkpoint_saved else 'no'}")
        print(f"LR reduced: {'yes' if was_lr_reduced else 'no'}")

        if epochs_without_accuracy_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epochs_without_accuracy_improvement} epochs "
                "without validation accuracy improvement."
            )
            break

    print(f"\nTraining finished. Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
