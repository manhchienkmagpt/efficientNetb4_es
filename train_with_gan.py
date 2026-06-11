import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader

from datasets import DeepfakeFrameDataset, GANFrameDataset, get_eval_transform, get_train_transform
from models import build_model
from train import current_lrs, load_config, lr_reduced, resolve_device, run_one_epoch
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import format_metrics
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a configurable backbone with FF++ and GAN data")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    return parser.parse_args()


def _required_gan_dir(config: Dict, key: str) -> str:
    value = config.get(key)
    if value in (None, "", "null"):
        raise ValueError(f"Missing {key} in config.")
    return str(value)


def build_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    image_size = int(config["image_size"])
    train_transform = get_train_transform(image_size)
    eval_transform = get_eval_transform(image_size)

    ffpp_train_dataset = DeepfakeFrameDataset(
        root_dir=config["data_root"],
        split=config["train_dir"],
        dataset_type="ffpp",
        train_transform=train_transform,
        eval_transform=eval_transform,
        original_upsample_factor=int(config.get("original_upsample_factor", 0)),
        mode="train",
    )
    gan_train_dataset = GANFrameDataset(
        fake_dir=_required_gan_dir(config, "gan_fake_dir"),
        real_dir=_required_gan_dir(config, "gan_real_dir"),
        train_transform=train_transform,
        eval_transform=eval_transform,
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

    train_dataset = ConcatDataset([ffpp_train_dataset, gan_train_dataset])
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


def main():
    args = parse_args()
    config = load_config(args.config)
    if not config.get("gan_fake_dir") or not config.get("gan_real_dir"):
        raise ValueError("Set gan_fake_dir and gan_real_dir in config.")

    set_seed(int(config.get("seed", 42)))

    device = resolve_device(str(config.get("device", "cuda")))
    threshold = float(config.get("threshold", 0.5))
    save_dir = Path(config.get("gan_save_dir") or config.get("save_dir", "checkpoints"))
    checkpoint_name = str(config.get("gan_checkpoint_name", "best_model_with_gan.pth"))
    checkpoint_path = save_dir / checkpoint_name

    train_loader, val_loader = build_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    backbone = str(config.get("backbone", "efficientnetb4_es"))
    print(f"Backbone: {backbone}")
    model = build_model(
        backbone=backbone,
        pretrained=bool(config.get("pretrained", True)),
        dropout=float(config.get("dropout", 0.4)),
        image_size=int(config["image_size"]),
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
