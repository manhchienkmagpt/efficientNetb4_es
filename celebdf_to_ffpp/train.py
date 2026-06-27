import argparse
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader

from common import DEFAULT_CONFIG_PATH

from datasets import DeepfakeFrameDataset, get_eval_transform, get_train_transform
from train import _count_labels, load_config, resolve_device, run_training_loop
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train on CelebDF train/val splits")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    return parser.parse_args()


def build_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    image_size = int(config["image_size"])
    train_transform = get_train_transform(image_size)
    eval_transform = get_eval_transform(image_size)

    train_dataset = DeepfakeFrameDataset(
        root_dir=config["celebdf_root"],
        split=config.get("celebdf_train_dir", "train"),
        dataset_type="celebdf",
        train_transform=train_transform,
        eval_transform=eval_transform,
        original_upsample_factor=0,
        mode="train",
    )
    val_dataset = DeepfakeFrameDataset(
        root_dir=config["celebdf_root"],
        split=config.get("celebdf_val_dir", "val"),
        dataset_type="celebdf",
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


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = resolve_device(str(config.get("device", "cuda")))
    save_dir = Path(config.get("save_dir", "checkpoints_celebdf_to_ffpp"))
    checkpoint_path = save_dir / str(config.get("checkpoint_name", "best_celebdf_to_ffpp.pth"))

    train_loader, val_loader = build_loaders(config)
    train_real_count, train_fake_count = _count_labels(train_loader.dataset)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Train class distribution — real: {train_real_count}, fake: {train_fake_count}")
    print(f"Val samples: {len(val_loader.dataset)}")

    run_training_loop(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=checkpoint_path,
        device=device,
        resume_path=args.resume or config.get("resume_from"),
    )


if __name__ == "__main__":
    main()
