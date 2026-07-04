import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from datasets import DeepfakeFrameDataset, GANFrameDataset, get_eval_transform
from models import build_model
from train import load_config, resolve_device
from utils.checkpoint import load_checkpoint
from utils.metrics import binary_confusion_matrix, compute_binary_metrics, format_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a LightGBM head on frozen feature vectors from trained backbones"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Backbone checkpoint path. Repeat this argument for each trained backbone.",
    )
    parser.add_argument("--output-dir", type=str, default="lightgbm_outputs", help="Where to save outputs")
    parser.add_argument("--include-gan", action="store_true", help="Append configured GAN real/fake data to train set")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional feature cache directory")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing feature cache and recompute")
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--test-origin", action="store_true", help="Evaluate on config test_dir")
    parser.add_argument("--test-cross", action="store_true", help="Evaluate on config cross_dataset_root")
    return parser.parse_args()


def build_ffpp_dataset(config: Dict, split_key: str, mode: str):
    return DeepfakeFrameDataset(
        root_dir=config["data_root"],
        split=config[split_key],
        dataset_type="ffpp",
        train_transform=None,
        eval_transform=get_eval_transform(int(config["image_size"])),
        original_upsample_factor=config.get("original_upsample_factor") if mode == "train" else 0,
        train_real_percent=config.get("train_real_percent", 100),
        seed=int(config.get("seed", 42)),
        mode=mode,
    )


def build_train_dataset(config: Dict, include_gan: bool):
    train_dataset = build_ffpp_dataset(config, "train_dir", "train")
    if not include_gan:
        return train_dataset

    gan_dataset = GANFrameDataset(
        fake_dir=config["gan_fake_dir"],
        real_dir=config["gan_real_dir"],
        train_transform=None,
        eval_transform=get_eval_transform(int(config["image_size"])),
        mode="train",
    )
    return ConcatDataset([train_dataset, gan_dataset])


def build_cross_dataset(config: Dict):
    return DeepfakeFrameDataset(
        root_dir=config["cross_dataset_root"],
        split=None,
        dataset_type="cross",
        train_transform=None,
        eval_transform=get_eval_transform(int(config["image_size"])),
        original_upsample_factor=0,
        mode="test",
    )


def persistent_loader_kwargs(num_workers: int) -> Dict:
    if num_workers <= 0:
        return {}
    return {"persistent_workers": True}


def make_loader(config: Dict, dataset, shuffle: bool = False) -> DataLoader:
    num_workers = int(config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        **persistent_loader_kwargs(num_workers),
    )


def model_config(base_config: Dict, checkpoint: Dict) -> Dict:
    ckpt_config = checkpoint.get("config")
    if not isinstance(ckpt_config, dict):
        return base_config
    merged = dict(base_config)
    for key in ["backbone", "dropout", "image_size", "model_kwargs"]:
        if key in ckpt_config:
            merged[key] = ckpt_config[key]
    return merged


def load_feature_model(checkpoint_path: str, base_config: Dict, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    cfg = model_config(base_config, checkpoint)
    model = build_model(
        backbone=str(cfg.get("backbone", "efficientnetb4")),
        pretrained=False,
        dropout=float(cfg.get("dropout", 0.4)),
        image_size=int(cfg["image_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, str(cfg.get("backbone", "unknown"))


def cache_path(cache_dir: Optional[str], split_name: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"{split_name}_features.npz"


def model_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "extract_features"):
        return model.extract_features(images)
    if hasattr(model, "backbone"):
        return model.backbone(images)
    return model(images)


def extract_features(
    models: List[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    split_name: str,
    cache_file: Optional[Path] = None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if cache_file is not None and cache_file.exists() and use_cache:
        cached = np.load(cache_file, allow_pickle=True)
        return cached["features"], cached["labels"], cached["paths"].astype(str).tolist()

    feature_chunks = []
    label_chunks = []
    image_paths: List[str] = []

    with torch.inference_mode():
        for images, labels, paths in tqdm(loader, desc=f"Extract {split_name}", leave=False):
            images = images.to(device, non_blocking=True)
            batch_features = []
            for model in models:
                features = model_features(model, images)
                batch_features.append(features.flatten(1).detach().cpu().numpy())
            feature_chunks.append(np.concatenate(batch_features, axis=1).astype(np.float32))
            label_chunks.append(labels.numpy().astype(np.int64))
            image_paths.extend(paths)

    features_np = np.concatenate(feature_chunks, axis=0)
    labels_np = np.concatenate(label_chunks, axis=0)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            features=features_np,
            labels=labels_np,
            paths=np.asarray(image_paths),
        )

    return features_np, labels_np, image_paths


def train_lightgbm(args, config: Dict, x_train, y_train, x_val, y_val):
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    except ImportError as exc:
        raise ImportError("Install LightGBM first: pip install lightgbm") from exc

    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        class_weight="balanced" if config.get("use_pos_weight", False) else None,
        random_state=int(config.get("seed", 42)),
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(args.early_stopping_rounds, first_metric_only=True),
            log_evaluation(period=25),
        ],
    )
    return model


def best_val_auc(model) -> Optional[float]:
    score = getattr(model, "best_score_", None)
    if not isinstance(score, dict):
        return None
    valid_scores = score.get("valid_0") or score.get("validation_0")
    if not isinstance(valid_scores, dict) or "auc" not in valid_scores:
        return None
    return float(valid_scores["auc"])


def evaluate_split(model, x, y, paths, threshold: float, output_csv: Path, title: str):
    probs = model.predict_proba(x)[:, 1]
    metrics = compute_binary_metrics(y, probs, threshold=threshold)
    cm = binary_confusion_matrix(y, probs, threshold=threshold)
    preds = (probs >= threshold).astype(int)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "image_path": paths,
            "label": y,
            "probability": probs,
            "prediction": preds,
        }
    ).to_csv(output_csv, index=False)
    print(f"{title} | {format_metrics(metrics)}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"Saved predictions to: {output_csv}")
    return metrics


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(str(config.get("device", "cuda")))
    threshold = float(args.threshold if args.threshold is not None else config.get("threshold", 0.5))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = []
    backbone_names = []
    for checkpoint_path in args.checkpoint:
        model, backbone_name = load_feature_model(checkpoint_path, config, device)
        models.append(model)
        backbone_names.append(backbone_name)
        print(f"Loaded {backbone_name}: {checkpoint_path}")

    train_loader = make_loader(config, build_train_dataset(config, args.include_gan))
    val_loader = make_loader(config, build_ffpp_dataset(config, "val_dir", "val"))
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    use_cache = not args.no_cache
    x_train, y_train, train_paths = extract_features(
        models,
        train_loader,
        device,
        "train",
        cache_path(args.cache_dir, "train"),
        use_cache,
    )
    x_val, y_val, val_paths = extract_features(
        models,
        val_loader,
        device,
        "val",
        cache_path(args.cache_dir, "val"),
        use_cache,
    )
    print(f"Feature shape: train={x_train.shape}, val={x_val.shape}")

    lightgbm_model = train_lightgbm(args, config, x_train, y_train, x_val, y_val)
    best_iteration = getattr(lightgbm_model, "best_iteration_", None)
    best_auc = best_val_auc(lightgbm_model)
    metadata = {
        "checkpoints": args.checkpoint,
        "backbones": backbone_names,
        "threshold": threshold,
        "feature_dim": int(x_train.shape[1]),
        "include_gan": bool(args.include_gan),
        "early_stopping_metric": "val_auc",
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "best_val_auc": best_auc,
    }

    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Install joblib or scikit-learn to save the LightGBM model.") from exc

    joblib.dump(lightgbm_model, output_dir / "lightgbm_model.joblib")
    if best_iteration is not None:
        lightgbm_model.booster_.save_model(
            str(output_dir / "lightgbm_best_auc.txt"),
            num_iteration=int(best_iteration),
        )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved LightGBM model to: {output_dir / 'lightgbm_model.joblib'}")
    if best_iteration is not None:
        print(f"Saved best-AUC LightGBM booster to: {output_dir / 'lightgbm_best_auc.txt'}")
        print(f"Best val AUC: {best_auc:.6f} at iteration {best_iteration}" if best_auc is not None else f"Best iteration: {best_iteration}")

    metrics = {
        "val": evaluate_split(
            lightgbm_model,
            x_val,
            y_val,
            val_paths,
            threshold,
            output_dir / "val_predictions.csv",
            "Val LightGBM",
        )
    }

    if args.test_origin:
        test_loader = make_loader(config, build_ffpp_dataset(config, "test_dir", "test"))
        x_test, y_test, test_paths = extract_features(
            models,
            test_loader,
            device,
            "test_origin",
            cache_path(args.cache_dir, "test_origin"),
            use_cache,
        )
        metrics["test_origin"] = evaluate_split(
            lightgbm_model,
            x_test,
            y_test,
            test_paths,
            threshold,
            output_dir / "origin_predictions.csv",
            "Origin Test LightGBM",
        )

    if args.test_cross:
        cross_loader = make_loader(config, build_cross_dataset(config))
        x_cross, y_cross, cross_paths = extract_features(
            models,
            cross_loader,
            device,
            "test_cross",
            cache_path(args.cache_dir, "test_cross"),
            use_cache,
        )
        metrics["test_cross"] = evaluate_split(
            lightgbm_model,
            x_cross,
            y_cross,
            cross_paths,
            threshold,
            output_dir / "cross_predictions.csv",
            "Cross Test LightGBM",
        )

    serializable_metrics = {
        split: {key: float(value) for key, value in split_metrics.items()}
        for split, split_metrics in metrics.items()
    }
    (output_dir / "metrics.json").write_text(json.dumps(serializable_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
