# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies** (install PyTorch separately with the appropriate CUDA version first):
```bash
pip install -r requirements.txt
```

**Train on FF++ dataset:**
```bash
python train.py --config configs/config.yaml
python train.py --config configs/config.yaml --resume checkpoints_efficientnetb4/best_efficientnetb4.pth
```

**Train with additional GAN data:**
```bash
python train_with_gan.py --config configs/config.yaml
```

**Evaluate on FF++ test split:**
```bash
python test_origin_dataset.py --config configs/config.yaml --checkpoint checkpoints_efficientnetb4/best_efficientnetb4.pth
```

**Evaluate cross-dataset (CelebDF):**
```bash
python test_cross_dataset.py --config configs/config.yaml --checkpoint checkpoints_efficientnetb4/best_efficientnetb4.pth
```

**CelebDF → FF++ transfer workflow:**
```bash
python celebdf_to_ffpp/train.py --config celebdf_to_ffpp/config.yaml
python celebdf_to_ffpp/test_origin_dataset.py --config celebdf_to_ffpp/config.yaml --checkpoint checkpoints_celebdf_to_ffpp/best_celebdf_to_ffpp.pth
python celebdf_to_ffpp/test_cross_dataset.py --config celebdf_to_ffpp/config.yaml --checkpoint checkpoints_celebdf_to_ffpp/best_celebdf_to_ffpp.pth
```

## Architecture

Binary deepfake frame classifier (real=0, fake=1) built on **timm** backbones. Model outputs a single raw logit; `BCEWithLogitsLoss` is used during training and `torch.sigmoid()` at inference.

**`models/backbones.py`** — `build_model(config)` dispatches to `TimmBackbone` (EfficientNet-B4, ResNet-50, Swin-Tiny) or `SwinTransformerSmall`. All share the same structure: timm feature extractor → Flatten → Dropout → Linear(1).

**`datasets/deepfake_dataset.py`** — `DeepfakeFrameDataset` loads FF++ frames (train/val/test splits, with subdirectories `original/` for real and `Deepfakes/`, `Face2Face/`, `FaceShifter/`, `FaceSwap/`, `NeuralTextures/` for fake). `GANFrameDataset` loads from explicit `fake_dir`/`real_dir` paths. Both return `(image_tensor, label_float, image_path_str)`.

**Training loop** (`train.py`): AdamW optimizer, `ReduceLROnPlateau` scheduler tracking validation accuracy, early stopping, checkpoint saved on best validation accuracy.

**`train_with_gan.py`**: Combines FF++ training split with GAN data via `ConcatDataset`; validation remains FF++ only.

## Configuration

All paths and hyperparameters live in `configs/config.yaml`. Key knobs:

| Key | Default | Effect |
|---|---|---|
| `backbone` | `efficientnetb4` | `efficientnetb4`, `resnet50`, `swin_tiny`, `swin_small` |
| `original_upsample_factor` | `3` | Adds N augmented copies of each real training image (set 0 to disable) |
| `train_real_percent` | `100` | % of real training images to use |
| `early_stopping_patience` | `5` | Epochs without val-acc improvement before stopping |
| `threshold` | `0.5` | Sigmoid threshold for binary prediction at inference |
| `gan_fake_dir` / `gan_real_dir` | — | Required only for `train_with_gan.py` |

Update `data_root` and `cross_dataset_root` to point to local dataset paths before running.

## Dataset Layout

**FF++ (primary):**
```
data_root/
├── train/
│   ├── original/          # real frames
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceShifter/
│   ├── FaceSwap/
│   └── NeuralTextures/
├── val/
└── test/
```

**CelebDF (cross-dataset test):**
```
cross_dataset_root/
├── real/
└── fake/
```

Directory matching is case-insensitive.

## Key Design Notes

- Best checkpoint is selected by **validation accuracy** (not AUC); AUC is also tracked and saved.
- `original_upsample_factor` only applies to the training real split — upsampled copies get the same augmentation pipeline as all other training images.
- `train_real_percent` downsampling uses a fixed seed (`seed` in config) for reproducibility.
- Augmentation uses **albumentations**: HorizontalFlip, Rotate(±10°), Scale, Translate, BrightnessContrast, GaussNoise, GaussianBlur, ImageCompression at train time; Resize(224) + ImageNet normalization only at eval/test time.
- `utils/metrics.py::compute_binary_metrics()` returns accuracy, F1, precision, recall, AUC; `safe_auc()` returns NaN when only one class is present.
