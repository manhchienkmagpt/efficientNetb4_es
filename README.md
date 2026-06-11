# Deepfake Detection Backbones

PyTorch project for binary deepfake frame detection with a configurable ImageNet backbone, ECA attention, and SCConv. The model outputs one raw logit. Training uses `BCEWithLogitsLoss`; inference converts logits with `torch.sigmoid`.

Labels:

- `original` / `real`: `0`
- fake classes: `1`

## Project Structure

```text
deepfake_efficientnetb4_es/
├── configs/config.yaml
├── datasets/deepfake_dataset.py
├── models/efficientnetb4_es.py
├── models/modules.py
├── utils/
├── train.py
├── test_ffpp.py
├── test_celebdf.py
├── requirements.txt
└── README.md
```

## Install

```bash
cd deepfake_efficientnetb4_es
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install the CUDA build of PyTorch that matches your machine if needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Data Layout

FF++ data is expected at `D:/duong_huy_ct7/deepfake-data`:

```text
train|val|test/
├── original
├── Deepfakes
├── Face2Face
├── FaceShifter
├── FaceSwap
└── NeuralTextures
```

CelebDF test data is expected at `D:/duong_huy_ct7/deepfake-data/celeb-df/test`:

```text
test/
├── fake
└── real
```

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

GAN training data can be configured with `gan_fake_dir` and `gan_real_dir` in `configs/config.yaml`.
Images under `gan_fake_dir` use label `1`; images under `gan_real_dir` use label `0`.
The default local layout is:

```text
+-- gan_fake/
|   `-- fake_001.jpg
`-- gan_real/
    `-- real_001.jpg
```

Set the two folders directly:

```yaml
gan_fake_dir: "path/to/gan_fake"
gan_real_dir: "path/to/gan_real"
```

## Train

Edit `configs/config.yaml` if needed, then run:

```bash
python train.py --config configs/config.yaml
```

Choose a backbone in `configs/config.yaml`:

```yaml
backbone: "efficientnetb4_es"  # options: efficientnetb4_es, resnet50, swin_tiny
```

Resume from a checkpoint:

```bash
python train.py --config configs/config.yaml --resume checkpoints/best_model.pth
```

Or set `resume_from` in `configs/config.yaml`.

The best checkpoint is saved by validation accuracy to:

```text
checkpoints/best_model.pth
```

Training uses:

- `BCEWithLogitsLoss`
- `AdamW`
- `ReduceLROnPlateau(mode="max")` tracking validation accuracy
- Early stopping tracking validation accuracy
- Strong augmentation for fake train samples and upsampled original samples
- Base eval transform only for original train samples that are not upsampled

`original_upsample_factor: N` keeps all original samples and adds `N` extra augmented copies for each original training image.

## Train With GAN Data

Edit `gan_data_root` in `configs/config.yaml`, then run:

```bash
python train_with_gan.py --config configs/config.yaml
```

Resume from a checkpoint:

```bash
python train_with_gan.py --config configs/config.yaml --resume checkpoints/best_model_with_gan.pth
```

This script trains on FF++ train data plus GAN train data and keeps FF++ validation unchanged. The best checkpoint defaults to:

```text
checkpoints/best_model_with_gan.pth
```

## Test FF++

```bash
python test_ffpp.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

Optional CSV path:

```bash
python test_ffpp.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --output-csv outputs/ffpp_predictions.csv
```

## Test CelebDF

```bash
python test_celebdf.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

Optional CSV path:

```bash
python test_celebdf.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --output-csv outputs/celebdf_predictions.csv
```

Both test scripts print Accuracy, F1, Precision, Recall, AUC, and the confusion matrix, and save per-image predictions with:

- `image_path`
- `label`
- `probability`
- `prediction`
