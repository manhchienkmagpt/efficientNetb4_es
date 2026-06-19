# Deepfake Detection Backbones

PyTorch project for binary deepfake frame detection with a configurable ImageNet backbone, ECA attention, and SCConv. The model outputs one raw logit. Training uses `BCEWithLogitsLoss`; inference converts logits with `torch.sigmoid`.

Labels:

- `real`: `0`
- `fake`: `1`

## Project Structure

```text
deepfake_efficientnetb4_es/
|-- configs/config.yaml
|-- datasets/deepfake_dataset.py
|-- models/efficientnetb4_es.py
|-- models/modules.py
|-- utils/
|-- train.py
|-- train_with_gan.py
|-- test_origin_dataset.py
|-- test_cross_dataset.py
|-- requirements.txt
`-- README.md
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

Origin train, validation, and test data are expected under `data_root`:

```text
root/
|-- train/
|   |-- real/
|   `-- fake/
|-- val/
|   |-- real/
|   `-- fake/
`-- test/
    |-- real/
    `-- fake/
```

The cross-dataset test root is configured with `cross_dataset_root` and should contain:

```text
cross_dataset_root/
|-- real/
`-- fake/
```

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

GAN training data can be configured with `gan_fake_dir` and `gan_real_dir` in `configs/config.yaml`.
Images under `gan_fake_dir` use label `1`; images under `gan_real_dir` use label `0`.

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
- Strong augmentation for all train samples, including real, fake, and upsampled real samples

Set `original_upsample_factor: null` to disable upsampling. Set it to `N` to keep all real samples and add `N` extra augmented copies for each real training image.

Set `train_real_percent` to control how many real images from `train_dir/real` are used during training:

```yaml
train_real_percent: 50  # use 50% of train/real
```

This only affects the origin training split. Validation, test, and GAN real data are unchanged.

## Train With GAN Data

Edit `gan_fake_dir` and `gan_real_dir` in `configs/config.yaml`, then run:

```bash
python train_with_gan.py --config configs/config.yaml
```

Resume from a checkpoint:

```bash
python train_with_gan.py --config configs/config.yaml --resume checkpoints/best_model_with_gan.pth
```

This script trains on origin train data plus GAN train data and keeps origin validation unchanged. The best checkpoint defaults to:

```text
checkpoints/best_model_with_gan.pth
```

## Test Origin Dataset

```bash
python test_origin_dataset.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

Optional CSV path:

```bash
python test_origin_dataset.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --output-csv outputs/origin_predictions.csv
```

## Test Cross Dataset

```bash
python test_cross_dataset.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

Optional CSV path:

```bash
python test_cross_dataset.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --output-csv outputs/cross_predictions.csv
```

Both test scripts print Accuracy, F1, Precision, Recall, AUC, and the confusion matrix, and save per-image predictions with:

- `image_path`
- `label`
- `probability`
- `prediction`
