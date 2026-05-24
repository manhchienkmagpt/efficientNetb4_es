# EfficientNetB4-ES Deepfake Detection

PyTorch project for binary deepfake frame detection with an EfficientNet-B4 ImageNet backbone, ECA attention, and SCConv. The model outputs one raw logit. Training uses `BCEWithLogitsLoss`; inference converts logits with `torch.sigmoid`.

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

## Train

Edit `configs/config.yaml` if needed, then run:

```bash
python train.py --config configs/config.yaml
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
