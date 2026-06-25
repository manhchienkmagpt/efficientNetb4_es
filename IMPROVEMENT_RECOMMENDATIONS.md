# Deepfake Detection — Improvement Recommendations

## Current Results

**Model:** ResNet-50 | **Trained on:** FF++ | **Cross-tested on:** CelebDF

| Dataset | Accuracy | F1 | Precision | Recall | AUC |
|---------|----------|----|-----------|--------|-----|
| FF++ (in-distribution) | 0.9336 | 0.9605 | 0.9525 | 0.9686 | 0.9669 |
| CelebDF (cross-dataset) | 0.6938 | 0.7708 | 0.7578 | 0.7842 | 0.7199 |

**Key weaknesses:**
- AUC drop of 0.25 from in-distribution to cross-dataset → poor generalization
- FP rate on real images: 24.2% (FF++) → 47.9% (CelebDF) → strong fake-prediction bias
- ResNet-50 lacks the fine-grained local feature extraction that EfficientNet-B4 provides, making it more susceptible to dataset shift
- Model likely learned FF++-specific compression artifacts rather than generalizable forgery cues

---

## Recommendations (Prioritized)

### 1. Frequency-Domain Input

The model relies on RGB pixel artifacts that are dataset-specific. Adding a frequency branch forces it to learn generalizable forgery signals.

```python
import numpy as np, cv2

def dct_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.dct(gray)  # feed as extra channel alongside RGB
```

Or use **SRM (Steganalysis Rich Model)** filters as a fixed preprocessing layer — widely used in deepfake generalization literature.

---

### 2. Fix Class Imbalance with Weighted Loss

The 1:5 real:fake ratio causes the fake-prediction bias (47.9% FP on CelebDF real images).

```python
# train.py
pos_weight = torch.tensor([num_real / num_fake])  # ~0.2
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

Or switch to **Focal Loss** to focus on hard examples (alpha=0.25, gamma=2 are standard starting points).

---

### 3. Stronger JPEG/Compression Augmentation

FF++ uses specific compression levels; CelebDF differs. Randomize heavily during training:

```python
# datasets/deepfake_dataset.py — training transform
A.ImageCompression(quality_lower=30, quality_upper=95, p=0.8),
A.RandomGamma(gamma_limit=(80, 120), p=0.5),
```

---

### 4. Multi-Dataset Training

Mix CelebDF into training (keep a held-out test portion). The `ConcatDataset` pattern in `train_with_gan.py` already supports this:

```yaml
# configs/config.yaml
extra_train_dirs:
  - /path/to/celebdf/train/real
  - /path/to/celebdf/train/fake
```

---

### 5. Face-Region Attention / Landmark Cropping

The model processes full frames. Cropping tightly to face + adding landmark-guided attention improves cross-dataset robustness:

- Use `insightface` or `mediapipe` to extract face crops with margin
- Or add a **CBAM attention block** after the backbone feature extractor

---

### 6. Label Smoothing

Reduces overconfidence on FF++-specific features:

```python
# Implement manually since BCEWithLogitsLoss doesn't have label_smoothing
targets = targets * (1 - smoothing) + smoothing * 0.5
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

### 7. Test-Time Augmentation (TTA)

Quick win at inference — average predictions over flipped/rotated versions:

```python
# test_cross_dataset.py / test_origin_dataset.py
preds = []
for aug in [original, hflip, rotate5, rotate_neg5]:
    preds.append(torch.sigmoid(model(aug)))
final = torch.stack(preds).mean(0)
```

---

### 8. Tune Decision Threshold

Given the model's fake-prediction bias, tune the threshold on a validation set to reduce FP rate:

```yaml
# configs/config.yaml
threshold: 0.65  # instead of 0.5 — tune on val set
```

---

## Priority Summary

| Priority | Change | Effort | Expected Gain |
|----------|--------|--------|---------------|
| 1 | Weighted loss / focal loss | Low | Fixes FP bias immediately |
| 2 | Stronger compression augmentation | Low | +2–5 AUC cross-dataset |
| 3 | Tune decision threshold | Low | +1–3 AUC free |
| 4 | TTA at inference | Low | +1–2 AUC free |
| 5 | Multi-dataset training | Medium | +5–8 AUC |
| 6 | Frequency branch / SRM filters | Medium | +5–10 AUC cross-dataset |
| 7 | Face crop + attention | High | +3–7 AUC long term |

> Start with priorities 1–4 (low effort, no architectural changes) to validate the training loop before committing to larger changes.
