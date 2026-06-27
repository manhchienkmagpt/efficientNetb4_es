# Repository Guidelines

## Project Structure & Module Organization
Core training and evaluation entrypoints for the main FF++ workflow live at the repository root: `train.py`, `train_with_gan.py`, `test_origin_dataset.py`, and `test_cross_dataset.py`. Model definitions are in `models/`, dataset loading and Albumentations pipelines are in `datasets/`, and shared helpers such as checkpoints, metrics, and seeding are in `utils/`. Runtime settings are centralized in `configs/config.yaml`. The `celebdf_to_ffpp/` subfolder is a separate workflow with its own files: `train.py`, `train_with_gan.py`, `test_origin_dataset.py`, `test_cross_dataset.py`, `ffpp_dataset.py`, and `config.yaml`. Keep generated checkpoints and CSV predictions out of source folders; use dedicated output directories such as `checkpoints_efficientnetb4/`, `checkpoints_celebdf_to_ffpp/`, `outputs/`, or `celebdf_to_ffpp/outputs/`.

## Build, Test, and Development Commands
Create an environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Train on the base dataset:
```bash
python train.py --config configs/config.yaml
```
Train with additional GAN data:
```bash
python train_with_gan.py --config configs/config.yaml
```
Run evaluation scripts:
```bash
python test_origin_dataset.py --config configs/config.yaml --checkpoint checkpoints_efficientnetb4/best_efficientnetb4.pth
python test_cross_dataset.py --config configs/config.yaml --checkpoint checkpoints_efficientnetb4/best_efficientnetb4.pth
```

For the separate CelebDF to FF++ workflow:
```bash
python celebdf_to_ffpp/train.py --config celebdf_to_ffpp/config.yaml
python celebdf_to_ffpp/test_origin_dataset.py --config celebdf_to_ffpp/config.yaml --checkpoint checkpoints_celebdf_to_ffpp/best_celebdf_to_ffpp.pth
python celebdf_to_ffpp/test_cross_dataset.py --config celebdf_to_ffpp/config.yaml --checkpoint checkpoints_celebdf_to_ffpp/best_celebdf_to_ffpp.pth
```

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and concise type hints where useful. Prefer `pathlib.Path` for filesystem paths and keep configuration-driven values in `configs/config.yaml` instead of hardcoding them in scripts. Match existing naming patterns such as `*_dataset.py`, `test_*.py`, and descriptive checkpoint names like `best_efficientnetb4_gan.pth`.

## Testing Guidelines
This repo uses runnable evaluation scripts rather than a dedicated unit-test framework. For the main FF++ workflow, treat `test_origin_dataset.py` and `test_cross_dataset.py` as validation entrypoints. For `celebdf_to_ffpp/`, use `test_origin_dataset.py` for CelebDF test evaluation and `test_cross_dataset.py` for FF++ evaluation. Run at least the relevant script before submitting changes that affect data loading, metrics, or inference. When changing training behavior, verify the config still loads and document the exact command used.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `update dataset` and `update model`. Keep commits focused and descriptive, for example `adjust ffpp label mapping` or `add gan checkpoint naming`. Pull requests should include: what changed, why it changed, which config or dataset layout assumptions apply, and the commands used for verification. Include metric deltas or sample output paths when behavior changes affect training or evaluation results.

## Configuration & Data Notes
`configs/config.yaml` contains machine-specific dataset paths and device settings for the main workflow, while `celebdf_to_ffpp/config.yaml` configures the CelebDF to FF++ pipeline. Do not commit private local paths or large generated artifacts. Verify folder layouts match the expected dataset structure before running training or evaluation: the main workflow expects `train/`, `val/`, `test/`, `real/`, and `fake/`, while the root `README.md` documents the separate CelebDF and FF++ layout used by `celebdf_to_ffpp/`.
