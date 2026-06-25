# CelebDF to FF++ Workflow

This folder keeps a separate workflow for training on CelebDF and evaluating on FF++ without changing the existing FF++ scripts in the repository root.

## Expected Layout

CelebDF root:
```text
celebdf_root/
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

FF++ root:
```text
ffpp_root/
`-- test/
    |-- original/
    |-- Deepfakes/
    |-- Face2Face/
    |-- FaceShifter/
    |-- FaceSwap/
    `-- NeuralTextures/
```

## Commands

Train on CelebDF:
```bash
python celebdf_to_ffpp/train_celebdf.py --config celebdf_to_ffpp/config.yaml
```

Test the saved checkpoint on FF++:
```bash
python celebdf_to_ffpp/test_ffpp.py --config celebdf_to_ffpp/config.yaml
```

`test_ffpp.py` writes prediction CSVs under `celebdf_to_ffpp/outputs/` by default.
