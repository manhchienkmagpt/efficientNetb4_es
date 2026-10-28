"""Video-level evaluation for the FA-ViT-CNN paper (ICCIES 2027 revision).

CHUA CHAY THU (viet tren may khong co Python/GPU). Chay `--selftest` truoc.

Pipeline
  1. Predict per-frame fake probability for every (model, dataset) in the manifest
     (or reuse an existing per-frame CSV from test_*.py).
  2. Group frames by video (video id parsed from the frame filename).
  3. Report frame-level and video-level (mean of frame probabilities) metrics,
     stratified video bootstrap CIs, paired DeLong / paired bootstrap between models.

No threshold or hyperparameter is tuned on cross-dataset data: the optional
video-level threshold is chosen on the dataset marked `role: val` (FF++ val) only.

Usage (run from efficientNetb4_es/, the source used for the paper):
  python eval_video_level.py --selftest
  python eval_video_level.py --manifest video_eval_manifest.yaml
  python eval_video_level.py --manifest video_eval_manifest.yaml --analyze-only
"""

import argparse
import datetime
import hashlib
import itertools
import json
import math
import platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# <dataset>_<split>_<source>_<video_id>_sample000012_frame000345.jpg  ->  group "video"
FRAME_RE = re.compile(r"^(?P<video>.+)_sample\d+_frame\d+$")
CLASS_DIR_NAMES = {
    "real", "fake", "original", "deepfakes", "face2face", "faceshifter", "faceswap", "neuraltextures", "0", "1",
}


def label_mapping(dataset_entry: Dict) -> Optional[Dict[str, int]]:
    """Optional `label_dirs: {"0": 0, "1": 1}` for folders not named real/fake (0 = real, 1 = fake)."""
    dirs = dataset_entry.get("label_dirs")
    return {str(k): int(v) for k, v in dirs.items()} if dirs else None


# Module level (not inside a function) so DataLoader workers can pickle it on Windows (spawn).
try:
    from datasets import DeepfakeFrameDataset as _BaseFrameDataset
except ImportError:  # --selftest / --analyze-only without the training dependencies
    _BaseFrameDataset = None

if _BaseFrameDataset is not None:
    class MappedFrameDataset(_BaseFrameDataset):
        """Same loader as the frame-level test; only the folder-name -> label map can be overridden."""

        def __init__(self, *args, label_map: Optional[Dict[str, int]] = None, **kwargs) -> None:
            self.label_map = label_map
            super().__init__(*args, **kwargs)

        def _get_class_mapping(self) -> Dict[str, int]:
            return dict(self.label_map) if self.label_map else super()._get_class_mapping()
else:
    MappedFrameDataset = None


RESERVED_COLUMNS = {"image_path", "label", "prob", "video_id", "source"}
MAIN_THRESHOLD = 0.5


# --------------------------------------------------------------------------- prediction

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


EXCLUDED_CHECKPOINT_TAGS = ("gan", "cross", "stage_2", "stage2")


def resolve_checkpoint(path_str: str) -> Path:
    """Accept a .pth file or a checkpoint folder holding exactly one train.py checkpoint."""
    path = Path(path_str)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    candidates = sorted(p for p in path.rglob("*.pth")
                        if not any(tag in p.name.lower() for tag in EXCLUDED_CHECKPOINT_TAGS))
    if len(candidates) != 1:
        raise ValueError(
            f"{path}: expected exactly one train.py checkpoint (excluding names with "
            f"{EXCLUDED_CHECKPOINT_TAGS}), found {[str(p) for p in candidates]}. Put the exact .pth path instead."
        )
    return candidates[0]


def load_model(model_entry: Dict):
    import inspect

    import torch  # noqa: F401  (heavy imports only when predicting)
    from models import build_model
    from train import load_config, resolve_device
    from utils.checkpoint import load_checkpoint

    model_entry = {**model_entry, "checkpoint": str(resolve_checkpoint(model_entry["checkpoint"]))}
    print(f"[{model_entry['name']}] checkpoint: {model_entry['checkpoint']}", flush=True)
    checkpoint = load_checkpoint(model_entry["checkpoint"], "cpu")
    # Default: the training config saved inside the checkpoint (the most reliable record of how it was built).
    if model_entry.get("config"):
        config = load_config(model_entry["config"])
    elif isinstance(checkpoint.get("config"), dict):
        config = dict(checkpoint["config"])
    else:
        raise ValueError(f"{model_entry['name']}: checkpoint has no saved config; set `config:` in the manifest.")
    config.update(model_entry.get("config_overrides") or {})
    device = resolve_device(str(config.get("device", "cuda")))

    kwargs = {
        "backbone": str(config.get("backbone", "efficientnetb4")),
        "pretrained": False,
        "dropout": float(config.get("dropout", 0.4)),
        "image_size": int(config["image_size"]),
        "freq_in_channels": int(config.get("freq_in_channels", 3)),
        "freq_dim": int(config.get("freq_dim", 128)),
        "use_freq": bool(config.get("use_freq", True)),
        "favit_checkpoint": config.get("favit_checkpoint"),
        "swin_checkpoint": config.get("swin_checkpoint"),
    }
    accepted = inspect.signature(build_model).parameters
    model = build_model(**{k: v for k, v in kwargs.items() if k in accepted}).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])  # strict: fails loudly on a wrong backbone
    model.eval()

    info = {
        "checkpoint": str(Path(model_entry["checkpoint"]).resolve()),
        "checkpoint_sha256": sha256_of(Path(model_entry["checkpoint"])),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_auc": checkpoint.get("best_auc"),
        "checkpoint_best_accuracy": checkpoint.get("best_accuracy"),
        "checkpoint_train_config": {k: (checkpoint.get("config") or {}).get(k) for k in (
            "backbone", "data_root", "cross_dataset_root", "save_dir", "checkpoint_name", "seed", "lambda_fal",
            "resume_from", "use_freq", "use_pos_weight", "pos_weight", "label_smoothing", "lr", "epochs")},
        "config_source": model_entry.get("config") or "checkpoint['config']",
        "backbone": config.get("backbone"),
        "image_size": config.get("image_size"),
        "seed": config.get("seed"),
        "tta": bool(model_entry.get("tta", False)),
    }
    return model, device, config, info


def predict_frames(model, device, config: Dict, dataset_entry: Dict, model_entry: Dict,
                   batch_size: Optional[int], num_workers: Optional[int]) -> pd.DataFrame:
    from torch.utils.data import DataLoader
    from datasets import get_eval_transform
    from utils.inference import predict, predict_tta

    if MappedFrameDataset is None:
        raise ImportError("Cannot import datasets.DeepfakeFrameDataset; run from the efficientNetb4_es folder.")
    dataset = MappedFrameDataset(
        label_map=label_mapping(dataset_entry),
        root_dir=dataset_entry["root"],
        split=dataset_entry.get("split"),
        dataset_type=dataset_entry.get("type", "cross"),
        eval_transform=get_eval_transform(int(config["image_size"])),
        original_upsample_factor=0,
        mode="test",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or config["batch_size"]),
        shuffle=False,
        num_workers=int(num_workers if num_workers is not None else config["num_workers"]),
        pin_memory=True,
    )
    run_predict = predict_tta if model_entry.get("tta", False) else predict
    paths, labels, probs = run_predict(model, loader, device,
                                       desc=f"{model_entry['name']} | {dataset_entry['name']}")
    return pd.DataFrame({"image_path": paths, "label": np.asarray(labels).astype(int), "prob": probs})


def read_frames_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "prob" not in df.columns and "probability" in df.columns:
        df = df.rename(columns={"probability": "prob"})
    missing = {"image_path", "label", "prob"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df[["image_path", "label", "prob"]].copy()
    df["label"] = df["label"].astype(float).round().astype(int)
    return df


# --------------------------------------------------------------------------- grouping

def find_source(path: Path) -> str:
    for parent in path.parents:
        if parent.name.lower() in CLASS_DIR_NAMES:
            return parent.name
    return "unknown"


def video_key(path: Path, regex: re.Pattern) -> Optional[str]:
    match = regex.match(path.stem)
    if match:
        return f"{path.parent.name}/{match.group('video')}"
    # Nested layout root/{real,fake}/<video>/<frame>.jpg: the parent folder is the video.
    if path.parent.name.lower() not in CLASS_DIR_NAMES:
        return f"{path.parent.parent.name}/{path.parent.name}"
    return None


def add_video_ids(df: pd.DataFrame, regex: re.Pattern, dataset_name: str) -> pd.DataFrame:
    df = df.drop_duplicates(subset="image_path").copy()
    paths = [Path(p) for p in df["image_path"]]
    keys = [video_key(p, regex) for p in paths]
    unresolved = [str(p) for p, k in zip(paths, keys) if k is None]
    if unresolved:
        raise ValueError(
            f"[{dataset_name}] cannot parse video id for {len(unresolved)} frames, e.g. {unresolved[:3]}. "
            "Set `video_regex` for this dataset in the manifest (named group `video`)."
        )
    df["video_id"] = keys
    df["source"] = [find_source(p) for p in paths]
    return df


def build_wide(dfs: Dict[str, pd.DataFrame], models: List[str], dataset_name: str,
               sanity: List[str]) -> pd.DataFrame:
    base = dfs[models[0]][["image_path", "label", "video_id", "source"]].set_index("image_path")
    probs = pd.concat({m: dfs[m].set_index("image_path")["prob"] for m in models}, axis=1)
    wide = base.join(probs, how="left")
    n_before = len(wide)
    wide = wide.dropna(subset=models)
    if len(wide) != n_before:
        sanity.append(f"[WARN] {dataset_name}: {n_before - len(wide)} frames missing in some model; dropped.")
    for m in models[1:]:
        extra = len(dfs[m]) - len(wide)
        if extra:
            sanity.append(f"[WARN] {dataset_name}: model {m} has {extra} frames not in the aligned set.")
    for m in models[1:]:
        labels_m = dfs[m].set_index("image_path")["label"].reindex(wide.index)
        if (labels_m.values != wide["label"].values).any():
            raise ValueError(f"[{dataset_name}] label mismatch between {models[0]} and {m}.")
    return wide.reset_index()


def build_video_table(wide: pd.DataFrame, models: List[str], dataset_name: str) -> pd.DataFrame:
    groups = wide.groupby("video_id", sort=True)
    vid = pd.DataFrame({
        "label": groups["label"].first(),
        "label_nunique": groups["label"].nunique(),
        "n_frames": groups.size(),
        "source": groups["source"].first(),
    })
    if (vid["label_nunique"] > 1).any():
        bad = vid.index[vid["label_nunique"] > 1][:3].tolist()
        raise ValueError(f"[{dataset_name}] videos with mixed labels (wrong video id parsing?): {bad}")
    vid = vid.drop(columns="label_nunique")
    for m in models:
        vid[f"{m}__mean"] = groups[m].mean()
        vid[f"{m}__median"] = groups[m].median()
    return vid


# --------------------------------------------------------------------------- statistics

def binary_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")
    return {"acc": (tp + tn) / len(y), "auc": auc, "f1": f1, "precision": prec, "recall": rec,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    best = int(np.argmax(j[1:]) + 1)  # thr[0] is +inf
    return float(min(thr[best], 1.0))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_paired(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> Tuple[float, float, float, float]:
    """Fast DeLong (Sun & Xu, 2014) for two correlated AUCs. Returns auc1, auc2, z, p (two-sided)."""
    pos, neg = y == 1, y == 0
    x = np.vstack([s1[pos], s2[pos]])
    yy = np.vstack([s1[neg], s2[neg]])
    m, n = x.shape[1], yy.shape[1]
    tx = np.vstack([_midrank(r) for r in x])
    ty = np.vstack([_midrank(r) for r in yy])
    tz = np.vstack([_midrank(np.concatenate([x[k], yy[k]])) for k in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    for k, s in enumerate((s1, s2)):
        assert abs(aucs[k] - roc_auc_score(y, s)) < 1e-9, "DeLong AUC disagrees with sklearn"
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan"), float("nan")
    z = (aucs[0] - aucs[1]) / math.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(math.erfc(abs(z) / math.sqrt(2)))


def _stratified_sample(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_video(y: np.ndarray, scores: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((n_boot, scores.shape[1]))
    for b in range(n_boot):
        idx = _stratified_sample(y, rng)
        for j in range(scores.shape[1]):
            out[b, j] = roc_auc_score(y[idx], scores[idx, j])
    return out


def bootstrap_frame_cluster(y_video: np.ndarray, frame_lists: List[np.ndarray], y_frame: np.ndarray,
                            scores: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Resample whole videos (stratified by label) and pool their frames: honest CI for frame-level AUC."""
    out = np.empty((n_boot, scores.shape[1]))
    for b in range(n_boot):
        idx = _stratified_sample(y_video, rng)
        fidx = np.concatenate([frame_lists[i] for i in idx])
        for j in range(scores.shape[1]):
            out[b, j] = roc_auc_score(y_frame[fidx], scores[fidx, j])
    return out


def ci(values: np.ndarray) -> Tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def boot_p(delta: np.ndarray) -> float:
    return float(min(1.0, 2 * min((delta <= 0).mean(), (delta >= 0).mean())))


# --------------------------------------------------------------------------- analysis

REPRO_TOL_PP = 0.01  # paper tables are rounded to 2 decimals
METRIC_ORDER = ("acc", "auc", "f1", "precision", "recall")  # column order of Tables 1-3


def expected_metrics(model_entry: Dict) -> Dict[str, Dict[str, float]]:
    """Paper frame-level numbers: `expected_frame_metrics: {ds: [Acc, AUC, F1, Prec, Rec]}`
    or the older `expected_frame_auc: {ds: AUC}`."""
    out = {ds: dict(zip(METRIC_ORDER, map(float, vals)))
           for ds, vals in (model_entry.get("expected_frame_metrics") or {}).items()}
    for ds, auc in (model_entry.get("expected_frame_auc") or {}).items():
        out.setdefault(ds, {"auc": float(auc)})
    return out


def analyze(frames: Dict[Tuple[str, str], pd.DataFrame], manifest: Dict, out_dir: Path,
            n_boot: int, seed: int) -> None:
    models = [m["name"] for m in manifest["models"]]
    datasets = manifest["datasets"]
    expected = {m["name"]: expected_metrics(m) for m in manifest["models"]}
    n_checks = n_fail = 0
    pairs = manifest.get("compare") or [list(p) for p in itertools.combinations(models, 2)]
    rng = np.random.default_rng(seed)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    sanity: List[str] = []
    summary, comparisons, per_source = [], [], []
    wides, vids = {}, {}

    for d in datasets:
        dn = d["name"]
        regex = re.compile(d["video_regex"]) if d.get("video_regex") else FRAME_RE
        dfs = {m: add_video_ids(frames[(m, dn)], regex, dn) for m in models}
        wide = build_wide(dfs, models, dn, sanity)
        vid = build_video_table(wide, models, dn)
        wides[dn], vids[dn] = wide, vid
        vid.to_csv(out_dir / "videos" / f"{dn}.csv")

        n_real = int((vid["label"] == 0).sum())
        n_fake = int((vid["label"] == 1).sum())
        nf = vid["n_frames"]
        sanity.append(
            f"{dn}: {len(wide)} frames | videos {len(vid)} (real {n_real}, fake {n_fake}) | "
            f"frames/video min {nf.min()} median {nf.median():.0f} max {nf.max()} | "
            f"videos with < half the median frames: {int((nf < 0.5 * nf.median()).sum())}"
        )
        if d.get("expected_videos") and int(d["expected_videos"]) != len(vid):
            sanity.append(f"[WARN] {dn}: expected {d['expected_videos']} videos, found {len(vid)}. "
                          "Check video id parsing or missing videos (no face detected).")
        if d.get("type", "cross") in {"ffpp", "origin", "original"}:
            counts = vid.groupby("source").size().to_dict()
            sanity.append(f"{dn}: videos per source {counts}")

    val_name = next((d["name"] for d in datasets if d.get("role") == "val"), None)
    val_thr = {}
    if val_name:
        v = vids[val_name]
        for m in models:
            val_thr[m] = youden_threshold(v["label"].values.astype(int), v[f"{m}__mean"].values)
        sanity.append(f"Video-level thresholds chosen on {val_name} (Youden J, mean aggregation): "
                      + ", ".join(f"{m}={t:.4f}" for m, t in val_thr.items()))

    for d in datasets:
        dn = d["name"]
        if d.get("role") == "val":
            continue
        wide, vid = wides[dn], vids[dn]
        y_v = vid["label"].values.astype(int)
        s_v = vid[[f"{m}__mean" for m in models]].values
        y_f = wide["label"].values.astype(int)
        s_f = wide[models].values
        positions = wide.groupby("video_id", sort=True).indices
        frame_lists = [np.asarray(positions[v]) for v in vid.index]

        print(f"Bootstrap {dn}: {n_boot} resamples x {len(models)} models ...", flush=True)
        boot_v = bootstrap_video(y_v, s_v, n_boot, rng)
        boot_f = bootstrap_frame_cluster(y_v, frame_lists, y_f, s_f, n_boot, rng)

        common = {"dataset": dn, "n_videos": len(vid), "n_real_videos": int((y_v == 0).sum()),
                  "n_fake_videos": int((y_v == 1).sum()), "n_frames": len(wide)}
        for j, m in enumerate(models):
            rows = [
                ("frame", "-", "0.5", MAIN_THRESHOLD, y_f, s_f[:, j], boot_f[:, j]),
                ("video", "mean", "0.5", MAIN_THRESHOLD, y_v, s_v[:, j], boot_v[:, j]),
                ("video", "median", "0.5", MAIN_THRESHOLD, y_v, vid[f"{m}__median"].values, None),
            ]
            if m in val_thr:
                rows.append(("video", "mean", f"val({val_name})", val_thr[m], y_v, s_v[:, j], boot_v[:, j]))
            for level, agg, thr_name, thr, y, s, boot in rows:
                met = binary_metrics(y, s, thr)
                lo, hi = ci(boot) if boot is not None else (float("nan"), float("nan"))
                row = {"model": m, "level": level, "aggregation": agg, "threshold_rule": thr_name,
                       "threshold": thr, **common, **met, "auc_ci_lo": lo, "auc_ci_hi": hi}
                if level == "frame" and dn in expected[m]:
                    row["paper_frame_auc"] = float(expected[m][dn]["auc"])
                    row["diff_vs_paper_pp"] = met["auc"] * 100 - row["paper_frame_auc"]
                    for key, paper_value in expected[m][dn].items():
                        diff = met[key] * 100 - float(paper_value)
                        n_checks += 1
                        if abs(diff) > REPRO_TOL_PP:
                            n_fail += 1
                            sanity.append(f"[WARN] {m} on {dn}: frame {key} {met[key] * 100:.2f} vs paper "
                                          f"{float(paper_value):.2f} (diff {diff:+.3f} pp). "
                                          "Pipeline differs from the frame-level run (checkpoint/config/TTA/data?)")
                summary.append(row)

        for a, b in pairs:
            if a not in models or b not in models:
                sanity.append(f"[WARN] compare pair {a} vs {b}: unknown model name, skipped.")
                continue
            ia, ib = models.index(a), models.index(b)
            auc_a, auc_b, z, p = delong_paired(y_v, s_v[:, ia], s_v[:, ib])
            dv = boot_v[:, ia] - boot_v[:, ib]
            comparisons.append({"dataset": dn, "level": "video(mean)", "model_a": a, "model_b": b,
                                "auc_a": auc_a, "auc_b": auc_b, "delta_pp": (auc_a - auc_b) * 100,
                                "delta_ci_lo_pp": ci(dv)[0] * 100, "delta_ci_hi_pp": ci(dv)[1] * 100,
                                "boot_p": boot_p(dv), "delong_z": z, "delong_p": p, "n_units": len(vid)})
            fa = roc_auc_score(y_f, s_f[:, ia])
            fb = roc_auc_score(y_f, s_f[:, ib])
            df_ = boot_f[:, ia] - boot_f[:, ib]
            comparisons.append({"dataset": dn, "level": "frame(cluster-bootstrap)", "model_a": a, "model_b": b,
                                "auc_a": fa, "auc_b": fb, "delta_pp": (fa - fb) * 100,
                                "delta_ci_lo_pp": ci(df_)[0] * 100, "delta_ci_hi_pp": ci(df_)[1] * 100,
                                "boot_p": boot_p(df_), "delong_z": float("nan"), "delong_p": float("nan"),
                                "n_units": len(vid)})

        if d.get("type", "cross") in {"ffpp", "origin", "original"}:
            for src in sorted(vid.loc[vid["label"] == 1, "source"].unique()):
                vmask = ((vid["label"] == 0) | (vid["source"] == src)).values
                fmask = ((wide["label"] == 0) | (wide["source"] == src)).values
                for j, m in enumerate(models):
                    per_source.append({"dataset": dn, "source": src, "model": m,
                                       "n_fake_videos": int((vid["source"] == src).sum()),
                                       "video_auc": roc_auc_score(y_v[vmask], s_v[vmask, j]),
                                       "frame_auc": roc_auc_score(y_f[fmask], s_f[fmask, j])})

    if n_checks:
        verdict = "REPRODUCTION OK" if n_fail == 0 else "REPRODUCTION FAILED"
        sanity.insert(0, f"{verdict}: {n_checks - n_fail}/{n_checks} frame-level numbers match the paper "
                         f"(tolerance {REPRO_TOL_PP} pp). Video-level results are valid only if this is OK.")
    summary_df = pd.DataFrame(summary)
    comp_df = pd.DataFrame(comparisons)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    comp_df.to_csv(out_dir / "comparisons.csv", index=False)
    if per_source:
        pd.DataFrame(per_source).to_csv(out_dir / "per_source_ffpp.csv", index=False)
    (out_dir / "sanity.txt").write_text("\n".join(sanity) + "\n", encoding="utf-8")
    write_paper_tables(summary_df, comp_df, models, out_dir / "paper_tables.md", n_boot)
    print("\n".join(sanity))
    print(f"\nDone. Results in: {out_dir.resolve()}")


def write_paper_tables(summary: pd.DataFrame, comp: pd.DataFrame, models: List[str], path: Path,
                       n_boot: int) -> None:
    pct = lambda v: "--" if pd.isna(v) else f"{v * 100:.2f}"  # noqa: E731
    lines = ["# Video-level results (auto-generated)", "",
             f"Video score = mean of frame probabilities. 95% CI: stratified bootstrap over videos, "
             f"{n_boot} resamples. Frame CI uses cluster (video) bootstrap. Threshold 0.5 unless stated.", ""]
    for dn in summary["dataset"].unique():
        s = summary[summary["dataset"] == dn]
        first = s.iloc[0]
        lines += [f"## {dn}  (videos: {first['n_videos']} = {first['n_real_videos']} real + "
                  f"{first['n_fake_videos']} fake; frames: {first['n_frames']})", "",
                  "| Model | Frame AUC (paper) | Frame AUC [95% CI] | Video AUC [95% CI] | Video Acc | "
                  "Video F1 | Video Prec | Video Rec | Video AUC (median agg) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m in models:
            fr = s[(s["model"] == m) & (s["level"] == "frame")].iloc[0]
            vm = s[(s["model"] == m) & (s["aggregation"] == "mean") & (s["threshold_rule"] == "0.5")].iloc[0]
            vmed = s[(s["model"] == m) & (s["aggregation"] == "median")].iloc[0]
            paper = fr.get("paper_frame_auc", float("nan"))
            paper_txt = "--" if pd.isna(paper) else f"{paper:.2f}"
            lines.append(
                f"| {m} | {paper_txt} | {pct(fr['auc'])} [{pct(fr['auc_ci_lo'])}, {pct(fr['auc_ci_hi'])}] | "
                f"{pct(vm['auc'])} [{pct(vm['auc_ci_lo'])}, {pct(vm['auc_ci_hi'])}] | {pct(vm['acc'])} | "
                f"{pct(vm['f1'])} | {pct(vm['precision'])} | {pct(vm['recall'])} | {pct(vmed['auc'])} |")
        val_rows = s[s["threshold_rule"].str.startswith("val")]
        if len(val_rows):
            lines += ["", f"Video metrics at threshold chosen on {val_rows.iloc[0]['threshold_rule']}:", "",
                      "| Model | thr | Acc | F1 | Prec | Rec |", "|---|---|---|---|---|---|"]
            for _, r in val_rows.iterrows():
                lines.append(f"| {r['model']} | {r['threshold']:.4f} | {pct(r['acc'])} | {pct(r['f1'])} | "
                             f"{pct(r['precision'])} | {pct(r['recall'])} |")
        c = comp[comp["dataset"] == dn]
        if len(c):
            lines += ["", "| A vs B | level | AUC A | AUC B | Δ (pp) [95% CI] | boot p | DeLong p |",
                      "|---|---|---|---|---|---|---|"]
            for _, r in c.iterrows():
                dp = "--" if pd.isna(r["delong_p"]) else f"{r['delong_p']:.4f}"
                lines.append(f"| {r['model_a']} vs {r['model_b']} | {r['level']} | {pct(r['auc_a'])} | "
                             f"{pct(r['auc_b'])} | {r['delta_pp']:+.2f} [{r['delta_ci_lo_pp']:+.2f}, "
                             f"{r['delta_ci_hi_pp']:+.2f}] | {r['boot_p']:.4f} | {dp} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- driver

def preflight_datasets(datasets: List[Dict]) -> None:
    """Fail before loading any model if a dataset folder does not match the frame-level loader's layout."""
    from datasets.deepfake_dataset import CROSS_DATASET_LABELS, collect_class_images, find_class_dir

    problems = []
    for d in datasets:
        split_dir = Path(d["root"]) / d["split"] if d.get("split") else Path(d["root"])
        if not split_dir.is_dir():
            problems.append(f"[{d['name']}] folder does not exist: {split_dir}")
            continue
        counts = {}
        for class_name, label in (label_mapping(d) or CROSS_DATASET_LABELS).items():
            class_dir = find_class_dir(split_dir, class_name)
            if class_dir.exists():
                counts[class_dir.name] = (label, len(collect_class_images(class_dir)))
        by_label = {lab: sum(n for l, n in counts.values() if l == lab) for lab in (0, 1)}
        print(f"[preflight] {d['name']}: {split_dir} -> {counts}", flush=True)
        if by_label[0] == 0 or by_label[1] == 0:
            children = sorted(p.name + ("/" if p.is_dir() else "") for p in split_dir.iterdir())[:20]
            problems.append(
                f"[{d['name']}] need images under real/fake (or original + manipulation) folders in {split_dir}; "
                f"found real={by_label[0]}, fake={by_label[1]}. Contents: {children}")
    if problems:
        raise RuntimeError("Dataset folder check failed:\n  " + "\n  ".join(problems))


def run(manifest_path: Path, out_override: Optional[str], analyze_only: bool, overwrite: bool,
        n_boot_override: Optional[int]) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not analyze_only:
        preflight_datasets(manifest["datasets"])
    models = [m["name"] for m in manifest["models"]]
    clash = RESERVED_COLUMNS.intersection(models)
    if clash or len(set(models)) != len(models):
        raise ValueError(f"Model names must be unique and not in {sorted(RESERVED_COLUMNS)}")

    out_dir = Path(out_override or manifest.get("output_dir", "results_video_level"))
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    n_boot = int(n_boot_override or manifest.get("n_boot", 2000))
    seed = int(manifest.get("seed", 42))

    run_info = {"started": datetime.datetime.now().isoformat(timespec="seconds"),
                "python": sys.version.split()[0], "platform": platform.platform(),
                "manifest": manifest, "n_boot": n_boot, "models": {}}
    try:
        import torch
        run_info["torch"] = torch.__version__
        run_info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass

    frames: Dict[Tuple[str, str], pd.DataFrame] = {}
    for m in manifest["models"]:
        loaded = None
        imported = m.get("frames_csv") or {}
        for d in manifest["datasets"]:
            key = (m["name"], d["name"])
            cache = out_dir / "frames" / f"{m['name']}__{d['name']}.csv"
            if d["name"] in imported:
                df = read_frames_csv(Path(imported[d["name"]]))
                df.to_csv(cache, index=False)
                run_info["models"].setdefault(m["name"], {})[f"imported_{d['name']}"] = imported[d["name"]]
            elif cache.exists() and not overwrite:
                df = read_frames_csv(cache)
            else:
                if analyze_only:
                    raise FileNotFoundError(f"--analyze-only but no cached predictions: {cache}")
                if loaded is None:
                    loaded = load_model(m)
                    run_info["models"].setdefault(m["name"], {}).update(loaded[3])
                model, device, config, _ = loaded
                df = predict_frames(model, device, config, d, m,
                                    manifest.get("batch_size"), manifest.get("num_workers"))
                df.to_csv(cache, index=False)
            frames[key] = df
        if loaded is not None:
            import torch
            del loaded
            torch.cuda.empty_cache()

    train_roots = {name: (info.get("checkpoint_train_config") or {}).get("data_root")
                   for name, info in run_info["models"].items() if "checkpoint_train_config" in info}
    if len(set(train_roots.values())) > 1:
        msg = (f"[WARN] models were trained on different data_root (detector mismatch?): {train_roots}. "
               "The 5-model comparison is only valid if all were trained/tested on the same face crops.")
        run_info["warnings"] = [msg]
        print(msg)
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")
    analyze(frames, manifest, out_dir, n_boot, seed)


def selftest() -> None:
    """Synthetic data: checks parsing, DeLong vs sklearn, bootstrap and report writing (no GPU, no data)."""
    rng = np.random.default_rng(0)
    manifest = {
        "models": [{"name": "model_a", "expected_frame_auc": {"syn_test": 50.0}}, {"name": "model_b"}],
        "datasets": [{"name": "syn_val", "role": "val", "type": "ffpp", "expected_videos": 120},
                     {"name": "syn_test", "type": "ffpp", "expected_videos": 120}],
        "compare": [["model_a", "model_b"]],
    }
    frames = {}
    for d in ("syn_val", "syn_test"):
        rows = {"model_a": [], "model_b": []}
        paths, labels = [], []
        for v in range(120):
            label = int(v >= 60)
            source = "original" if label == 0 else ("Deepfakes" if v % 2 else "FaceSwap")
            latent = rng.normal(1.0 * label, 1.0)
            for k in range(int(rng.integers(5, 12))):
                paths.append(f"/data/{d}/{source}/ffpp_test_{source}_{v:03d}_sample{k:06d}_frame{k * 7:06d}.jpg")
                labels.append(label)
                rows["model_a"].append(1 / (1 + math.exp(-(latent + rng.normal(0, 0.5)))))
                rows["model_b"].append(1 / (1 + math.exp(-(0.6 * latent + rng.normal(0, 1.0)))))
        for m in rows:
            frames[(m, d)] = pd.DataFrame({"image_path": paths, "label": labels, "prob": rows[m]})

    y = rng.integers(0, 2, 300)
    s1, s2 = rng.random(300) + 0.3 * y, rng.random(300) + 0.1 * y
    a1, a2, z, p = delong_paired(y, s1, s2)
    assert abs(a1 - roc_auc_score(y, s1)) < 1e-12 and 0 <= p <= 1

    with tempfile.TemporaryDirectory() as tmp:
        analyze(frames, manifest, Path(tmp), n_boot=200, seed=0)
        summary = pd.read_csv(Path(tmp) / "summary_metrics.csv")
        assert set(summary["level"]) == {"frame", "video"}
        assert (summary.loc[summary["level"] == "video", "n_videos"] == 120).all()
        print((Path(tmp) / "paper_tables.md").read_text(encoding="utf-8"))
    print("SELFTEST OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frame- and video-level evaluation with CIs and DeLong")
    parser.add_argument("--manifest", type=str, default="video_eval_manifest.yaml")
    parser.add_argument("--out", type=str, default=None, help="Override output_dir in the manifest")
    parser.add_argument("--analyze-only", action="store_true", help="Reuse cached frame CSVs, no GPU")
    parser.add_argument("--overwrite", action="store_true", help="Re-predict even if cached CSVs exist")
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    run(Path(args.manifest), args.out, args.analyze_only, args.overwrite, args.n_boot)


if __name__ == "__main__":
    main()
