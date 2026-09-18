"""Shared helpers for frozen-feature classifier evaluation (KNN / MLP)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.dataset.TensionDataset import TensionDataset

DATASET_KEYS = {
    "feature",
    "root",
    "annotations",
    "label_mode",
    "flatten",
    "require_feature",
    "include_all_clips",
    "feature_root",
}
EVALUATION_KEYS = {
    "k_values",
    "k_fold",
    "n_splits",
    "test_fraction",
    "seed",
    "split",
    "group_by",
    "output_csv",
    "merge_video_groups",
    "classifier",
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "patience",
    "val_fraction",
}


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    path = Path(path)
    with path.open() as handle:
        if path.suffix.lower() == ".json":
            return json.load(handle)

        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install pyyaml or use a JSON config file.") from exc
        value = yaml.safe_load(handle)
        return value or {}


def merged_config(args) -> dict[str, Any]:
    config = load_config(args.config)

    dataset = dict(config.get("dataset", {}))
    evaluation = dict(config.get("evaluation", config.get("knn", config.get("mlp", {}))))
    mlp_block = dict(config.get("mlp", {}))
    evaluation.update(mlp_block)
    dataset.update({key: config[key] for key in DATASET_KEYS if key in config})
    evaluation.update({key: config[key] for key in EVALUATION_KEYS if key in config})

    if getattr(args, "feature", None) is not None:
        dataset["feature"] = args.feature
    if getattr(args, "annotations", None) is not None:
        dataset["annotations"] = args.annotations
    if getattr(args, "root", None) is not None:
        dataset["root"] = args.root
    if getattr(args, "label_mode", None) is not None:
        dataset["label_mode"] = args.label_mode
    if getattr(args, "include_all_clips", False):
        dataset["include_all_clips"] = True
    if getattr(args, "feature_root", None) is not None:
        dataset["feature_root"] = args.feature_root
    if getattr(args, "classifier", None) is not None:
        evaluation["classifier"] = args.classifier
    if getattr(args, "k_values", None) is not None:
        evaluation["k_values"] = args.k_values
    if getattr(args, "n_splits", None) is not None:
        evaluation["n_splits"] = args.n_splits
    if getattr(args, "seed", None) is not None:
        evaluation["seed"] = args.seed
    if getattr(args, "group_by", None) is not None:
        evaluation["group_by"] = args.group_by
    if getattr(args, "split", None) is not None:
        evaluation["split"] = args.split
    if getattr(args, "test_fraction", None) is not None:
        evaluation["test_fraction"] = args.test_fraction
    if getattr(args, "output_csv", None) is not None:
        evaluation["output_csv"] = args.output_csv
    if getattr(args, "k_fold", False):
        evaluation["k_fold"] = True
    if getattr(args, "no_k_fold", False):
        evaluation["k_fold"] = False

    for name in ("hidden_dim", "dropout", "lr", "weight_decay", "batch_size", "max_epochs", "patience", "val_fraction"):
        value = getattr(args, name, None)
        if value is not None:
            evaluation[name] = value

    if "runs" in config:
        return {"dataset": dataset, "evaluation": evaluation, "runs": config["runs"]}

    if "feature" not in dataset:
        raise ValueError(f"Set dataset.feature in config or pass --feature. Choices: {TensionDataset.feature_names()}")

    return {"dataset": dataset, "evaluation": evaluation}


def normalize_run_config(global_config: Mapping[str, Any], run_config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = dict(global_config.get("dataset", {}))
    evaluation = dict(global_config.get("evaluation", {}))

    dataset.update(run_config.get("dataset", {}))
    evaluation.update(run_config.get("evaluation", run_config.get("knn", run_config.get("mlp", {}))))
    dataset.update({key: run_config[key] for key in DATASET_KEYS if key in run_config})
    evaluation.update({key: run_config[key] for key in EVALUATION_KEYS if key in run_config})

    if "feature" not in dataset:
        raise ValueError(f"Each run needs a feature. Choices: {TensionDataset.feature_names()}")
    return {"dataset": dataset, "evaluation": evaluation}


def load_arrays(dataset: TensionDataset, group_by: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, groups = [], [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        features.append(sample["feature"].detach().cpu().numpy().astype("float32", copy=False))
        labels.append(int(sample["label"]))

        if group_by == "video":
            groups.append(sample["video"])
        elif group_by == "clip":
            groups.append(sample["clip_name"])
        elif group_by == "none":
            groups.append(str(index))
        else:
            raise ValueError("group_by must be 'video', 'clip', or 'none'")

    x = np.nan_to_num(np.stack(features).astype("float32"))
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype=object)
    return x, y, group_array


def merge_groups(groups: np.ndarray, group_merges: Sequence[Sequence[str]]) -> np.ndarray:
    if not group_merges:
        return groups

    merged = groups.astype(object, copy=True)
    for group_merge in group_merges:
        if not group_merge:
            continue
        canonical = "+".join(group_merge)
        merged[np.isin(merged, list(group_merge))] = canonical
    return merged


def standardize_l2(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_train = x_train / (np.linalg.norm(x_train, axis=1, keepdims=True) + 1e-6)
    x_test = x_test / (np.linalg.norm(x_test, axis=1, keepdims=True) + 1e-6)
    return x_train, x_test


def standardize_zscore(
    x_train: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_eval - mean) / std


def stratified_train_val_split(
    y: np.ndarray,
    indices: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) < 4 or val_fraction <= 0:
        return indices, indices[:0]

    y_sub = y[indices]
    min_class = int(np.bincount(y_sub).min()) if len(y_sub) else 0
    if min_class < 2 and val_fraction > 0:
        val_size = max(1, int(round(len(indices) * val_fraction)))
        val_idx = indices[-val_size:]
        train_idx = indices[:-val_size]
        return train_idx, val_idx

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_rel, val_rel = next(splitter.split(np.zeros((len(indices), 1)), y_sub))
    return indices[train_rel], indices[val_rel]


def write_results(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    column_order: Sequence[str],
) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = {key for row in rows for key in row}
    fieldnames = [name for name in column_order if name in all_keys]
    fieldnames.extend(sorted(all_keys - set(fieldnames)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
