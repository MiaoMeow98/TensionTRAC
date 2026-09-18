from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.TensionDataset import TensionDataset
from src.knn_evaluation.metrics import CORE_METRICS, CSV_COLUMN_ORDER
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


def merged_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)

    dataset = dict(config.get("dataset", {}))
    evaluation = dict(config.get("evaluation", config.get("knn", {})))
    dataset.update({key: config[key] for key in DATASET_KEYS if key in config})
    evaluation.update({key: config[key] for key in EVALUATION_KEYS if key in config})

    if args.feature is not None:
        dataset["feature"] = args.feature
    if args.annotations is not None:
        dataset["annotations"] = args.annotations
    if args.root is not None:
        dataset["root"] = args.root
    if args.label_mode is not None:
        dataset["label_mode"] = args.label_mode
    if getattr(args, "include_all_clips", False):
        dataset["include_all_clips"] = True
    if getattr(args, "feature_root", None) is not None:
        dataset["feature_root"] = args.feature_root
    if args.k_values is not None:
        evaluation["k_values"] = args.k_values
    if args.n_splits is not None:
        evaluation["n_splits"] = args.n_splits
    if args.seed is not None:
        evaluation["seed"] = args.seed
    if args.group_by is not None:
        evaluation["group_by"] = args.group_by
    if args.split is not None:
        evaluation["split"] = args.split
    if args.test_fraction is not None:
        evaluation["test_fraction"] = args.test_fraction
    if args.output_csv is not None:
        evaluation["output_csv"] = args.output_csv
    if args.k_fold:
        evaluation["k_fold"] = True
    if args.no_k_fold:
        evaluation["k_fold"] = False

    if "runs" in config:
        return {"dataset": dataset, "evaluation": evaluation, "runs": config["runs"]}

    if "feature" not in dataset:
        raise ValueError(f"Set dataset.feature in config or pass --feature. Choices: {TensionDataset.feature_names()}")

    return {"dataset": dataset, "evaluation": evaluation}


def normalize_run_config(global_config: Mapping[str, Any], run_config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = dict(global_config.get("dataset", {}))
    evaluation = dict(global_config.get("evaluation", {}))

    dataset.update(run_config.get("dataset", {}))
    evaluation.update(run_config.get("evaluation", run_config.get("knn", {})))
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


def split_indices(
    y: np.ndarray,
    groups: np.ndarray,
    split: str,
    k_fold: bool,
    n_splits: int,
    test_fraction: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dummy_x = np.zeros((len(y), 1), dtype=np.float32)

    if k_fold and split == "stratified":
        min_class_count = int(np.bincount(y).min())
        n_splits = min(max(2, n_splits), min_class_count)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(dummy_x, y))

    if k_fold and split == "video":
        n_splits = min(max(2, n_splits), len(np.unique(groups)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(dummy_x, y, groups))

    if k_fold and split == "clip":
        n_splits = min(max(2, n_splits), len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(dummy_x, y, groups))

    if split == "stratified":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        return list(splitter.split(dummy_x, y))

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    return list(splitter.split(dummy_x, y, groups))


def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int,
    num_classes: int,
) -> np.ndarray:
    similarity = x_test @ x_train.T
    neighbors = np.argsort(-similarity, axis=1)[:, :k]
    predictions = []
    for row, scores in zip(neighbors, np.take_along_axis(similarity, neighbors, axis=1)):
        votes = np.zeros(num_classes, dtype=np.float32)
        for label, score in zip(y_train[row], scores):
            votes[label] += 1.0 / (1.0 - score + 1e-6)
        predictions.append(int(votes.argmax()))
    return np.asarray(predictions, dtype=np.int64)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    num_classes = int(max(y_true.max(), y_pred.max())) + 1
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    accuracy = float((y_true == y_pred).mean())

    recalls = []
    precisions = []
    f1_scores = []
    for label in range(num_classes):
        tp = matrix[label, label]
        fp = matrix[:, label].sum() - tp
        fn = matrix[label, :].sum() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1))

    out: dict[str, Any] = {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "confusion_matrix": matrix.tolist(),
    }

    if num_classes == 2:
        out.update(
            {
                "precision": precisions[1],
                "recall": recalls[1],
                "f1": f1_scores[1],
                "specificity": recalls[0],
            }
        )
    return out


def evaluate_knn(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    k_values: Sequence[int] = (1, 3, 5, 7),
    split: str = "video",
    k_fold: bool = True,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    seed: int = 0,
    merge_video_groups: Sequence[Sequence[str]] = (),
) -> list[dict[str, Any]]:
    if split == "video":
        groups = merge_groups(groups, merge_video_groups)

    folds = split_indices(
        y,
        groups,
        split=split,
        k_fold=k_fold,
        n_splits=n_splits,
        test_fraction=test_fraction,
        seed=seed,
    )

    results = []
    num_classes = int(y.max()) + 1

    for k in k_values:
        k = int(k)
        if k < 1:
            continue

        predictions = np.full(len(y), -1, dtype=np.int64)
        evaluated = np.zeros(len(y), dtype=bool)
        for train_idx, test_idx in folds:
            if len(train_idx) < k:
                continue
            overlap = np.intersect1d(groups[train_idx], groups[test_idx])
            if split in {"video", "clip"} and len(overlap):
                raise ValueError(f"Train/test group leakage: {overlap.tolist()}")

            x_train, x_test = standardize_l2(x[train_idx], x[test_idx])
            predictions[test_idx] = knn_predict(x_train, y[train_idx], x_test, k, num_classes)
            evaluated[test_idx] = True

        if not evaluated.any():
            continue

        metrics = summarize_metrics(y[evaluated], predictions[evaluated])
        metrics.update({"k": k, "evaluated_clips": int(evaluated.sum()), "folds": len(folds)})
        results.append(metrics)

    return results


def print_results(run_name: str, feature: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, results: Sequence[Mapping[str, Any]]) -> None:
    print(
        f"\n{run_name}: feature={feature} clips={len(y)} groups={len(np.unique(groups))} "
        f"dim={x.shape[1]} labels={np.bincount(y).tolist()}"
    )
    for result in results:
        metric_parts = " ".join(f"{name}={result[name]:.3f}" for name in CORE_METRICS if name in result)
        line = f"k={result['k']} {metric_parts}"
        if "f1" in result:
            line += (
                f" tension_precision={result['precision']:.3f}"
                f" tension_recall={result['recall']:.3f}"
                f" tension_f1={result['f1']:.3f}"
            )
        print(line)

    if results:
        best = max(results, key=lambda item: item.get("f1", item["macro_f1"]))
        print(f"best_k={best['k']} confusion_matrix={best['confusion_matrix']}")


def write_results(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = {key for row in rows for key in row}
    fieldnames = [name for name in CSV_COLUMN_ORDER if name in all_keys]
    fieldnames.extend(sorted(all_keys - set(fieldnames)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_once(name: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset_config = dict(config["dataset"])
    evaluation_config = dict(config.get("evaluation", {}))
    dataset = TensionDataset.from_config({"dataset": dataset_config})
    split = evaluation_config.get("split", "video")
    if "group_by" in evaluation_config:
        group_by = evaluation_config["group_by"]
    elif split == "video":
        group_by = "video"
    elif split == "stratified":
        group_by = "none"
    else:
        group_by = "clip"
    x, y, groups = load_arrays(dataset, group_by)
    merge_video_groups = evaluation_config.get("merge_video_groups", ())
    if split == "video":
        groups = merge_groups(groups, merge_video_groups)

    results = evaluate_knn(
        x,
        y,
        groups,
        k_values=evaluation_config.get("k_values", (1, 3, 5, 7)),
        split=split,
        k_fold=evaluation_config.get("k_fold", True),
        n_splits=int(evaluation_config.get("n_splits", 5)),
        test_fraction=float(evaluation_config.get("test_fraction", 0.2)),
        seed=int(evaluation_config.get("seed", 0)),
        merge_video_groups=(),
    )

    print_results(name, dataset.feature, x, y, groups, results)
    rows = []
    for result in results:
        row = dict(result)
        row.update({"run": name, "feature": dataset.feature, "split": split, "group_by": group_by})
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KNN evaluation for extracted tension-recognition features.")
    parser.add_argument("--config", type=Path, help="YAML/JSON config from the repository root.")
    parser.add_argument("--feature", choices=TensionDataset.feature_names(), help="Feature source to evaluate.")
    parser.add_argument("--root", type=Path, help="Dataset root. Defaults to current directory.")
    parser.add_argument("--annotations", type=Path, help="Annotation CSV relative to root.")
    parser.add_argument(
        "--feature-root",
        type=str,
        help="Override feature directory relative to data/ (e.g. TRAC_features/2d_sem).",
    )
    parser.add_argument(
        "--label-mode",
        choices=("binary", "multiclass", "cascade"),
        help="binary: no tension vs tension. multiclass: tension levels 1-4 only. "
        "cascade: all clips with tension_level 0-4 (used by knn_cascade.py).",
    )
    parser.add_argument(
        "--include-all-clips",
        action="store_true",
        help="Skip cross-backbone feature alignment; still load only CSV-matched "
        "classification clips (~3549). Default also requires slowfast∩timesformer∩video_swin.",
    )
    parser.add_argument("--k-values", type=int, nargs="+")
    parser.add_argument("--k-fold", action="store_true", help="Use grouped k-fold evaluation.")
    parser.add_argument("--no-k-fold", action="store_true", help="Use one grouped train/test split.")
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("video", "stratified", "clip"))
    parser.add_argument("--group-by", choices=("video", "clip", "none"))
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merged_config(args)

    if "runs" in config:
        all_rows = []
        for index, run_config in enumerate(config["runs"]):
            run_name = run_config.get("name", f"run_{index}")
            all_rows.extend(run_once(run_name, normalize_run_config(config, run_config)))
        output_csv = config.get("output_csv") or config.get("evaluation", {}).get("output_csv")
    else:
        all_rows = run_once(config["dataset"]["feature"], config)
        output_csv = config.get("evaluation", {}).get("output_csv")

    if output_csv:
        write_results(output_csv, all_rows)
        print(f"saved results: {output_csv}")


if __name__ == "__main__":
    main()
