"""Two-stage cascade MLP: binary tension detection then level grading (1-4).

Step 1 (3549 clips): no tension (0) vs tension (1).
Step 2 (train on tension-present clips only): level 1 / 2 / 3 / 4.
End-to-end prediction on all clips maps to tension_level 0-4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.TensionDataset import EXPECTED_CLASSIFICATION_CLIPS, TensionDataset
from src.knn_evaluation.eval_common import (
    merge_groups,
    merged_config,
    normalize_run_config,
    standardize_zscore,
    stratified_train_val_split,
    write_results,
)
from src.knn_evaluation.knn import split_indices, summarize_metrics
from src.knn_evaluation.metrics import (
    CASCADE_E2E_FIELDS,
    CASCADE_STEP1_FIELDS,
    CASCADE_STEP2_FIELDS,
    CORE_METRICS,
    MLP_CASCADE_CSV_COLUMN_ORDER,
)
from src.knn_evaluation.mlp import predict_mlp, train_mlp


def load_cascade_arrays(
    dataset: TensionDataset,
    group_by: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return features, tension levels (0-4), binary labels, and group ids."""
    features: list[np.ndarray] = []
    tension_levels: list[int] = []
    groups: list[str] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        features.append(sample["feature"].detach().cpu().numpy().astype("float32", copy=False))
        tension_levels.append(int(sample["tension_level"]))

        if group_by == "video":
            groups.append(sample["video"])
        elif group_by == "clip":
            groups.append(sample["clip_name"])
        elif group_by == "none":
            groups.append(str(index))
        else:
            raise ValueError("group_by must be 'video', 'clip', or 'none'")

    x = np.nan_to_num(np.stack(features).astype("float32"))
    levels = np.asarray(tension_levels, dtype=np.int64)
    y_binary = (levels > 0).astype(np.int64)
    group_array = np.asarray(groups, dtype=object)
    return x, levels, y_binary, group_array


def _train_val_indices(
    y: np.ndarray,
    train_idx: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    fit_idx, val_idx = stratified_train_val_split(y, train_idx, val_fraction, seed)
    if len(val_idx) == 0:
        val_size = max(1, int(round(len(train_idx) * val_fraction)))
        val_idx = train_idx[-val_size:]
        fit_idx = train_idx[:-val_size]
    if len(fit_idx) == 0:
        fit_idx, val_idx = train_idx, train_idx
    return fit_idx, val_idx


def cascade_predict_mlp(
    x_train: np.ndarray,
    levels_train: np.ndarray,
    x_test: np.ndarray,
    *,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one cascade fold; return binary, level (0-3), final (0-4)."""
    train_rel = np.arange(len(x_train), dtype=np.int64)
    y_binary_train = (levels_train > 0).astype(np.int64)

    fit_idx, val_idx = _train_val_indices(y_binary_train, train_rel, val_fraction, seed)
    binary_model = train_mlp(
        x_train[fit_idx],
        y_binary_train[fit_idx],
        x_train[val_idx],
        y_binary_train[val_idx],
        num_classes=2,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        device=device,
    )
    binary_pred = predict_mlp(binary_model, x_test, device, batch_size=batch_size)

    tension_mask = levels_train > 0
    n_tension = int(tension_mask.sum())
    if n_tension < 8:
        raise ValueError(f"Not enough tension-present training clips ({n_tension}) for level MLP")

    x_train_tension = x_train[tension_mask]
    y_level_train = levels_train[tension_mask] - 1
    tension_rel = np.arange(len(x_train_tension), dtype=np.int64)
    fit_idx, val_idx = _train_val_indices(y_level_train, tension_rel, val_fraction, seed + 1)

    level_model = train_mlp(
        x_train_tension[fit_idx],
        y_level_train[fit_idx],
        x_train_tension[val_idx],
        y_level_train[val_idx],
        num_classes=4,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed + 1,
        device=device,
    )
    level_pred = predict_mlp(level_model, x_test, device, batch_size=batch_size)

    final_pred = np.zeros(len(x_test), dtype=np.int64)
    for index, is_tension in enumerate(binary_pred):
        final_pred[index] = 0 if is_tension == 0 else int(level_pred[index]) + 1
    return binary_pred, level_pred, final_pred


def step2_metrics_on_true_tension(
    levels_true: np.ndarray,
    level_pred: np.ndarray,
) -> dict[str, Any]:
    """Level-only metrics on clips with ground-truth tension (levels 1-4)."""
    mask = levels_true > 0
    if not mask.any():
        return {"step2_macro_f1": 0.0, "step2_macro_precision": 0.0, "step2_macro_recall": 0.0}

    y_true = levels_true[mask] - 1
    y_pred = level_pred[mask]
    metrics = summarize_metrics(y_true, y_pred)
    return {
        "step2_accuracy": metrics["accuracy"],
        "step2_balanced_accuracy": metrics["balanced_accuracy"],
        "step2_macro_precision": metrics["macro_precision"],
        "step2_macro_recall": metrics["macro_recall"],
        "step2_macro_f1": metrics["macro_f1"],
        "step2_confusion_matrix": metrics["confusion_matrix"],
        "step2_evaluated_clips": int(mask.sum()),
    }


def evaluate_cascade_mlp(
    x: np.ndarray,
    levels: np.ndarray,
    groups: np.ndarray,
    *,
    split: str = "video",
    k_fold: bool = True,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    seed: int = 0,
    merge_video_groups: Sequence[Sequence[str]] = (),
    hidden_dim: int = 256,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 15,
    val_fraction: float = 0.15,
) -> list[dict[str, Any]]:
    if split == "video":
        groups = merge_groups(groups, merge_video_groups)

    folds = split_indices(
        levels,
        groups,
        split=split,
        k_fold=k_fold,
        n_splits=n_splits,
        test_fraction=test_fraction,
        seed=seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    binary_pred = np.full(len(levels), -1, dtype=np.int64)
    level_pred = np.full(len(levels), -1, dtype=np.int64)
    final_pred = np.full(len(levels), -1, dtype=np.int64)
    evaluated = np.zeros(len(levels), dtype=bool)

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        if len(train_idx) < 16:
            continue
        overlap = np.intersect1d(groups[train_idx], groups[test_idx])
        if split in {"video", "clip"} and len(overlap):
            raise ValueError(f"Train/test group leakage: {overlap.tolist()}")

        x_train_raw, x_test_raw = x[train_idx], x[test_idx]
        x_train, x_test = standardize_zscore(x_train_raw, x_test_raw)
        fold_binary, fold_level, fold_final = cascade_predict_mlp(
            x_train,
            levels[train_idx],
            x_test,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            val_fraction=val_fraction,
            seed=seed + fold_id,
            device=device,
        )
        binary_pred[test_idx] = fold_binary
        level_pred[test_idx] = fold_level
        final_pred[test_idx] = fold_final
        evaluated[test_idx] = True

    if not evaluated.any():
        return []

    y_true = levels[evaluated]
    y_binary_true = (y_true > 0).astype(np.int64)
    y_binary_pred = binary_pred[evaluated]
    y_final = final_pred[evaluated]
    y_level_pred = level_pred[evaluated]

    step1 = summarize_metrics(y_binary_true, y_binary_pred)
    end_to_end = summarize_metrics(y_true, y_final)
    step2 = step2_metrics_on_true_tension(y_true, y_level_pred)

    row: dict[str, Any] = {
        "seed": seed,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "val_fraction": val_fraction,
        "classifier": "mlp",
        "evaluated_clips": int(evaluated.sum()),
        "folds": len(folds),
        "step1_accuracy": step1["accuracy"],
        "step1_balanced_accuracy": step1["balanced_accuracy"],
        "step1_macro_precision": step1["macro_precision"],
        "step1_macro_recall": step1["macro_recall"],
        "step1_macro_f1": step1["macro_f1"],
        "step1_precision": step1["precision"],
        "step1_recall": step1["recall"],
        "step1_f1": step1["f1"],
        "step1_confusion_matrix": step1["confusion_matrix"],
        "cascade_accuracy": end_to_end["accuracy"],
        "cascade_balanced_accuracy": end_to_end["balanced_accuracy"],
        "cascade_macro_precision": end_to_end["macro_precision"],
        "cascade_macro_recall": end_to_end["macro_recall"],
        "cascade_macro_f1": end_to_end["macro_f1"],
        "cascade_confusion_matrix": end_to_end["confusion_matrix"],
    }
    row.update(step2)
    return [row]


def _format_core_metrics(result: Mapping[str, Any], field_map: Mapping[str, str]) -> str:
    return " ".join(
        f"{name}={float(result[field_map[name]]):.3f}"
        for name in CORE_METRICS
        if field_map[name] in result
    )


def print_cascade_results(
    run_name: str,
    feature: str,
    levels: np.ndarray,
    groups: np.ndarray,
    results: Sequence[Mapping[str, Any]],
) -> None:
    level_counts = np.bincount(levels, minlength=5).tolist()
    print(
        f"\n{run_name}: classifier=mlp feature={feature} clips={len(levels)} "
        f"groups={len(np.unique(groups))} tension_levels={level_counts}"
    )
    for result in results:
        print(
            f"seed={result['seed']} hidden={result['hidden_dim']} "
            f"step1 {{{_format_core_metrics(result, CASCADE_STEP1_FIELDS)}}} "
            f"step2 {{{_format_core_metrics(result, CASCADE_STEP2_FIELDS)}}} "
            f"e2e {{{_format_core_metrics(result, CASCADE_E2E_FIELDS)}}}"
        )
    if results:
        best = results[0]
        print(f"cascade_confusion_matrix={best['cascade_confusion_matrix']}")


def run_once(name: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset_config = dict(config["dataset"])
    evaluation_config = dict(config.get("evaluation", {}))
    dataset_config["label_mode"] = "cascade"
    dataset = TensionDataset.from_config({"dataset": dataset_config})

    if len(dataset) != EXPECTED_CLASSIFICATION_CLIPS:
        raise ValueError(
            f"cascade expects {EXPECTED_CLASSIFICATION_CLIPS} clips, got {len(dataset)}"
        )

    split = evaluation_config.get("split", "video")
    if "group_by" in evaluation_config:
        group_by = evaluation_config["group_by"]
    elif split == "video":
        group_by = "video"
    elif split == "stratified":
        group_by = "none"
    else:
        group_by = "clip"

    x, levels, _y_binary, groups = load_cascade_arrays(dataset, group_by)
    merge_video_groups = evaluation_config.get("merge_video_groups", ())

    results = evaluate_cascade_mlp(
        x,
        levels,
        groups,
        split=split,
        k_fold=evaluation_config.get("k_fold", True),
        n_splits=int(evaluation_config.get("n_splits", 5)),
        test_fraction=float(evaluation_config.get("test_fraction", 0.2)),
        seed=int(evaluation_config.get("seed", 0)),
        merge_video_groups=merge_video_groups if split == "video" else (),
        hidden_dim=int(evaluation_config.get("hidden_dim", 256)),
        dropout=float(evaluation_config.get("dropout", 0.3)),
        lr=float(evaluation_config.get("lr", 1e-3)),
        weight_decay=float(evaluation_config.get("weight_decay", 1e-4)),
        batch_size=int(evaluation_config.get("batch_size", 64)),
        max_epochs=int(evaluation_config.get("max_epochs", 100)),
        patience=int(evaluation_config.get("patience", 15)),
        val_fraction=float(evaluation_config.get("val_fraction", 0.15)),
    )

    print_cascade_results(name, dataset.feature, levels, groups, results)
    rows = []
    for result in results:
        row = dict(result)
        row.update({"run": name, "feature": dataset.feature, "split": split, "group_by": group_by})
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage cascade MLP: binary tension detection + level grading."
    )
    parser.add_argument("--config", type=Path, help="YAML/JSON config from the repository root.")
    parser.add_argument("--feature", choices=TensionDataset.feature_names())
    parser.add_argument("--root", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--k-fold", action="store_true")
    parser.add_argument("--no-k-fold", action="store_true")
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("video", "stratified", "clip"))
    parser.add_argument("--group-by", choices=("video", "clip", "none"))
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--val-fraction", type=float)
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
        write_results(output_csv, all_rows, MLP_CASCADE_CSV_COLUMN_ORDER)
        print(f"saved results: {output_csv}")


if __name__ == "__main__":
    main()
