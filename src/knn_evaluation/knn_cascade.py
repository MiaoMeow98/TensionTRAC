"""Two-stage cascade KNN: binary tension detection then level grading (1-4).

Step 1 (3549 clips): no tension (0) vs tension (1).
Step 2 (train on tension-present clips only): level 1 / 2 / 3 / 4.
End-to-end prediction on all clips maps to tension_level 0-4.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.TensionDataset import EXPECTED_CLASSIFICATION_CLIPS, TensionDataset
from src.knn_evaluation.knn import (
    knn_predict,
    merge_groups,
    merged_config,
    normalize_run_config,
    split_indices,
    standardize_l2,
    summarize_metrics,
)
from src.knn_evaluation.metrics import CASCADE_CSV_COLUMN_ORDER, CASCADE_E2E_FIELDS, CORE_METRICS


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
        level = int(sample["tension_level"])
        tension_levels.append(level)

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


def cascade_predict(
    x_train: np.ndarray,
    levels_train: np.ndarray,
    x_test: np.ndarray,
    k_binary: int,
    k_level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one cascade fold; return binary, level (0-3, unused if binary=0), final (0-4)."""
    y_binary_train = (levels_train > 0).astype(np.int64)
    binary_pred = knn_predict(x_train, y_binary_train, x_test, k_binary, 2)

    tension_mask = levels_train > 0
    if int(tension_mask.sum()) < k_level:
        raise ValueError(
            f"Not enough tension-present training clips ({int(tension_mask.sum())}) for k_level={k_level}"
        )

    x_train_tension = x_train[tension_mask]
    y_level_train = levels_train[tension_mask] - 1
    level_pred = knn_predict(x_train_tension, y_level_train, x_test, k_level, 4)

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


def evaluate_cascade_knn(
    x: np.ndarray,
    levels: np.ndarray,
    groups: np.ndarray,
    *,
    k_values: Sequence[int] = (1, 3, 5, 7),
    k_binary_values: Sequence[int] | None = None,
    k_level_values: Sequence[int] | None = None,
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
        levels,
        groups,
        split=split,
        k_fold=k_fold,
        n_splits=n_splits,
        test_fraction=test_fraction,
        seed=seed,
    )

    k_binary_values = list(k_binary_values or k_values)
    k_level_values = list(k_level_values or k_values)
    if len(k_binary_values) != len(k_level_values):
        raise ValueError("k_binary_values and k_level_values must have the same length")

    results: list[dict[str, Any]] = []
    for k_binary, k_level in zip(k_binary_values, k_level_values):
        k_binary = int(k_binary)
        k_level = int(k_level)
        if k_binary < 1 or k_level < 1:
            continue

        binary_pred = np.full(len(levels), -1, dtype=np.int64)
        level_pred = np.full(len(levels), -1, dtype=np.int64)
        final_pred = np.full(len(levels), -1, dtype=np.int64)
        evaluated = np.zeros(len(levels), dtype=bool)

        for train_idx, test_idx in folds:
            if len(train_idx) < max(k_binary, k_level):
                continue
            overlap = np.intersect1d(groups[train_idx], groups[test_idx])
            if split in {"video", "clip"} and len(overlap):
                raise ValueError(f"Train/test group leakage: {overlap.tolist()}")

            x_train, x_test = standardize_l2(x[train_idx], x[test_idx])
            fold_binary, fold_level, fold_final = cascade_predict(
                x_train,
                levels[train_idx],
                x_test,
                k_binary,
                k_level,
            )
            binary_pred[test_idx] = fold_binary
            level_pred[test_idx] = fold_level
            final_pred[test_idx] = fold_final
            evaluated[test_idx] = True

        if not evaluated.any():
            continue

        y_true = levels[evaluated]
        y_binary_true = (y_true > 0).astype(np.int64)
        y_binary_pred = binary_pred[evaluated]
        y_final = final_pred[evaluated]
        y_level_pred = level_pred[evaluated]

        step1 = summarize_metrics(y_binary_true, y_binary_pred)
        end_to_end = summarize_metrics(y_true, y_final)
        step2 = step2_metrics_on_true_tension(y_true, y_level_pred)

        row: dict[str, Any] = {
            "k": k_binary,
            "k_binary": k_binary,
            "k_level": k_level,
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
        results.append(row)

    return results


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
    from src.knn_evaluation.metrics import CASCADE_STEP1_FIELDS, CASCADE_STEP2_FIELDS

    level_counts = np.bincount(levels, minlength=5).tolist()
    print(
        f"\n{run_name}: feature={feature} clips={len(levels)} groups={len(np.unique(groups))} "
        f"tension_levels={level_counts}"
    )
    for result in results:
        print(
            f"k=({result['k_binary']},{result['k_level']}) "
            f"step1 {{{_format_core_metrics(result, CASCADE_STEP1_FIELDS)}}} "
            f"step2 {{{_format_core_metrics(result, CASCADE_STEP2_FIELDS)}}} "
            f"e2e {{{_format_core_metrics(result, CASCADE_E2E_FIELDS)}}}"
        )
    if results:
        best = max(results, key=lambda item: item["cascade_macro_f1"])
        print(
            f"best_k=({best['k_binary']},{best['k_level']}) "
            f"cascade_confusion_matrix={best['cascade_confusion_matrix']}"
        )


def write_cascade_results(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = {key for row in rows for key in row}
    fieldnames = [name for name in CASCADE_CSV_COLUMN_ORDER if name in all_keys]
    fieldnames.extend(sorted(all_keys - set(fieldnames)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

    results = evaluate_cascade_knn(
        x,
        levels,
        groups,
        k_values=evaluation_config.get("k_values", (1, 3, 5, 7)),
        k_binary_values=evaluation_config.get("k_binary_values"),
        k_level_values=evaluation_config.get("k_level_values"),
        split=split,
        k_fold=evaluation_config.get("k_fold", True),
        n_splits=int(evaluation_config.get("n_splits", 5)),
        test_fraction=float(evaluation_config.get("test_fraction", 0.2)),
        seed=int(evaluation_config.get("seed", 0)),
        merge_video_groups=merge_video_groups if split == "video" else (),
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
        description="Two-stage cascade KNN: binary tension detection + level grading."
    )
    parser.add_argument("--config", type=Path, help="YAML/JSON config from the repository root.")
    parser.add_argument("--feature", choices=TensionDataset.feature_names())
    parser.add_argument("--root", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--label-mode", choices=("binary", "multiclass", "cascade"))
    parser.add_argument("--include-all-clips", action="store_true")
    parser.add_argument("--k-values", type=int, nargs="+")
    parser.add_argument("--k-binary-values", type=int, nargs="+")
    parser.add_argument("--k-level-values", type=int, nargs="+")
    parser.add_argument("--k-fold", action="store_true")
    parser.add_argument("--no-k-fold", action="store_true")
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

    if args.k_binary_values is not None:
        config.setdefault("evaluation", {})["k_binary_values"] = args.k_binary_values
    if args.k_level_values is not None:
        config.setdefault("evaluation", {})["k_level_values"] = args.k_level_values

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
        write_cascade_results(output_csv, all_rows)
        print(f"saved results: {output_csv}")


if __name__ == "__main__":
    main()
