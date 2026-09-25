"""Print comparison tables for core classifier metrics (KNN / MLP)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.knn_evaluation.metrics import (
    CASCADE_E2E_FIELDS,
    CASCADE_STEP1_FIELDS,
    CASCADE_STEP2_FIELDS,
    CORE_METRIC_LABELS,
    CORE_METRICS,
)

FEATURE_ORDER = ("slowfast", "timesformer", "video_swin", "tension_trac")
FEATURE_LABELS = {
    "slowfast": "SlowFast",
    "timesformer": "TimeSformer",
    "video_swin": "Video Swin-3D",
    "tension_trac": "TensionTRAC",
}
SPLIT_ORDER = ("stratified", "video")

TASK_FILES = {
    "knn": {
        "binary": ("knn_stratified_split.csv", "knn_video_split.csv"),
        "multiclass": ("knn_stratified_split_multiclass.csv", "knn_video_split_multiclass.csv"),
        "cascade": ("knn_cascade_stratified_split.csv", "knn_cascade_video_split.csv"),
        "cascade_step1": ("knn_cascade_stratified_split.csv", "knn_cascade_video_split.csv"),
        "cascade_step2": ("knn_cascade_stratified_split.csv", "knn_cascade_video_split.csv"),
    },
    "mlp": {
        "binary": ("mlp_stratified_split.csv", "mlp_video_split.csv"),
        "multiclass": ("mlp_stratified_split_multiclass.csv", "mlp_video_split_multiclass.csv"),
        "cascade": ("mlp_cascade_stratified_split.csv", "mlp_cascade_video_split.csv"),
        "cascade_step1": ("mlp_cascade_stratified_split.csv", "mlp_cascade_video_split.csv"),
        "cascade_step2": ("mlp_cascade_stratified_split.csv", "mlp_cascade_video_split.csv"),
    },
}

TASK_FIELDS = {
    "binary": {name: name for name in CORE_METRICS},
    "multiclass": {name: name for name in CORE_METRICS},
    "cascade": CASCADE_E2E_FIELDS,
    "cascade_step1": CASCADE_STEP1_FIELDS,
    "cascade_step2": CASCADE_STEP2_FIELDS,
}

TASK_TITLES = {
    "binary": "Binary classification",
    "multiclass": "Tension level classification (levels 1-4)",
    "cascade": "Cascade end-to-end (levels 0-4, 3549 clips)",
    "cascade_step1": "Cascade step 1: binary tension detection",
    "cascade_step2": "Cascade step 2: tension level grading (GT tension clips)",
}


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pick_row(
    rows: list[dict[str, str]],
    feature: str,
    split: str,
    classifier: str,
    k: int | None,
    seed: int | None,
) -> dict[str, str] | None:
    matches = [row for row in rows if row["feature"] == feature and row["split"] == split]
    if classifier:
        matches = [row for row in matches if row.get("classifier", "knn") == classifier]
    if k is not None and matches and "k" in matches[0]:
        matches = [row for row in matches if int(row["k"]) == k]
    if seed is not None and matches and "seed" in matches[0]:
        matches = [row for row in matches if int(float(row["seed"])) == seed]
    return matches[0] if matches else None


def format_percent(value: str) -> str:
    return f"{float(value) * 100.0:.1f}"


def render_table(
    title: str,
    stratified_rows: list[dict[str, str]],
    video_rows: list[dict[str, str]],
    classifier: str,
    field_map: Mapping[str, str],
    k: int | None,
    seed: int | None,
) -> str:
    metric_width = 12
    split_width = metric_width * len(CORE_METRICS)
    setting = f"k={k}" if k is not None else f"seed={seed}"
    lines = [
        f"{title} ({classifier.upper()}, {setting})",
        f"{'Method':<16}{'Stratified 5-fold':>{split_width}}{'Video 4-fold':>{split_width}}",
        f"{'':16}"
        + "".join(f"{CORE_METRIC_LABELS[m]:>{metric_width}}" for m in CORE_METRICS)
        + "".join(f"{CORE_METRIC_LABELS[m]:>{metric_width}}" for m in CORE_METRICS),
        "-" * (16 + split_width * 2),
    ]

    for feature in FEATURE_ORDER:
        line = f"{FEATURE_LABELS[feature]:<16}"
        for split in SPLIT_ORDER:
            rows = stratified_rows if split == "stratified" else video_rows
            row = pick_row(rows, feature, split, classifier, k, seed)
            if row is None:
                line += f"{'--':>{split_width}}"
                continue
            line += "".join(
                f"{format_percent(row[field_map[metric]]):>{metric_width}}" for metric in CORE_METRICS
            )
        lines.append(line)
    return "\n".join(lines)


TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "cascade_all": ("cascade", "cascade_step1", "cascade_step2"),
    "all": ("binary", "multiclass", "cascade", "cascade_step1", "cascade_step2"),
}


def resolve_tasks(task: str) -> tuple[str, ...]:
    if task in TASK_GROUPS:
        return TASK_GROUPS[task]
    return (task,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print core metric comparison tables.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--classifier", choices=("knn", "mlp"), default="knn")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task",
        choices=(
            "all",
            "binary",
            "multiclass",
            "cascade",
            "cascade_step1",
            "cascade_step2",
            "cascade_all",
        ),
        default="all",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    k = args.k if args.classifier == "knn" else None
    seed = None if args.classifier == "knn" else args.seed
    selected = resolve_tasks(args.task)

    sections: list[str] = []
    for task in selected:
        if task not in TASK_FILES[args.classifier]:
            sections.append(f"{task}: not available for classifier={args.classifier}")
            continue
        stratified_name, video_name = TASK_FILES[args.classifier][task]
        stratified_path = args.results_dir / stratified_name
        video_path = args.results_dir / video_name
        if not stratified_path.exists() or not video_path.exists():
            sections.append(f"{task}: missing {stratified_name} or {video_name}")
            continue
        sections.append(
            render_table(
                TASK_TITLES[task],
                load_rows(stratified_path),
                load_rows(video_path),
                args.classifier,
                TASK_FIELDS[task],
                k,
                seed,
            )
        )

    text = "\n\n".join(sections)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
