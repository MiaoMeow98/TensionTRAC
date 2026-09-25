"""Plot KNN core metrics vs. k for all feature-extractor baselines."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

FEATURE_LABELS = {
    "slowfast": "SlowFast",
    "timesformer": "TimeSformer",
    "video_swin": "Video Swin-3D",
    "tension_trac": "TensionTRAC",
}
FEATURE_ORDER = ("slowfast", "timesformer", "video_swin", "tension_trac")
METHOD_COLORS = {
    "slowfast": "#1f77b4",
    "timesformer": "#ff7f0e",
    "video_swin": "black",
    "tension_trac": "#2ca02c",
}
SPLIT_STYLES = {
    "stratified": "-",
    "video": "--",
}
SPLIT_LABELS = {
    "stratified": "Stratified 5-fold",
    "video": "Video 4-fold",
}

TASK_CONFIG = {
    "binary": {
        "stratified_csv": "knn_stratified_split.csv",
        "video_csv": "knn_video_split.csv",
        "title": "Binary KNN: core metrics vs. $k$",
        "output": "knn_k_sensitivity_binary.pdf",
        "fields": {name: name for name in CORE_METRICS},
    },
    "multiclass": {
        "stratified_csv": "knn_stratified_split_multiclass.csv",
        "video_csv": "knn_video_split_multiclass.csv",
        "title": "Tension level KNN (2675 clips, levels 1--4): core metrics vs. $k$",
        "output": "knn_k_sensitivity_multiclass.pdf",
        "fields": {name: name for name in CORE_METRICS},
    },
    "cascade": {
        "stratified_csv": "knn_cascade_stratified_split.csv",
        "video_csv": "knn_cascade_video_split.csv",
        "title": "Cascade end-to-end KNN (3549 clips): core metrics vs. $k$",
        "output": "knn_k_sensitivity_cascade.pdf",
        "fields": CASCADE_E2E_FIELDS,
    },
    "cascade_step1": {
        "stratified_csv": "knn_cascade_stratified_split.csv",
        "video_csv": "knn_cascade_video_split.csv",
        "title": "Cascade step 1 (binary): core metrics vs. $k$",
        "output": "knn_k_sensitivity_cascade_step1.pdf",
        "fields": CASCADE_STEP1_FIELDS,
    },
    "cascade_step2": {
        "stratified_csv": "knn_cascade_stratified_split.csv",
        "video_csv": "knn_cascade_video_split.csv",
        "title": "Cascade step 2 (level 1--4): core metrics vs. $k$",
        "output": "knn_k_sensitivity_cascade_step2.pdf",
        "fields": CASCADE_STEP2_FIELDS,
    },
}


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def series_by_k(
    rows: list[dict[str, str]],
    feature: str,
    split: str,
    csv_field: str,
) -> tuple[list[int], list[float]]:
    subset = [row for row in rows if row["feature"] == feature and row["split"] == split]
    subset.sort(key=lambda row: int(row["k"]))
    ks = [int(row["k"]) for row in subset]
    values = [float(row[csv_field]) * 100.0 for row in subset]
    return ks, values


def plot_metric_panel(
    ax: plt.Axes,
    stratified_rows: list[dict[str, str]],
    video_rows: list[dict[str, str]],
    csv_field: str,
    metric_label: str,
    highlight_k: int | None,
) -> None:
    all_rows = stratified_rows + video_rows
    for feature in FEATURE_ORDER:
        for split, style in SPLIT_STYLES.items():
            rows = stratified_rows if split == "stratified" else video_rows
            ks, values = series_by_k(rows, feature, split, csv_field)
            ax.plot(
                ks,
                values,
                linestyle=style,
                marker="o",
                linewidth=1.8,
                markersize=5.5,
                color=METHOD_COLORS[feature],
            )

    if highlight_k is not None:
        ax.axvline(highlight_k, color="#888888", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.set_xticks([1, 3, 5, 7])
    ax.set_title(metric_label, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)

    ymin = min(float(row[csv_field]) for row in all_rows) * 100.0
    ymax = max(float(row[csv_field]) for row in all_rows) * 100.0
    margin = max(2.0, (ymax - ymin) * 0.12)
    ax.set_ylim(max(0.0, ymin - margin), min(100.0, ymax + margin))


def plot_task_metrics(
    stratified_csv: Path,
    video_csv: Path,
    title: str,
    output_path: Path,
    field_map: Mapping[str, str],
    highlight_k: int | None = 3,
) -> None:
    stratified_rows = load_rows(stratified_csv)
    video_rows = load_rows(video_csv)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), sharex=True)
    axes_list = axes.flatten()
    for index, metric in enumerate(CORE_METRICS):
        plot_metric_panel(
            axes_list[index],
            stratified_rows,
            video_rows,
            field_map[metric],
            CORE_METRIC_LABELS[metric],
            highlight_k,
        )

    axes_list[-1].axis("off")
    method_handles = [
        mlines.Line2D([], [], color=METHOD_COLORS[f], linewidth=1.8, marker="o", label=FEATURE_LABELS[f])
        for f in FEATURE_ORDER
    ]
    split_handles = [
        mlines.Line2D([], [], color="black", linestyle=SPLIT_STYLES[s], linewidth=1.8, label=SPLIT_LABELS[s])
        for s in ("stratified", "video")
    ]
    axes_list[-1].legend(handles=method_handles + split_handles, loc="center", fontsize=9, frameon=False)

    for ax in axes_list[:3]:
        ax.set_xlabel("Number of neighbors ($k$)")
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KNN core metrics vs. k.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    parser.add_argument(
        "--task",
        choices=("all", "binary", "multiclass", "cascade", "cascade_step1", "cascade_step2"),
        default="all",
    )
    args = parser.parse_args()

    if args.task == "all":
        selected = ("binary", "multiclass", "cascade", "cascade_step1", "cascade_step2")
    else:
        selected = (args.task,)

    for name in selected:
        config = TASK_CONFIG[name]
        stratified_csv = args.results_dir / config["stratified_csv"]
        video_csv = args.results_dir / config["video_csv"]
        if not stratified_csv.exists() or not video_csv.exists():
            print(f"Skipping {name}: missing {stratified_csv.name} or {video_csv.name}")
            continue
        plot_task_metrics(
            stratified_csv,
            video_csv,
            config["title"],
            args.output_dir / config["output"],
            config["fields"],
        )

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
