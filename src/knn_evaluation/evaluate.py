"""Unified entry point for frozen-feature classifiers (KNN or MLP)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.TensionDataset import TensionDataset
from src.knn_evaluation.eval_common import merged_config, normalize_run_config, write_results
from src.knn_evaluation.metrics import CSV_COLUMN_ORDER, MLP_CSV_COLUMN_ORDER


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen-feature classifier (KNN or MLP) on tension recognition."
    )
    parser.add_argument(
        "--classifier",
        choices=("knn", "mlp"),
        required=True,
        help="Classifier head to use on extracted features.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--feature", choices=TensionDataset.feature_names())
    parser.add_argument("--root", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--label-mode", choices=("binary", "multiclass"))
    parser.add_argument("--include-all-clips", action="store_true")
    parser.add_argument("--feature-root", type=str, help="Override feature directory under data/ (TRAC variants).")
    parser.add_argument("--k-values", type=int, nargs="+")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = merged_config(args)

    if args.classifier == "knn":
        from src.knn_evaluation.knn import run_once

        column_order = CSV_COLUMN_ORDER
    else:
        from src.knn_evaluation.mlp import run_once

        column_order = MLP_CSV_COLUMN_ORDER

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
        write_results(output_csv, all_rows, column_order)
        print(f"saved results: {output_csv}")


if __name__ == "__main__":
    main()
