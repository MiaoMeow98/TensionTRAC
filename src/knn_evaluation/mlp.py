"""MLP classifier evaluation on frozen tension-recognition features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.TensionDataset import TensionDataset
from src.knn_evaluation.eval_common import (
    load_arrays,
    merge_groups,
    merged_config,
    normalize_run_config,
    standardize_zscore,
    stratified_train_val_split,
    write_results,
)
from src.knn_evaluation.knn import split_indices, summarize_metrics
from src.knn_evaluation.metrics import CORE_METRICS, MLP_CSV_COLUMN_ORDER


class FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = len(y) / (num_classes * np.maximum(counts, 1.0))
    return torch.from_numpy(weights)


def macro_f1_from_logits(logits: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1).cpu().numpy()
    y_np = y_true.cpu().numpy()
    return float(summarize_metrics(y_np, preds)["macro_f1"])


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    num_classes: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> FeatureMLP:
    torch.manual_seed(seed)
    model = FeatureMLP(x_train.shape[1], hidden_dim, num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, num_classes).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=min(batch_size, len(y_train)),
        shuffle=True,
    )

    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    best_state = None
    best_score = -1.0
    stale = 0

    for _epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            score = macro_f1_from_logits(val_logits, y_val_t, num_classes)

        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_mlp(model: FeatureMLP, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            preds.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds).astype(np.int64)


def evaluate_mlp(
    x: np.ndarray,
    y: np.ndarray,
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
        y,
        groups,
        split=split,
        k_fold=k_fold,
        n_splits=n_splits,
        test_fraction=test_fraction,
        seed=seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(y.max()) + 1
    predictions = np.full(len(y), -1, dtype=np.int64)
    evaluated = np.zeros(len(y), dtype=bool)

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        if len(train_idx) < max(8, num_classes * 2):
            continue
        overlap = np.intersect1d(groups[train_idx], groups[test_idx])
        if split in {"video", "clip"} and len(overlap):
            raise ValueError(f"Train/test group leakage: {overlap.tolist()}")

        fit_idx, val_idx = stratified_train_val_split(y, train_idx, val_fraction, seed + fold_id)
        if len(val_idx) == 0:
            val_size = max(1, int(round(len(train_idx) * val_fraction)))
            val_idx = train_idx[-val_size:]
            fit_idx = train_idx[:-val_size]
        if len(fit_idx) == 0:
            fit_idx, val_idx = train_idx, train_idx

        x_fit_raw, x_val_raw = x[fit_idx], x[val_idx]
        x_test_raw = x[test_idx]
        x_fit, x_other = standardize_zscore(x_fit_raw, np.concatenate([x_val_raw, x_test_raw], axis=0))
        x_val = x_other[: len(x_val_raw)]
        x_test = x_other[len(x_val_raw) :]

        model = train_mlp(
            x_fit,
            y[fit_idx],
            x_val,
            y[val_idx],
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed + fold_id,
            device=device,
        )
        predictions[test_idx] = predict_mlp(model, x_test, device, batch_size=batch_size)
        evaluated[test_idx] = True

    if not evaluated.any():
        return []

    metrics = summarize_metrics(y[evaluated], predictions[evaluated])
    metrics.update(
        {
            "seed": seed,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "val_fraction": val_fraction,
            "evaluated_clips": int(evaluated.sum()),
            "folds": len(folds),
            "classifier": "mlp",
        }
    )
    return [metrics]


def print_mlp_results(
    run_name: str,
    feature: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    results: Sequence[Mapping[str, Any]],
) -> None:
    print(
        f"\n{run_name}: classifier=mlp feature={feature} clips={len(y)} "
        f"groups={len(np.unique(groups))} dim={x.shape[1]} labels={np.bincount(y).tolist()}"
    )
    for result in results:
        metric_parts = " ".join(f"{name}={result[name]:.3f}" for name in CORE_METRICS if name in result)
        print(f"seed={result['seed']} hidden={result['hidden_dim']} {metric_parts}")
    if results:
        best = results[0]
        print(f"confusion_matrix={best['confusion_matrix']}")


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

    results = evaluate_mlp(
        x,
        y,
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

    print_mlp_results(name, dataset.feature, x, y, groups, results)
    rows = []
    for result in results:
        row = dict(result)
        row.update({"run": name, "feature": dataset.feature, "split": split, "group_by": group_by})
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP evaluation for extracted tension-recognition features.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--feature", choices=TensionDataset.feature_names())
    parser.add_argument("--root", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--feature-root",
        type=str,
        help="Override feature directory relative to data/ (e.g. TRAC_features/2d_sem).",
    )
    parser.add_argument("--label-mode", choices=("binary", "multiclass"))
    parser.add_argument("--include-all-clips", action="store_true")
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
        write_results(output_csv, all_rows, MLP_CSV_COLUMN_ORDER)
        print(f"saved results: {output_csv}")


if __name__ == "__main__":
    main()
