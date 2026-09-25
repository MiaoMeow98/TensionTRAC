"""Shared metric names for KNN evaluation outputs and reporting."""

from __future__ import annotations

CORE_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)

CORE_METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1": "Macro F1",
}

BINARY_EXTRA_METRICS: tuple[str, ...] = ("precision", "recall", "f1", "specificity")

CSV_COLUMN_ORDER: tuple[str, ...] = (
    "run",
    "feature",
    "split",
    "group_by",
    "k",
    "evaluated_clips",
    "folds",
    *CORE_METRICS,
    *BINARY_EXTRA_METRICS,
    "confusion_matrix",
)

CASCADE_E2E_FIELDS: dict[str, str] = {
    "accuracy": "cascade_accuracy",
    "balanced_accuracy": "cascade_balanced_accuracy",
    "macro_precision": "cascade_macro_precision",
    "macro_recall": "cascade_macro_recall",
    "macro_f1": "cascade_macro_f1",
}

CASCADE_STEP1_FIELDS: dict[str, str] = {
    "accuracy": "step1_accuracy",
    "balanced_accuracy": "step1_balanced_accuracy",
    "macro_precision": "step1_macro_precision",
    "macro_recall": "step1_macro_recall",
    "macro_f1": "step1_macro_f1",
}

CASCADE_STEP2_FIELDS: dict[str, str] = {
    "accuracy": "step2_accuracy",
    "balanced_accuracy": "step2_balanced_accuracy",
    "macro_precision": "step2_macro_precision",
    "macro_recall": "step2_macro_recall",
    "macro_f1": "step2_macro_f1",
}

CASCADE_CSV_COLUMN_ORDER: tuple[str, ...] = (
    "run",
    "feature",
    "split",
    "group_by",
    "k",
    "k_binary",
    "k_level",
    "evaluated_clips",
    "folds",
    *CASCADE_E2E_FIELDS.values(),
    *CASCADE_STEP1_FIELDS.values(),
    "step1_precision",
    "step1_recall",
    "step1_f1",
    *CASCADE_STEP2_FIELDS.values(),
    "step2_evaluated_clips",
    "cascade_confusion_matrix",
    "step1_confusion_matrix",
    "step2_confusion_matrix",
)

MLP_CSV_COLUMN_ORDER: tuple[str, ...] = (
    "run",
    "feature",
    "split",
    "group_by",
    "classifier",
    "seed",
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "patience",
    "val_fraction",
    "evaluated_clips",
    "folds",
    *CORE_METRICS,
    *BINARY_EXTRA_METRICS,
    "confusion_matrix",
)

MLP_CASCADE_CSV_COLUMN_ORDER: tuple[str, ...] = (
    "run",
    "feature",
    "split",
    "group_by",
    "classifier",
    "seed",
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "patience",
    "val_fraction",
    "evaluated_clips",
    "folds",
    *CASCADE_E2E_FIELDS.values(),
    *CASCADE_STEP1_FIELDS.values(),
    "step1_precision",
    "step1_recall",
    "step1_f1",
    *CASCADE_STEP2_FIELDS.values(),
    "step2_evaluated_clips",
    "cascade_confusion_matrix",
    "step1_confusion_matrix",
    "step2_confusion_matrix",
)
