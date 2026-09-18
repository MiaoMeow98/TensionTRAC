from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# Cross-backbone folders used when ``include_all_clips=False``.
ALIGNMENT_FEATURE_DIRS = (
    "slowfast_4x16_feature",
    "timesformer_feature",
    "video_swin_feature",
)

FEATURE_DIR_NAMES = {
    "slowfast": "slowfast_4x16_feature",
    "timesformer": "timesformer_feature",
    "video_swin": "video_swin_feature",
}


def is_tension_trac_feature(feature: str) -> bool:
    return feature == "tension_trac" or feature.startswith("tension_trac_")


def tension_trac_embedding_dir(feature: str) -> str:
    """Relative path under ``data/`` for one TRAC variant's per-clip embeddings.

    Each ``embedding.pt`` stores a 1-D fusion-dim vector (ST patch tokens pooled
    with visibility-weighted mean). Variant-specific ``D`` follows the active
    fusion branches (e.g. 2304 for ``2d_intra_cross_sem``).
    """
    from src.trajectory_based_model.checkpoint_variants import (
        embedding_dirname,
        variant_from_feature,
    )

    return embedding_dirname(variant_from_feature(feature))

DEFAULT_ANNOTATIONS = "data/annotation/tension_clip_annotations.csv"

# CSV rows with class labels 0-4 after excluding reverse (208) and unsure (66).
EXPECTED_CLASSIFICATION_CLIPS = 3549
# Tension-present clips used for 4-class level grading (levels 1-4 only).
EXPECTED_TENSION_LEVEL_CLIPS = 2675


def is_reverse_annotation(row: Mapping[str, str]) -> bool:
    """Return True for reverse-tension clips (excluded from classification)."""
    text = (row.get("annotation") or "").strip().lower()
    return "tension 100-0" in text


def is_unsure_annotation(row: Mapping[str, str]) -> bool:
    """Return True for unclear / unsure clips (excluded from classification)."""
    text = (row.get("annotation") or "").strip().lower()
    return text == "unsure"


def is_excluded_from_classification(row: Mapping[str, str]) -> bool:
    """Annotations present in CSV but never used for training or evaluation."""
    text = (row.get("annotation") or "").strip().lower()
    if not text or text == "not_annotated":
        return True
    return is_reverse_annotation(row) or is_unsure_annotation(row)


def parse_tension_level(row: Mapping[str, str]) -> int | None:
    """Parse class label 0-4 from CSV columns; None for excluded annotations."""
    if is_excluded_from_classification(row):
        return None

    raw_level = (row.get("tension_level") or "").strip()
    if raw_level:
        try:
            level = int(float(raw_level))
            if 0 <= level <= 4:
                return level
        except ValueError:
            pass

    text = (row.get("annotation") or "").strip().lower()
    if "no tension" in text:
        return 0
    for level in (4, 3, 2, 1):
        if f"level {level}" in text:
            return level
    return None


def resolve_clip_dir_name(row: Mapping[str, str], clip_dirs: set[str] | None = None) -> str:
    """Map CSV ``clip_name`` to ``data/clips`` / feature folder name."""
    clip_name = row.get("clip_name", "")
    if not clip_name:
        return ""

    if clip_dirs and clip_name in clip_dirs:
        return clip_name

    stem = clip_name[:-4] if clip_name.endswith(".mp4") else clip_name
    video = row.get("video", "")
    if video:
        candidate = f"{video}_{stem}"
        if clip_dirs is None or candidate in clip_dirs:
            return candidate

    return clip_name


def aligned_clip_ids(root: Path) -> set[str]:
    """Clip ids present under all backbone feature folders (3823 by default)."""
    name_sets: list[set[str]] = []
    for feature_dir in ALIGNMENT_FEATURE_DIRS:
        folder = root / "data" / feature_dir
        if not folder.is_dir():
            return set()
        name_sets.append({entry.name for entry in folder.iterdir() if entry.is_dir()})
    if not name_sets:
        return set()
    aligned = name_sets[0]
    for other in name_sets[1:]:
        aligned &= other
    return aligned


def build_annotation_index(
    annotations: Path,
    clip_dirs: set[str],
) -> dict[str, dict[str, str]]:
    """Map resolved ``clip_dir_name`` to CSV rows with parseable class labels."""
    index: dict[str, dict[str, str]] = {}
    with annotations.open(newline="") as handle:
        for row in csv.DictReader(handle):
            clip_dir_name = resolve_clip_dir_name(row, clip_dirs)
            if not clip_dir_name:
                continue
            tension_level = parse_tension_level(row)
            if tension_level is None:
                continue
            enriched = dict(row)
            enriched["clip_dir_name"] = clip_dir_name
            enriched["tension_level"] = str(tension_level)
            index[clip_dir_name] = enriched
    return index


class TensionDataset(Dataset):
    """Classification dataset driven only by the annotation CSV.

    Loads clips whose ``clip_name`` resolves to a folder under ``data/clips`` and
    carries a usable label in ``{0, 1, 2, 3, 4}``. Reverse (``tension 100-0``),
    ``unsure``, and the 44 on-disk clips without CSV rows are never included.

    Default settings yield **3549** clips (= 3823 CSV rows - 208 reverse - 66 unsure).

    ``label_mode='binary'`` maps level 0 -> no tension, levels 1-4 -> tension.
    ``label_mode='multiclass'`` keeps only tension-present clips (levels 1-4,
    **2675** clips) and uses zero-based class ids 0-3 corresponding to
    tension levels 1-4.
    ``label_mode='cascade'`` loads all **3549** clips; each sample exposes
    ``tension_level`` (0-4) for end-to-end cascade evaluation.

    ``include_all_clips=False`` (default) additionally requires the clip to appear
    under slowfast, timesformer, and video_swin feature folders. Set
    ``include_all_clips=True`` to skip that cross-backbone alignment check while
    still loading only CSV-matched classification clips.
    """

    def __init__(
        self,
        feature: str,
        root: str | Path = ".",
        annotations: str | Path = DEFAULT_ANNOTATIONS,
        label_mode: str = "binary",
        flatten: bool = True,
        require_feature: bool = True,
        include_all_clips: bool = False,
        feature_root: str | None = None,
    ) -> None:
        if feature not in self.feature_names():
            raise ValueError(f"feature must be one of {self.feature_names()}")

        self.feature = feature
        self.root = Path(root)
        self.annotations = Path(annotations)
        if not self.annotations.is_absolute():
            self.annotations = self.root / self.annotations
        self.label_mode = label_mode
        self.flatten = flatten
        self.include_all_clips = include_all_clips
        self.feature_root = feature_root

        clip_dirs = self._clip_dir_names()
        aligned_ids = aligned_clip_ids(self.root)
        annotation_index = build_annotation_index(self.annotations, clip_dirs)

        rows: list[dict[str, str]] = []
        for clip_dir_name, row in sorted(annotation_index.items()):
            if not include_all_clips and clip_dir_name not in aligned_ids:
                continue
            rows.append(dict(row))

        if require_feature:
            rows = [row for row in rows if self.feature_path(row).exists()]

        if label_mode == "multiclass":
            rows = [row for row in rows if int(row["tension_level"]) > 0]

        self.rows = rows
        self.aligned_clip_count = len(aligned_ids)
        self.classification_clip_count = len(annotation_index)
        if not self.rows:
            mode = "all_csv_classification" if include_all_clips else "aligned_csv_classification"
            raise ValueError(
                f"No rows with {feature!r} features found in {self.annotations} "
                f"(mode={mode}, csv_classification={len(annotation_index)}, "
                f"aligned_ids={len(aligned_ids)})"
            )

        self._validate_trac_feature_dim()

    def _validate_trac_feature_dim(self) -> None:
        """Reject legacy 768-d global embeddings when a wider fusion dim is expected."""
        if not is_tension_trac_feature(self.feature):
            return

        from src.trajectory_based_model.checkpoint_variants import expected_feature_dim

        expected = expected_feature_dim(self.feature)
        if expected is None:
            return

        path = self.feature_path(self.rows[0])
        tensor = self.load_feature(path)
        actual = int(tensor.reshape(-1).numel())
        if actual == expected:
            return

        hint = ""
        if actual == 768 and expected > 768:
            hint = " Likely legacy global_embedding; re-extract with TRAC_OVERWRITE=1."
        raise ValueError(
            f"TensionTRAC feature {self.feature!r} expected dim={expected}, "
            f"got dim={actual} in {path}.{hint}"
        )

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        from src.trajectory_based_model.checkpoint_variants import all_feature_names

        return ("slowfast", "timesformer", "video_swin", *all_feature_names())

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TensionDataset":
        config = dict(config.get("dataset", config))
        kwargs = {
            "feature": config["feature"],
            "root": config.get("root", "."),
            "annotations": config.get("annotations", DEFAULT_ANNOTATIONS),
            "label_mode": config.get("label_mode", "binary"),
            "flatten": config.get("flatten", True),
            "require_feature": config.get("require_feature", True),
            "include_all_clips": bool(config.get("include_all_clips", False)),
            "feature_root": config.get("feature_root"),
        }
        return cls(**kwargs)

    def _clip_dir_names(self) -> set[str]:
        clips_root = self.root / "data" / "clips"
        if not clips_root.is_dir():
            return set()
        return {entry.name for entry in clips_root.iterdir() if entry.is_dir()}

    def __len__(self) -> int:
        return len(self.rows)

    def feature_path(self, row: Mapping[str, str]) -> Path:
        clip_dir_name = row.get("clip_dir_name") or resolve_clip_dir_name(row)
        if is_tension_trac_feature(self.feature):
            feature_dir = self.feature_root or tension_trac_embedding_dir(self.feature)
            filename = "embedding.pt"
        else:
            feature_dir = FEATURE_DIR_NAMES[self.feature]
            filename = "feature.pt"

        raw_path = ""
        if self.feature == "slowfast":
            raw_path = row.get("slowfast_4x16_feature_path", "")
        elif self.feature == "timesformer":
            raw_path = row.get("timesformer_feature_path", "")
        elif self.feature == "video_swin":
            raw_path = row.get("video_swin_feature_path", "")
        elif is_tension_trac_feature(self.feature):
            raw_path = row.get(f"{feature_dir}_path", "")

        if raw_path:
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.root / path
            if path.suffix.lower() not in {".pt", ".pth", ".pyth", ".npy", ".npz"}:
                path = path / filename
            return path

        path = self.root / "data" / feature_dir / clip_dir_name / filename
        return path

    def load_feature(self, path: Path) -> torch.Tensor:
        if path.suffix.lower() in {".pt", ".pth", ".pyth"}:
            value = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(value, Mapping):
                for key in ("embedding", "global_embedding", "feature", "features"):
                    if key in value:
                        value = value[key]
                        break
                else:
                    tensors = [item for item in value.values() if torch.is_tensor(item)]
                    if len(tensors) != 1:
                        raise KeyError(f"Could not find a single feature tensor in {path}")
                    value = tensors[0]
            return torch.as_tensor(value).float()

        if path.suffix.lower() == ".npy":
            return torch.from_numpy(np.load(path, allow_pickle=False)).float()

        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as data:
                for key in ("embedding", "global_embedding", "feature", "features"):
                    if key in data.files:
                        return torch.from_numpy(np.asarray(data[key])).float()
                if len(data.files) == 1:
                    return torch.from_numpy(np.asarray(data[data.files[0]])).float()
                raise KeyError(f"Could not find an embedding array in {path}; found {data.files}")

        raise ValueError(f"Unsupported feature file: {path}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.feature_path(row)
        feature = self.load_feature(path)
        if self.flatten:
            feature = feature.reshape(-1)

        tension_level = int(float(row["tension_level"]))
        if tension_level < 0 or tension_level > 4:
            raise ValueError(f"Expected tension level in 0..4, got {tension_level}")

        if self.label_mode == "binary":
            label = int(tension_level > 0)
        elif self.label_mode == "multiclass":
            if tension_level < 1 or tension_level > 4:
                raise ValueError(
                    f"multiclass expects tension levels 1-4, got {tension_level}"
                )
            label = tension_level - 1
        elif self.label_mode == "cascade":
            label = tension_level
        else:
            raise ValueError("label_mode must be 'binary', 'multiclass', or 'cascade'")

        clip_path = Path(row["clip_path"]) if row.get("clip_path") else Path()
        if str(clip_path) and not clip_path.is_absolute():
            clip_path = self.root / clip_path

        return {
            "feature": feature,
            "label": torch.tensor(label, dtype=torch.long),
            "clip_name": row.get("clip_dir_name", row.get("clip_name", "")),
            "video": row.get("video", ""),
            "tension_level": tension_level,
            "feature_path": str(path),
            "clip_path": str(clip_path) if str(clip_path) else "",
        }
