"""TensionTRAC checkpoint variant registry for ablation studies."""

from __future__ import annotations

from pathlib import Path

# Short keys used by CLI (--variant) and feature names (tension_trac_2d_intra).
TRAC_VARIANTS: dict[str, str] = {
    "2d_sem": "TensionTRAC_2d_sem",
    "2d_intra": "TensionTRAC_2d_intra",
    "2d_cross": "TensionTRAC_2d_cross",
    "2d_intra_sem": "TensionTRAC_2d_intra_sem",
    "2d_cross_sem": "TensionTRAC_2d_cross_sem",
    "2d_intra_cross": "TensionTRAC_2d_intra_cross",
    "2d_intra_cross_sem": "TensionTRAC_2d_intra_cross_sem",
}

DEFAULT_VARIANT = "2d_intra_cross_sem"
ALL_VARIANTS: tuple[str, ...] = tuple(TRAC_VARIANTS)

# Fusion width after concat of active branches (each branch = 768).
EXPECTED_FUSION_DIM: dict[str, int] = {
    "2d_sem": 768,
    "2d_intra": 768,
    "2d_cross": 768,
    "2d_intra_sem": 1536,
    "2d_cross_sem": 1536,
    "2d_intra_cross": 1536,
    "2d_intra_cross_sem": 2304,
}

# All extracted embeddings live under data/TRAC_features/<variant>/ on DATA_ROOT.
TRAC_FEATURES_ROOT = "TRAC_features"


def validate_variant(variant: str) -> str:
    if variant not in TRAC_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {ALL_VARIANTS}")
    return variant


def feature_name(variant: str) -> str:
    """Dataset / eval ``feature`` key for a TRAC variant."""
    validate_variant(variant)
    if variant == DEFAULT_VARIANT:
        return "tension_trac"
    return f"tension_trac_{variant}"


def expected_fusion_dim(variant: str) -> int:
    """Return the fusion feature width for a checkpoint variant."""
    validate_variant(variant)
    return EXPECTED_FUSION_DIM[variant]


def expected_feature_dim(feature: str) -> int | None:
    """Return expected 1-D embedding size for a TensionTRAC feature key, or None."""
    if feature == "tension_trac" or feature.startswith("tension_trac_"):
        return expected_fusion_dim(variant_from_feature(feature))
    return None


def variant_from_feature(feature: str) -> str:
    """Map eval feature key back to checkpoint variant slug."""
    if feature == "tension_trac":
        return DEFAULT_VARIANT
    if feature.startswith("tension_trac_"):
        return validate_variant(feature.removeprefix("tension_trac_"))
    raise ValueError(f"Not a TensionTRAC feature key: {feature!r}")


def embedding_dirname(variant: str) -> str:
    """Relative path under ``data/`` for one variant's per-clip embeddings.

    Each clip stores ``embedding.pt`` as a 1-D fusion-dim vector produced by
    ST-transformer patch tokens + visibility-weighted mean pooling.
    """
    validate_variant(variant)
    return f"{TRAC_FEATURES_ROOT}/{variant}"


def embedding_root(data_root: Path, variant: str) -> Path:
    """Absolute embedding directory for a variant."""
    return Path(data_root) / "data" / embedding_dirname(variant)


def all_feature_names() -> tuple[str, ...]:
    return tuple(feature_name(v) for v in ALL_VARIANTS)


def resolve_paths(repo_root: Path, variant: str) -> tuple[Path, Path]:
    """Return ``(checkpoint_path, config_path)`` for a variant."""
    validate_variant(variant)
    folder = TRAC_VARIANTS[variant]
    ckpt_dir = repo_root / "TRAC_checkpoints" / folder
    return ckpt_dir / "checkpoint_best.pyth", ckpt_dir / f"{folder}.yaml"
