from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from fvcore.common.config import CfgNode

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from TensionTRAC.config.defaults import assert_and_infer_cfg, get_cfg  # noqa: E402
from TensionTRAC.datasets.hod import get_orientation_hist  # noqa: E402
from TensionTRAC.models.pointformer import Pointformer  # noqa: E402
from checkpoint_variants import (  # noqa: E402
    ALL_VARIANTS,
    DEFAULT_VARIANT,
    embedding_dirname,
    feature_name,
    resolve_paths,
    validate_variant,
)

DEFAULT_CHECKPOINT = REPO_ROOT / "TRAC_checkpoints" / "checkpoint.pyth"
DEFAULT_CONFIG = REPO_ROOT / "TRAC_checkpoints" / "TensionTRAC_checkpoint.yaml"


def uses_semantic_features(cfg) -> bool:
    return bool(getattr(cfg.MODEL.FUSION, "USE_SEMANTIC_FEATURES", True))


def _cfg_from_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        node = CfgNode()
        for key, item in value.items():
            node[key] = _cfg_from_mapping(item)
        return node
    if isinstance(value, list):
        return [_cfg_from_mapping(item) for item in value]
    return value


def cfg_from_checkpoint(checkpoint: Any) -> CfgNode | None:
    """Return the training cfg stored in a TensionTRAC checkpoint, if present."""
    if not isinstance(checkpoint, Mapping) or "cfg" not in checkpoint:
        return None
    embedded = checkpoint["cfg"]
    if isinstance(embedded, CfgNode):
        return embedded
    if isinstance(embedded, Mapping):
        return _cfg_from_mapping(dict(embedded))
    if isinstance(embedded, str):
        load_cfg = getattr(CfgNode, "load_cfg", None)
        if callable(load_cfg):
            return load_cfg(embedded)
        import yaml

        loaded = yaml.safe_load(embedded)
        if not isinstance(loaded, dict):
            raise TypeError("Checkpoint cfg string did not parse to a mapping.")
        return _cfg_from_mapping(loaded)
    raise TypeError(f"Unsupported checkpoint cfg type: {type(embedded)!r}")


def load_config(
    config_path: str | Path | None = DEFAULT_CONFIG,
    *,
    checkpoint: Any | None = None,
    checkpoint_path: str | Path | None = None,
    load_dino_on_init: bool | None = None,
):
    cfg = get_cfg()
    cfg.set_new_allowed(True)

    checkpoint_obj = checkpoint
    if checkpoint_obj is None and checkpoint_path is not None:
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    embedded = cfg_from_checkpoint(checkpoint_obj) if checkpoint_obj is not None else None
    if embedded is not None:
        cfg.merge_from_other_cfg(embedded)

    if config_path is not None:
        overlay = Path(config_path)
        if overlay.is_file():
            cfg.merge_from_file(str(overlay))
        elif embedded is None:
            raise FileNotFoundError(
                f"Missing config overlay {overlay} and checkpoint has no embedded cfg."
            )

    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
    if load_dino_on_init is None:
        load_dino_on_init = uses_semantic_features(cfg)
    cfg.MODEL.LOAD_DINO_ON_INIT = load_dino_on_init
    return assert_and_infer_cfg(cfg)


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model", "state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }


def _align_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], list[str], list[str], list[str]]:
    model_state = model.state_dict()
    aligned: dict[str, torch.Tensor] = {}
    mismatched: list[str] = []
    unexpected: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            mismatched.append(
                f"{key}: checkpoint{tuple(value.shape)} vs model{tuple(model_state[key].shape)}"
            )
            continue
        aligned[key] = value

    missing = [key for key in model_state if key not in aligned]
    return aligned, mismatched, missing, unexpected


def _assert_overlay_matches_checkpoint(cfg, state_dict: Mapping[str, Any]) -> None:
    """Catch POINT_INFO overlays that change CrossMotionModule input width."""
    weight = state_dict.get("cross_motion_module.fc2.c_fc.weight")
    if weight is None:
        return
    expected_in = int(weight.shape[1])
    actual_in = int(cfg.POINT_INFO.POINT_DIM) * int(cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE)
    if expected_in == actual_in:
        return
    implied_points = expected_in // max(int(cfg.POINT_INFO.POINT_DIM), 1)
    raise ValueError(
        "TensionTRAC_checkpoint.yaml POINT_INFO does not match checkpoint weights: "
        f"NUM_POINTS_TO_SAMPLE={cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE} implies "
        f"cross-motion in_features={actual_in}, but the checkpoint has {expected_in} "
        f"(~{implied_points} points). Set NUM_POINTS_TO_SAMPLE/GRID_SIZE to match "
        "the .pyth you are loading."
    )


def load_tension_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    config_path: str | Path | None = DEFAULT_CONFIG,
    *,
    device: str | torch.device | None = None,
    load_dino_on_init: bool | None = None,
    strict: bool = True,
) -> Pointformer:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = load_config(
        config_path,
        checkpoint=checkpoint,
        load_dino_on_init=load_dino_on_init,
    )
    state_dict = _checkpoint_state_dict(checkpoint)
    _assert_overlay_matches_checkpoint(cfg, state_dict)
    model = Pointformer(cfg).to(device)
    aligned, mismatched, missing, unexpected = _align_state_dict(state_dict, model)

    if mismatched:
        preview = "; ".join(mismatched[:8])
        raise RuntimeError(
            f"Checkpoint weights do not match model architecture ({len(mismatched)} mismatches). "
            f"Examples: {preview}"
        )

    model.load_state_dict(aligned, strict=False)

    if strict:
        critical_missing = [
            key
            for key in missing
            if not (key.startswith("dino.") and model.dino is None)
        ]
        if critical_missing:
            preview = ", ".join(critical_missing[:8])
            raise RuntimeError(
                f"Checkpoint is missing {len(critical_missing)} required model tensors. "
                f"Examples: {preview}"
            )

    if unexpected:
        print(
            f"Loaded checkpoint with {len(aligned)} tensors; "
            f"ignored {len(unexpected)} unexpected keys."
        )
    if missing and not strict:
        print(f"Warning: {len(missing)} model tensors were not present in the checkpoint.")

    model.eval()
    return model


def load_tension_model_for_variant(
    variant: str = DEFAULT_VARIANT,
    *,
    repo_root: Path | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> Pointformer:
    validate_variant(variant)
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    checkpoint_path, config_path = resolve_paths(root, variant)
    return load_tension_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
        strict=strict,
    )


def build_metadata(
    tracks: torch.Tensor | np.ndarray,
    visibility: torch.Tensor | np.ndarray,
    *,
    query_mask: torch.Tensor | np.ndarray | None = None,
    hod_feat: torch.Tensor | np.ndarray | None = None,
    num_bins: int = 32,
) -> dict[str, torch.Tensor]:
    """Build the metadata dict expected by Pointformer.

    ``tracks`` must be normalized to the ``[-1, 1]`` grid-sample coordinate
    system and shaped ``[B, T, N, 2]``. ``visibility`` is shaped ``[B, T, N]``.
    """
    tracks_t = torch.as_tensor(tracks, dtype=torch.float32)
    visibility_t = torch.as_tensor(visibility, dtype=torch.float32)
    if tracks_t.ndim != 4 or tracks_t.shape[-1] != 2:
        raise ValueError(f"Expected tracks shape [B, T, N, 2], got {tuple(tracks_t.shape)}")
    if visibility_t.shape != tracks_t.shape[:3]:
        raise ValueError(
            "Expected visibility shape [B, T, N], "
            f"got {tuple(visibility_t.shape)} for tracks {tuple(tracks_t.shape)}"
        )

    if query_mask is None:
        query_mask_t = visibility_t.bool()
    else:
        query_mask_t = torch.as_tensor(query_mask).bool()

    if hod_feat is None:
        hod_batches = []
        for batch_tracks in tracks_t.cpu().numpy():
            points = np.transpose(batch_tracks, (1, 0, 2))
            hod = get_orientation_hist(points, num_bins, preserve_temporal=True)
            hod_batches.append(torch.from_numpy(hod).unsqueeze(0))
        hod_feat_t = torch.cat(hod_batches, dim=0).float()
    else:
        hod_feat_t = torch.as_tensor(hod_feat, dtype=torch.float32)

    return {
        "pred_tracks": tracks_t,
        "pred_visibility": visibility_t,
        "pred_query_mask": query_mask_t,
        "hod_feat": hod_feat_t,
    }


def _model_inputs(
    model: Pointformer,
    video: torch.Tensor,
    tracks: torch.Tensor | np.ndarray,
    visibility: torch.Tensor | np.ndarray,
    *,
    query_mask: torch.Tensor | np.ndarray | None = None,
    hod_feat: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.device, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = next(model.parameters()).device
    metadata = build_metadata(
        tracks,
        visibility,
        query_mask=query_mask,
        hod_feat=hod_feat,
        num_bins=model.cfg.POINT_INFO.HOD.NUM_BINS,
    )
    metadata = {key: value.to(device) for key, value in metadata.items()}
    batch = {"video": video.to(device), "metadata": metadata}
    return device, metadata, batch


@torch.no_grad()
def forward_tension(
    model: Pointformer,
    video: torch.Tensor,
    tracks: torch.Tensor | np.ndarray,
    visibility: torch.Tensor | np.ndarray,
    *,
    query_mask: torch.Tensor | np.ndarray | None = None,
    hod_feat: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass and return ``(logits, trajectory_embeddings)``."""
    _device, _metadata, batch = _model_inputs(
        model, video, tracks, visibility, query_mask=query_mask, hod_feat=hod_feat
    )
    output = model(batch)
    if isinstance(output, tuple):
        return output
    return output, torch.empty(0, device=_device)


@torch.no_grad()
def extract_patch_embedding(
    model: Pointformer,
    video: torch.Tensor,
    tracks: torch.Tensor | np.ndarray,
    visibility: torch.Tensor | np.ndarray,
    *,
    query_mask: torch.Tensor | np.ndarray | None = None,
    hod_feat: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """Return raw ST-transformer patch tokens for one clip, shaped ``[T, N, D]``.

    Tokens come from ``pt_forward`` (after spatio-temporal transformer + norm) and
    do **not** pass through ``pre_logits`` or ``head``.
    """
    _logits, patch_x = forward_tension(
        model,
        video,
        tracks,
        visibility,
        query_mask=query_mask,
        hod_feat=hod_feat,
    )
    if patch_x.numel() == 0:
        raise RuntimeError("Failed to capture TensionTRAC patch tokens from model output.")
    return patch_x.squeeze(0).float().cpu()


def pool_patch_tokens(
    patch_x: torch.Tensor,
    visibility: torch.Tensor | np.ndarray,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pool ``[T, N, D]`` patch tokens to a clip-level ``[D]`` vector.

    Uses a visibility-weighted mean over time and trajectory points so occluded or
    missing tracks do not bias the embedding.
    """
    if patch_x.ndim == 4:
        if patch_x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for patch_x, got {patch_x.shape[0]}")
        patch_x = patch_x.squeeze(0)
    visibility_t = torch.as_tensor(visibility, dtype=torch.float32, device=patch_x.device)
    if visibility_t.ndim == 3:
        if visibility_t.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for visibility, got {visibility_t.shape[0]}")
        visibility_t = visibility_t.squeeze(0)
    if patch_x.ndim != 3:
        raise ValueError(f"Expected patch_x shape [T, N, D], got {tuple(patch_x.shape)}")
    if visibility_t.shape != patch_x.shape[:2]:
        raise ValueError(
            f"visibility shape {tuple(visibility_t.shape)} must match patch_x [T, N] "
            f"{tuple(patch_x.shape[:2])}"
        )

    weights = visibility_t.clamp(min=0.0)
    denom = weights.sum().clamp_min(eps)
    pooled = (patch_x * weights.unsqueeze(-1)).sum(dim=(0, 1)) / denom
    return pooled.float().cpu()


@torch.no_grad()
def extract_tension_feature(
    model: Pointformer,
    video: torch.Tensor,
    tracks: torch.Tensor | np.ndarray,
    visibility: torch.Tensor | np.ndarray,
    *,
    query_mask: torch.Tensor | np.ndarray | None = None,
    hod_feat: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """Return a clip-level trajectory feature for KNN/MLP, shaped ``[D]``.

    Pipeline: sem + intra + inter fusion -> ST transformer -> visibility-weighted
    mean pool over ``(T, N)``. Does **not** use ``pre_logits`` or ``head``.
    ``D`` equals the active fusion width (e.g. 2304 for ``2d_intra_cross_sem``).
    """
    patch_x = extract_patch_embedding(
        model,
        video,
        tracks,
        visibility,
        query_mask=query_mask,
        hod_feat=hod_feat,
    )
    return pool_patch_tokens(patch_x, visibility)


@torch.no_grad()
def extract_global_embedding(
    model: Pointformer,
    video: torch.Tensor,
    tracks: torch.Tensor | np.ndarray,
    visibility: torch.Tensor | np.ndarray,
    *,
    query_mask: torch.Tensor | np.ndarray | None = None,
    hod_feat: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """Return a 768-d global embedding (``pre_logits`` output, pre-64-way head)."""
    _device, _metadata, batch = _model_inputs(
        model, video, tracks, visibility, query_mask=query_mask, hod_feat=hod_feat
    )
    captured: dict[str, torch.Tensor] = {}

    def _capture_head_input(_module, inputs) -> None:
        captured["embedding"] = inputs[0].detach()

    handle = model.head.register_forward_pre_hook(_capture_head_input)
    try:
        model(batch)
    finally:
        handle.remove()

    if "embedding" not in captured:
        raise RuntimeError("Failed to capture TensionTRAC global embedding from model.head input.")
    return captured["embedding"].squeeze(0).float().cpu()


__all__ = [
    "ALL_VARIANTS",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_CONFIG",
    "DEFAULT_VARIANT",
    "build_metadata",
    "embedding_dirname",
    "extract_global_embedding",
    "extract_patch_embedding",
    "extract_tension_feature",
    "pool_patch_tokens",
    "feature_name",
    "forward_tension",
    "cfg_from_checkpoint",
    "load_config",
    "load_tension_model",
    "load_tension_model_for_variant",
    "variant_from_feature",
    "embedding_root",
    "resolve_paths",
    "uses_semantic_features",
    "validate_variant",
]
