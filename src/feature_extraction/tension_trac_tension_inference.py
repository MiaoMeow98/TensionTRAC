from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.TensionDataset import (  # noqa: E402
    aligned_clip_ids,
    build_annotation_index,
)
from src.trajectory_based_model.checkpoint_variants import (  # noqa: E402
    ALL_VARIANTS,
    DEFAULT_VARIANT,
    TRAC_VARIANTS,
    embedding_dirname,
    expected_fusion_dim,
    resolve_paths,
    validate_variant,
)
from src.trajectory_based_model.inference import (  # noqa: E402
    extract_tension_feature,
    load_tension_model,
)
from src.trajectory_based_model.point_tracking.extract_trajectories import load_frames  # noqa: E402

DEFAULT_TRAJECTORY_SUBDIR = "cotracker_uniform_400"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract TensionTRAC clip features (ST patch tokens + visibility-weighted "
            "mean pool) for frozen KNN/MLP evaluation."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/.../surgical_video_assessment/turbo_data"),
        help="Dataset root containing data/clips and trajectories/.",
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=None,
        help="Trajectory root (default: data-root/trajectories/cotracker_uniform_400).",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help=(
            "TensionTRAC checkpoint variant key "
            f"({', '.join(ALL_VARIANTS)}) or 'all' to extract every variant."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path (overrides --variant).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional explicit config path (overrides --variant).",
    )
    parser.add_argument("--num-frames", type=int, default=0, help="Override NUM_FRAMES (0 = use config).")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--aligned-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only process clips aligned across slowfast/timesformer/video_swin.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N clips (0 = all).")
    return parser.parse_args()


def temporal_indices(source_frames: int, target_frames: int) -> np.ndarray:
    if source_frames <= 0:
        raise ValueError("source_frames must be positive")
    if source_frames == target_frames:
        return np.arange(source_frames, dtype=np.int64)
    return np.linspace(0, source_frames - 1, target_frames).round().astype(np.int64)


def preprocess_frames(frames: np.ndarray, cfg) -> torch.Tensor:
    """Resize, optionally reverse channels, and normalize frames to match training."""
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    if tensor.max() > 1.5:
        tensor = tensor / 255.0

    crop_size = int(cfg.DATA.TRAIN_CROP_SIZE)
    if tensor.shape[-2] != crop_size or tensor.shape[-1] != crop_size:
        tensor = F.interpolate(
            tensor,
            size=(crop_size, crop_size),
            mode="bilinear",
            align_corners=False,
        )

    if bool(getattr(cfg.DATA, "REVERSE_INPUT_CHANNEL", False)):
        tensor = torch.flip(tensor, dims=(1,))

    mean = torch.tensor(cfg.DATA.MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(cfg.DATA.STD, dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def prepare_clip_tensors(
    frames: np.ndarray,
    tracks_normalized: np.ndarray,
    visibility: np.ndarray,
    *,
    cfg,
    num_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_frames = min(len(frames), tracks_normalized.shape[0], visibility.shape[0])
    indices = temporal_indices(source_frames, num_frames)

    sampled_frames = frames[indices]
    sampled_tracks = tracks_normalized[indices]
    sampled_visibility = visibility[indices]

    video = preprocess_frames(sampled_frames, cfg).unsqueeze(0)
    tracks = torch.from_numpy(sampled_tracks).float().unsqueeze(0)
    vis = torch.from_numpy(sampled_visibility).float().unsqueeze(0)
    return video, tracks, vis


def iter_clip_dirs(data_root: Path, aligned_only: bool) -> list[str]:
    clips_root = data_root / "data" / "clips"
    if not clips_root.is_dir():
        raise FileNotFoundError(f"Missing clips directory: {clips_root}")

    clip_dirs = {entry.name for entry in clips_root.iterdir() if entry.is_dir()}
    annotations = data_root / "annotations" / "tension_clip_annotations total 3824.csv"
    if not annotations.is_file():
        raise FileNotFoundError(f"Missing annotations CSV: {annotations}")

    classification_clips = set(build_annotation_index(annotations, clip_dirs))
    if aligned_only:
        aligned = aligned_clip_ids(data_root)
        if not aligned:
            raise RuntimeError(
                "No aligned clip ids found across slowfast/timesformer/video_swin feature folders."
            )
        classification_clips &= aligned

    return sorted(classification_clips)


def _variant_from_config_path(config_path: Path) -> str:
    folder_name = config_path.parent.name
    for variant, checkpoint_folder in TRAC_VARIANTS.items():
        if checkpoint_folder == folder_name:
            return variant
    slug = folder_name.removeprefix("TensionTRAC_")
    return slug


def resolve_variant_jobs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.checkpoint is not None and args.config is not None:
        variant = _variant_from_config_path(args.config.resolve())
        return [(variant, args.checkpoint.resolve(), args.config.resolve())]

    if args.variant == "all":
        return [(variant, *resolve_paths(REPO_ROOT, variant)) for variant in ALL_VARIANTS]

    variant = validate_variant(args.variant)
    checkpoint_path, config_path = resolve_paths(REPO_ROOT, variant)
    return [(variant, checkpoint_path, config_path)]


def extract_variant(
    *,
    variant: str,
    checkpoint_path: Path,
    config_path: Path,
    data_root: Path,
    trajectory_root: Path,
    clip_names: list[str],
    device: str,
    num_frames_override: int,
    overwrite: bool,
) -> None:
    model = load_tension_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )
    num_frames = num_frames_override if num_frames_override > 0 else int(model.cfg.DATA.NUM_FRAMES)
    output_root = data_root / "data" / embedding_dirname(variant)
    output_root.mkdir(parents=True, exist_ok=True)

    processed = skipped = missing_traj = failed = 0
    feature_dim: int | None = None
    for clip_name in tqdm(clip_names, desc=f"TensionTRAC[{variant}]"):
        out_path = output_root / clip_name / "embedding.pt"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        traj_path = trajectory_root / clip_name / "trajectories.npz"
        clip_dir = data_root / "data" / "clips" / clip_name
        if not traj_path.is_file():
            missing_traj += 1
            continue

        try:
            frames, _fps, _source = load_frames(clip_dir)
            with np.load(traj_path) as traj:
                tracks_normalized = traj["tracks_normalized"].astype(np.float32)
                visibility = traj["visibility"].astype(np.float32)

            video, tracks, visibility_t = prepare_clip_tensors(
                frames,
                tracks_normalized,
                visibility,
                cfg=model.cfg,
                num_frames=num_frames,
            )
            embedding = extract_tension_feature(model, video, tracks, visibility_t)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embedding, out_path)
            if feature_dim is None:
                feature_dim = int(embedding.numel())
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[{variant}] failed {clip_name}: {exc}")

    print(
        f"[{variant}] done: processed={processed} skipped={skipped} "
        f"missing_traj={missing_traj} failed={failed} feature_dim={feature_dim} "
        f"output={output_root}"
    )
    if processed > 0 or skipped > 0:
        verify_variant_embeddings(output_root, variant, expected_dim=expected_fusion_dim(variant))


def verify_variant_embeddings(
    output_root: Path,
    variant: str,
    *,
    expected_dim: int,
) -> None:
    """Smoke-check saved embeddings: count on disk and fusion width."""
    files = sorted(output_root.glob("*/embedding.pt"))
    if not files:
        raise RuntimeError(f"[{variant}] no embedding.pt files under {output_root}")

    sample = torch.load(files[0], map_location="cpu", weights_only=False)
    actual = int(torch.as_tensor(sample).reshape(-1).numel())
    if actual != expected_dim:
        raise RuntimeError(
            f"[{variant}] expected fusion dim={expected_dim}, got {actual} in {files[0]}. "
            "Re-run with --overwrite if old global embeddings remain."
        )
    print(f"[{variant}] verified: {len(files)} embeddings on disk, dim={actual}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    trajectory_root = (
        args.trajectory_root.resolve()
        if args.trajectory_root is not None
        else data_root / "trajectories" / DEFAULT_TRAJECTORY_SUBDIR
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    clip_names = iter_clip_dirs(data_root, aligned_only=args.aligned_only)
    if args.limit > 0:
        clip_names = clip_names[: args.limit]

    for variant, checkpoint_path, config_path in resolve_variant_jobs(args):
        extract_variant(
            variant=variant,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_root=data_root,
            trajectory_root=trajectory_root,
            clip_names=clip_names,
            device=device,
            num_frames_override=args.num_frames,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
