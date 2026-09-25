from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
# extract_trajectories.py lives at src/trajectory_based_model/point_tracking/
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COTRACKER_REPO = REPO_ROOT / "co-tracker"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class TrajectoryResult:
    frames: np.ndarray
    tracks: np.ndarray
    tracks_normalized: np.ndarray
    visibility: np.ndarray
    fps: float
    source: Path


def read_frame_dir(frames_dir: Path) -> tuple[np.ndarray, float]:
    frame_paths = sorted(
        path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
    )
    if not frame_paths:
        raise ValueError(f"No image frames found in {frames_dir}")
    frames = [np.array(Image.open(path).convert("RGB")) for path in frame_paths]
    return np.stack(frames), 10.0


def read_video(video_path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 10.0)
    frames = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return np.stack(frames), fps


def load_frames(source: Path, frames_subdir: str = "frames") -> tuple[np.ndarray, float, Path]:
    source = Path(source)
    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        frames, fps = read_video(source)
        return frames, fps, source
    if source.is_dir() and any(path.suffix.lower() in IMAGE_EXTS for path in source.iterdir()):
        frames, fps = read_frame_dir(source)
        return frames, fps, source
    frames_dir = source / frames_subdir
    if frames_dir.is_dir():
        frames, fps = read_frame_dir(frames_dir)
        return frames, fps, frames_dir
    clip_video = source / "clip.mp4"
    if clip_video.is_file():
        frames, fps = read_video(clip_video)
        return frames, fps, clip_video
    raise ValueError(f"Cannot find frames or clip.mp4 under {source}")


def normalize_tracks(tracks: np.ndarray, width: int, height: int) -> np.ndarray:
    tracks = tracks.astype(np.float32, copy=True)
    tracks[..., 0] = 2.0 * tracks[..., 0] / max(width - 1, 1) - 1.0
    tracks[..., 1] = 2.0 * tracks[..., 1] / max(height - 1, 1) - 1.0
    return np.clip(tracks, -1.0, 1.0)


def denormalize_tracks(tracks_normalized: np.ndarray, width: int, height: int) -> np.ndarray:
    tracks = tracks_normalized.astype(np.float32, copy=True)
    tracks[..., 0] = (tracks[..., 0] + 1.0) * 0.5 * max(width - 1, 1)
    tracks[..., 1] = (tracks[..., 1] + 1.0) * 0.5 * max(height - 1, 1)
    return tracks


def load_cotracker(cotracker_repo: Path = DEFAULT_COTRACKER_REPO, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(str(cotracker_repo), "cotracker3_offline", source="local")
    return model.to(device).eval()


@torch.inference_mode()
def track_frames(
    frames: np.ndarray,
    model=None,
    *,
    cotracker_repo: Path = DEFAULT_COTRACKER_REPO,
    grid_size: int = 10,
    query_frame: int = 0,
    backward_tracking: bool = False,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = load_cotracker(cotracker_repo, device)

    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    pred_tracks, pred_visibility = model(
        video,
        grid_size=grid_size,
        grid_query_frame=query_frame,
        backward_tracking=backward_tracking,
    )
    return (
        pred_tracks[0].detach().cpu().numpy().astype(np.float32),
        pred_visibility[0].detach().cpu().numpy().astype(np.float32),
    )


def save_trajectories(
    output_path: Path,
    *,
    tracks: np.ndarray,
    tracks_normalized: np.ndarray,
    visibility: np.ndarray,
    frames: np.ndarray,
    fps: float,
    source: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    np.savez_compressed(
        output_path,
        tracks=tracks,
        tracks_normalized=tracks_normalized,
        visibility=visibility,
        frame_size=np.array([height, width], dtype=np.int64),
        fps=np.array([fps], dtype=np.float32),
        source=np.array([str(source)]),
    )


def load_trajectories(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_debug_video(
    frames: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    output_path: Path,
    fps: float = 10.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps or 10.0,
        (width, height),
    )
    for frame_idx, frame in enumerate(frames):
        canvas = frame.copy()
        for point_idx in range(tracks.shape[1]):
            if visibility[frame_idx, point_idx] < 0.5:
                continue
            x, y = tracks[frame_idx, point_idx]
            cv2.circle(canvas, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()


def iter_clip_sources(inputs: list[Path], frames_subdir: str = "frames"):
    for item in inputs:
        item = Path(item)
        if item.is_file():
            yield item.name, item
            continue
        if (item / frames_subdir).is_dir() or (item / "clip.mp4").is_file():
            yield item.name, item
            continue
        if item.is_dir() and any(path.suffix.lower() in IMAGE_EXTS for path in item.iterdir()):
            yield item.name, item
            continue
        for child in sorted(path for path in item.iterdir() if path.is_dir() or path.is_file()):
            if child.is_file() and child.suffix.lower() in VIDEO_EXTS:
                yield child.stem, child
            elif child.is_dir() and (
                (child / frames_subdir).is_dir()
                or (child / "clip.mp4").is_file()
                or any(path.suffix.lower() in IMAGE_EXTS for path in child.iterdir())
            ):
                yield child.name, child


def extract_source(
    source: Path,
    output_path: Path,
    *,
    model=None,
    cotracker_repo: Path = DEFAULT_COTRACKER_REPO,
    frames_subdir: str = "frames",
    grid_size: int = 10,
    query_frame: int = 0,
    backward_tracking: bool = False,
    device: str | None = None,
    save_video: bool = False,
) -> TrajectoryResult:
    frames, fps, resolved_source = load_frames(source, frames_subdir)
    tracks, visibility = track_frames(
        frames,
        model,
        cotracker_repo=cotracker_repo,
        grid_size=grid_size,
        query_frame=query_frame,
        backward_tracking=backward_tracking,
        device=device,
    )
    height, width = frames.shape[1:3]
    tracks_normalized = normalize_tracks(tracks, width, height)
    save_trajectories(
        output_path,
        tracks=tracks,
        tracks_normalized=tracks_normalized,
        visibility=visibility,
        frames=frames,
        fps=fps,
        source=resolved_source,
    )
    if save_video:
        save_debug_video(frames, tracks, visibility, output_path.with_suffix(".mp4"), fps)
    return TrajectoryResult(frames, tracks, tracks_normalized, visibility, fps, resolved_source)


def run_tension_model(result: TrajectoryResult, checkpoint: Path, config: Path, device: str | None = None):
    from trajectory_based_model.inference import forward_tension, load_tension_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_tension_model(
        checkpoint_path=checkpoint,
        config_path=config,
        device=device,
        load_dino_on_init=True,
    )
    video = torch.from_numpy(result.frames).permute(0, 3, 1, 2)[None].float()
    visibility = result.visibility[None]
    tracks = result.tracks_normalized[None]
    return forward_tension(model, video, tracks, visibility)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CoTracker trajectories for TensionTRAC.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Video files, frame dirs, clip dirs, or roots.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cotracker-repo", type=Path, default=DEFAULT_COTRACKER_REPO)
    parser.add_argument("--frames-subdir", default="frames")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--query-frame", type=int, default=0)
    parser.add_argument("--backward-tracking", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/TensionTRAC/TRAC_checkpoints/checkpoint.pyth"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/TensionTRAC/TRAC_checkpoints/TensionTRAC_checkpoint.yaml"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tracker = load_cotracker(args.cotracker_repo, device)

    for clip_name, source in iter_clip_sources(args.inputs, args.frames_subdir):
        output_path = args.output_root / clip_name / "trajectories.npz"
        if output_path.exists() and not args.run_model:
            print(f"exists: {output_path}")
            continue
        result = extract_source(
            source,
            output_path,
            model=tracker,
            cotracker_repo=args.cotracker_repo,
            frames_subdir=args.frames_subdir,
            grid_size=args.grid_size,
            query_frame=args.query_frame,
            backward_tracking=args.backward_tracking,
            device=device,
            save_video=args.save_video,
        )
        print(f"saved: {output_path}")
        if args.run_model:
            logits, embedding = run_tension_model(result, args.checkpoint, args.config, device)
            np.savez_compressed(
                output_path.parent / "tension_prediction.npz",
                logits=logits.detach().cpu().numpy(),
                embedding=embedding.detach().cpu().numpy(),
            )


if __name__ == "__main__":
    main()
