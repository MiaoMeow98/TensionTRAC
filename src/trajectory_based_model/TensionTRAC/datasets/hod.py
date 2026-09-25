"""2D histogram-of-displacements features for tracked trajectories."""

from __future__ import annotations

import numpy as np


def get_orientation_hist(
    points: np.ndarray,
    num_bins: int,
    preserve_temporal: bool = True,
    orientation_hist_type: str = "2d",
) -> np.ndarray:
    """Compute 2D HoD features from normalized trajectories.

    Args:
        points: Array shaped ``[num_points, num_frames, 2]``.
        num_bins: Number of angular bins over ``[-pi, pi)``.
        preserve_temporal: If true, return one histogram per frame.
        orientation_hist_type: Kept for config compatibility; only ``"2d"`` is supported.
    """
    if orientation_hist_type != "2d":
        raise ValueError("This inference package only supports 2D HoD features.")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"Expected points with shape [N, T, 2], got {points.shape}.")

    num_points, num_frames, _ = points.shape
    if num_frames == 0:
        raise ValueError("At least one frame is required.")

    displacements = np.diff(points, axis=1)
    angles = np.arctan2(displacements[..., 1], displacements[..., 0])
    valid = np.isfinite(angles)
    bin_ids = np.floor((angles + np.pi) / (2 * np.pi) * num_bins).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)

    if preserve_temporal:
        hist = np.zeros((num_points, num_frames, num_bins), dtype=np.float32)
        for point_idx in range(num_points):
            for time_idx in range(num_frames - 1):
                if valid[point_idx, time_idx]:
                    hist[point_idx, time_idx + 1, bin_ids[point_idx, time_idx]] = 1.0
        return hist

    hist = np.zeros((num_points, num_bins), dtype=np.float32)
    for point_idx in range(num_points):
        for time_idx in range(num_frames - 1):
            if valid[point_idx, time_idx]:
                hist[point_idx, bin_ids[point_idx, time_idx]] += 1.0
    denom = hist.sum(axis=-1, keepdims=True)
    return hist / np.maximum(denom, 1.0)


def compute_temporal_pyramid(points: np.ndarray, num_bins: int) -> np.ndarray:
    """Compatibility helper for callers that expect a HoD pyramid function."""
    return get_orientation_hist(points, num_bins, preserve_temporal=True)
