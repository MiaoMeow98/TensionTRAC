"""Minimal trajectory statistics for tracked point clips."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

EPS = 1e-6

def mean(values):
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

def maximum(values):
    return float(np.nanmax(values)) if np.isfinite(values).any() else float("nan")

def variance(values):
    return float(np.nanvar(values)) if np.isfinite(values).any() else float("nan")

def stddev(values):
    return float(np.nanstd(values)) if np.isfinite(values).any() else float("nan")

def load_tracks(npz_path: Path):
    data = np.load(npz_path)
    return data["tracks"].astype(np.float32), data["visibility"].astype(bool)

def compute_velocity_and_acceleration(tracks, visibility):
    velocities = np.diff(tracks, axis=0)
    velocity_valid = visibility[:-1] & visibility[1:]
    velocity_mag = np.where(velocity_valid, np.linalg.norm(velocities, axis=-1), np.nan)

    accelerations = np.diff(velocities, axis=0)
    acceleration_valid = visibility[:-2] & visibility[1:-1] & visibility[2:]
    acceleration_mag = np.where(
        acceleration_valid,
        np.linalg.norm(accelerations, axis=-1),
        np.nan,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        velocity_mean_per_point = np.nanmean(velocity_mag, axis=0)
        acceleration_mean_per_point = np.nanmean(acceleration_mag, axis=0)
        velocity_mean_per_frame = np.nanmean(velocity_mag, axis=1)
        acceleration_mean_per_frame = np.nanmean(acceleration_mag, axis=1)

    return {
        "velocity_magnitude": velocity_mag,
        "acceleration_magnitude": acceleration_mag,
        "velocity_mean": mean(velocity_mag),
        "velocity_max": maximum(velocity_mag),
        "velocity_variance": variance(velocity_mag),
        "acceleration_mean": mean(acceleration_mag),
        "acceleration_max": maximum(acceleration_mag),
        "acceleration_variance": variance(acceleration_mag),
        "velocity_mean_per_point": velocity_mean_per_point,
        "acceleration_mean_per_point": acceleration_mean_per_point,
        "velocity_mean_per_frame": velocity_mean_per_frame,
        "acceleration_mean_per_frame": acceleration_mean_per_frame,
    }

def compute_pairwise_deformation(tracks, visibility):
    num_points = tracks.shape[1]
    pairwise_xy = tracks[:, :, None, :] - tracks[:, None, :, :]
    pairwise_dist = np.linalg.norm(pairwise_xy, axis=-1)

    pair_valid = visibility[:, :, None] & visibility[:, None, :]
    upper_mask = np.triu(np.ones((num_points, num_points), dtype=bool), k=1)
    baseline_dist = pairwise_dist[0]
    baseline_valid = pair_valid[0] & upper_mask & (baseline_dist > EPS)

    distance_change = pairwise_dist - baseline_dist[None]
    stretch_ratio = pairwise_dist / np.where(baseline_dist > EPS, baseline_dist, np.nan)[None]
    strain = distance_change / np.where(baseline_dist > EPS, baseline_dist, np.nan)[None]
    valid = pair_valid & baseline_valid[None]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        distance_change_pairs = np.nanmean(np.where(valid, distance_change, np.nan), axis=0)
        distance_std_pairs = np.nanstd(np.where(valid, pairwise_dist, np.nan), axis=0)
        max_stretch_pairs = np.nanmax(np.where(valid, stretch_ratio, np.nan), axis=0)
        strain_energy_per_point = np.nanmean(np.where(valid, strain, np.nan) ** 2, axis=2)

    strain_energy_per_frame = np.nanmean(strain_energy_per_point, axis=1)
    upper_values = upper_mask
    return {
        "strain_energy_per_point": strain_energy_per_point,
        "strain_energy_per_frame": strain_energy_per_frame,
        "mean_distance_change": mean(distance_change_pairs[upper_values]),
        "std_distance_over_time": mean(distance_std_pairs[upper_values]),
        "max_stretch_ratio": maximum(max_stretch_pairs[upper_values]),
        "mean_max_stretch_ratio": mean(max_stretch_pairs[upper_values]),
        "strain_energy_mean": mean(strain_energy_per_frame),
        "strain_energy_max": maximum(strain_energy_per_frame),
    }

def compute_motion_coherence(tracks, visibility):
    velocities = np.diff(tracks, axis=0)
    velocity_valid = visibility[:-1] & visibility[1:]
    values = []
    for t in range(velocities.shape[0]):
        valid = velocity_valid[t]
        if valid.sum() < 2:
            values.append(np.nan)
            continue
        frame_vel = velocities[t, valid]
        numerator = np.linalg.norm(frame_vel.sum(axis=0))
        denominator = np.linalg.norm(frame_vel, axis=1).sum() + EPS
        values.append(numerator / denominator)
    values = np.asarray(values, dtype=np.float32)
    return {
        "motion_coherence_per_frame": values,
        "motion_coherence_mean": mean(values),
        "motion_coherence_std": stddev(values),
    }

def compute_spatial_dispersion(tracks, visibility):
    values = []
    for t in range(tracks.shape[0]):
        valid = visibility[t]
        if valid.sum() == 0:
            values.append(np.nan)
            continue
        pts = tracks[t, valid]
        centroid = pts.mean(axis=0)
        values.append(np.linalg.norm(pts - centroid, axis=1).mean())
    values = np.asarray(values, dtype=np.float32)
    return {
        "spatial_dispersion_per_frame": values,
        "spatial_dispersion_mean": mean(values),
        "spatial_dispersion_max": maximum(values),
    }

def compute_summary_from_arrays(tracks, visibility):
    motion = compute_velocity_and_acceleration(tracks, visibility)
    deform = compute_pairwise_deformation(tracks, visibility)
    coherence = compute_motion_coherence(tracks, visibility)
    dispersion = compute_spatial_dispersion(tracks, visibility)

    summary = {
        "num_frames": int(tracks.shape[0]),
        "num_points": int(tracks.shape[1]),
        "num_visible_points_mean": float(visibility.sum(axis=1).mean()),
        "velocity_mean": motion["velocity_mean"],
        "velocity_max": motion["velocity_max"],
        "velocity_variance": motion["velocity_variance"],
        "acceleration_mean": motion["acceleration_mean"],
        "acceleration_max": motion["acceleration_max"],
        "acceleration_variance": motion["acceleration_variance"],
        "mean_distance_change": deform["mean_distance_change"],
        "std_distance_over_time": deform["std_distance_over_time"],
        "max_stretch_ratio": deform["max_stretch_ratio"],
        "mean_max_stretch_ratio": deform["mean_max_stretch_ratio"],
        "strain_energy_mean": deform["strain_energy_mean"],
        "strain_energy_max": deform["strain_energy_max"],
        "motion_coherence_mean": coherence["motion_coherence_mean"],
        "motion_coherence_std": coherence["motion_coherence_std"],
        "spatial_dispersion_mean": dispersion["spatial_dispersion_mean"],
        "spatial_dispersion_max": dispersion["spatial_dispersion_max"],
    }
    arrays = {
        "velocity_magnitude": motion["velocity_magnitude"],
        "acceleration_magnitude": motion["acceleration_magnitude"],
        "velocity_mean_per_point": motion["velocity_mean_per_point"],
        "acceleration_mean_per_point": motion["acceleration_mean_per_point"],
        "velocity_mean_per_frame": motion["velocity_mean_per_frame"],
        "acceleration_mean_per_frame": motion["acceleration_mean_per_frame"],
        "strain_energy_per_point": deform["strain_energy_per_point"],
        "strain_energy_per_frame": deform["strain_energy_per_frame"],
        "motion_coherence_per_frame": coherence["motion_coherence_per_frame"],
        "spatial_dispersion_per_frame": dispersion["spatial_dispersion_per_frame"],
    }
    return summary, arrays

def compute_summary(npz_path: Path):
    tracks, visibility = load_tracks(npz_path)
    return compute_summary_from_arrays(tracks, visibility)

def save(output_dir: Path, summary, arrays):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    np.savez_compressed(output_dir / "stats_arrays.npz", **arrays)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    summary, arrays = compute_summary(args.tracks_npz)
    output_dir = args.output_dir or args.tracks_npz.parent / "trajectory_stats"
    save(output_dir, summary, arrays)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
