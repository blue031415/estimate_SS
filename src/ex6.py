"""Experiment 6: estimate self-video poses and run the Experiment 2 analyses."""

from pathlib import Path

import matplotlib
import numpy as np

# Experiment 6 only writes GIF files and must also run on machines without Tk.
matplotlib.use("Agg")

from experiments.common_subspace import (  # noqa: E402
    average_projection_matrix,
    build_shape_subspaces,
    eigendecompose_descending,
    project_to_common_subspace,
    select_eigen_indices,
)
from experiments.config import load_experiment_config, parse_config_path  # noqa: E402
from experiments.dissimilarity import (  # noqa: E402
    calculate_difference_scores,
    calculate_geodestic,
    calculate_second_order_difference_DS,
)
from experiments.pose_estimation import MediaPipePoseEstimator, MMPosePoseEstimator  # noqa: E402
from utils.display import display_motion_score_contribution, display_motion_some_score  # noqa: E402
from utils.pose_display import save_pose_comparison_gif  # noqa: E402
from utils.video import VideoReader  # noqa: E402


DEFAULT_CONFIG = {
    "input_dir": Path("data/self_videos"),
    "video_pattern": "*.mp4",
    "model_path": Path("data/models/pose_landmarker_lite.task"),
    "estimator": "mediapipe",
    "mmpose_pose3d_model": "human3d",
    "mmpose_pose3d_weights": None,
    "mmpose_detector": None,
    "mmpose_device": "cpu",
    "mmpose_max_input_size": 1024,
    "output_dir": Path("out/ex6"),
    "frame_stride": 3,
    "max_frames": None,
    "gif_fps": 10,
    "panel_size": 512,
    "min_detection_confidence": 0.5,
    "min_presence_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "display_visibility_threshold": 0.5,
    "first_dimension": 4,
    "second_dimension": 5,
    "geodestic_dimension": 5,
    "frame_interval": 3,
    "contribution_threshold": 0.999,
}


def _estimator_context(config):
    if config["estimator"] == "mediapipe":
        return MediaPipePoseEstimator(
            config["model_path"],
            min_detection_confidence=config["min_detection_confidence"],
            min_presence_confidence=config["min_presence_confidence"],
            min_tracking_confidence=config["min_tracking_confidence"],
        )
    if config["estimator"] == "mmpose":
        return MMPosePoseEstimator(
            pose3d_model=config["mmpose_pose3d_model"],
            pose3d_weights=config["mmpose_pose3d_weights"],
            detector=config["mmpose_detector"],
            device=config["mmpose_device"],
            max_input_size=config["mmpose_max_input_size"],
        )
    raise ValueError("estimator must be 'mediapipe' or 'mmpose'")


def _estimate_video(video_path, config):
    rgb_frames = []
    frame_indices = []
    timestamps_ms = []
    estimates = []

    with VideoReader(video_path) as video:
        source_fps = video.fps
        with _estimator_context(config) as estimator:
            landmark_names = estimator.landmark_names
            connections = estimator.connections
            estimator_name = estimator.name
            for frame_index, rgb_frame in video.iter_sampled_frames(
                frame_stride=config["frame_stride"], max_frames=config["max_frames"]
            ):
                timestamp_ms = round(frame_index * 1000 / source_fps)
                estimate = estimator.estimate(rgb_frame, timestamp_ms)
                rgb_frames.append(rgb_frame)
                frame_indices.append(frame_index)
                timestamps_ms.append(timestamp_ms)
                estimates.append(estimate)

    if not rgb_frames:
        raise ValueError(f"No frames were decoded from {video_path}")

    landmark_count = len(landmark_names)
    image_landmarks = np.full((len(estimates), landmark_count, 3), np.nan)
    world_landmarks = np.full((len(estimates), landmark_count, 3), np.nan)
    visibility = np.full((len(estimates), landmark_count), np.nan)
    presence = np.full((len(estimates), landmark_count), np.nan)
    for index, estimate in enumerate(estimates):
        if estimate is not None:
            image_landmarks[index] = estimate.image_landmarks
            world_landmarks[index] = estimate.world_landmarks
            visibility[index] = estimate.visibility
            presence[index] = estimate.presence

    return {
        "rgb_frames": rgb_frames,
        "frame_indices": np.asarray(frame_indices),
        "timestamps_ms": np.asarray(timestamps_ms),
        "world_landmarks": world_landmarks,
        "image_landmarks": image_landmarks,
        "visibility": visibility,
        "presence": presence,
        "landmark_names": np.asarray(landmark_names),
        "connections": np.asarray(connections),
        "source_fps": source_fps,
        "estimator_name": estimator_name,
    }


def _projected_subspaces(data, dimension, threshold):
    shape_subspaces = build_shape_subspaces(data)
    projection_matrix = average_projection_matrix(shape_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)
    indices, _, cumulative_ratio = select_eigen_indices(eigen_values, threshold=threshold)
    if dimension > len(indices):
        raise ValueError(
            f"Configured dimension {dimension} exceeds the {len(indices)} "
            "eigenvectors selected by contribution_threshold"
        )
    print(f"selected indices: {indices}")
    print(f"cumulative contribution: {cumulative_ratio[indices[-1]]}")
    common_subspace = eigen_vectors[:, indices[:dimension]]
    return project_to_common_subspace(shape_subspaces, common_subspace)


def _upright_landmarks(world_landmarks):
    """Convert (X, Y-down, Z-depth) to the XYZ layout used by ex2 plots."""
    points = np.asarray(world_landmarks)
    return np.stack((points[..., 0], points[..., 2], -points[..., 1]), axis=-1)


def _run_ex2_analyses(data, output_dir, output_stem, config, title):
    frame_interval = config["frame_interval"]
    interval_tag = f"fi{frame_interval}"
    plot_title = f"{title}, frame interval={frame_interval}"
    origin_output_dir = output_dir / "origin" / interval_tag
    origin_output_dir.mkdir(parents=True, exist_ok=True)
    if len(data) <= 2 * frame_interval:
        raise ValueError(
            f"At least {2 * frame_interval + 1} fully detected frames are required; got {len(data)}"
        )

    # Repeat ex2_first_origin.py directly on the original shape subspaces.
    shape_subspaces = build_shape_subspaces(data)
    magnitudes, contributions = calculate_difference_scores(shape_subspaces, frame_interval)
    display_motion_score_contribution(
        data,
        range(len(shape_subspaces) - frame_interval),
        magnitudes,
        contributions,
        str(origin_output_dir / f"{output_stem}_first_origin_{interval_tag}.gif"),
        title=plot_title,
    )

    # Repeat ex2_second_origin.py on the original shape subspaces.
    origin_analysis_frames = range(frame_interval, len(shape_subspaces) - frame_interval)
    magnitudes, contributions = calculate_second_order_difference_DS(shape_subspaces, frame_interval)
    display_motion_score_contribution(
        data,
        origin_analysis_frames,
        magnitudes,
        contributions,
        str(origin_output_dir / f"{output_stem}_second_origin_{interval_tag}.gif"),
        title=plot_title,
    )

    # Repeat ex2_geodestic_origin.py on the original shape subspaces.
    magnitudes_along, contributions_along, magnitudes_orth, contributions_orth = calculate_geodestic(
        shape_subspaces, frame_interval
    )
    display_motion_score_contribution(
        data,
        origin_analysis_frames,
        magnitudes_along,
        contributions_along,
        str(origin_output_dir / f"{output_stem}_along_geodestic_origin_{interval_tag}.gif"),
        title=plot_title,
    )
    display_motion_score_contribution(
        data,
        origin_analysis_frames,
        magnitudes_orth,
        contributions_orth,
        str(origin_output_dir / f"{output_stem}_orth_geodestic_origin_{interval_tag}.gif"),
        title=plot_title,
    )
    display_motion_some_score(
        data,
        origin_analysis_frames,
        [magnitudes_along, magnitudes_orth],
        [
            f"along geodesic(origin, interval={frame_interval})",
            f"orth to geodesic(origin, interval={frame_interval})",
        ],
        str(origin_output_dir / f"{output_stem}_geodestic_origin_{interval_tag}.gif"),
        title=plot_title,
    )

    # Repeat ex2_first.py with the pose landmarks loaded from the generated NPZ.
    first_dimensions = config["first_dimension"]
    if isinstance(first_dimensions, int):
        first_dimensions = [first_dimensions]
    for dimension in first_dimensions:
        dimension_output_dir = output_dir / f"d{dimension}" / interval_tag
        dimension_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dimention of first-order common subspace: {dimension}")
        projected = _projected_subspaces(data, dimension, config["contribution_threshold"])
        magnitudes, contributions = calculate_difference_scores(projected, frame_interval)
        display_motion_score_contribution(
            data, range(len(projected) - frame_interval), magnitudes, contributions,
            str(dimension_output_dir / f"{output_stem}_first_d{dimension}_{interval_tag}.gif"),
            title=plot_title,
        )

    # Repeat ex2_second.py.
    second_dimensions = config["second_dimension"]
    if isinstance(second_dimensions, int):
        second_dimensions = [second_dimensions]
    for dimension in second_dimensions:
        dimension_output_dir = output_dir / f"d{dimension}" / interval_tag
        dimension_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dimention of second-order common subspace: {dimension}")
        projected = _projected_subspaces(data, dimension, config["contribution_threshold"])
        magnitudes, contributions = calculate_second_order_difference_DS(projected, frame_interval)
        analysis_frames = range(frame_interval, len(projected) - frame_interval)
        display_motion_score_contribution(
            data, analysis_frames, magnitudes, contributions,
            str(dimension_output_dir / f"{output_stem}_second_d{dimension}_{interval_tag}.gif"),
            title=plot_title,
        )

    # Repeat ex2_geodestic.py, preserving its public/output spelling.
    geodestic_dimensions = config["geodestic_dimension"]
    if isinstance(geodestic_dimensions, int):
        geodestic_dimensions = [geodestic_dimensions]
    for dimension in geodestic_dimensions:
        dimension_output_dir = output_dir / f"d{dimension}" / interval_tag
        dimension_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dimention of geodestic common subspace: {dimension}")
        projected = _projected_subspaces(data, dimension, config["contribution_threshold"])
        magnitudes_along, contributions_along, magnitudes_orth, contributions_orth = calculate_geodestic(
            projected, frame_interval
        )
        analysis_frames = range(frame_interval, len(projected) - frame_interval)
        display_motion_score_contribution(
            data, analysis_frames, magnitudes_along, contributions_along,
            str(dimension_output_dir / f"{output_stem}_along_geodestic_d{dimension}_{interval_tag}.gif"),
            title=plot_title,
        )
        display_motion_score_contribution(
            data, analysis_frames, magnitudes_orth, contributions_orth,
            str(dimension_output_dir / f"{output_stem}_orth_geodestic_d{dimension}_{interval_tag}.gif"),
            title=plot_title,
        )
        display_motion_some_score(
            data, analysis_frames, [magnitudes_along, magnitudes_orth],
            [f"along geodesic(D={dimension}, interval={frame_interval})",
             f"orth to geodesic(D={dimension}, interval={frame_interval})"],
            str(dimension_output_dir / f"{output_stem}_geodestic_d{dimension}_{interval_tag}.gif"),
            title=plot_title,
        )


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    input_dir = config["input_dir"]
    video_paths = sorted(input_dir.glob(config["video_pattern"]), key=lambda path: path.name)
    if not video_paths:
        raise ValueError(f"No videos matching {config['video_pattern']} were found in {input_dir}")

    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for video_path in video_paths:
        print(f"reading: {video_path}")
        result = _estimate_video(video_path, config)
        suffix = "" if config["estimator"] == "mediapipe" else "_mmpose"
        output_stem = f"{video_path.stem}{suffix}"
        npz_path = output_dir / f"{output_stem}.npz"
        np.savez_compressed(
            npz_path,
            world_landmarks=result["world_landmarks"],
            image_landmarks=result["image_landmarks"],
            visibility=result["visibility"],
            presence=result["presence"],
            frame_indices=result["frame_indices"],
            timestamps_ms=result["timestamps_ms"],
            landmark_names=result["landmark_names"],
            connections=result["connections"],
            source_fps=result["source_fps"],
            source_video=str(video_path),
            estimator=result["estimator_name"],
        )

        pose_gif_path = output_dir / f"{output_stem}_pose.gif"
        save_pose_comparison_gif(
            result["rgb_frames"],
            result["world_landmarks"],
            result["visibility"],
            result["connections"],
            pose_gif_path,
            output_fps=config["gif_fps"],
            panel_size=config["panel_size"],
            visibility_threshold=config["display_visibility_threshold"],
            estimator_label=result["estimator_name"],
        )

        # Keep source alignment in NPZ, but exclude failed detections from SVD.
        valid_frames = np.isfinite(result["world_landmarks"]).all(axis=(1, 2))
        data = _upright_landmarks(result["world_landmarks"][valid_frames])
        detected = int(valid_frames.sum())
        print(f"detected {detected}/{len(valid_frames)} sampled frames")
        print(f"saved: {npz_path}")
        print(f"saved: {pose_gif_path}")
        _run_ex2_analyses(data, output_dir, output_stem, config, video_path.name)


if __name__ == "__main__":
    main(parse_config_path(__doc__))
