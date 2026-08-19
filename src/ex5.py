"""Experiment 5: estimate and visualize a 3D pose from GaHu-Video S001."""

from pathlib import Path

import numpy as np

from experiments.config import load_experiment_config, parse_config_path
from experiments.pose_estimation import MediaPipePoseEstimator, MMPosePoseEstimator
from utils.pose_display import save_pose_comparison_gif
from utils.video import VideoReader

DEFAULT_CONFIG = {
    "input_path": Path("data/GaHu-Video/S001.avi"),
    "model_path": Path("data/models/pose_landmarker_lite.task"),
    "estimator": "mediapipe",
    "mmpose_pose3d_model": "human3d",
    "mmpose_pose3d_weights": None,
    "mmpose_detector": None,
    "mmpose_device": "cpu",
    "mmpose_max_input_size": 1024,
    "npz_path": Path("out/ex5/S001.npz"),
    "gif_path": Path("out/ex5/S001.gif"),
    "frame_stride": 3,
    "max_frames": None,
    "gif_fps": 10,
    "panel_size": 512,
    "min_detection_confidence": 0.5,
    "min_presence_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "display_visibility_threshold": 0.5,
}


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    input_path = config["input_path"]
    print(f"reading: {input_path}")
    rgb_frames = []
    frame_indices = []
    timestamps_ms = []
    estimates = []

    # 1. Decode the GaHu video in source order, sampling through its full duration.
    with VideoReader(input_path) as video:
        source_fps = video.fps

        # 2. Select one explicit adapter and estimate a 3D skeleton per frame.
        if config["estimator"] == "mediapipe":
            estimator_context = MediaPipePoseEstimator(
                config["model_path"],
                min_detection_confidence=config["min_detection_confidence"],
                min_presence_confidence=config["min_presence_confidence"],
                min_tracking_confidence=config["min_tracking_confidence"],
            )
        elif config["estimator"] == "mmpose":
            estimator_context = MMPosePoseEstimator(
                pose3d_model=config["mmpose_pose3d_model"],
                pose3d_weights=config["mmpose_pose3d_weights"],
                detector=config["mmpose_detector"],
                device=config["mmpose_device"],
                max_input_size=config["mmpose_max_input_size"],
            )
        else:
            raise ValueError("estimator must be 'mediapipe' or 'mmpose'")

        with estimator_context as estimator:
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
        raise ValueError(f"No frames were decoded from {input_path}")

    # 3. Keep failed detections as NaN so all arrays retain source-frame alignment.
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

    npz_path = config["npz_path"]
    if config["estimator"] == "mmpose":
        npz_path = npz_path.with_stem(f"{npz_path.stem}_mmpose")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        world_landmarks=world_landmarks,
        image_landmarks=image_landmarks,
        visibility=visibility,
        presence=presence,
        frame_indices=np.asarray(frame_indices),
        timestamps_ms=np.asarray(timestamps_ms),
        landmark_names=np.asarray(landmark_names),
        connections=np.asarray(connections),
        source_fps=source_fps,
        source_video=str(input_path),
        estimator=estimator_name,
    )

    # 4. Save the original 16:9 frames beside a fixed-axis 3D pose animation.
    gif_path = config["gif_path"]
    if config["estimator"] == "mmpose":
        gif_path = gif_path.with_stem(f"{gif_path.stem}_mmpose")
    save_pose_comparison_gif(
        rgb_frames,
        world_landmarks,
        visibility,
        connections,
        gif_path,
        output_fps=config["gif_fps"],
        panel_size=config["panel_size"],
        visibility_threshold=config["display_visibility_threshold"],
        estimator_label=estimator_name,
    )
    detected = int(np.isfinite(world_landmarks).all(axis=2).any(axis=1).sum())
    print(f"detected {detected}/{len(rgb_frames)} sampled frames")
    print(f"saved: {npz_path}")
    print(f"saved: {gif_path}")


if __name__ == "__main__":
    main(parse_config_path(__doc__))
