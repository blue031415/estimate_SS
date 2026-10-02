"""Experiment 10: project video joints into a noisy-C3D common subspace."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from experiments.common_subspace import (  # noqa: E402
    average_projection_matrix,
    build_shape_subspaces,
    eigendecompose_descending,
    select_eigen_indices,
)
from experiments.config import load_experiment_config, parse_config_path  # noqa: E402
from experiments.coordinate_projection import (  # noqa: E402
    project_coordinate_frames,
    upright_pose_coordinates,
)
from experiments.joint_correspondence import (  # noqa: E402
    extract_c3d_common_joints,
    extract_pose_common_joints,
)
from experiments.noise import add_noise_to_markers, marker_indices  # noqa: E402
from utils.c3d import read_c3d  # noqa: E402
from utils.projected_display import plot_coordinates, plot_projected_coordinates  # noqa: E402


DEFAULT_CONFIG = {
    "c3d_path": Path("data/07_01.c3d"),
    "pose_npz_path": Path("out/ex6/歩行動作3/歩行動作3.npz"),
    "output_path": Path("out/ex10/video_projected_to_c3d.png"),
    "original_output_path": Path("out/ex10/video_original.png"),
    "dimension": 5,
    "frame_index": 60,
    "contribution_threshold": 0.999,
    "noise": {
        "start_frame": 150,
        "end_frame": 250,
        "scale": 30,
        "body_parts": ["left_arm"],
        "mean": [0.0, 0.0, 0.0],
        "covariance": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    },
}


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    noise = config["noise"]

    # 1. Reproduce Experiment 3's noisy C3D input.
    c3d_data, c3d_labels = read_c3d(str(config["c3d_path"]), return_labels=True)
    noise_indices = marker_indices(c3d_labels, noise["body_parts"])
    noisy_c3d = add_noise_to_markers(
        c3d_data,
        noise_indices,
        noise["start_frame"],
        noise["end_frame"],
        noise["scale"],
        noise["mean"],
        noise["covariance"],
    )

    # 2. Extract corresponding joints, averaging paired wrist/waist markers.
    c3d_common_joints = extract_c3d_common_joints(noisy_c3d, c3d_labels)
    with np.load(config["pose_npz_path"]) as contents:
        pose_world_landmarks = contents["world_landmarks"]
        landmark_names = contents["landmark_names"]
    valid = np.isfinite(pose_world_landmarks).all(axis=(1, 2))
    pose_common_joints = extract_pose_common_joints(pose_world_landmarks[valid], landmark_names)
    pose_common_joints = upright_pose_coordinates(pose_common_joints)

    # 3. Plot the original video joints without common-subspace projection.
    frame_index = config["frame_index"]
    if frame_index >= len(pose_common_joints):
        raise ValueError(f"frame_index {frame_index} exceeds {len(pose_common_joints)} valid pose frames")
    plot_coordinates(
        pose_common_joints[frame_index],
        config["original_output_path"],
        f"Original video joints, frame {frame_index}",
    )

    # 4. Estimate a common subspace only from C3D and project video coordinates into it.
    c3d_subspaces = build_shape_subspaces(c3d_common_joints)
    projection_matrix = average_projection_matrix(c3d_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)
    indices, _, _ = select_eigen_indices(eigen_values, threshold=config["contribution_threshold"])
    dimension = config["dimension"]
    if dimension > len(indices):
        raise ValueError(f"dimension {dimension} exceeds {len(indices)} selected C3D eigenvectors")
    c3d_common_subspace = eigen_vectors[:, indices[:dimension]]
    projected_pose = project_coordinate_frames(pose_common_joints, c3d_common_subspace)

    # 5. Plot the same video frame before and after projection into the C3D subspace.
    plot_projected_coordinates(
        pose_common_joints[frame_index],
        projected_pose[frame_index],
        config["output_path"],
        f"Video joints projected into noisy C3D common subspace, D={dimension}",
    )


if __name__ == "__main__":
    main(parse_config_path(__doc__))
