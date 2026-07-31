"""Experiment 3: geodesic decomposition after injecting marker noise."""

import os
from pathlib import Path

from experiments.common_subspace import (
    average_projection_matrix,
    build_shape_subspaces,
    eigendecompose_descending,
    project_to_common_subspace,
    select_eigen_indices,
)
from experiments.config import load_experiment_config, parse_config_path
from experiments.dissimilarity import calculate_geodestic
from experiments.noise import add_noise_to_markers, marker_indices
from utils.display import display_motion_score_contribution, display_motion_some_score
from utils.c3d import read_c3d

DEFAULT_CONFIG = {
    "data_path": Path("data") / "07_01.c3d",
    "dimension": 11,
    "output_path_along": Path("out") / "ex3" / "07_01_along_geodestic_d11.gif",
    "output_path_orth": Path("out") / "ex3" / "07_01_orth_geodestic_d11.gif",
    "output_path": Path("out") / "ex3" / "07_01_geodestic_d11.gif",
    "frame_interval": 5,
    "contribution_threshold": 0.999,
    "noise": {
        "start_frame": 200,
        "end_frame": 250,
        "scale": 30,
        "body_parts": ["left_arm"],
        "mean": [0.0, 0.0, 0.0],
        "covariance": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    },
}


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    data_path = config["data_path"]
    dimension = config["dimension"]
    frame_interval = config["frame_interval"]
    noise = config["noise"]

    # 1. Read the data and inject the configured marker noise.
    print(f"Dimention of common subspace: {dimension}")
    data, labels = read_c3d(str(data_path), return_labels=True)
    print(f"frame num: {data.shape[0]}")
    noise_indices = marker_indices(labels, noise["body_parts"])
    print(f"noise body parts: {noise['body_parts']}")
    print(f"noise markers: {[labels[index] for index in noise_indices]}")
    noisy_data = add_noise_to_markers(
        data,
        noise_indices,
        noise["start_frame"],
        noise["end_frame"],
        noise["scale"],
        noise["mean"],
        noise["covariance"],
    )

    # 2. Build and apply the common subspace.
    shape_subspaces = build_shape_subspaces(noisy_data)
    projection_matrix = average_projection_matrix(shape_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)
    indices, _, cumulative_ratio = select_eigen_indices(eigen_values, threshold=config["contribution_threshold"])
    print(f"selected indices: {indices}")
    print(f"cumulative contribution: {cumulative_ratio[indices[-1]]}")
    common_subspace = eigen_vectors[:, indices[:dimension]]
    projected_subspaces = project_to_common_subspace(shape_subspaces, common_subspace)

    # 3. Decompose the second-order change into geodesic components.
    magnitudes_along, contributions_along, magnitudes_orth, contributions_orth = calculate_geodestic(
        projected_subspaces, frame_interval
    )
    frames = range(frame_interval, len(projected_subspaces) - frame_interval)

    # 4. Render each component and their overlay.
    output_path_along = config["output_path_along"]
    output_path_orth = config["output_path_orth"]
    output_path = config["output_path"]
    output_path_along.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating {os.path.basename(output_path_along)}")
    display_motion_score_contribution(
        noisy_data,
        frames,
        magnitudes_along,
        contributions_along,
        str(output_path_along),
        title=data_path.name,
    )
    output_path_orth.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating {os.path.basename(output_path_orth)}")
    display_motion_score_contribution(
        noisy_data,
        frames,
        magnitudes_orth,
        contributions_orth,
        str(output_path_orth),
        title=data_path.name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating {os.path.basename(output_path)}")
    display_motion_some_score(
        noisy_data,
        frames,
        [magnitudes_along, magnitudes_orth],
        [f"along geodesic(D={dimension})", f"orth to geodesic(D={dimension})"],
        str(output_path),
        title=data_path.name,
    )


if __name__ == "__main__":
    main(parse_config_path(__doc__))
