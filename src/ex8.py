"""Experiment 8: visualize noisy C3D coordinates projected into their common subspace."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from experiments.common_subspace import (  # noqa: E402
    average_projection_matrix,
    build_shape_subspaces,
    eigendecompose_descending,
    select_eigen_indices,
)
from experiments.config import load_experiment_config, parse_config_path  # noqa: E402
from experiments.coordinate_projection import project_coordinate_frames  # noqa: E402
from experiments.noise import add_noise_to_markers, marker_indices  # noqa: E402
from utils.c3d import read_c3d  # noqa: E402
from utils.projected_display import plot_projected_coordinates  # noqa: E402


DEFAULT_CONFIG = {
    "data_path": Path("data/07_01.c3d"),
    "output_path": Path("out/ex8/projected_coordinates.png"),
    "dimension": 5,
    "frame_index": 200,
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

    # 1. Repeat Experiment 3's C3D loading, marker selection, and noise injection.
    data, labels = read_c3d(str(config["data_path"]), return_labels=True)
    noise_indices = marker_indices(labels, noise["body_parts"])
    noisy_data = add_noise_to_markers(
        data,
        noise_indices,
        noise["start_frame"],
        noise["end_frame"],
        noise["scale"],
        noise["mean"],
        noise["covariance"],
    )

    # 2. Repeat Experiment 3's common-subspace estimation.
    shape_subspaces = build_shape_subspaces(noisy_data)
    projection_matrix = average_projection_matrix(shape_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)
    indices, _, _ = select_eigen_indices(eigen_values, threshold=config["contribution_threshold"])
    dimension = config["dimension"]
    if dimension > len(indices):
        raise ValueError(f"dimension {dimension} exceeds {len(indices)} selected eigenvectors")
    common_subspace = eigen_vectors[:, indices[:dimension]]

    # 3. Project marker coordinates with U U^T and plot one configured frame.
    projected_data = project_coordinate_frames(noisy_data, common_subspace)
    frame_index = config["frame_index"]
    plot_projected_coordinates(
        noisy_data[frame_index],
        projected_data[frame_index],
        config["output_path"],
        f"Noisy C3D frame {frame_index}, D={dimension}",
    )


if __name__ == "__main__":
    main(parse_config_path(__doc__))
