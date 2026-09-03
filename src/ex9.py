"""Experiment 9: visualize Experiment 6 poses projected into their common subspaces."""

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
from utils.projected_display import plot_projected_coordinates  # noqa: E402


DEFAULT_CONFIG = {
    "input_dir": Path("out/ex6"),
    "npz_pattern": "*.npz",
    "output_dir": Path("out/ex9"),
    "dimension": 5,
    "frame_index": 60,
    "contribution_threshold": 0.999,
}


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    npz_paths = sorted(config["input_dir"].glob(config["npz_pattern"]), key=lambda path: path.name)
    if not npz_paths:
        raise ValueError(f"No NPZ files were found in {config['input_dir']}")

    for npz_path in npz_paths:
        # 1. Load fully detected pose frames without adding noise.
        with np.load(npz_path) as contents:
            world_landmarks = contents["world_landmarks"]
        valid = np.isfinite(world_landmarks).all(axis=(1, 2))
        data = upright_pose_coordinates(world_landmarks[valid])

        # 2. Estimate this video's common subspace and project its coordinates.
        shape_subspaces = build_shape_subspaces(data)
        projection_matrix = average_projection_matrix(shape_subspaces)
        eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)
        indices, _, _ = select_eigen_indices(eigen_values, threshold=config["contribution_threshold"])
        dimension = config["dimension"]
        if dimension > len(indices):
            raise ValueError(f"dimension {dimension} exceeds {len(indices)} selected eigenvectors for {npz_path}")
        common_subspace = eigen_vectors[:, indices[:dimension]]
        projected_data = project_coordinate_frames(data, common_subspace)

        # 3. Plot one configured valid-frame index.
        frame_index = config["frame_index"]
        if frame_index >= len(data):
            raise ValueError(f"frame_index {frame_index} is outside {npz_path} with {len(data)} valid frames")
        output_path = config["output_dir"] / f"{npz_path.stem}_projected_d{dimension}.png"
        plot_projected_coordinates(
            data[frame_index],
            projected_data[frame_index],
            output_path,
            f"{npz_path.stem} frame {frame_index}, D={dimension}",
        )


if __name__ == "__main__":
    main(parse_config_path(__doc__))
