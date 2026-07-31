"""Experiment 1: estimate and evaluate the common shape subspace."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from experiments.common_subspace import (
    average_projection_matrix,
    build_shape_subspaces,
    eigendecompose_descending,
    project_to_common_subspace,
    select_eigen_indices,
)
from subspace.utils import cal_magnitude
from experiments.config import load_experiment_config, parse_config_path
from utils.c3d import read_c3d

DEFAULT_CONFIG = {
    "data_path": Path("data") / "07_01.c3d",
    "output_path": Path("out") / "ex1" / "eigen_val.png",
    "contribution_threshold": 0.999,
}


def plot_eigen_values(eigen_values, path):
    plt.figure()
    plt.plot(np.arange(1, len(eigen_values) + 1), eigen_values, marker="o")
    plt.xticks(np.arange(1, len(eigen_values) + 1))
    plt.xlabel("rank")
    plt.ylabel("eigenvalue")
    plt.title("Eigenvalues of averaged projection matrix")
    plt.grid(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    data_path = config["data_path"]

    # 1. Build one shape subspace per input frame.
    data = read_c3d(str(data_path))
    print(f"frame num: {data.shape[0]}")
    shape_subspaces = build_shape_subspaces(data)

    # 2. Estimate the common subspace and select dimensions by cumulative contribution.
    projection_matrix = average_projection_matrix(shape_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)
    indices, _, cumulative_ratio = select_eigen_indices(eigen_values, threshold=config["contribution_threshold"])
    print(f"selected indices: {indices}")
    print(f"cumulative contribution: {cumulative_ratio[indices[-1]]}")
    plot_eigen_values(eigen_values[indices], config["output_path"])

    # 3. Project every frame and measure the distance from its original subspace.
    common_subspace = eigen_vectors[:, indices]
    projected_subspaces = project_to_common_subspace(shape_subspaces, common_subspace)
    magnitudes = [
        cal_magnitude(projected, original) for projected, original in zip(projected_subspaces, shape_subspaces)
    ]
    print(f"average magnitude: {np.mean(magnitudes)}")
    print(f"max magnitude: {np.max(magnitudes)} at frame {np.argmax(magnitudes)}")
    print(f"min magnitude: {np.min(magnitudes)} at frame {np.argmin(magnitudes)}")


if __name__ == "__main__":
    main(parse_config_path(__doc__))
