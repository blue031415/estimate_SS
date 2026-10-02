"""Experiment 7: plot common-subspace eigenvalues from Experiment 6 NPZ files."""

from pathlib import Path

import matplotlib
import numpy as np

# Experiment 7 only writes PNG files and must also run on machines without Tk.
matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from experiments.common_subspace import (  # noqa: E402
    average_projection_matrix,
    build_shape_subspaces,
    eigendecompose_descending,
    project_to_common_subspace,
    select_eigen_indices,
)
from experiments.config import load_experiment_config, parse_config_path  # noqa: E402
from subspace import cal_magnitude  # noqa: E402
from utils.plot_style import configure_matplotlib_font  # noqa: E402


configure_matplotlib_font()


DEFAULT_CONFIG = {
    "input_dir": Path("out/ex6"),
    "npz_pattern": "*/*.npz",
    "output_dir": Path("out/ex7"),
    "contribution_threshold": 0.999,
}


def plot_eigen_values(eigen_values, path, title):
    plt.figure()
    ranks = np.arange(1, len(eigen_values) + 1)
    plt.plot(ranks, eigen_values, marker="o")
    plt.xticks(ranks)
    plt.xlabel("rank")
    plt.ylabel("eigenvalue")
    plt.title(title)
    plt.grid(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    input_dir = config["input_dir"]
    npz_paths = sorted(input_dir.glob(config["npz_pattern"]), key=lambda path: path.name)
    if not npz_paths:
        raise ValueError(f"No NPZ files matching {config['npz_pattern']} were found in {input_dir}")

    output_dir = config["output_dir"]
    for npz_path in npz_paths:
        # 1. Load ex6 world landmarks without modifying the source NPZ.
        with np.load(npz_path) as contents:
            world_landmarks = contents["world_landmarks"]
        valid_frames = np.isfinite(world_landmarks).all(axis=(1, 2))
        data = world_landmarks[valid_frames]
        if not len(data):
            raise ValueError(f"No fully detected frames were found in {npz_path}")
        print(f"[{npz_path.stem}] frame num: {len(data)}")

        # 2. Repeat ex1's common-subspace eigenvalue calculation.
        shape_subspaces = build_shape_subspaces(data)
        projection_matrix = average_projection_matrix(shape_subspaces)
        eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)
        indices, _, cumulative_ratio = select_eigen_indices(
            eigen_values, threshold=config["contribution_threshold"]
        )
        print(f"[{npz_path.stem}] selected indices: {indices}")
        print(f"[{npz_path.stem}] cumulative contribution: {cumulative_ratio[indices[-1]]}")
        output_path = output_dir / f"{npz_path.stem}_eigen_val.png"
        plot_eigen_values(
            eigen_values[indices],
            output_path,
            f"Eigenvalues of {npz_path.stem} averaged projection matrix",
        )

        # 3. Repeat ex1's projection-error evaluation.
        common_subspace = eigen_vectors[:, indices]
        projected_subspaces = project_to_common_subspace(shape_subspaces, common_subspace)
        magnitudes = [
            cal_magnitude(projected, original)
            for projected, original in zip(projected_subspaces, shape_subspaces)
        ]
        print(f"[{npz_path.stem}] average magnitude: {np.mean(magnitudes)}")
        print(f"[{npz_path.stem}] max magnitude: {np.max(magnitudes)} at frame {np.argmax(magnitudes)}")
        print(f"[{npz_path.stem}] min magnitude: {np.min(magnitudes)} at frame {np.argmin(magnitudes)}")
        print(f"[{npz_path.stem}] saved: {output_path}")


if __name__ == "__main__":
    main(parse_config_path(__doc__))
