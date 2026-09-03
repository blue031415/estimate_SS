"""Experiment 12: compare Experiment 6 and 11 difference-score graphs."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from ex6 import _projected_subspaces, _upright_landmarks  # noqa: E402
from experiments.common_subspace import (  # noqa: E402
    average_projection_matrix,
    build_shape_subspaces,
    eigendecompose_descending,
    project_to_common_subspace,
    select_eigen_indices,
)
from experiments.config import load_experiment_config, parse_config_path  # noqa: E402
from experiments.coordinate_projection import upright_pose_coordinates  # noqa: E402
from experiments.dissimilarity import (  # noqa: E402
    calculate_difference_scores,
    calculate_geodestic,
    calculate_second_order_difference_DS,
)
from experiments.joint_correspondence import (  # noqa: E402
    extract_c3d_common_joints,
    extract_pose_common_joints,
)
from utils.c3d import read_c3d  # noqa: E402
from utils.display import save_score_comparison  # noqa: E402


DEFAULT_CONFIG = {
    "c3d_path": Path("data/07_01.c3d"),
    "input_dir": Path("out/ex6"),
    "npz_pattern": "*.npz",
    "output_dir": Path("out/ex12"),
    "dimensions": [4, 5],
    "frame_interval": 2,
    "contribution_threshold": 0.999,
}


def _scores(subspaces, frame_interval):
    first, _ = calculate_difference_scores(subspaces, frame_interval)
    second, _ = calculate_second_order_difference_DS(subspaces, frame_interval)
    along, _, orth, _ = calculate_geodestic(subspaces, frame_interval)
    return {"first": first, "second": second, "along_geodestic": along, "orth_geodestic": orth}


def _save_comparisons(ex6_scores, ex11_scores, output_dir, stem, dimension, frame_interval):
    for score_name in ex6_scores:
        start = 0 if score_name == "first" else frame_interval
        x = range(start, start + len(ex6_scores[score_name]))
        save_score_comparison(
            x,
            [ex6_scores[score_name], ex11_scores[score_name]],
            ["ex6: pose common subspace", "ex11: clean-C3D common subspace"],
            output_dir / f"{stem}_{score_name}_d{dimension}_fi{frame_interval}.png",
            f"{stem}: {score_name}, D={dimension}, frame interval={frame_interval}",
        )
    geodestic_x = range(frame_interval, frame_interval + len(ex6_scores["along_geodestic"]))
    save_score_comparison(
        geodestic_x,
        [ex6_scores["along_geodestic"], ex6_scores["orth_geodestic"],
         ex11_scores["along_geodestic"], ex11_scores["orth_geodestic"]],
        ["ex6: along", "ex6: orthogonal", "ex11: along", "ex11: orthogonal"],
        output_dir / f"{stem}_geodestic_d{dimension}_fi{frame_interval}.png",
        f"{stem}: geodestic components, D={dimension}, frame interval={frame_interval}",
    )


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)

    # 1. Reproduce the clean-C3D common-subspace construction used by ex11.
    c3d_data, c3d_labels = read_c3d(str(config["c3d_path"]), return_labels=True)
    c3d_joints = extract_c3d_common_joints(c3d_data, c3d_labels)
    c3d_subspaces = build_shape_subspaces(c3d_joints)
    projection_matrix = average_projection_matrix(c3d_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)
    indices, _, _ = select_eigen_indices(eigen_values, config["contribution_threshold"])
    if max(config["dimensions"]) > len(indices):
        raise ValueError("configured dimension exceeds the selected clean-C3D eigenvectors")

    # 2. Reuse each saved estimate; no pose inference is performed.
    npz_paths = sorted(config["input_dir"].glob(config["npz_pattern"]), key=lambda path: path.name)
    if not npz_paths:
        raise ValueError(f"No NPZ files were found in {config['input_dir']}")
    for npz_path in npz_paths:
        with np.load(npz_path) as contents:
            world_landmarks = contents["world_landmarks"]
            landmark_names = contents["landmark_names"]
        valid = np.isfinite(world_landmarks).all(axis=(1, 2))
        ex6_data = _upright_landmarks(world_landmarks[valid])
        ex11_data = extract_pose_common_joints(world_landmarks[valid], landmark_names)
        ex11_data = upright_pose_coordinates(ex11_data)
        ex11_shape_subspaces = build_shape_subspaces(ex11_data)

        # 3. Calculate the same four score series as ex6/ex11 and only overlay their graphs.
        for dimension in config["dimensions"]:
            ex6_projected = _projected_subspaces(
                ex6_data, dimension, config["contribution_threshold"]
            )
            c3d_common_subspace = eigen_vectors[:, indices[:dimension]]
            ex11_projected = project_to_common_subspace(ex11_shape_subspaces, c3d_common_subspace)
            condition_dir = config["output_dir"] / f"d{dimension}" / f"fi{config['frame_interval']}"
            condition_dir.mkdir(parents=True, exist_ok=True)
            _save_comparisons(
                _scores(ex6_projected, config["frame_interval"]),
                _scores(ex11_projected, config["frame_interval"]),
                condition_dir,
                npz_path.stem,
                dimension,
                config["frame_interval"],
            )


if __name__ == "__main__":
    main(parse_config_path(__doc__))
