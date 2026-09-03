"""Experiment 11: analyze video differences in a clean-C3D common subspace."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

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
from utils.display import display_motion_score_contribution, display_motion_some_score  # noqa: E402


DEFAULT_CONFIG = {
    "c3d_path": Path("data/07_01.c3d"),
    "input_dir": Path("out/ex6"),
    "npz_pattern": "*.npz",
    "output_dir": Path("out/ex11"),
    "dimension": 5,
    "frame_interval": 5,
    "contribution_threshold": 0.999,
}


def _analyze_video(data, subspaces, output_dir, output_stem, config, output_tag, analysis_label):
    frame_interval = config["frame_interval"]
    plot_title = f"{output_stem}, frame interval={frame_interval}"

    # 1. First-order differences for the supplied original or projected subspaces.
    magnitudes, contributions = calculate_difference_scores(subspaces, frame_interval)
    display_motion_score_contribution(
        data,
        range(len(subspaces) - frame_interval),
        magnitudes,
        contributions,
        str(output_dir / f"{output_stem}_first_{output_tag}.gif"),
        title=plot_title,
    )

    # 2. Second-order differences after the same projection.
    analysis_frames = range(frame_interval, len(subspaces) - frame_interval)
    magnitudes, contributions = calculate_second_order_difference_DS(subspaces, frame_interval)
    display_motion_score_contribution(
        data,
        analysis_frames,
        magnitudes,
        contributions,
        str(output_dir / f"{output_stem}_second_{output_tag}.gif"),
        title=plot_title,
    )

    # 3. Along/orthogonal geodestic components after the same projection.
    magnitudes_along, contributions_along, magnitudes_orth, contributions_orth = calculate_geodestic(
        subspaces, frame_interval
    )
    display_motion_score_contribution(
        data,
        analysis_frames,
        magnitudes_along,
        contributions_along,
        str(output_dir / f"{output_stem}_along_geodestic_{output_tag}.gif"),
        title=plot_title,
    )
    display_motion_score_contribution(
        data,
        analysis_frames,
        magnitudes_orth,
        contributions_orth,
        str(output_dir / f"{output_stem}_orth_geodestic_{output_tag}.gif"),
        title=plot_title,
    )
    display_motion_some_score(
        data,
        analysis_frames,
        [magnitudes_along, magnitudes_orth],
        [
            f"along geodesic({analysis_label}, interval={frame_interval})",
            f"orth to geodesic({analysis_label}, interval={frame_interval})",
        ],
        str(output_dir / f"{output_stem}_geodestic_{output_tag}.gif"),
        title=plot_title,
    )


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)

    # 1. Build one common subspace from corresponding joints in clean C3D data.
    c3d_data, c3d_labels = read_c3d(str(config["c3d_path"]), return_labels=True)
    c3d_common_joints = extract_c3d_common_joints(c3d_data, c3d_labels)
    c3d_subspaces = build_shape_subspaces(c3d_common_joints)
    projection_matrix = average_projection_matrix(c3d_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)
    indices, _, cumulative_ratio = select_eigen_indices(
        eigen_values, threshold=config["contribution_threshold"]
    )
    dimensions = config["dimension"]
    if isinstance(dimensions, int):
        dimensions = [dimensions]
    if max(dimensions) > len(indices):
        raise ValueError(f"dimension {max(dimensions)} exceeds {len(indices)} selected clean-C3D eigenvectors")
    print(f"clean C3D selected indices: {indices}")
    print(f"clean C3D cumulative contribution: {cumulative_ratio[indices[-1]]}")

    # 2. Extract the same joints from each pose NPZ in the same marker order.
    npz_paths = sorted(config["input_dir"].glob(config["npz_pattern"]), key=lambda path: path.name)
    if not npz_paths:
        raise ValueError(f"No NPZ files were found in {config['input_dir']}")
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for npz_path in npz_paths:
        with np.load(npz_path) as contents:
            world_landmarks = contents["world_landmarks"]
            landmark_names = contents["landmark_names"]
        valid = np.isfinite(world_landmarks).all(axis=(1, 2))
        pose_joints = extract_pose_common_joints(world_landmarks[valid], landmark_names)
        pose_joints = upright_pose_coordinates(pose_joints)
        if len(pose_joints) <= 2 * config["frame_interval"]:
            raise ValueError(f"Not enough valid frames in {npz_path}")

        # 3. Analyze the original video shape subspaces without dimensional reduction.
        pose_subspaces = build_shape_subspaces(pose_joints)
        origin_output_dir = output_dir / "origin" / f"fi{config['frame_interval']}"
        origin_output_dir.mkdir(parents=True, exist_ok=True)
        _analyze_video(
            pose_joints,
            pose_subspaces,
            origin_output_dir,
            npz_path.stem,
            config,
            f"origin_fi{config['frame_interval']}",
            "origin",
        )

        # 4. Project video shape subspaces into the clean-C3D common subspace.
        for dimension in dimensions:
            c3d_common_subspace = eigen_vectors[:, indices[:dimension]]
            projected_subspaces = project_to_common_subspace(pose_subspaces, c3d_common_subspace)

            # 5. Analyze the projected subspaces and write distinct C3D-tagged outputs.
            print(f"[{npz_path.stem}] analyzing {len(pose_joints)} valid frames, D={dimension}")
            condition_output_dir = output_dir / "c3d" / f"d{dimension}" / f"fi{config['frame_interval']}"
            condition_output_dir.mkdir(parents=True, exist_ok=True)
            _analyze_video(
                pose_joints, projected_subspaces, condition_output_dir, npz_path.stem, config,
                f"c3d_d{dimension}_fi{config['frame_interval']}", f"C3D, D={dimension}",
            )


if __name__ == "__main__":
    main(parse_config_path(__doc__))
