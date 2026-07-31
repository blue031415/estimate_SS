"""Second-order differences after projection into a common shape subspace."""

from pathlib import Path

from experiments.analysis import (
    average_projection_matrix,
    build_shape_subspaces,
    calculate_second_order_difference_DS,
    eigendecompose_descending,
    project_to_common_subspace,
    select_eigen_indices,
)
from utils.display import display_motion_score_contribution
from utils.experiment_config import load_experiment_config, parse_config_path
from utils.readc3d import read_c3d

DEFAULT_CONFIG = {
    "data_path": Path("data") / "07_01.c3d",
    "dimension": 5,
    "output_path": Path("out") / "ex2" / "07_01_second_d5.gif",
    "frame_interval": 5,
    "contribution_threshold": 0.999,
}


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    data_path = config["data_path"]
    output_path = config["output_path"]
    dimension = config["dimension"]
    frame_interval = config["frame_interval"]

    print(f"Dimention of common subspace: {dimension}")
    data = read_c3d(str(data_path), return_labels=False)
    print(f"frame num: {data.shape[0]}")  # type: ignore

    # Build the common subspace from all frames.
    shape_subspaces = build_shape_subspaces(data)
    projection_matrix = average_projection_matrix(shape_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)

    indices, _, cumulative_ratio = select_eigen_indices(eigen_values, threshold=config["contribution_threshold"])
    print(f"selected indices: {indices}")
    print(f"cumulative contribution: {cumulative_ratio[indices[-1]]}")

    # Project into the configured dimension and calculate second-order differences.
    common_subspace = eigen_vectors[:, indices[:dimension]]
    projected_subspaces = project_to_common_subspace(shape_subspaces, common_subspace)
    magnitudes, contributions = calculate_second_order_difference_DS(projected_subspaces, frame_interval)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_motion_score_contribution(
        data,
        range(frame_interval, len(projected_subspaces) - frame_interval),
        magnitudes,
        contributions,
        str(output_path),
        title=data_path.name,
    )


if __name__ == "__main__":
    main(parse_config_path(__doc__))
