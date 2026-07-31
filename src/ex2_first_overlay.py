"""Compare first-order differences before and after common-subspace projection."""

from pathlib import Path

from experiments.analysis import (
    average_projection_matrix,
    build_shape_subspaces,
    calculate_difference_scores,
    eigendecompose_descending,
    project_to_common_subspace,
    select_eigen_indices,
)
from utils.display import display_motion_some_score
from utils.experiment_config import load_experiment_config, parse_config_path
from utils.readc3d import read_c3d

DEFAULT_CONFIG = {
    "data_path": Path("data") / "07_01.c3d",
    "dimensions": [4, 5, 6],
    "output_path": Path("out") / "ex2" / "07_01_first_d_456.gif",
    "frame_interval": 5,
    "contribution_threshold": 0.999,
}


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    data_path = config["data_path"]
    output_path = config["output_path"]
    frame_interval = config["frame_interval"]

    # Establish the common-subspace candidates shared by every comparison.
    data = read_c3d(str(data_path))
    print(f"frame num: {data.shape[0]}")
    shape_subspaces = build_shape_subspaces(data)
    projection_matrix = average_projection_matrix(shape_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)
    indices, _, cumulative_ratio = select_eigen_indices(eigen_values, threshold=config["contribution_threshold"])
    print(f"selected indices: {indices}")
    print(f"cumulative contribution: {cumulative_ratio[indices[-1]]}")

    # Compare the original score with every configured projection dimension.
    magnitudes_origin, _ = calculate_difference_scores(shape_subspaces, frame_interval)
    magnitudes_list = {"origin": magnitudes_origin}
    for dimension in config["dimensions"]:
        common_subspace = eigen_vectors[:, indices[:dimension]]
        projected_subspaces = project_to_common_subspace(shape_subspaces, common_subspace)
        magnitudes, _ = calculate_difference_scores(projected_subspaces, frame_interval)
        magnitudes_list[f"D = {dimension}"] = magnitudes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_motion_some_score(
        data,
        range(len(shape_subspaces) - frame_interval),
        list(magnitudes_list.values()),
        list(magnitudes_list.keys()),
        str(output_path),
        title=data_path.name,
    )


if __name__ == "__main__":
    main(parse_config_path(__doc__))
