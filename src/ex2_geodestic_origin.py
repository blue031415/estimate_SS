"""Geodesic decomposition on the original, unprojected shape subspaces."""

from pathlib import Path

from experiments.common_subspace import build_shape_subspaces
from experiments.config import load_experiment_config, parse_config_path
from experiments.dissimilarity import calculate_geodestic
from utils.display import display_motion_score_contribution, display_motion_some_score
from utils.c3d import read_c3d

DEFAULT_CONFIG = {
    "data_path": Path("data") / "07_01.c3d",
    "output_path_along": Path("out") / "ex2" / "07_01_along_geodestic_origin.gif",
    "output_path_orth": Path("out") / "ex2" / "07_01_orth_geodestic_origin.gif",
    "output_path": Path("out") / "ex2" / "07_01_geodestic_origin.gif",
    "frame_interval": 5,
}


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    data_path = config["data_path"]
    frame_interval = config["frame_interval"]

    # Decompose changes in the original shape subspaces.
    data = read_c3d(str(data_path))
    print(f"frame num: {data.shape[0]}")
    shape_subspaces = build_shape_subspaces(data)
    magnitudes_along, contributions_along, magnitudes_orth, contributions_orth = calculate_geodestic(
        shape_subspaces, frame_interval
    )
    frames = range(frame_interval, len(shape_subspaces) - frame_interval)

    output_path_along = config["output_path_along"]
    output_path_orth = config["output_path_orth"]
    output_path = config["output_path"]
    output_path_along.parent.mkdir(parents=True, exist_ok=True)
    display_motion_score_contribution(
        data,
        frames,
        magnitudes_along,
        contributions_along,
        str(output_path_along),
        title=data_path.name,
    )
    output_path_orth.parent.mkdir(parents=True, exist_ok=True)
    display_motion_score_contribution(
        data,
        frames,
        magnitudes_orth,
        contributions_orth,
        str(output_path_orth),
        title=data_path.name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_motion_some_score(
        data,
        frames,
        [magnitudes_along, magnitudes_orth],
        ["along geodesic", "orth to geodesic"],
        str(output_path),
        title=data_path.name,
    )


if __name__ == "__main__":
    main(parse_config_path(__doc__))
