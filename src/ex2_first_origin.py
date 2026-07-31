"""First-order difference on the original, unprojected shape subspaces."""

from pathlib import Path

from experiments.common_subspace import build_shape_subspaces
from experiments.config import load_experiment_config, parse_config_path
from experiments.dissimilarity import calculate_difference_scores
from utils.display import display_motion_score_contribution
from utils.c3d import read_c3d

DEFAULT_CONFIG = {
    "data_path": Path("data") / "07_01.c3d",
    "output_path": Path("out") / "ex2" / "07_01_first_origin.gif",
    "frame_interval": 5,
}


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    data_path = config["data_path"]
    output_path = config["output_path"]
    frame_interval = config["frame_interval"]

    # Build the original shape subspaces and compare frames directly.
    data = read_c3d(str(data_path))
    print(f"frame num: {data.shape[0]}")
    shape_subspaces = build_shape_subspaces(data)
    magnitudes, contributions = calculate_difference_scores(shape_subspaces, frame_interval)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_motion_score_contribution(
        data,
        range(len(shape_subspaces) - frame_interval),
        magnitudes,
        contributions,
        str(output_path),
        title=data_path.name,
    )


if __name__ == "__main__":
    main(parse_config_path(__doc__))
