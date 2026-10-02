"""Experiment 6 analysis: rerun numerical analyses from saved pose NPZ files."""

from pathlib import Path

import matplotlib
import numpy as np

# This script only writes GIF files and must also run on machines without Tk.
matplotlib.use("Agg")

from ex6 import _run_ex2_analyses, _upright_landmarks  # noqa: E402
from experiments.config import load_experiment_config, parse_config_path  # noqa: E402


DEFAULT_CONFIG = {
    "input_dir": Path("out/ex6"),
    "npz_pattern": "*/*.npz",
    "output_dir": Path("out/ex6"),
    "first_dimension": 4,
    "second_dimension": 5,
    "geodestic_dimension": 5,
    "frame_interval": 3,
    "contribution_threshold": 0.999,
}


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)
    input_dir = config["input_dir"]
    npz_paths = sorted(input_dir.glob(config["npz_pattern"]), key=lambda path: path.name)
    if not npz_paths:
        raise ValueError(f"No NPZ files matching {config['npz_pattern']} were found in {input_dir}")

    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for npz_path in npz_paths:
        # 1. Load the frame-aligned skeleton estimated by ex6.
        with np.load(npz_path) as contents:
            world_landmarks = contents["world_landmarks"]

        # 2. Exclude failed detections from SVD and rotate only the display axes.
        valid_frames = np.isfinite(world_landmarks).all(axis=(1, 2))
        data = _upright_landmarks(world_landmarks[valid_frames])
        print(f"[{npz_path.stem}] detected {len(data)}/{len(world_landmarks)} sampled frames")

        # 3. Run each configured frame interval without repeating pose inference.
        frame_intervals = config["frame_interval"]
        if isinstance(frame_intervals, int):
            frame_intervals = [frame_intervals]
        for frame_interval in frame_intervals:
            interval_config = dict(config)
            interval_config["frame_interval"] = frame_interval
            video_output_dir = output_dir / npz_path.stem
            _run_ex2_analyses(
                data, video_output_dir, npz_path.stem, interval_config, npz_path.name
            )


if __name__ == "__main__":
    main(parse_config_path(__doc__))
