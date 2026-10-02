"""Experiment 6 pose GIF: redraw a saved pose estimate without inference."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from experiments.config import load_experiment_config, parse_config_path  # noqa: E402
from utils.pose_display import save_pose_comparison_gif  # noqa: E402
from utils.video import VideoReader  # noqa: E402


DEFAULT_CONFIG = {
    "input_npz": Path("out/ex6/歩行動画奥から手前２/歩行動画奥から手前２.npz"),
    "output_path": Path(
        "out/ex6/歩行動画奥から手前２/歩行動画奥から手前２_pose_rotated_90.gif"
    ),
    "rotate_pose_90_degrees": False,
    "gif_fps": 10,
    "panel_size": 512,
    "display_visibility_threshold": 0.5,
}


def _decode_saved_frames(video_path, frame_indices):
    expected_indices = [int(index) for index in frame_indices]
    frames = []
    next_expected = 0
    with VideoReader(video_path) as video:
        for frame_index, rgb_frame in video.iter_sampled_frames():
            if next_expected >= len(expected_indices):
                break
            expected = expected_indices[next_expected]
            if frame_index == expected:
                frames.append(rgb_frame)
                next_expected += 1
            elif frame_index > expected:
                raise ValueError(f"Source video is missing saved frame {expected}")
    if next_expected != len(expected_indices):
        missing = expected_indices[next_expected]
        raise ValueError(f"Source video ended before saved frame {missing}")
    return frames


def main(config_path=None):
    config = load_experiment_config(DEFAULT_CONFIG, config_path)

    # 1. Load the frame-aligned pose estimate created by Experiment 6.
    with np.load(config["input_npz"]) as contents:
        world_landmarks = contents["world_landmarks"]
        visibility = contents["visibility"]
        connections = contents["connections"]
        frame_indices = contents["frame_indices"]
        source_video = Path(str(contents["source_video"]))
        estimator_label = str(contents["estimator"])

    # 2. Decode exactly the source frames recorded in the NPZ.
    rgb_frames = _decode_saved_frames(source_video, frame_indices)

    # 3. Render the standard view or the additive 90-degree rotated view.
    rotation_degrees = 90 if config["rotate_pose_90_degrees"] else 0
    save_pose_comparison_gif(
        rgb_frames,
        world_landmarks,
        visibility,
        connections,
        config["output_path"],
        output_fps=config["gif_fps"],
        panel_size=config["panel_size"],
        visibility_threshold=config["display_visibility_threshold"],
        estimator_label=estimator_label,
        azimuth_rotation_degrees=rotation_degrees,
    )
    print(f"saved: {config['output_path']}")


if __name__ == "__main__":
    main(parse_config_path(__doc__))
