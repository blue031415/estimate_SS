import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ex4
from experiments.pose_estimation import PoseEstimate


class _FakeVideoReader:
    fps = 25.0

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def iter_sampled_frames(self, frame_stride, max_frames):
        self.arguments = frame_stride, max_frames
        yield 0, np.zeros((4, 4, 3), dtype=np.uint8)
        yield frame_stride, np.zeros((4, 4, 3), dtype=np.uint8)


class _FakeEstimator:
    name = "fake"
    landmark_names = ("left", "right")
    connections = ((0, 1),)

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def estimate(self, rgb_frame, timestamp_ms):
        points = np.full((2, 3), timestamp_ms, dtype=float)
        return PoseEstimate(points, points, np.ones(2), np.ones(2))


class Ex4FlowTest(unittest.TestCase):
    @patch("ex4.save_pose_comparison_gif")
    @patch("ex4.MediaPipePoseEstimator", _FakeEstimator)
    @patch("ex4.VideoReader", _FakeVideoReader)
    def test_each_configured_video_is_saved_as_npz_and_gif(self, save_gif):
        output_dir = Path("out/test_ex4_flow")
        config = {
            "input_videos": {"primary": Path("one.avi"), "validation": Path("two.avi")},
            "model_path": Path("pose.task"),
            "estimator": "mediapipe",
            "mmpose_pose3d_model": "human3d",
            "mmpose_pose3d_weights": None,
            "mmpose_detector": None,
            "mmpose_device": "cpu",
            "mmpose_max_input_size": 1024,
            "output_dir": output_dir,
            "frame_stride": 5,
            "max_frames": 2,
            "gif_fps": 10,
            "panel_size": 64,
            "min_detection_confidence": 0.5,
            "min_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "display_visibility_threshold": 0.5,
        }
        with patch("ex4.load_experiment_config", return_value=config):
            ex4.main("ignored.yaml")

        self.assertEqual(save_gif.call_count, 2)
        for video_name in config["input_videos"]:
            with np.load(output_dir / f"{video_name}.npz") as data:
                self.assertEqual(data["world_landmarks"].shape, (2, 2, 3))
                np.testing.assert_array_equal(data["frame_indices"], [0, 5])
                np.testing.assert_array_equal(data["timestamps_ms"], [0, 200])
                self.assertEqual(data["estimator"].item(), "fake")


if __name__ == "__main__":
    unittest.main()
