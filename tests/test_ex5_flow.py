import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ex5
from experiments.pose_estimation import PoseEstimate


class _FakeVideoReader:
    fps = 15.0

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def iter_sampled_frames(self, frame_stride, max_frames):
        self.arguments = frame_stride, max_frames
        yield 0, np.zeros((4, 8, 3), dtype=np.uint8)
        yield frame_stride, np.zeros((4, 8, 3), dtype=np.uint8)


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


class Ex5FlowTest(unittest.TestCase):
    @patch("ex5.save_pose_comparison_gif")
    @patch("ex5.MediaPipePoseEstimator", _FakeEstimator)
    @patch("ex5.VideoReader", _FakeVideoReader)
    def test_gahu_video_is_saved_as_npz_and_gif(self, save_gif):
        output_dir = Path("out/test_ex5_flow")
        config = {
            "input_path": Path("S001.avi"),
            "model_path": Path("pose.task"),
            "estimator": "mediapipe",
            "mmpose_pose3d_model": "human3d",
            "mmpose_pose3d_weights": None,
            "mmpose_detector": None,
            "mmpose_device": "cpu",
            "mmpose_max_input_size": 1024,
            "npz_path": output_dir / "S001.npz",
            "gif_path": output_dir / "S001.gif",
            "frame_stride": 3,
            "max_frames": None,
            "gif_fps": 10,
            "panel_size": 64,
            "min_detection_confidence": 0.5,
            "min_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "display_visibility_threshold": 0.5,
        }
        with patch("ex5.load_experiment_config", return_value=config):
            ex5.main("ignored.yaml")

        save_gif.assert_called_once()
        with np.load(output_dir / "S001.npz") as data:
            self.assertEqual(data["world_landmarks"].shape, (2, 2, 3))
            np.testing.assert_array_equal(data["frame_indices"], [0, 3])
            np.testing.assert_array_equal(data["timestamps_ms"], [0, 200])
            self.assertEqual(data["source_fps"].item(), 15.0)

    @patch("ex5.save_pose_comparison_gif")
    @patch("ex5.MMPosePoseEstimator", _FakeEstimator)
    @patch("ex5.VideoReader", _FakeVideoReader)
    def test_mmpose_backend_uses_distinct_output_names(self, save_gif):
        output_dir = Path("out/test_ex5_flow")
        config = {
            "input_path": Path("S001.avi"),
            "model_path": Path("pose.task"),
            "estimator": "mmpose",
            "mmpose_pose3d_model": "human3d",
            "mmpose_pose3d_weights": None,
            "mmpose_detector": None,
            "mmpose_device": "cpu",
            "mmpose_max_input_size": 1024,
            "npz_path": output_dir / "S001.npz",
            "gif_path": output_dir / "S001.gif",
            "frame_stride": 3,
            "max_frames": 2,
            "gif_fps": 10,
            "panel_size": 64,
            "min_detection_confidence": 0.5,
            "min_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "display_visibility_threshold": 0.5,
        }
        with patch("ex5.load_experiment_config", return_value=config):
            ex5.main("ignored.yaml")

        self.assertTrue((output_dir / "S001_mmpose.npz").is_file())
        self.assertEqual(save_gif.call_args.args[4], output_dir / "S001_mmpose.gif")


if __name__ == "__main__":
    unittest.main()
