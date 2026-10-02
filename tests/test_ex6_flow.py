import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ex6


class Ex6FlowTest(unittest.TestCase):
    def test_world_landmarks_are_rotated_upright_for_ex2_plots(self):
        landmarks = np.asarray([[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]])

        upright = ex6._upright_landmarks(landmarks)

        np.testing.assert_array_equal(upright, [[[1.0, 3.0, -2.0], [-1.0, -3.0, 2.0]]])

    @patch("ex6._run_ex2_analyses")
    @patch("ex6.save_pose_comparison_gif")
    @patch("ex6._estimate_video")
    def test_each_video_is_saved_then_analyzed(self, estimate_video, save_gif, run_analyses):
        output_dir = Path("out/test_ex6_flow")
        input_dir = Path("data/self_videos")
        landmark_count = 6
        frame_count = 12
        world_landmarks = np.ones((frame_count, landmark_count, 3))
        world_landmarks[1] = np.nan
        estimate_video.return_value = {
            "rgb_frames": [np.zeros((4, 8, 3), dtype=np.uint8)] * frame_count,
            "frame_indices": np.arange(frame_count) * 3,
            "timestamps_ms": np.arange(frame_count) * 100,
            "world_landmarks": world_landmarks,
            "image_landmarks": world_landmarks.copy(),
            "visibility": np.ones((frame_count, landmark_count)),
            "presence": np.ones((frame_count, landmark_count)),
            "landmark_names": np.asarray([f"joint_{index}" for index in range(landmark_count)]),
            "connections": np.asarray([[0, 1]]),
            "source_fps": 30.0,
            "estimator_name": "fake",
        }
        config = dict(ex6.DEFAULT_CONFIG)
        config.update(
            {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "video_pattern": "*.mp4",
            }
        )

        with patch("ex6.load_experiment_config", return_value=config):
            ex6.main("ignored.yaml")

        video_count = len(list(input_dir.glob(config["video_pattern"])))
        self.assertEqual(estimate_video.call_count, video_count)
        self.assertEqual(save_gif.call_count, video_count)
        for call in save_gif.call_args_list:
            self.assertEqual(call.kwargs["azimuth_rotation_degrees"], 0)
            self.assertTrue(str(call.args[4]).endswith("_pose.gif"))
        self.assertEqual(run_analyses.call_count, video_count)
        for call in run_analyses.call_args_list:
            self.assertEqual(call.args[0].shape, (frame_count - 1, landmark_count, 3))
        for video_path in input_dir.glob(config["video_pattern"]):
            with np.load(output_dir / video_path.stem / f"{video_path.stem}.npz") as data:
                self.assertEqual(data["world_landmarks"].shape, (frame_count, landmark_count, 3))
                self.assertTrue(np.isnan(data["world_landmarks"][1]).all())

    @patch("ex6.display_motion_some_score")
    @patch("ex6.display_motion_score_contribution")
    @patch("ex6.calculate_geodestic")
    @patch("ex6.calculate_second_order_difference_DS")
    @patch("ex6.calculate_difference_scores")
    @patch("ex6._projected_subspaces")
    def test_runs_all_ex2_variants(
        self,
        projected_subspaces,
        first_scores,
        second_scores,
        geodestic_scores,
        display_contribution,
        display_some_score,
    ):
        data = np.ones((12, 6, 3))
        projected_subspaces.return_value = [np.eye(6, 3)] * len(data)
        first_scores.return_value = [0.0] * 7, [np.ones(6)] * 7
        second_scores.return_value = [0.0] * 2, [np.ones(6)] * 2
        geodestic_scores.return_value = (
            [0.0] * 2,
            [np.ones(6)] * 2,
            [0.0] * 2,
            [np.ones(6)] * 2,
        )

        ex6._run_ex2_analyses(
            data, Path("out/ex6/video"), "video", ex6.DEFAULT_CONFIG, "video.mp4"
        )

        self.assertEqual(projected_subspaces.call_count, 3)
        self.assertEqual(first_scores.call_count, 2)
        self.assertIs(first_scores.call_args_list[1].args[0], projected_subspaces.return_value)
        self.assertEqual(second_scores.call_count, 2)
        self.assertEqual(geodestic_scores.call_count, 2)
        self.assertEqual(display_contribution.call_count, 8)
        self.assertEqual(
            Path(display_contribution.call_args_list[0].args[4]),
            Path("out/ex6/video/fi3/origin/video_first_origin.gif"),
        )
        self.assertIn("frame interval=3", display_contribution.call_args_list[0].kwargs["title"])
        self.assertEqual(
            Path(display_contribution.call_args_list[4].args[4]),
            Path("out/ex6/video/fi3/d4/video_first.gif"),
        )
        self.assertEqual(display_some_score.call_count, 2)


if __name__ == "__main__":
    unittest.main()
