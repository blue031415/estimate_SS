import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ex6_pose


class Ex6PoseFlowTest(unittest.TestCase):
    @patch("ex6_pose.save_pose_comparison_gif")
    @patch("ex6_pose._decode_saved_frames")
    def test_saved_pose_can_be_redrawn_with_90_degree_rotation(self, decode, save_gif):
        input_npz = Path("out/test_ex6_pose/motion.npz")
        output_path = Path("out/test_ex6_pose/motion_pose_rotated_90.gif")
        input_npz.parent.mkdir(parents=True, exist_ok=True)
        world_landmarks = np.ones((2, 3, 3))
        visibility = np.ones((2, 3))
        connections = np.asarray([[0, 1], [1, 2]])
        frame_indices = np.asarray([0, 3])
        np.savez_compressed(
            input_npz,
            world_landmarks=world_landmarks,
            visibility=visibility,
            connections=connections,
            frame_indices=frame_indices,
            source_video="source.mp4",
            estimator="fake_estimator",
        )
        rgb_frames = [np.zeros((4, 8, 3), dtype=np.uint8)] * 2
        decode.return_value = rgb_frames
        config = dict(ex6_pose.DEFAULT_CONFIG)
        config.update(
            {
                "input_npz": input_npz,
                "output_path": output_path,
                "rotate_pose_90_degrees": True,
            }
        )

        with patch("ex6_pose.load_experiment_config", return_value=config):
            ex6_pose.main("ignored.yaml")

        decode.assert_called_once()
        self.assertEqual(decode.call_args.args[0], Path("source.mp4"))
        np.testing.assert_array_equal(decode.call_args.args[1], frame_indices)
        self.assertIs(save_gif.call_args.args[0], rgb_frames)
        np.testing.assert_array_equal(save_gif.call_args.args[1], world_landmarks)
        self.assertEqual(save_gif.call_args.args[4], output_path)
        self.assertEqual(save_gif.call_args.kwargs["azimuth_rotation_degrees"], 90)

    @patch("ex6_pose.VideoReader")
    def test_only_saved_source_frames_are_decoded(self, video_reader):
        reader = video_reader.return_value.__enter__.return_value
        reader.iter_sampled_frames.return_value = [
            (index, np.full((1, 1, 3), index, dtype=np.uint8)) for index in range(7)
        ]

        frames = ex6_pose._decode_saved_frames(Path("source.mp4"), [0, 3, 6])

        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [0, 3, 6])


if __name__ == "__main__":
    unittest.main()
