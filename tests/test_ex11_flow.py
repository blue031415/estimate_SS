import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ex11


class Ex11FlowTest(unittest.TestCase):
    @patch("ex11._analyze_video")
    @patch("ex11.project_to_common_subspace")
    @patch("ex11.select_eigen_indices")
    @patch("ex11.eigendecompose_descending")
    @patch("ex11.average_projection_matrix")
    @patch("ex11.build_shape_subspaces")
    @patch("ex11.extract_pose_common_joints")
    @patch("ex11.extract_c3d_common_joints")
    @patch("ex11.read_c3d")
    def test_clean_c3d_subspace_is_used_for_each_pose_npz(
        self, read_c3d, extract_c3d, extract_pose, build_subspaces,
        average_projection, eigendecompose, select_indices, project_subspaces, analyze,
    ):
        input_dir = Path("out/test_ex11_input")
        output_dir = Path("out/test_ex11_output")
        video_input_dir = input_dir / "motion"
        video_input_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            video_input_dir / "motion.npz",
            world_landmarks=np.ones((12, 33, 3)),
            landmark_names=np.asarray([f"joint_{index}" for index in range(33)]),
        )
        c3d_subspaces = [np.eye(12, 3)] * 20
        pose_subspaces = [np.eye(12, 3)] * 12
        projected = [np.eye(12, 3)] * 12
        read_c3d.return_value = np.ones((20, 16, 3)), [f"M{index}" for index in range(16)]
        extract_c3d.return_value = np.ones((20, 12, 3))
        extract_pose.return_value = np.ones((12, 12, 3))
        build_subspaces.side_effect = [c3d_subspaces, pose_subspaces]
        average_projection.return_value = np.eye(12)
        eigendecompose.return_value = np.ones(12), np.eye(12)
        select_indices.return_value = np.arange(6), None, np.linspace(0.2, 1.0, 12)
        project_subspaces.return_value = projected
        config = dict(ex11.DEFAULT_CONFIG)
        config.update(
            {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "dimension": [4, 5, 6],
                "frame_interval": [2, 3],
            }
        )

        with patch("ex11.load_experiment_config", return_value=config):
            ex11.main("ignored.yaml")

        read_c3d.assert_called_once_with(str(config["c3d_path"]), return_labels=True)
        self.assertEqual(project_subspaces.call_count, 6)
        self.assertIs(project_subspaces.call_args.args[0], pose_subspaces)
        np.testing.assert_array_equal(project_subspaces.call_args.args[1], np.eye(12)[:, :6])
        self.assertEqual(analyze.call_count, 8)
        self.assertIs(analyze.call_args_list[0].args[1], pose_subspaces)
        self.assertEqual(analyze.call_args_list[0].args[5], "origin")
        self.assertEqual(
            analyze.call_args_list[0].args[2], output_dir / "motion" / "fi2" / "origin"
        )
        self.assertIs(analyze.call_args_list[1].args[1], projected)
        self.assertEqual(analyze.call_args_list[1].args[5], "c3d")
        self.assertEqual(
            analyze.call_args_list[1].args[2], output_dir / "motion" / "fi2" / "d4"
        )
        self.assertEqual(
            analyze.call_args_list[4].args[2], output_dir / "motion" / "fi3" / "origin"
        )

    @patch("ex11.display_motion_some_score")
    @patch("ex11.display_motion_score_contribution")
    @patch("ex11.calculate_geodestic")
    @patch("ex11.calculate_second_order_difference_DS")
    @patch("ex11.calculate_difference_scores")
    def test_all_ex6_difference_variants_are_generated(
        self, first, second, geodestic, display_contribution, display_some
    ):
        data = np.ones((12, 12, 3))
        subspaces = [np.eye(12, 3)] * 12
        first.return_value = [0.0] * 7, [np.ones(12)] * 7
        second.return_value = [0.0] * 2, [np.ones(12)] * 2
        geodestic.return_value = [0.0] * 2, [np.ones(12)] * 2, [0.0] * 2, [np.ones(12)] * 2

        ex11._analyze_video(
            data, subspaces, Path("out/ex11"), "motion", ex11.DEFAULT_CONFIG, "origin", "origin"
        )

        first.assert_called_once_with(subspaces, 5)
        second.assert_called_once_with(subspaces, 5)
        geodestic.assert_called_once_with(subspaces, 5)
        self.assertEqual(display_contribution.call_count, 4)
        self.assertIn("frame interval=5", display_contribution.call_args_list[0].kwargs["title"])
        self.assertEqual(
            Path(display_contribution.call_args_list[0].args[4]),
            Path("out/ex11/motion_first_origin.gif"),
        )
        display_some.assert_called_once()


if __name__ == "__main__":
    unittest.main()
