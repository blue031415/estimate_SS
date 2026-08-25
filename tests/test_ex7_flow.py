import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ex7


class Ex7FlowTest(unittest.TestCase):
    @patch("ex7.plot_eigen_values")
    @patch("ex7.cal_magnitude", return_value=0.25)
    @patch("ex7.project_to_common_subspace")
    @patch("ex7.select_eigen_indices")
    @patch("ex7.eigendecompose_descending")
    @patch("ex7.average_projection_matrix")
    @patch("ex7.build_shape_subspaces")
    def test_each_ex6_npz_is_eigendecomposed_and_plotted(
        self,
        build_subspaces,
        average_projection,
        eigendecompose,
        select_indices,
        project_subspaces,
        cal_magnitude,
        plot_eigen_values,
    ):
        input_dir = Path("out/test_ex7_input")
        output_dir = Path("out/test_ex7_output")
        input_dir.mkdir(parents=True, exist_ok=True)
        points = np.ones((3, 4, 3))
        points[1] = np.nan
        for stem in ("first", "second"):
            np.savez_compressed(input_dir / f"{stem}.npz", world_landmarks=points)

        subspaces = [np.eye(4, 3), np.eye(4, 3)]
        build_subspaces.return_value = subspaces
        average_projection.return_value = np.eye(4)
        eigendecompose.return_value = np.asarray([1.0, 0.1]), np.eye(2)
        select_indices.return_value = np.asarray([0]), None, np.asarray([0.999])
        project_subspaces.return_value = subspaces
        config = {
            "input_dir": input_dir,
            "npz_pattern": "*.npz",
            "output_dir": output_dir,
            "contribution_threshold": 0.999,
        }

        with patch("ex7.load_experiment_config", return_value=config):
            ex7.main("ignored.yaml")

        self.assertEqual(build_subspaces.call_count, 2)
        for call in build_subspaces.call_args_list:
            self.assertEqual(call.args[0].shape, (2, 4, 3))
        self.assertEqual(eigendecompose.call_count, 2)
        self.assertEqual(plot_eigen_values.call_count, 2)
        self.assertEqual(cal_magnitude.call_count, 4)
        output_names = {call.args[1].name for call in plot_eigen_values.call_args_list}
        self.assertEqual(output_names, {"first_eigen_val.png", "second_eigen_val.png"})


if __name__ == "__main__":
    unittest.main()
