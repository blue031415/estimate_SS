import unittest
from unittest.mock import patch

import numpy as np

import ex3


class Ex3FlowTest(unittest.TestCase):
    @patch("ex3.plot_eigen_values")
    @patch("ex3.cal_magnitude", return_value=0.25)
    @patch("ex3.project_to_common_subspace")
    @patch("ex3.select_eigen_indices")
    @patch("ex3.eigendecompose_descending")
    @patch("ex3.average_projection_matrix")
    @patch("ex3.build_shape_subspaces")
    @patch("ex3.add_noise_to_markers")
    @patch("ex3.marker_indices", return_value=[0])
    @patch("ex3.read_c3d")
    def test_yaml_values_are_used_by_the_experiment_script(
        self,
        read_c3d,
        marker_indices,
        add_noise,
        build_subspaces,
        average_projection,
        eigendecompose,
        select_indices,
        project_subspaces,
        cal_magnitude,
        plot_eigen_values,
    ):
        data = np.zeros((3, 2, 3))
        labels = ["subject:LSHO", "subject:RSHO"]
        subspaces = [np.eye(2) for _ in range(3)]
        read_c3d.return_value = data, labels
        add_noise.return_value = data
        build_subspaces.return_value = subspaces
        average_projection.return_value = np.eye(2)
        eigendecompose.return_value = np.array([1.0, 0.1]), np.eye(2)
        select_indices.return_value = np.array([0]), None, np.array([0.999])
        project_subspaces.return_value = subspaces

        ex3.main("configs/ex3.yaml")

        marker_indices.assert_called_once_with(labels, ["left_arm"])
        add_noise.assert_called_once_with(
            data,
            [0],
            150,
            250,
            30,
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        select_indices.assert_called_once()
        self.assertEqual(select_indices.call_args.kwargs["threshold"], 0.999)
        plot_eigen_values.assert_called_once()
        self.assertEqual(cal_magnitude.call_count, 3)


if __name__ == "__main__":
    unittest.main()
