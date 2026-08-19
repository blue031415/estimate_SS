import unittest
from unittest.mock import patch

import numpy as np

from experiments.noise import add_noise_to_markers
from utils.c3d import remove_nan


class DataProcessingTest(unittest.TestCase):
    def test_remove_nan_keeps_original_truncation_behavior(self):
        data = np.arange(24, dtype=float).reshape(4, 3, 2)
        data[:, 2, 0] = np.nan

        result = remove_nan(data)

        expected = np.array([data[:, :2, frame] for frame in range(2)]).transpose(1, 2, 0)
        np.testing.assert_array_equal(result, expected)

    @patch("experiments.noise.np.random.multivariate_normal")
    def test_noise_generation_is_applied_once_per_selected_frame(self, generate_noise):
        generate_noise.return_value = np.ones((1, 3))
        data = np.zeros((4, 2, 3))

        result = add_noise_to_markers(
            data,
            indices=[1],
            start_frame=1,
            end_frame=3,
            scale=2,
            mean=[0, 0, 0],
            covariance=np.eye(3),
        )

        np.testing.assert_array_equal(result[0], np.zeros((2, 3)))
        np.testing.assert_array_equal(result[1:3, 1, :], np.full((2, 3), 2.0))
        np.testing.assert_array_equal(result[3], np.zeros((2, 3)))
        self.assertEqual(generate_noise.call_count, 2)


if __name__ == "__main__":
    unittest.main()
