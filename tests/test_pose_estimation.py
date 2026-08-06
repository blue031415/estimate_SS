import unittest

import numpy as np

from experiments.pose_estimation import _mmpose_prediction_to_estimate


class PoseEstimationTest(unittest.TestCase):
    def test_mmpose_coordinates_are_converted_for_common_display(self):
        points = np.zeros((17, 3))
        points[0] = [1.0, 2.0, 3.0]
        scores = np.linspace(0.1, 0.9, 17)

        estimate = _mmpose_prediction_to_estimate(
            {"keypoints": points.tolist(), "keypoint_scores": scores.tolist()}
        )

        np.testing.assert_array_equal(estimate.world_landmarks[0], [1.0, -3.0, 2.0])
        self.assertTrue(np.isnan(estimate.image_landmarks).all())
        np.testing.assert_allclose(estimate.visibility, scores)
        np.testing.assert_allclose(estimate.presence, scores)

    def test_mmpose_rejects_non_human36m_output(self):
        with self.assertRaisesRegex(ValueError, "17, 3"):
            _mmpose_prediction_to_estimate({"keypoints": np.zeros((33, 3))})


if __name__ == "__main__":
    unittest.main()
