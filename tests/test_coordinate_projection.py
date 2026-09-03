import unittest

import numpy as np

from experiments.coordinate_projection import project_coordinate_frames, upright_pose_coordinates
from experiments.joint_correspondence import extract_c3d_common_joints, extract_pose_common_joints


class CoordinateProjectionTest(unittest.TestCase):
    def test_coordinate_projection_preserves_centroid(self):
        data = np.asarray([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
        common_subspace = np.asarray([[1.0], [0.0], [0.0]])

        projected = project_coordinate_frames(data, common_subspace)

        np.testing.assert_allclose(np.mean(projected[0], axis=0), np.mean(data[0], axis=0))

    def test_common_joint_extractors_use_the_same_order_and_interpolate(self):
        labels = [
            "S:LSHO", "S:RSHO", "S:LELB", "S:RELB", "S:LWRA", "S:LWRB",
            "S:RWRA", "S:RWRB", "S:LFWT", "S:LBWT", "S:RFWT", "S:RBWT",
            "S:LKNE", "S:RKNE", "S:LANK", "S:RANK",
        ]
        c3d = np.arange(1 * len(labels) * 3, dtype=float).reshape(1, len(labels), 3)
        c3d_joints = extract_c3d_common_joints(c3d, labels)
        np.testing.assert_allclose(c3d_joints[0, 4], np.mean(c3d[0, [4, 5]], axis=0))
        np.testing.assert_allclose(c3d_joints[0, 6], np.mean(c3d[0, [8, 9]], axis=0))

        names = np.asarray([
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
            "right_knee", "left_ankle", "right_ankle",
        ])
        pose = np.arange(1 * len(names) * 3, dtype=float).reshape(1, len(names), 3)
        np.testing.assert_array_equal(extract_pose_common_joints(pose, names), pose)

    def test_pose_coordinates_are_made_upright(self):
        points = np.asarray([[[1.0, 2.0, 3.0]]])
        np.testing.assert_array_equal(upright_pose_coordinates(points), [[[1.0, 3.0, -2.0]]])


if __name__ == "__main__":
    unittest.main()
