import unittest

import numpy as np

import subspace


class SubspacePackageTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.first = np.linalg.qr(rng.normal(size=(7, 3)))[0]
        self.middle = np.linalg.qr(rng.normal(size=(7, 3)))[0]
        self.last = np.linalg.qr(rng.normal(size=(7, 3)))[0]

    def test_public_api_is_available_from_package(self):
        expected = {
            "cal_svd",
            "gen_shape_subspace",
            "gen_shape_difference_subspace",
            "gen_shape_principal_com_subspace",
            "cal_magnitude",
            "orthonormal_basis",
            "along_geodesic",
            "orth_decomposition_geodesic",
        }

        self.assertTrue(expected.issubset(set(subspace.__all__)))
        for name in expected:
            self.assertTrue(callable(getattr(subspace, name)))

    def test_along_geodesic_matches_original_sequence(self):
        neighbor_span = subspace.orthonormal_basis(
            np.concatenate([self.first, self.last], axis=1)
        )
        projected = neighbor_span @ (neighbor_span.T @ self.middle)
        projected = subspace.orthonormal_basis(projected)
        principal = subspace.gen_shape_principal_com_subspace(self.first, self.last)
        expected_magnitude = subspace.cal_magnitude(principal, projected)
        difference = subspace.gen_shape_difference_subspace(principal, projected)
        contribution_basis = difference @ (difference.T @ self.middle)
        contribution_basis = subspace.orthonormal_basis(contribution_basis)
        expected_contribution = np.sum(np.square(contribution_basis), axis=1)

        magnitude, contribution = subspace.along_geodesic(
            self.first, self.middle, self.last
        )

        self.assertAlmostEqual(magnitude, expected_magnitude)
        np.testing.assert_allclose(contribution, expected_contribution)

    def test_shape_subspace_remains_three_dimensional(self):
        rng = np.random.default_rng(7)
        points = rng.normal(size=(10, 3))

        result = subspace.gen_shape_subspace(points)

        self.assertEqual(result.shape, (10, 3))
        np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
