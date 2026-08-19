"""Public API for shape-subspace calculations."""

from subspace.decomposition import cal_svd, error_cal_svd, gram_schmidt, orthonormal_basis
from subspace.geodesic import along_geodesic, orth_decomposition_geodesic
from subspace.shape import (
    cal_magnitude,
    gen_shape_difference_subspace,
    gen_shape_principal_com_subspace,
    gen_shape_subspace,
)

__all__ = [
    "along_geodesic",
    "cal_magnitude",
    "cal_svd",
    "error_cal_svd",
    "gen_shape_difference_subspace",
    "gen_shape_principal_com_subspace",
    "gen_shape_subspace",
    "gram_schmidt",
    "orth_decomposition_geodesic",
    "orthonormal_basis",
]
