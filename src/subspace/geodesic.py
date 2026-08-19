"""Geodesic and orthogonal decomposition between shape subspaces."""

import numpy as np

from subspace.decomposition import orthonormal_basis
from subspace.shape import (
    cal_magnitude,
    gen_shape_difference_subspace,
    gen_shape_principal_com_subspace,
)


def _project_middle_to_neighbor_span(first, middle, last):
    neighbor_span = orthonormal_basis(np.concatenate([first, last], axis=1))
    projected_middle = neighbor_span @ (neighbor_span.T @ middle)
    return orthonormal_basis(projected_middle)


def _contribution(reference, projected, middle):
    difference = gen_shape_difference_subspace(reference, projected)
    basis = difference @ (difference.T @ middle)
    basis = orthonormal_basis(basis)
    return np.sum(np.square(basis), axis=1)


def along_geodesic(first, middle, last):
    projected_middle = _project_middle_to_neighbor_span(first, middle, last)
    principal = gen_shape_principal_com_subspace(first, last)
    magnitude = cal_magnitude(principal, projected_middle)
    contribution = _contribution(principal, projected_middle, middle)
    return magnitude, contribution


def orth_decomposition_geodesic(first, middle, last):
    projected_middle = _project_middle_to_neighbor_span(first, middle, last)
    magnitude = cal_magnitude(middle, projected_middle)
    contribution = _contribution(middle, projected_middle, middle)
    return magnitude, contribution
