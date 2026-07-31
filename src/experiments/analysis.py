import numpy as np
from scipy.linalg import eigh

from subspace.utils import (
    along_geodesic,
    cal_magnitude,
    gen_shape_difference_subspace,
    gen_shape_principal_com_subspace,
    gen_shape_subspace,
    orth_decomposition_geodesic,
    orthonormal_basis,
)


def select_eigen_indices(eigen_values, threshold=0.95):
    contribution_ratio = eigen_values / np.sum(eigen_values)
    cumulative_ratio = np.cumsum(contribution_ratio)
    dim = np.searchsorted(cumulative_ratio, threshold) + 1
    return np.arange(dim), contribution_ratio, cumulative_ratio


def build_shape_subspaces(data):
    return [gen_shape_subspace(data[frame, :, :]) for frame in range(data.shape[0])]


def average_projection_matrix(subspaces):
    projection_sum = np.zeros((subspaces[0].shape[0], subspaces[0].shape[0]))
    for subspace in subspaces:
        projection_sum += subspace @ subspace.T
    return projection_sum / len(subspaces)


def eigendecompose_descending(matrix):
    eigen_values, eigen_vectors = eigh(matrix)
    return eigen_values[::-1], eigen_vectors[:, ::-1]


def project_to_common_subspace(shape_subspaces, common_subspace):
    return [
        orthonormal_basis(common_subspace @ (common_subspace.T @ shape_subspace)) for shape_subspace in shape_subspaces
    ]


def contribution_by_marker(difference_subspace, subspace):
    projected = difference_subspace @ (difference_subspace.T @ subspace)
    basis = orthonormal_basis(projected)
    return np.sum(np.square(basis), axis=1)


def calculate_difference_scores(subspaces, frame_interval):
    magnitudes = []
    contributions = []
    for frame in range(len(subspaces) - frame_interval):
        first = subspaces[frame]
        second = subspaces[frame + frame_interval]
        difference_subspace = gen_shape_difference_subspace(first, second)
        magnitudes.append(cal_magnitude(first, second))
        contributions.append(contribution_by_marker(difference_subspace, first))
    return magnitudes, contributions


def calculate_second_order_difference_DS(subspaces, frame_interval):
    magnitudes = []
    contributions = []
    for frame in range(frame_interval, len(subspaces) - frame_interval):
        first = subspaces[frame - frame_interval]
        middle = subspaces[frame]
        last = subspaces[frame + frame_interval]
        principal = gen_shape_principal_com_subspace(first, last)
        difference_subspace = gen_shape_difference_subspace(middle, principal)
        magnitudes.append(cal_magnitude(principal, middle))
        contributions.append(contribution_by_marker(difference_subspace, middle))
    return magnitudes, contributions


def calculate_geodestic(subspaces, frame_interval):
    magnitudes_along = []
    magnitudes_orth = []
    contributions_along = []
    contributions_orth = []
    for frame in range(frame_interval, len(subspaces) - frame_interval):
        first = subspaces[frame - frame_interval]
        middle = subspaces[frame]
        last = subspaces[frame + frame_interval]
        mag_along, contribution_along = along_geodesic(first, middle, last)
        mag_orth, contribution_orth = orth_decomposition_geodesic(first, middle, last)
        magnitudes_along.append(mag_along)
        magnitudes_orth.append(mag_orth)
        contributions_along.append(contribution_along)
        contributions_orth.append(contribution_orth)
    return magnitudes_along, contributions_along, magnitudes_orth, contributions_orth
