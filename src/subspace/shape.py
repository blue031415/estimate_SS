"""Shape-subspace construction, combinations, and dissimilarity magnitude."""

import numpy as np
from scipy.linalg import eigh

from subspace.decomposition import cal_svd


def gen_shape_subspace(data):
    centered = data - np.mean(data, axis=0)
    left_vectors, _, _ = cal_svd(centered)
    return left_vectors[:, 0:3]


def gen_shape_difference_subspace(first, second):
    projection_sum = first @ first.T + second @ second.T
    eigen_values, eigen_vectors = eigh(projection_sum)
    indices = np.where((1e-6 < eigen_values) & (eigen_values < 1))[0]
    return eigen_vectors[:, indices]


def gen_shape_principal_com_subspace(first, second):
    projection_sum = first @ first.T + second @ second.T
    eigen_values, eigen_vectors = eigh(projection_sum)
    indices = np.where(1 <= eigen_values)[0]
    return eigen_vectors[:, indices]


def cal_magnitude(first, second):
    _, singular_values, _ = cal_svd(first.T @ second)
    return np.sum(2 * (1 - singular_values))
