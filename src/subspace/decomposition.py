"""Numerically robust matrix decomposition and basis construction."""

import numpy as np
import scipy
from numpy.linalg import LinAlgError


def error_cal_svd(matrix):
    """Compute an SVD-compatible decomposition when NumPy's SVD fails."""
    eigen_values, eigen_vectors = scipy.linalg.eigh(matrix.T @ matrix)
    eigen_values = eigen_values[::-1]
    eigen_values[eigen_values < 0] = 0
    eigen_vectors = eigen_vectors[:, ::-1]

    singular_values = np.sqrt(eigen_values)
    left_vectors = matrix @ eigen_vectors
    inverse = np.where(singular_values > 0, 1.0 / singular_values, 0)
    left_vectors = left_vectors @ np.diag(inverse)
    right_vectors = eigen_vectors.T
    return left_vectors, singular_values, right_vectors


def cal_svd(matrix):
    """Compute an SVD, falling back to an eigenvalue-based implementation."""
    try:
        return np.linalg.svd(matrix)
    except LinAlgError:
        return error_cal_svd(matrix)


def gram_schmidt(arr):
    """Return the Gram-Schmidt orthonormalization used by the original experiments."""
    arr = np.array(arr, dtype=np.float64)
    column_count = arr.shape[1]
    vector = arr[:, [0]]
    norm = scipy.linalg.norm(vector)
    if norm < 1e-8:
        return np.zeros_like(arr)
    basis = vector / norm

    for column in range(1, column_count):
        vector = arr[:, [column]]
        for previous in range(column):
            vector -= np.dot(basis[:, previous], arr[:, column]) * basis[:, [previous]]
        norm = scipy.linalg.norm(vector)
        normalized = np.zeros_like(vector) if norm < 1e-8 else vector / norm
        basis = np.append(basis, normalized, axis=1)
    return basis


def orthonormal_basis(arr):
    """Return an orthonormal basis for the column space of a 2-D array."""
    matrix = np.asarray(arr, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("arr must be a 2-dimensional array.")

    left_vectors, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = np.count_nonzero(singular_values)
    return left_vectors[:, :rank]
