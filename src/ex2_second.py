from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from subspace.utils import (
    cal_magnitude,
    gen_shape_difference_subspace,
    gen_shape_principal_com_subspace,
    gen_shape_subspace,
    orthonormal_basis,
)
from utils.display import display_motion_score_contribution
from utils.readc3d import read_c3d

DATA_PATH = Path("data") / "07_01.c3d"
Dim = 5
OUTPUT_PATH = Path("out") / "ex2" / f"07_01_second_d{Dim}.gif"
FRAME_INTERVAL = 5
CONTRIBUTION_THRESHOLD = 0.999


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


def contribution_by_marker(difference_subspace, S):
    projected = difference_subspace @ (difference_subspace.T @ S)
    basis = orthonormal_basis(projected)
    return np.sum(np.square(basis), axis=1)


def calculate_second_order_difference_DS(subspaces, frame_interval):
    magnitudes = []
    contributions = []

    for frame in range(frame_interval, len(subspaces) - frame_interval):
        S1 = subspaces[frame - frame_interval]
        S2 = subspaces[frame]
        S3 = subspaces[frame + frame_interval]

        M = gen_shape_principal_com_subspace(S1, S3)

        difference_subspace = gen_shape_difference_subspace(S2, M)

        magnitudes.append(cal_magnitude(M, S2))
        contributions.append(contribution_by_marker(difference_subspace, S2))

    return magnitudes, contributions


def main():
    print(f"Dimention of common subspace: {Dim}")
    data = read_c3d(str(DATA_PATH), return_labels=False)
    print(f"frame num: {data.shape[0]}")  # type: ignore

    shape_subspaces = build_shape_subspaces(data)
    projection_matrix = average_projection_matrix(shape_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)

    indices, _, cumulative_ratio = select_eigen_indices(eigen_values, threshold=CONTRIBUTION_THRESHOLD)
    print(f"selected indices: {indices}")
    print(f"cumulative contribution: {cumulative_ratio[indices[-1]]}")

    common_subspace = eigen_vectors[:, indices[:Dim]]
    projected_subspaces = project_to_common_subspace(shape_subspaces, common_subspace)
    magnitudes, contributions = calculate_second_order_difference_DS(projected_subspaces, FRAME_INTERVAL)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    display_motion_score_contribution(
        data,
        range(FRAME_INTERVAL, len(projected_subspaces) - FRAME_INTERVAL),
        magnitudes,
        contributions,
        str(OUTPUT_PATH),
        title=DATA_PATH.name,
    )


if __name__ == "__main__":
    main()
