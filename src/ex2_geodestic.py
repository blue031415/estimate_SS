import os
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from subspace.utils import (
    along_geodesic,
    gen_shape_subspace,
    orth_decomposition_geodesic,
    orthonormal_basis,
)
from utils.display import display_motion_score_contribution, display_motion_some_score
from utils.readc3d import read_c3d

DATA_PATH = Path("data") / "07_01.c3d"
Dim = 5
OUTPUT_PATH_ALONG = Path("out") / "ex2" / f"07_01_along_geodestic_d{Dim}.gif"
OUTPUT_PATH_ORTH = Path("out") / "ex2" / f"07_01_orth_geodestic_d{Dim}.gif"
OUTPUT_PATH = Path("out") / "ex2" / f"07_01_geodestic_d{Dim}.gif"
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


def calculate_geodestic(subspaces, frame_interval):
    magnitudes_along = []
    magnitudes_orth = []
    contributions_along = []
    contributions_orth = []

    for frame in range(frame_interval, len(subspaces) - frame_interval):
        S1 = subspaces[frame - frame_interval]
        S2 = subspaces[frame]
        S3 = subspaces[frame + frame_interval]

        mag_along, contribution_along = along_geodesic(S1, S2, S3)
        mag_orth, contribution_orth = orth_decomposition_geodesic(S1, S2, S3)

        magnitudes_along.append(mag_along)
        magnitudes_orth.append(mag_orth)
        contributions_along.append(contribution_along)
        contributions_orth.append(contribution_orth)

    return magnitudes_along, contributions_along, magnitudes_orth, contributions_orth


def main():
    print(f"Dimention of common subspace: {Dim}")
    data = read_c3d(str(DATA_PATH))
    print(f"frame num: {data.shape[0]}")

    shape_subspaces = build_shape_subspaces(data)
    projection_matrix = average_projection_matrix(shape_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)

    indices, _, cumulative_ratio = select_eigen_indices(eigen_values, threshold=CONTRIBUTION_THRESHOLD)
    print(f"selected indices: {indices}")
    print(f"cumulative contribution: {cumulative_ratio[indices[-1]]}")

    common_subspace = eigen_vectors[:, indices[:Dim]]
    projected_subspaces = project_to_common_subspace(shape_subspaces, common_subspace)
    magnitudes_along, contributions_along, magnitudes_orth, contributions_orth = calculate_geodestic(
        projected_subspaces, FRAME_INTERVAL
    )

    OUTPUT_PATH_ALONG.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating {(os.path.basename(OUTPUT_PATH_ALONG))}")
    display_motion_score_contribution(
        data,
        range(FRAME_INTERVAL, len(projected_subspaces) - FRAME_INTERVAL),
        magnitudes_along,
        contributions_along,
        str(OUTPUT_PATH_ALONG),
        title=DATA_PATH.name,
    )
    OUTPUT_PATH_ORTH.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating {(os.path.basename(OUTPUT_PATH_ORTH))}")
    display_motion_score_contribution(
        data,
        range(FRAME_INTERVAL, len(projected_subspaces) - FRAME_INTERVAL),
        magnitudes_orth,
        contributions_orth,
        str(OUTPUT_PATH_ORTH),
        title=DATA_PATH.name,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating {(os.path.basename(OUTPUT_PATH))}")
    display_motion_some_score(
        data,
        range(FRAME_INTERVAL, len(projected_subspaces) - FRAME_INTERVAL),
        [magnitudes_along, magnitudes_orth],
        [f"along geodesic(D={Dim})", f"orth to geodesic(D={Dim})"],
        str(OUTPUT_PATH),
        title=DATA_PATH.name,
    )


if __name__ == "__main__":
    main()
