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
from utils.gen_noise import gen_noise
from utils.readc3d import read_c3d

DATA_PATH = Path("data") / "07_01.c3d"
DIM = 11
OUTPUT_PATH_ALONG = Path("out") / "ex3" / f"07_01_along_geodestic_d{DIM}.gif"
OUTPUT_PATH_ORTH = Path("out") / "ex3" / f"07_01_orth_geodestic_d{DIM}.gif"
OUTPUT_PATH = Path("out") / "ex3" / f"07_01_geodestic_d{DIM}.gif"
FRAME_INTERVAL = 5
CONTRIBUTION_THRESHOLD = 0.999
mean = np.array([0.0, 0.0, 0.0])
cov = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
NOISE_START_FRAME = 200
NOISE_END_FRAME = 250
NOISE_SCALE = 30
NOISE_BODY_PARTS = ["left_arm"]

BODY_PART_MARKERS = {
    "head": ["RFHD", "LFHD", "RBHD", "LBHD"],
    "torso": ["C7", "T10", "CLAV", "STRN", "RBAC", "RFWT", "LFWT", "RBWT", "LBWT"],
    "left_arm": ["LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN"],
    "right_arm": ["RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN"],
    "left_leg": ["LTHI", "LKNE", "LSHN", "LANK"],
    "right_leg": ["RTHI", "RKNE", "RSHN", "RANK"],
    "left_foot": ["LHEE", "LTOE", "LMT5"],
    "right_foot": ["RHEE", "RTOE", "RMT5"],
}


def select_eigen_indices(eigen_values, threshold=0.95):
    contribution_ratio = eigen_values / np.sum(eigen_values)
    cumulative_ratio = np.cumsum(contribution_ratio)
    dim = np.searchsorted(cumulative_ratio, threshold) + 1
    return np.arange(dim), contribution_ratio, cumulative_ratio


def build_shape_subspaces(data):
    return [gen_shape_subspace(data[frame, :, :]) for frame in range(data.shape[0])]


def marker_name(label):
    return label.split(":")[-1]


def marker_indices(labels, body_parts):
    unknown_parts = sorted(set(body_parts) - set(BODY_PART_MARKERS))
    if unknown_parts:
        raise ValueError(f"Unknown body parts: {unknown_parts}")

    target_markers = {marker for body_part in body_parts for marker in BODY_PART_MARKERS[body_part]}
    indices = [i for i, label in enumerate(labels) if marker_name(label) in target_markers]
    if not indices:
        raise ValueError(f"No markers found for body parts: {body_parts}")
    return indices


def add_noise_to_markers(data, indices, start_frame, end_frame, scale):
    noisy_data = data.copy()

    for frame in range(start_frame, end_frame):
        noisy_data[frame, indices, :] += scale * gen_noise(mean, cov, len(indices))

    return noisy_data


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
    print(f"Dimention of common subspace: {DIM}")
    data, labels = read_c3d(str(DATA_PATH), return_labels=True)
    print(f"frame num: {data.shape[0]}")

    noise_indices = marker_indices(labels, NOISE_BODY_PARTS)
    print(f"noise body parts: {NOISE_BODY_PARTS}")
    print(f"noise markers: {[labels[i] for i in noise_indices]}")
    noisy_data = add_noise_to_markers(
        data,
        noise_indices,
        NOISE_START_FRAME,
        NOISE_END_FRAME,
        NOISE_SCALE,
    )

    shape_subspaces = build_shape_subspaces(noisy_data)
    projection_matrix = average_projection_matrix(shape_subspaces)
    eigen_values, eigen_vectors = eigendecompose_descending(projection_matrix)

    indices, _, cumulative_ratio = select_eigen_indices(eigen_values, threshold=CONTRIBUTION_THRESHOLD)
    print(f"selected indices: {indices}")
    print(f"cumulative contribution: {cumulative_ratio[indices[-1]]}")

    common_subspace = eigen_vectors[:, indices[:DIM]]
    projected_subspaces = project_to_common_subspace(shape_subspaces, common_subspace)
    magnitudes_along, contributions_along, magnitudes_orth, contributions_orth = calculate_geodestic(
        projected_subspaces, FRAME_INTERVAL
    )

    OUTPUT_PATH_ALONG.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating {(os.path.basename(OUTPUT_PATH_ALONG))}")
    display_motion_score_contribution(
        noisy_data,
        range(FRAME_INTERVAL, len(projected_subspaces) - FRAME_INTERVAL),
        magnitudes_along,
        contributions_along,
        str(OUTPUT_PATH_ALONG),
        title=DATA_PATH.name,
    )
    OUTPUT_PATH_ORTH.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating {(os.path.basename(OUTPUT_PATH_ORTH))}")
    display_motion_score_contribution(
        noisy_data,
        range(FRAME_INTERVAL, len(projected_subspaces) - FRAME_INTERVAL),
        magnitudes_orth,
        contributions_orth,
        str(OUTPUT_PATH_ORTH),
        title=DATA_PATH.name,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating {(os.path.basename(OUTPUT_PATH))}")
    display_motion_some_score(
        noisy_data,
        range(FRAME_INTERVAL, len(projected_subspaces) - FRAME_INTERVAL),
        [magnitudes_along, magnitudes_orth],
        [f"along geodesic(D={DIM})", f"orth to geodesic(D={DIM})"],
        str(OUTPUT_PATH),
        title=DATA_PATH.name,
    )


if __name__ == "__main__":
    main()
