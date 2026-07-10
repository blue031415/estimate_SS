# from util.display import display_motion
from pathlib import Path
from subspace.utils import gen_shape_subspace, cal_magnitude, orthonormal_basis
from utils.readc3d import read_c3d
from utils.gen_noise import gen_noise
import numpy as np
from matplotlib import pyplot as plt

from scipy.linalg import eigh

DATA_PATH = Path("data") / "07_01.c3d"
OUTPUT_PATH = Path("out") / "ex3" / "eigen_val_noise.png"
CONTRIBUTION_THRESHOLD = 0.999
mean = np.array([0.0, 0.0, 0.0])
cov = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
NOISE_START_FRAME = 150
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


def plot_eigen_val_log(eigen_val):
    eigen_val = eigen_val[eigen_val > 0]
    plt.figure()
    plt.plot(np.arange(1, len(eigen_val) + 1), eigen_val, marker="o")
    x = np.arange(1, len(eigen_val) + 1)
    plt.xticks(x)
    plt.yscale("log")
    plt.xlabel("rank")
    plt.ylabel("eigenvalue")
    plt.title("Eigenvalues of averaged projection matrix")
    plt.grid(True)
    plt.show()


def plot_eigen_val(eigen_val, path=None):
    plt.figure()
    plt.plot(np.arange(1, len(eigen_val) + 1), eigen_val, marker="o")
    x = np.arange(1, len(eigen_val) + 1)
    plt.xticks(x)
    plt.xlabel("rank")
    plt.ylabel("eigenvalue")
    plt.title("Eigenvalues of averaged projection matrix")
    plt.grid(True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def select_eigen_indices(eigen_val, threshold=0.95):
    contribution_ratio = eigen_val / np.sum(eigen_val)
    cumulative_ratio = np.cumsum(contribution_ratio)
    dim = np.searchsorted(cumulative_ratio, threshold) + 1
    return np.arange(dim), contribution_ratio, cumulative_ratio


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


def main():
    data, labels = read_c3d(str(DATA_PATH), return_labels=True)
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

    shape_subspace_list = []
    print(f"frame num: {noisy_data.shape[0]}")
    P = np.zeros((noisy_data.shape[1], noisy_data.shape[1]))
    for i in range(noisy_data.shape[0]):
        S = gen_shape_subspace(noisy_data[i, :, :])
        shape_subspace_list.append(S)
        P += S @ S.T

    P /= noisy_data.shape[0]
    eigen_val, eigen_vec = eigh(P)
    eigen_val, eigen_vec = eigen_val[::-1], eigen_vec[:, ::-1]

    indices, _, cumulative_ratio = select_eigen_indices(eigen_val, threshold=CONTRIBUTION_THRESHOLD)

    print(f"selected indices: {indices}")
    print(f"cumulative contribution: {cumulative_ratio[indices[-1]]}")
    plot_eigen_val(eigen_val[indices], str(OUTPUT_PATH))

    # 以下部分空間の次元は５とする。これは固有値の寄与率が0.99を超える最小の次元である。

    common_subspace = eigen_vec[:, indices]
    # print(common_subspace.shape)
    P_common = common_subspace @ common_subspace.T

    p_list = [orthonormal_basis(P_common @ shape_subspace) for shape_subspace in shape_subspace_list]

    magnitudes = [cal_magnitude(p, shape_subspace) for p, shape_subspace in zip(p_list, shape_subspace_list)]

    mag_max, mag_maxidx = np.max(magnitudes), np.argmax(magnitudes)
    mag_min, mag_minidx = np.min(magnitudes), np.argmin(magnitudes)
    print(f"average magnitude: {np.mean(magnitudes)}")
    print(f"max magnitude: {mag_max} at frame {mag_maxidx}")
    print(f"min magnitude: {mag_min} at frame {mag_minidx}")


if __name__ == "__main__":
    main()
