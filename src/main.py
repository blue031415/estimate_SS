# from util.display import display_motion
from pathlib import Path
from subspace.utils import gen_shape_subspace, cal_magnitude, orthonormal_basis
from utils.readc3d import read_c3d
import numpy as np
from matplotlib import pyplot as plt

from scipy.linalg import eigh

path = str(Path("data") / "07_01.c3d")


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


def plot_eigen_val(eigen_val):
    plt.figure()
    plt.plot(np.arange(1, len(eigen_val) + 1), eigen_val, marker="o")
    x = np.arange(1, len(eigen_val) + 1)
    plt.xticks(x)
    plt.xlabel("rank")
    plt.ylabel("eigenvalue")
    plt.title("Eigenvalues of averaged projection matrix")
    plt.grid(True)
    plt.show()


def select_eigen_indices(eigen_val, threshold=0.95):
    contribution_ratio = eigen_val / np.sum(eigen_val)
    cumulative_ratio = np.cumsum(contribution_ratio)
    dim = np.searchsorted(cumulative_ratio, threshold) + 1
    return np.arange(dim), contribution_ratio, cumulative_ratio


def main():
    data = read_c3d(path)
    # print(data.shape)
    shape_subspace_list = []
    print(f"frame num: {data.shape[2]}")
    P = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[2]):
        S = gen_shape_subspace(data[:, :, i])
        shape_subspace_list.append(S)
        P += S @ S.T

    P /= data.shape[2]
    eigen_val, eigen_vec = eigh(P)
    eigen_val, eigen_vec = eigen_val[::-1], eigen_vec[:, ::-1]

    indices, _, cumulative_ratio = select_eigen_indices(eigen_val, threshold=0.99)

    print(f"selected indices: {indices}")
    print(f"cumulative contribution: {cumulative_ratio[indices[-1]]}")
    # plot_eigen_val(eigen_val[indices])

    # 以下部分空間の次元は５とする。これは固有値の寄与率が0.99を超える最小の次元である。

    common_subspace = eigen_vec[:, indices]
    # print(common_subspace.shape)
    P_common = common_subspace @ common_subspace.T

    p_list = [
        orthonormal_basis(P_common @ shape_subspace) for shape_subspace in shape_subspace_list
    ]

    magnitudes = [
        cal_magnitude(p, shape_subspace) for p, shape_subspace in zip(p_list, shape_subspace_list)
    ]

    mag_max, mag_maxidx = np.max(magnitudes), np.argmax(magnitudes)
    mag_min, mag_minidx = np.min(magnitudes), np.argmin(magnitudes)
    print(f"average magnitude: {np.mean(magnitudes)}")
    print(f"max magnitude: {mag_max} at frame {mag_maxidx}")
    print(f"min magnitude: {mag_min} at frame {mag_minidx}")


if __name__ == "__main__":
    main()
