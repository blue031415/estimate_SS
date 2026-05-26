# from util.display import display_motion
from pathlib import Path
from subspace.utils import gen_shape_subspace
from utils.preprocess import remove_nan
import numpy as np
from ezc3d import c3d
from matplotlib import pyplot as plt

from scipy.linalg import eigh

path = str(Path("data") / "07_01.c3d")


def read_c3d(path):
    c = c3d(path)
    point_data = remove_nan(c["data"]["points"])  # (XYZ1, num_mark, num_frame)
    data = np.swapaxes(point_data[0:3, :, :], 0, 1)  # (num_mark, XYZ, num_frame)
    return data


def plot_eigen_val_log(eigen_val):
    eigen_val = eigen_val[eigen_val > 0]
    plt.figure()
    plt.plot(np.arange(1, len(eigen_val) + 1), eigen_val, marker="o")
    plt.yscale("log")
    plt.xlabel("rank")
    plt.ylabel("eigenvalue")
    plt.title("Eigenvalues of averaged projection matrix")
    plt.grid(True)
    plt.show()


def plot_eigen_val(eigen_val):
    plt.figure()
    plt.plot(np.arange(1, len(eigen_val) + 1), eigen_val, marker="o")
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
    print("Hello from estimate-ss!")
    data = read_c3d(path)
    # print(data.shape)
    shape_subspace_list = []
    print(data.shape[2])
    P = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[2]):
        S = gen_shape_subspace(data[:, :, i])
        shape_subspace_list.append(S)
        P += S @ S.T

    P /= data.shape[2]
    eigen_val, eigen_vec = eigh(P)
    eigen_val = eigen_val[::-1]
    eigen_vec = eigen_vec[:, ::-1]

    indices, contribution_ratio, cumulative_ratio = select_eigen_indices(eigen_val, threshold=0.999)
    # selected_vec = eigen_vec[:, indices]

    print(f"selected indices: {indices}")
    print(f"cumulative contribution: {cumulative_ratio[indices[-1]]}")
    plot_eigen_val(eigen_val[indices])


if __name__ == "__main__":
    main()
