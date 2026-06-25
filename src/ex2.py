# from util.display import display_motion
from pathlib import Path
from subspace.utils import gen_shape_subspace, orthonormal_basis, gen_shape_difference_subspace
from utils.readc3d import read_c3d
import numpy as np

from scipy.linalg import eigh

# 実験の設定
path = str(Path("data") / "07_01.c3d")
t = 5


def select_eigen_indices(eigen_val, threshold=0.95):
    contribution_ratio = eigen_val / np.sum(eigen_val)
    cumulative_ratio = np.cumsum(contribution_ratio)
    dim = np.searchsorted(cumulative_ratio, threshold) + 1
    return np.arange(dim), contribution_ratio, cumulative_ratio


def main():
    data = read_c3d(path)
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

    # 以下部分空間の次元は５とする。これは固有値の寄与率が0.99を超える最小の次元である。

    common_subspace = eigen_vec[:, indices]
    # print(common_subspace.shape)
    P_common = common_subspace @ common_subspace.T

    p_list = [
        orthonormal_basis(P_common @ shape_subspace) for shape_subspace in shape_subspace_list
    ]

    D = [gen_shape_difference_subspace(p_list[i], p_list[i + t]) for i in range(len(p_list) - t)]
    print(len(D))


if __name__ == "__main__":
    main()
