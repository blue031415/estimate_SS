import numpy as np
import scipy
from scipy.linalg import eigh
from numpy.linalg import LinAlgError


# 収束のエラーが出た場合の特異値分解計算
def error_cal_svd(A):

    w, v = scipy.linalg.eigh(np.dot(A.T, A))

    w = w[::-1]
    w[w < 0] = 0
    v = v[:, ::-1]

    Y = np.sqrt(w)  # Y:特異値
    X = np.dot(A, v)  # X:左特異ベクトル行列

    epsilon = 0  # ゼロとみなす小さな値の閾値
    Y_inv = np.where(Y > epsilon, 1.0 / Y, 0)

    X = np.dot(X, np.diag(Y_inv))  #
    Z = v.T  # Z:右特異ベクトルの転置行列

    return X, Y, Z


# 収束エラーを考慮した特異値分解
def cal_svd(A):
    try:
        U, S, V = np.linalg.svd(A)
        return U, S, V
    except LinAlgError:
        U, S, V = error_cal_svd(A)
        return U, S, V


# 1frame内のポイントデータから形状部分空間を作成する。
def gen_shape_subspace(data):
    X = data
    mv = np.mean(X, axis=0)
    Xc = X - mv
    U, _, _ = cal_svd(Xc)

    return U[:, 0:3]


# 差分部分空間の生成
def gen_shape_difference_subspace(S1, S2):
    # U, S, Vt = cal_svd(S1.T @ S2)
    # S = np.diag(S)
    # I = np.eye(S.shape[0],S.shape[1])
    # D = ((S1 @ U) - (S2 @ Vt.T)) @ ((2 * (I - S))**(-0.5))
    G = S1 @ S1.T + S2 @ S2.T
    eigen_val, eigen_vec = eigh(G)
    idx = np.where((1e-6 < eigen_val) & (eigen_val < 1))[0]
    return eigen_vec[:, idx]


# 共通部分空間の作成
def gen_shape_principal_com_subspace(S1, S2):
    G = S1 @ S1.T + S2 @ S2.T
    eigen_val, eigen_vec = eigh(G)
    idx = np.where(1 <= eigen_val)[0]
    return eigen_vec[:, idx]


# 部分空間同士のマグニチュード(類似度)を計算
def cal_magnitude(S1, S2):
    _, S, _ = cal_svd(S1.T @ S2)
    mag = np.sum(2 * (1 - S))
    return mag


def gram_schmidt(arr):
    arr = np.array(arr, dtype=np.float64)
    k = arr.shape[1]
    u = arr[:, [0]]
    norm_u = scipy.linalg.norm(u)
    if norm_u < 1e-8:
        return np.zeros_like(arr)  # or handle appropriately
    q = u / norm_u

    for j in range(1, k):
        u = arr[:, [j]]
        for i in range(j):
            u -= np.dot(q[:, i], arr[:, j]) * q[:, [i]]
        norm_u = scipy.linalg.norm(u)
        if norm_u < 1e-8:
            qi = np.zeros_like(u)
        else:
            qi = u / norm_u
        q = np.append(q, qi, axis=1)
    # arr = np.asarray(arr, dtype=np.float64)
    # q, _ = np.linalg.qr(arr, mode="reduced")
    return q


def orthonormal_basis(arr):
    """
    入力行列 arr の列空間の正規直交基底を返す。

    計算誤差は考慮せず、非ゼロ特異値に対応する
    左特異ベクトルを基底として採用する。
    """
    A = np.asarray(arr, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("arr must be a 2-dimensional array.")

    U, singular_values, _ = np.linalg.svd(A, full_matrices=False)

    # 非ゼロ特異値の個数 = 行列のランク
    rank = np.count_nonzero(singular_values)

    # 非ゼロ特異値に対応する左特異ベクトル
    Q = U[:, :rank]

    return Q


# D(M, S2')の大きさが２階差分部分空間の測地線に沿った変動成分１
def along_geodesic(S1, S2, S3):
    # ６次元の和空間W(S1,S3)を作成
    W = np.concatenate([S1, S3], 1)
    # W = W / np.linalg.norm(W, axis=0)
    # W = gram_schmidt(W)
    W = orthonormal_basis(W)

    # 部分空間S2の３本の基底を６次元の和空間W(S1,S3)に射影する
    P = W @ W.T
    V = P @ S2

    # 射影された3本の基底に対してグラムシュミット直交化を適用してV:(S2’)を求める
    # # 射影した基底ベクトルを正規化
    # V = V / np.linalg.norm(V, axis=0)
    # # グラムシュミット直交化
    # V = gram_schmidt(V)
    V = orthonormal_basis(V)

    # Mをもとめる
    M = gen_shape_principal_com_subspace(S1, S3)

    # 変動成分：D(M, S2')の大きさが２階差分部分空間の測地線に沿った変動成分１
    mag = cal_magnitude(M, V)

    return mag


# D(S2, S2')の大きさが２階差分部分空間の測地線に直交する変動成分２
def orth_decomposition_geodesic(S1, S2, S3, cfg):
    # ６次元の和空間W(S1,S3)を作成
    W = np.concatenate([S1, S3], 1)
    # W = W / np.linalg.norm(W, axis=0)
    # W = gram_schmidt(W)
    W = orthonormal_basis(W)

    # 部分空間S2の３本の基底を６次元の和空間W(S1,S3)に射影する
    P = W @ W.T
    V = P @ S2

    # 射影された3本の基底に対してグラムシュミット直交化を適用してV:(S2’)を求める
    # # 射影した基底ベクトルを正規化
    # V = V / np.linalg.norm(V, axis=0)
    # # グラムシュミット直交化
    # V = gram_schmidt(V)
    V = orthonormal_basis(V)

    # 変動成分：D(S2, S2')の大きさが２階差分部分空間の測地線に直交する変動成分
    mag = cal_magnitude(S2, V)

    # S2とS2'の差分部分空間を求め射影し、各点の影響度を計算する。
    D = gen_shape_difference_subspace(S2, V, 3)
    V1 = D @ (D.T @ S2)
    # # 射影した基底ベクトルを正規化
    # V1 = V1 / np.linalg.norm(V1, axis=0)
    # # グラムシュミット直交化
    # V1 = gram_schmidt(V1)
    V1 = orthonormal_basis(V1)

    # Vの各要素を2乗
    V1 = np.square(V1)
    V1 = np.sum(V1, axis=1)

    return mag, V1


if __name__ == "__main__":
    S1 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    U, S, V = np.linalg.svd(S1)
    print(S)
    U, S, V = error_cal_svd(S1)
    print(S)
    U, S, V = cal_svd(S1)
    print(S)
