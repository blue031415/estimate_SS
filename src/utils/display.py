import os

import numpy as np

# import scipy
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# from scipy.linalg import eig
from utils.c3d import remove_nan


def _as_point_data(data):
    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError("data must be a 3D array")

    if data.shape[2] == 3:
        return data.transpose(2, 1, 0)  # (num_frame, num_mark, XYZ) -> (XYZ, num_mark, num_frame)

    if data.shape[1] == 3:
        return np.swapaxes(data, 0, 1)  # (num_mark, XYZ, num_frame) -> (XYZ, num_mark, num_frame)

    if data.shape[0] >= 3:
        return data[0:3, :, :]  # (XYZ or XYZ1, num_mark, num_frame)

    raise ValueError("data must have XYZ coordinates")


def _display_title(title):
    return os.path.basename(str(title)) if title is not None else ""


def contribution_to_size(cb, vmin, vmax, min_size=5, max_size=80):
    cb = np.asarray(cb)

    if np.isclose(vmax, vmin):
        return np.full(cb.shape, min_size, dtype=float)

    normalized = (cb - vmin) / (vmax - vmin)

    return min_size + normalized * (max_size - min_size)


def display_motion(data):
    point_data = _as_point_data(data)
    num_frame = point_data.shape[2]
    data = point_data[0:3, :, 0]

    x_range = np.nanmax(point_data[0, :, :]) - np.nanmin(point_data[0, :, :])
    y_range = np.nanmax(point_data[1, :, :]) - np.nanmin(point_data[1, :, :])
    z_range = np.nanmax(point_data[2, :, :]) - np.nanmin(point_data[2, :, :])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(data[0, :], data[1, :], data[2, :], s=5)  # type: ignore
    ft = ax.set_title(f"frame num {0}")

    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlim(np.nanmin(point_data[0, :, :]), np.nanmax(point_data[0, :, :]))
    ax.set_ylim(np.nanmin(point_data[1, :, :]), np.nanmax(point_data[1, :, :]))
    ax.set_zlim(np.nanmin(point_data[2, :, :]), np.nanmax(point_data[2, :, :]))

    def update(frame):
        data = point_data[0:3, :, frame]
        sc._offsets3d = (data[0, :], data[1, :], data[2, :])  # type: ignore
        ft.set_text(f"frame num {frame}")
        return (sc,)

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)  # type: ignore

    plt.show()
    return ani


def display_motion_score(data, x, y, save_path, title=None):
    point_data = _as_point_data(data)
    point_data = np.nan_to_num(point_data)
    num_frame = point_data.shape[2]
    data = point_data[0:3, :, 0]

    title = _display_title(title)

    x_range = np.max(point_data[0, :, :]) - np.min(point_data[0, :, :])
    y_range = np.max(point_data[1, :, :]) - np.min(point_data[1, :, :])
    z_range = np.max(point_data[2, :, :]) - np.min(point_data[2, :, :])

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    sc = ax1.scatter(data[0, :], data[1, :], data[2, :], s=5)  # type: ignore
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0, :, :]), np.max(point_data[0, :, :]))
    ax1.set_ylim(np.min(point_data[1, :, :]), np.max(point_data[1, :, :]))
    ax1.set_zlim(np.min(point_data[2, :, :]), np.max(point_data[2, :, :]))

    ax2 = fig.add_subplot(122)
    (line,) = ax2.plot([], [])
    ax2.set_xlim(np.min(x), np.max(x))
    ax2.set_ylim(np.min(y), np.max(y))
    ax2.set_xlabel("frame")
    ax2.set_ylabel("value")
    ax2.set_title(title)
    ax2.grid(True)

    def update(frame):
        data = point_data[0:3, :, frame]
        sc._offsets3d = (data[0, :], data[1, :], data[2, :])  # type: ignore
        if x[0] <= frame and frame < x[-1]:
            line.set_data(x[0 : frame - x[0]], y[0 : frame - x[0]])
        ft.set_text(f"frame num {frame}")
        return sc, line

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    ani.save(save_path, writer="pillow", fps=20)


def display_motion_score_contribution(data, x, y, contribution, save_path, title=None):
    # config = Config()
    title = _display_title(title)

    point_data = _as_point_data(data)
    point_data = remove_nan(point_data)

    num_frame = point_data.shape[2]
    data = point_data[0:3, :, 0]
    global_vmin = np.min(contribution)
    global_vmax = np.max(contribution)

    norm = colors.Normalize(
        vmin=global_vmin,
        vmax=global_vmax,
    )

    cb = contribution[0]
    sizes = contribution_to_size(
        cb,
        global_vmin,
        global_vmax,
    )

    x_range = np.max(point_data[0, :, :]) - np.min(point_data[0, :, :])
    y_range = np.max(point_data[1, :, :]) - np.min(point_data[1, :, :])
    z_range = np.max(point_data[2, :, :]) - np.min(point_data[2, :, :])

    fig = plt.figure(figsize=(11, 5))

    # fig.suptitle(f"{config.motion_description[title.split('.')[0]]} ({title})")
    fig.suptitle(f"({title})")
    gs = fig.add_gridspec(9, 11)
    ax1 = fig.add_subplot(gs[1:9, 0:7], projection="3d")

    sc = ax1.scatter(data[0, :], data[1, :], data[2, :], c=cb, norm=norm, cmap="jet", s=sizes)  # type: ignore
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0, :, :]), np.max(point_data[0, :, :]))
    ax1.set_ylim(np.min(point_data[1, :, :]), np.max(point_data[1, :, :]))
    ax1.set_zlim(np.min(point_data[2, :, :]), np.max(point_data[2, :, :]))

    ax2 = fig.add_subplot(gs[1:8, 7:11])
    (line,) = ax2.plot([], [])
    ax2.set_xlim(np.min(x), np.max(x))
    ax2.set_ylim(np.min(y), np.max(y))
    ax2.set_xlabel("frame")
    ax2.set_ylabel("value")
    ax2.set_title(f"Dissimilarity of {os.path.splitext(title)[0]} DS")
    ax2.grid(True)

    cbar = plt.colorbar(sc, pad=0.2)
    cbar.set_ticks([])

    def update(frame):
        data = point_data[0:3, :, frame]
        sc._offsets3d = (data[0, :], data[1, :], data[2, :])  # type: ignore

        if x[0] <= frame and frame < x[-1]:
            line.set_data(x[0 : frame - x[0] + 1], y[0 : frame - x[0] + 1])
            cb = contribution[frame - x[0] + 1]
            norm = colors.Normalize(vmin=min(cb), vmax=max(cb))
            sc.set_array(cb)
            sc.set_norm(norm)
            sc.set_sizes(
                contribution_to_size(
                    cb,
                    global_vmin,
                    global_vmax,
                )
            )
        ft.set_text(f"frame num {frame}")
        cbar.set_ticks([])

        return sc, line, cbar

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)  # type: ignore

    # plt.show()
    ani.save(save_path, writer="pillow", fps=20)


# 複数の波形を表示する
def display_motion_some_score(data, x, y, y_label, save_path, title=None):

    # yはarray
    # y[0] 値1
    # y[1] 値2
    # y_label[0] 値1のラベル
    # y_label[1] 値2のラベル

    title = _display_title(title)

    point_data = _as_point_data(data)
    point_data = remove_nan(point_data)

    num_frame = point_data.shape[2]
    data = point_data[0:3, :, 0]

    x_range = np.max(point_data[0, :, :]) - np.min(point_data[0, :, :])
    y_range = np.max(point_data[1, :, :]) - np.min(point_data[1, :, :])
    z_range = np.max(point_data[2, :, :]) - np.min(point_data[2, :, :])

    fig = plt.figure(figsize=(11, 5))
    # fig.suptitle(f"{cfg.motion_description[title.split('.')[0]]} ({title})")
    fig.suptitle(f"({title})")
    gs = fig.add_gridspec(9, 11)
    ax1 = fig.add_subplot(gs[0:9, 0:6], projection="3d")

    sc = ax1.scatter(data[0, :], data[1, :], data[2, :], s=5)  # type: ignore
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0, :, :]), np.max(point_data[0, :, :]))
    ax1.set_ylim(np.min(point_data[1, :, :]), np.max(point_data[1, :, :]))
    ax1.set_zlim(np.min(point_data[2, :, :]), np.max(point_data[2, :, :]))

    ax2 = fig.add_subplot(gs[1:7, 7:11])

    line = []
    for i in range(len(y)):
        # line.append(ax2.plot([], [], label=y_label[i]))
        if i == 0:
            line.append(ax2.plot([], [], label=y_label[i]))
        else:
            line.append(ax2.plot([], [], label=y_label[i], linestyle=":"))

    ax2.set_xlim(np.min(x), np.max(x))
    ax2.set_ylim(np.min(y), np.max(y))
    ax2.set_xlabel("frame")
    ax2.set_ylabel("value")
    ax2.set_title(f"Dissimilarity of {os.path.splitext(title)[0]} DS")
    ax2.grid(True)

    def update(frame):
        data = point_data[0:3, :, frame]
        sc._offsets3d = (data[0, :], data[1, :], data[2, :])  # type: ignore

        if x[0] <= frame and frame < x[-1]:
            for i in range(len(y)):
                line[i][0].set_data(x[0 : frame - x[0] + 1], y[i][0 : frame - x[0] + 1])
        ft.set_text(f"frame num {frame}")

        return sc, line

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)  # type: ignore
    plt.legend()
    # plt.show()
    ani.save(save_path, writer="pillow", fps=20)


def display_point(data, frame):
    point_data = _as_point_data(data)
    data = point_data[0:3, :, frame]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[0, :], data[1, :], data[2, :], s=5)  # type: ignore
    ax.set_box_aspect([1, 1, 2])
    plt.show()


def save_score_comparison(x, series, labels, save_path, title):
    """Save a static comparison of already-calculated dissimilarity scores."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for values, label in zip(series, labels):
        ax.plot(x, values, label=label)
    ax.set_xlabel("frame")
    ax.set_ylabel("dissimilarity")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
