"""Static 3D display for original and projected marker coordinates."""

import numpy as np
from matplotlib import pyplot as plt

from utils.plot_style import configure_matplotlib_font


configure_matplotlib_font()


def plot_coordinates(points, output_path, title):
    """Plot one unprojected 3D point frame."""
    points = np.asarray(points)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_projected_coordinates(original, projected, output_path, title):
    original = np.asarray(original)
    projected = np.asarray(projected)
    combined = np.concatenate((original, projected), axis=0)

    fig = plt.figure(figsize=(10, 5))
    for index, (points, panel_title) in enumerate(
        ((original, "Original coordinates"), (projected, "Projected coordinates")), start=1
    ):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20)
        ax.set_xlim(np.min(combined[:, 0]), np.max(combined[:, 0]))
        ax.set_ylim(np.min(combined[:, 1]), np.max(combined[:, 1]))
        ax.set_zlim(np.min(combined[:, 2]), np.max(combined[:, 2]))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(panel_title)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
