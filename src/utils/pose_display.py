"""Side-by-side RGB video and 3D pose GIF rendering."""

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from utils.plot_style import configure_matplotlib_font


configure_matplotlib_font()


def save_pose_comparison_gif(
    rgb_frames,
    world_landmarks,
    visibility,
    connections,
    output_path,
    output_fps=10,
    panel_size=512,
    visibility_threshold=0.5,
    estimator_label="3D pose",
    azimuth_rotation_degrees=0,
):
    """Save source frames beside an upright, camera-relative 3D skeleton."""
    if not rgb_frames:
        raise ValueError("At least one frame is required to create a GIF")
    if output_fps <= 0:
        raise ValueError("output_fps must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    axis_limit = _symmetric_axis_limit(world_landmarks)
    gif_frames = []
    for index, rgb_frame in enumerate(rgb_frames):
        pose_panel = _render_pose_panel(
            world_landmarks[index],
            visibility[index],
            connections,
            axis_limit,
            panel_size,
            visibility_threshold,
            index,
            estimator_label,
            azimuth_rotation_degrees,
        )
        source_panel = _fit_source_panel(rgb_frame, panel_size)
        combined = Image.new("RGB", (panel_size * 2, panel_size), "white")
        combined.paste(source_panel, (0, 0))
        combined.paste(pose_panel, (panel_size, 0))
        gif_frames.append(combined)

    duration_ms = max(1, round(1000 / output_fps))
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def _fit_source_panel(rgb_frame, panel_size):
    """Fit a source image into a square without changing its aspect ratio."""
    source = Image.fromarray(rgb_frame)
    source.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (panel_size, panel_size), "black")
    left = (panel_size - source.width) // 2
    top = (panel_size - source.height) // 2
    panel.paste(source, (left, top))
    return panel


def _symmetric_axis_limit(world_landmarks):
    finite = np.asarray(world_landmarks)[np.isfinite(world_landmarks)]
    if finite.size == 0:
        return 1.0
    return max(0.5, float(np.percentile(np.abs(finite), 99)) * 1.15)


def _render_pose_panel(
    points,
    visibility,
    connections,
    axis_limit,
    panel_size,
    threshold,
    frame_index,
    estimator_label,
    azimuth_rotation_degrees=0,
):
    dpi = 100
    fig = plt.figure(figsize=(panel_size / dpi, panel_size / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    valid = np.isfinite(points).all(axis=1) & (visibility >= threshold)

    if np.any(valid):
        # MediaPipe uses +Y downward. Display -Y as up and Z as depth.
        ax.scatter(points[valid, 0], points[valid, 2], -points[valid, 1], c="#e63946", s=18)
        for start, end in connections:
            if valid[start] and valid[end]:
                ax.plot(
                    points[[start, end], 0],
                    points[[start, end], 2],
                    -points[[start, end], 1],
                    color="#1d3557",
                    linewidth=2,
                )
    else:
        ax.text2D(0.5, 0.5, "Pose not detected", transform=ax.transAxes, ha="center")

    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-70 + azimuth_rotation_degrees)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title(f"{estimator_label} — sampled frame {frame_index}")
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return Image.fromarray(rgba[:, :, :3])
