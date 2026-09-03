"""Projection of marker coordinates into a common marker-space subspace."""

import numpy as np


def upright_pose_coordinates(world_landmarks):
    """Convert (X, Y-down, Z-depth) poses to plotting coordinates."""
    points = np.asarray(world_landmarks)
    return np.stack((points[..., 0], points[..., 2], -points[..., 1]), axis=-1)


def project_coordinate_frames(data, common_subspace):
    """Project centered marker coordinates with U U^T and restore centroids."""
    projected_frames = []
    for frame in data:
        centroid = np.mean(frame, axis=0)
        centered = frame - centroid
        projected = common_subspace @ (common_subspace.T @ centered)
        projected -= np.mean(projected, axis=0)
        projected_frames.append(projected + centroid)
    return np.asarray(projected_frames)
