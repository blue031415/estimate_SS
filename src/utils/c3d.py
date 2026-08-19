"""C3D loading and point-data cleanup."""

import numpy as np
from ezc3d import c3d


def remove_nan(data):
    """Remove invalid marker columns and keep a consistent marker count per frame."""
    frames = [data[:, :, frame] for frame in range(data.shape[2])]
    valid_frames = [frame[:, ~np.isnan(frame).any(axis=0)] for frame in frames]
    min_size = min(frame.shape[1] for frame in valid_frames)
    trimmed_frames = np.array([frame[:, :min_size] for frame in valid_frames])
    return np.transpose(trimmed_frames, (1, 2, 0))


def read_c3d(path, return_labels=False):
    contents = c3d(path)
    labels = contents["parameters"]["POINT"]["LABELS"]["value"]
    point_data = remove_nan(contents["data"]["points"])
    data = point_data[0:3, :, :].transpose(2, 1, 0)
    if return_labels:
        return data, labels[: data.shape[1]]
    return data
