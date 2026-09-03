"""Common joint extraction for C3D markers and MediaPipe landmarks."""

import numpy as np

from experiments.noise import marker_name


COMMON_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

C3D_MARKER_GROUPS = {
    "left_shoulder": ("LSHO",),
    "right_shoulder": ("RSHO",),
    "left_elbow": ("LELB",),
    "right_elbow": ("RELB",),
    "left_wrist": ("LWRA", "LWRB"),
    "right_wrist": ("RWRA", "RWRB"),
    "left_hip": ("LFWT", "LBWT"),
    "right_hip": ("RFWT", "RBWT"),
    "left_knee": ("LKNE",),
    "right_knee": ("RKNE",),
    "left_ankle": ("LANK",),
    "right_ankle": ("RANK",),
}


def extract_c3d_common_joints(data, labels):
    label_indices = {marker_name(label): index for index, label in enumerate(labels)}
    joint_frames = []
    for joint in COMMON_JOINTS:
        markers = C3D_MARKER_GROUPS[joint]
        missing = [marker for marker in markers if marker not in label_indices]
        if missing:
            raise ValueError(f"Missing C3D markers for {joint}: {missing}")
        indices = [label_indices[marker] for marker in markers]
        joint_frames.append(np.mean(data[:, indices, :], axis=1))
    return np.stack(joint_frames, axis=1)


def extract_pose_common_joints(world_landmarks, landmark_names):
    name_indices = {str(name): index for index, name in enumerate(landmark_names)}
    missing = [joint for joint in COMMON_JOINTS if joint not in name_indices]
    if missing:
        raise ValueError(f"Missing pose landmarks: {missing}")
    return world_landmarks[:, [name_indices[joint] for joint in COMMON_JOINTS], :]
