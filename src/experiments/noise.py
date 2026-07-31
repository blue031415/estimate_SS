import numpy as np

BODY_PART_MARKERS = {
    "head": ["RFHD", "LFHD", "RBHD", "LBHD"],
    "torso": ["C7", "T10", "CLAV", "STRN", "RBAC", "RFWT", "LFWT", "RBWT", "LBWT"],
    "left_arm": ["LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN"],
    "right_arm": ["RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN"],
    "left_leg": ["LTHI", "LKNE", "LSHN", "LANK"],
    "right_leg": ["RTHI", "RKNE", "RSHN", "RANK"],
    "left_foot": ["LHEE", "LTOE", "LMT5"],
    "right_foot": ["RHEE", "RTOE", "RMT5"],
}


def marker_name(label):
    return label.split(":")[-1]


def marker_indices(labels, body_parts):
    unknown_parts = sorted(set(body_parts) - set(BODY_PART_MARKERS))
    if unknown_parts:
        raise ValueError(f"Unknown body parts: {unknown_parts}")
    target_markers = {marker for part in body_parts for marker in BODY_PART_MARKERS[part]}
    indices = [index for index, label in enumerate(labels) if marker_name(label) in target_markers]
    if not indices:
        raise ValueError(f"No markers found for body parts: {body_parts}")
    return indices


def add_noise_to_markers(data, indices, start_frame, end_frame, scale, mean, covariance):
    noisy_data = data.copy()
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    for frame in range(start_frame, end_frame):
        noise = np.random.multivariate_normal(mean, covariance, len(indices))
        noisy_data[frame, indices, :] += scale * noise
    return noisy_data
