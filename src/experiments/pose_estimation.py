"""Replaceable 3D pose-estimator interface and MediaPipe implementation."""

from dataclasses import dataclass
import importlib.machinery
from pathlib import Path
import sys
import types
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PoseEstimate:
    """One frame of image-space and camera-relative 3D landmarks."""

    image_landmarks: np.ndarray
    world_landmarks: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray


class PoseEstimator(Protocol):
    """Interface that a future MediaPipe or MMPose adapter must implement."""

    name: str
    landmark_names: tuple[str, ...]
    connections: tuple[tuple[int, int], ...]

    def estimate(self, rgb_frame, timestamp_ms): ...

    def close(self): ...


class MediaPipePoseEstimator:
    """Estimate the first person's 33 MediaPipe pose landmarks in video mode."""

    name = "mediapipe_pose_landmarker"

    def __init__(
        self,
        model_path,
        min_detection_confidence=0.5,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("MediaPipe is required for ex4; install the project dependencies") from exc

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}. "
                "Download pose_landmarker_lite.task as described in docs/experiments.md."
            )

        self._mp = mp
        vision = mp.tasks.vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self.landmark_names = tuple(item.name.lower() for item in vision.PoseLandmark)
        self.connections = tuple(
            (connection.start, connection.end) for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS
        )

    def estimate(self, rgb_frame, timestamp_ms):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        if not result.pose_world_landmarks:
            return None

        image_landmarks = result.pose_landmarks[0]
        world_landmarks = result.pose_world_landmarks[0]
        return PoseEstimate(
            image_landmarks=np.asarray([(point.x, point.y, point.z) for point in image_landmarks], dtype=float),
            world_landmarks=np.asarray([(point.x, point.y, point.z) for point in world_landmarks], dtype=float),
            visibility=np.asarray([point.visibility for point in world_landmarks], dtype=float),
            presence=np.asarray([point.presence for point in world_landmarks], dtype=float),
        )

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class MMPosePoseEstimator:
    """Estimate a 17-joint Human3.6M pose with MMPose's 3D inferencer."""

    landmark_names = (
        "pelvis",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "spine",
        "thorax",
        "neck_base",
        "head",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    )
    connections = (
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 4),
        (4, 5),
        (5, 6),
        (0, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (8, 11),
        (11, 12),
        (12, 13),
        (8, 14),
        (14, 15),
        (15, 16),
    )

    def __init__(
        self,
        pose3d_model="human3d",
        pose3d_weights=None,
        detector=None,
        device="cpu",
        max_input_size=1024,
    ):
        _install_mmcv_lite_import_compatibility()
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError as exc:
            raise ImportError("MMPose is required; run `uv sync --extra mmpose`") from exc

        self.name = f"mmpose_{pose3d_model}"
        self._max_input_size = max_input_size
        self._inferencer = MMPoseInferencer(
            pose3d=pose3d_model,
            pose3d_weights=pose3d_weights,
            det_model=detector,
            device=device,
            show_progress=False,
        )

    def estimate(self, rgb_frame, timestamp_ms):
        del timestamp_ms  # MMPose image inference does not consume timestamps.
        bgr_frame = np.ascontiguousarray(rgb_frame[:, :, ::-1])
        bgr_frame = _resize_for_mmpose(bgr_frame, self._max_input_size)
        result = next(self._inferencer(bgr_frame, return_vis=False, show_progress=False))
        predictions = result.get("predictions", [])
        instances = predictions[0] if predictions else []
        if not instances:
            return None
        return _mmpose_prediction_to_estimate(instances[0])

    def close(self):
        self._inferencer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _mmpose_prediction_to_estimate(prediction):
    points = np.asarray(prediction["keypoints"], dtype=float)
    scores = np.asarray(prediction.get("keypoint_scores", np.ones(len(points))), dtype=float)
    if points.shape != (17, 3):
        raise ValueError(f"Expected MMPose Human3.6M keypoints with shape (17, 3), got {points.shape}")

    # MMPose returns (horizontal, depth, height). Convert it to the common
    # (X, Y-down, Z-depth) convention consumed by pose_display.py.
    world_landmarks = np.column_stack((points[:, 0], -points[:, 2], points[:, 1]))
    return PoseEstimate(
        image_landmarks=np.full((len(points), 3), np.nan),
        world_landmarks=world_landmarks,
        visibility=scores,
        presence=scores.copy(),
    )


def _resize_for_mmpose(frame, max_input_size):
    if max_input_size is None or max(frame.shape[:2]) <= max_input_size:
        return frame
    if max_input_size < 1:
        raise ValueError("mmpose_max_input_size must be positive or None")

    import cv2

    scale = max_input_size / max(frame.shape[:2])
    width = round(frame.shape[1] * scale)
    height = round(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _install_mmcv_lite_import_compatibility():
    """Allow MMPose models without native ops to import under mmcv-lite."""
    try:
        import mmcv._ext  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    extension = types.ModuleType("mmcv._ext")
    extension.__file__ = "mmcv/_ext_unavailable.py"
    extension.__spec__ = importlib.machinery.ModuleSpec("mmcv._ext", loader=None)

    def nms(boxes, scores, iou_threshold, offset=0):
        """Use TorchVision's CPU/CUDA NMS for the configured RTMDet model."""
        from torchvision.ops import nms as torchvision_nms

        if offset:
            boxes = boxes.clone()
            boxes[:, 2:] += offset
        return torchvision_nms(boxes, scores, iou_threshold)

    def unavailable(*args, **kwargs):
        raise RuntimeError(
            "This MMPose model called a native MMCV operation unavailable in mmcv-lite. "
            "Use the configured human3d/whole-image inference path or install a compatible full MMCV build."
        )

    def get_unavailable_operation(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return unavailable

    extension.nms = nms
    extension.__getattr__ = get_unavailable_operation
    sys.modules["mmcv._ext"] = extension
