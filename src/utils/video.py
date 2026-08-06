"""Video metadata and RGB frame decoding independent of pose estimation."""

from pathlib import Path

import cv2


class VideoReader:
    """Context-managed OpenCV reader with deterministic frame sampling."""

    def __init__(self, path):
        self.path = Path(path)
        self._capture = None

    def __enter__(self):
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise ValueError(f"Could not open video: {self.path}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def fps(self):
        self._require_open()
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise ValueError(f"Video reports an invalid frame rate: {self.path}")
        return fps

    def iter_sampled_frames(self, frame_stride=1, max_frames=None):
        """Yield ``(source frame index, RGB image)`` without seeking the codec."""
        self._require_open()
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be at least 1 or None")

        source_index = 0
        yielded = 0
        while True:
            success, bgr_frame = self._capture.read()
            if not success:
                break
            if source_index % frame_stride == 0:
                yield source_index, cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            source_index += 1

    def _require_open(self):
        if self._capture is None:
            raise RuntimeError("VideoReader must be used as a context manager")
