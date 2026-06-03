"""Orbbec capture wrapper.

Only keeps the user-verified working path:
`/dev/video4` + `BA81` + Bayer(BG) demosaic to BGR.
"""

import cv2
import numpy as np


class OrbbecCapture:
    def __init__(
        self,
        name: str,
        serial_number: str | None = None,
        dim: tuple[int, int] = (640, 360),
        fps: int = 30,
        exposure: int | None = None,
    ):
        # Keep ctor signature compatible with existing call sites.
        _ = (serial_number, exposure)
        self.name = name
        self.dim = dim
        self.fps = fps
        self.device_path = "/dev/video4"
        self._cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open Orbbec color device: {self.device_path}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.dim[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.dim[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        try:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"BA81"))
        except Exception:
            pass

        # Warm-up to avoid empty first frame.
        for _ in range(10):
            ok, _frame = self.read()
            if ok and _frame is not None:
                break

    def _read_raw_frame(self, trials: int = 8):
        for _ in range(trials):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return frame
        return None

    def _decode_ba81_to_bgr(self, frame):
        raw = np.asanyarray(frame).ravel()
        expected = self.dim[0] * self.dim[1]
        if raw.size < expected:
            return None
        bayer = raw[:expected].reshape(self.dim[1], self.dim[0])
        return cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2BGR)

    def read(self):
        if self._cap is None:
            return False, None
        raw = self._read_raw_frame(trials=8)
        if raw is None:
            return False, None
        bgr = self._decode_ba81_to_bgr(raw)
        if bgr is None:
            return False, None
        if bgr.shape[1] != self.dim[0] or bgr.shape[0] != self.dim[1]:
            bgr = cv2.resize(bgr, self.dim)
        return True, bgr

    def close(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
