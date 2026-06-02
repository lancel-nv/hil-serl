"""Orbbec depth-camera capture wrapper, RSCapture-compatible.

Mirrors `franka_env.camera.rs_capture.RSCapture`:
- `read() -> (bool, np.ndarray)` returning BGR HWC uint8 frames
- `close()` shuts the pipeline down

VideoCapture (the thread wrapper) consumes this contract identically.

pyorbbecsdk is imported lazily inside `__init__` so missing-dep failure raises a
clear error only when an Orbbec camera is actually requested in a task config.
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
        try:
            from pyorbbecsdk import (
                Pipeline,
                Config,
                OBSensorType,
                OBFormat,
                OBConvertFormat,
                FormatConvertFilter,
                Context,
                OBLogLevel,
            )
        except ImportError as e:
            raise ImportError(
                "pyorbbecsdk is required for OrbbecCapture but is not installed. "
                "See the install steps in "
                "/media/data/Projects/2024-05-21-Robotics/2026-02-24-UR/ur_utils/README.md"
            ) from e

        Context.set_logger_level(OBLogLevel.ERROR)

        self.name = name
        self.dim = dim
        self._OBFormat = OBFormat  # stash for _frame_to_bgr
        self._OBConvertFormat = OBConvertFormat
        self._FormatConvertFilter = FormatConvertFilter

        self.pipeline = Pipeline()
        config = Config()

        if serial_number is not None:
            try:
                config.enable_device_by_serial_number(serial_number)
            except Exception:
                # Older pyorbbecsdk versions may not expose this method; default device is fine
                # when only one Orbbec is attached.
                pass

        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is None:
            raise RuntimeError(f"Orbbec camera '{name}' has no color sensor profile list")

        try:
            color_profile = profile_list.get_video_stream_profile(dim[0], dim[1], OBFormat.MJPG, fps)
        except Exception:
            try:
                color_profile = profile_list.get_video_stream_profile(dim[0], dim[1], OBFormat.RGB, fps)
            except Exception:
                color_profile = profile_list.get_default_video_stream_profile()

        config.enable_stream(color_profile)
        self.pipeline.start(config)

        if exposure is not None:
            try:
                device = self.pipeline.get_device()
                device.set_int_property("color_exposure", int(exposure))
            except Exception:
                pass

        # Warm-up — first few frames after pipeline.start may be empty.
        for _ in range(5):
            try:
                f = self.pipeline.wait_for_frames(500)
                if f is not None and f.get_color_frame() is not None:
                    break
            except Exception:
                pass

    def _frame_to_bgr(self, frame) -> np.ndarray | None:
        OBFormat = self._OBFormat
        width = frame.get_width()
        height = frame.get_height()
        fmt = frame.get_format()
        data = np.asanyarray(frame.get_data())

        if fmt == OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif fmt == OBFormat.BGR:
            image = np.resize(data, (height, width, 3))
        elif fmt == OBFormat.YUYV:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        elif fmt == OBFormat.MJPG:
            convert_filter = self._FormatConvertFilter()
            convert_filter.set_format_convert_format(self._OBConvertFormat.MJPG_TO_RGB888)
            rgb_frame = convert_filter.process(frame)
            if rgb_frame is None:
                return None
            rgb_data = np.asanyarray(rgb_frame.get_data())
            image = np.resize(rgb_data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif fmt == OBFormat.UYVY:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        elif fmt == OBFormat.NV12:
            yuv = data.reshape((height * 3 // 2, width))
            image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        elif fmt == OBFormat.NV21:
            yuv = data.reshape((height * 3 // 2, width))
            image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        else:
            return None
        return image

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(500)
            if frames is None:
                return False, None
            color_frame = frames.get_color_frame()
            if color_frame is None:
                return False, None
            image = self._frame_to_bgr(color_frame)
            if image is None:
                return False, None
            if image.shape[1] != self.dim[0] or image.shape[0] != self.dim[1]:
                image = cv2.resize(image, self.dim)
            return True, image
        except Exception:
            return False, None

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
