"""Orbbec capture wrapper.

Uses the pyorbbecsdk Pipeline path (same approach as
`serl_robot_infra/tests/test_orbbec_camera.py`): grab color frames from the
Orbbec SDK and decode them to BGR via `frame_to_bgr_image`.
"""

from typing import Optional

import cv2
import numpy as np

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBError,
    VideoFrame,
)


def frame_to_bgr_image(frame: VideoFrame) -> Optional[np.ndarray]:
    """将视频帧转换为 BGR 格式的 numpy 数组"""
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    elif color_format == OBFormat.NV12:
        yuv = data.reshape((height * 3 // 2, width))
        image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    elif color_format == OBFormat.NV21:
        yuv = data.reshape((height * 3 // 2, width))
        image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    else:
        print(f"不支持的彩色格式: {color_format}")
        return None

    return image


class OrbbecCapture:
    def __init__(
        self,
        name: str,
        serial_number: str | None = None,
        dim: tuple[int, int] = (640, 360),
        fps: int = 30,
        exposure: int | None = None,
    ):
        # exposure is accepted for call-site compatibility but not applied here.
        _ = exposure
        self.name = name
        self.dim = dim
        self.fps = fps
        self.serial_number = serial_number

        self.pipeline = Pipeline()
        config = Config()

        if serial_number is not None:
            try:
                config.enable_device_by_serial_number(serial_number)
            except Exception:
                pass

        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is None:
            raise RuntimeError(f"No Orbbec color sensor found for camera '{name}'.")
        try:
            color_profile = profile_list.get_video_stream_profile(
                dim[0], dim[1], OBFormat.MJPG, fps
            )
        except OBError:
            color_profile = profile_list.get_video_stream_profile(
                dim[0], dim[1], OBFormat.RGB, fps
            )
        config.enable_stream(color_profile)
        self.pipeline.start(config)

        # Warm-up to avoid empty first frame.
        for _ in range(10):
            ok, _frame = self.read()
            if ok and _frame is not None:
                break

    def read(self):
        if self.pipeline is None:
            return False, None
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return False, None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return False, None
        bgr = frame_to_bgr_image(color_frame)
        if bgr is None:
            return False, None
        if bgr.shape[1] != self.dim[0] or bgr.shape[0] != self.dim[1]:
            bgr = cv2.resize(bgr, self.dim)
        return True, bgr

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
