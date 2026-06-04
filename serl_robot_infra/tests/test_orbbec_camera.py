"""
使用 pyorbbecsdk 实时显示 Orbbec 相机的彩色图像和深度图像
注意，pyorbbecsdk.so 需要是匹配当前版本编译才行。

操作说明:
- 按 'q' 或 ESC: 退出程序
"""

from typing import Optional

import cv2
import numpy as np

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBError,
    VideoFrame, DepthFrame
)

ESC_KEY = 27


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


def depth_frame_to_colormap(depth_frame: DepthFrame) -> Optional[np.ndarray]:
    """将深度帧转换为彩色可视化图像"""
    if depth_frame is None:
        return None

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()

    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    data = data.reshape((height, width))
    depth_data = data.astype(np.float32) * scale

    # 归一化到 0-255
    depth_vis = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    return depth_colormap


def main():
    # 初始化 Orbbec 相机
    pipeline = None
    has_color = False
    has_depth = False

    try:
        pipeline = Pipeline()
        config = Config()

        # 配置彩色传感器 (640x360)
        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                try:
                    color_profile = profile_list.get_video_stream_profile(640, 360, OBFormat.MJPG, 30)
                except OBError:
                    color_profile = profile_list.get_video_stream_profile(640, 360, OBFormat.RGB, 30)
                config.enable_stream(color_profile)
                has_color = True
                print(f"Orbbec 彩色传感器已启用: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()}fps")
        except OBError as e:
            print(f"Orbbec 彩色传感器配置错误: {e}")

        # 配置深度传感器 (640x360)
        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if profile_list is not None:
                depth_profile = profile_list.get_video_stream_profile(640, 360, OBFormat.Y16, 30)
                config.enable_stream(depth_profile)
                has_depth = True
                print(f"Orbbec 深度传感器已启用: {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()}fps")
        except OBError as e:
            print(f"Orbbec 深度传感器配置错误: {e}")

        if has_color or has_depth:
            pipeline.start(config)
            print("Orbbec 相机已启动")
        else:
            print("警告: Orbbec 相机没有可用的传感器")
            pipeline = None
    except Exception as e:
        print(f"Orbbec 相机初始化失败: {e}")
        pipeline = None

    if pipeline is None:
        print("错误: 没有可用的相机")
        return

    print("\n相机已启动! 按 'q' 或 ESC 退出\n")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(100)
                if frames is not None:
                    color_frame = frames.get_color_frame() if has_color else None
                    depth_frame = frames.get_depth_frame() if has_depth else None

                    # 显示彩色图像
                    if color_frame is not None:
                        color_image = frame_to_bgr_image(color_frame)
                        if color_image is not None:
                            cv2.imshow("Orbbec - Color", color_image)

                    # 显示深度图像
                    if depth_frame is not None:
                        depth_colormap = depth_frame_to_colormap(depth_frame)
                        if depth_colormap is not None:
                            cv2.imshow("Orbbec - Depth", depth_colormap)
            except Exception:
                pass  # 静默处理超时等错误

            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ESC_KEY:
                print("退出程序...")
                break

    except KeyboardInterrupt:
        print("\n程序被中断")
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("相机已关闭")


if __name__ == "__main__":
    main()
