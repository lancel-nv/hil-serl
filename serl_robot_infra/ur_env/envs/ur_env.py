"""Gym Interface for Universal Robots (UR) + Robotiq 2F.

Structural near-clone of `franka_env.envs.franka_env.FrankaEnv`. Differences:
1. `requests.Session()` for HTTP connection reuse against `ur_server.py`.
2. `CAMERAS` config dict carries a `type` discriminator so each entry is
   dispatched to either `RSCapture` or `OrbbecCapture`. The downstream
   `VideoCapture` thread wrapper is shared.
3. `_update_currpos` reshapes the jacobian to (6, 6) and reads (6,) joint
   vectors — UR has 6 DoF, Franka has 7.

Everything else (`step`, `reset`, `go_to_reset`, `interpolate_move`,
`clip_safety_box`, `compute_reward`, `_send_pos_command`,
`_send_gripper_command`, `_get_obs`, video saving, ImageDisplayer) is identical
to FrankaEnv. The robot-agnostic helpers (`rotations`, `VideoCapture`,
`RSCapture`) are imported from `franka_env` rather than duplicated, so any fix
there flows through to both arms.

Compliance with hil-serl quality bars for this port:
- IK is delegated to ur_server's `servoL`. Our only kinematics math is the
  quat <-> axis-angle conversion inside `ur_server`, verified by
  `tests/test_ur_pose_conversions.py` and `tests/test_ur_roundtrip.py`.
- Safety hardening (workspace bounds on server, FT watchdog, etc.) is deferred
  per the project memory `feedback_ur_port_quality_bars.md`.
"""
import copy
import os
import queue
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict

import cv2
import gymnasium as gym
import numpy as np
import requests
from scipy.spatial.transform import Rotation

from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler
from ur_env.camera.orbbec_capture import OrbbecCapture


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        super().__init__()
        self.queue = queue
        self.daemon = True
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break
            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k],
                axis=1,
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for UREnv. Fill in the values below.

    Camera entries are now of the form:
        {"type": "realsense", "serial_number": "...", "dim": (W, H), "exposure": int}
        {"type": "orbbec",    "dim": (W, H), "serial_number": "..." (optional)}
    """

    SERVER_URL: str = "http://127.0.0.1:5000/"
    CAMERAS: Dict[str, Dict] = {}
    IMAGE_CROP: Dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    # UR does not have impedance compliance/precision params. These remain as
    # empty dicts so FrankaEnv-style configs that POST them to /update_param
    # continue to work (the route is a no-op on the UR server).
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0


##############################################################################


class UREnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP

        self.session = requests.Session()

        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )

        # State that _update_currpos populates.
        self.currpos = np.zeros(7)
        self.currvel = np.zeros(6)
        self.currforce = np.zeros(3)
        self.currtorque = np.zeros(3)
        self.currjacobian = np.zeros((6, 6))
        self.q = np.zeros(6)
        self.dq = np.zeros(6)
        self.curr_gripper_pos = np.zeros(1)
        self.terminate = False

        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # Safety box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )

        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                        for key in config.CAMERAS
                    }
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self._update_currpos()

        self.cap = None
        self.init_cameras(config.CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        if set_load:
            input("Confirm robot is safe to set payload, then press enter.")
            self.session.post(self.url + "set_load", json=self.config.LOAD_PARAM)

        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.esc:
                self.terminate = True

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

        print("Initialized UREnv")

    # ---- helpers identical to FrankaEnv ----

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
        sign = np.sign(euler[0])
        euler[0] = sign * np.clip(
            np.abs(euler[0]),
            self.rpy_bounding_box.low[0],
            self.rpy_bounding_box.high[0],
        )
        euler[1:] = np.clip(euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:])
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()
        return pose

    def step(self, action: np.ndarray) -> tuple:
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
        self.nextpos[3:] = (
            Rotation.from_rotvec(action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        gripper_action = action[6] * self.action_scale[2]

        self._send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        return bool(np.all(delta < self._REWARD_THRESHOLD))

    def get_im(self) -> Dict[str, np.ndarray]:
        images = {}
        display_images = {}
        full_res_images = {}
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = (
                    self.config.IMAGE_CROP[key](rgb)
                    if key in self.config.IMAGE_CROP
                    else rgb
                )
                resized = cv2.resize(
                    cropped_rgb,
                    self.observation_space["images"][key].shape[:2][::-1],
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.CAMERAS)
                return self.get_im()

        if self.save_video:
            self.recording_frames.append(full_res_images)
        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        self.session.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        if joint_reset:
            print("JOINT RESET")
            self.session.post(self.url + "jointreset")
            time.sleep(0.5)

        if self.randomreset:
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            self.interpolate_move(self.resetpos.copy(), timeout=1)

        self.session.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        self.session.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists("./videos"):
                    os.makedirs("./videos")
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                for camera_key in self.recording_frames[0].keys():
                    video_path = f"./videos/ur_{camera_key}_{timestamp}.mp4"
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, cameras=None):
        if self.cap is not None:
            self.close_cameras()
        self.cap = OrderedDict()
        for cam_name, kwargs in (cameras or {}).items():
            kw = dict(kwargs)
            cam_type = kw.pop("type", "realsense")
            if cam_type == "realsense":
                # Lazy import so users with Orbbec-only setups don't need pyrealsense2.
                from franka_env.camera.rs_capture import RSCapture
                cap = RSCapture(name=cam_name, **kw)
            elif cam_type == "orbbec":
                cap = OrbbecCapture(name=cam_name, **kw)
            else:
                raise ValueError(f"Unknown camera type '{cam_type}' for camera '{cam_name}'")
            self.cap[cam_name] = VideoCapture(cap)

    def close_cameras(self):
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        self.session.post(self.url + "clearerr")

    def _send_pos_command(self, pos: np.ndarray):
        self._recover()
        arr = np.array(pos).astype(np.float32)
        self.session.post(self.url + "pose", json={"arr": arr.tolist()})

    def _send_gripper_command(self, pos: float, mode="binary"):
        if mode == "binary":
            if (
                (pos <= -0.5)
                and (self.curr_gripper_pos > 0.85)
                and (time.time() - self.last_gripper_act > self.gripper_sleep)
            ):
                self.session.post(self.url + "close_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (
                (pos >= 0.5)
                and (self.curr_gripper_pos < 0.85)
                and (time.time() - self.last_gripper_act > self.gripper_sleep)
            ):
                self.session.post(self.url + "open_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            else:
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self):
        """Read cached state from the UR server. Jacobian is (6,6) on UR vs (6,7) on Franka."""
        ps = self.session.post(self.url + "getstate").json()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])
        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 6))
        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])
        self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def update_currpos(self):
        self._update_currpos()

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
