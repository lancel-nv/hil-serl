"""Minimal UR + Robotiq example task.

A non-contact "reach a fixed pose" task — intentionally simple so the user can
verify the UR + Robotiq integration end-to-end (server boot, env reset,
keyboard teleop, demo recording) without depending on a physical fixture.

Before running:
1. Start ur_server.py against the UR.
2. Calibrate TARGET_POSE / RESET_POSE for your workspace by driving the arm
   to the desired pose with keyboard teleop and reading /getpos_euler.
3. Replace the camera serial numbers below with the ones for your setup.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    KeyboardIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    Quat2EulerWrapper,
)
from serl_launcher.networks.reward_classifier import load_classifier_func
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from experiments.config import DefaultTrainingConfig
from experiments.example_ur.wrapper import URExampleEnv
from ur_env.envs.ur_env import DefaultEnvConfig


class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    CAMERAS = {
        "wrist": {
            "type": "orbbec",
            "dim": (640, 360),
            # "serial_number": "<your-orbbec-serial>",  # optional when only one Orbbec is attached
        },
    }
    IMAGE_CROP = {}

    # Conservative placeholders — REPLACE with values from /getpos_euler at your safe workspace pose.
    # TARGET_POSE = np.array([0.40, 0.00, 0.30, np.pi, 0.0, 0.0])
    TARGET_POSE = np.array([0.964, -0.033, 0.225, -np.pi, 0.0, -np.pi/2])


    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.15, 0, 0, 0])
    REWARD_THRESHOLD = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.05, 0.05, 0.03, 0.2, 0.2, 0.3])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.05, 0.05, 0.20, 0.2, 0.2, 0.3])

    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = 0.00

    # Deliberately slow for first-run safety; raise once verified.
    ACTION_SCALE = (0.005, 0.03, 1.0)

    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    GRIPPER_SLEEP = 0.6
    JOINT_RESET_PERIOD = 0

    COMPLIANCE_PARAM = {}
    PRECISION_PARAM = {}
    RESET_PARAM = {}


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist"]
    classifier_keys = ["wrist"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = URExampleEnv(fake_env=fake_env, save_video=save_video, config=EnvConfig())
        env = GripperCloseEnv(env)
        if not fake_env:
            env = KeyboardIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # The training scripts (record_demos.py, train_bc.py, train_rlpd.py) all pass
        # classifier=True unconditionally. For this reach task we use the pose-based
        # binary reward built into UREnv.compute_reward, so we only wrap in the
        # vision classifier when a trained checkpoint actually exists.
        classifier_ckpt = os.path.abspath("classifier_ckpt/")
        if classifier and os.path.isdir(classifier_ckpt) and os.listdir(classifier_ckpt):
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=classifier_ckpt,
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier_fn(obs)) > 0.85)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        elif classifier:
            print("[example_ur] classifier=True but no classifier_ckpt/ found; "
                  "falling back to pose-based binary reward (REWARD_THRESHOLD).")
        return env
