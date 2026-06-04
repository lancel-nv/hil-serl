import numpy as np

from franka_env.envs.wrappers import GripperCloseEnv
from ur_env.envs.ur_env import UREnv


class PositionOnlyGripperCloseEnv(GripperCloseEnv):
    """Position-only control: keep the gripper closed AND zero the rotation action.

    Temporary workaround for the clip_safety_box roll-wraparound flip bug
    (see README_LANCEL.md "已知问题"). The 6-dim action space is preserved so the
    RelativeFrame / KeyboardIntervention dimensions are unchanged; the rotation
    dims (rx, ry, rz) simply have no effect, so the wrist stays at the reset
    orientation and clip_safety_box never triggers the flip.

    Revert to GripperCloseEnv once the clip bug is fixed to restore 6-DoF.
    """

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = super().action(action)
        new_action[3:6] = 0.0
        return new_action


class URExampleEnv(UREnv):
    """Reach-only example task on UR.

    Episode resets use the base UREnv.go_to_reset (servoL streaming via
    interpolate_move). For a safe startup, call move_to_safe() ONCE before the
    loop: it does a single blocking moveJ to the configured safe joint home so we
    never stream servoL from a possibly-singular startup pose. After that, all
    motion is servoL.
    """

    def move_to_safe(self):
        """One-time blocking moveJ to the safe joint home before any servoL streaming.

        Starting a servoL stream from an arbitrary/singular startup pose is risky;
        a joint-space moveJ is the safest way to reach a known pose. This reuses the
        server's /jointreset route (servoStop -> blocking moveJ to the launch
        script's --reset_joint_target -> settle/verify), so it works regardless of
        control_mode. Returns once the move is complete; all later resets/steps use
        servoL.
        """
        print("[example_ur] one-time moveJ to safe joint home (/jointreset)")
        self.session.post(self.url + "jointreset")
        self._update_currpos()
