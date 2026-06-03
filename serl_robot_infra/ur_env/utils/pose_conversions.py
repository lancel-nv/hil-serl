"""Pose conversions between hil-serl's quaternion convention and UR's axis-angle.

hil-serl uses scipy `Rotation` xyzw quaternions throughout (see franka_env.py).
UR's RTDE TCP-pose API uses axis-angle (rotation vector) for orientation:
`[x, y, z, rx, ry, rz]` where the rotation vector encodes both axis and angle.

These two helpers are the only kinematics conversions UREnv / URServer own —
everything else (IK, FK, trajectory interpolation) is delegated to the UR
controller via `rtde_control.servoL`. Correctness of these two functions is
therefore load-bearing for safe motion; tests/test_ur_pose_conversions.py
verifies round-trip accuracy to < 1e-7.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_to_axisangle(quat_xyzw):
    """Convert scipy-convention quaternion (x, y, z, w) to axis-angle (3-vector)."""
    return R.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_rotvec()


def axisangle_to_quat(rotvec):
    """Convert axis-angle (3-vector) to scipy-convention quaternion (x, y, z, w)."""
    return R.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_quat()


def pose7_to_ur6(pose7):
    """Convert [x, y, z, qx, qy, qz, qw] (7-vector) to UR [x, y, z, rx, ry, rz] (6-vector)."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    return np.concatenate([pose7[:3], quat_to_axisangle(pose7[3:])])


def ur6_to_pose7(ur6):
    """Convert UR [x, y, z, rx, ry, rz] (6-vector) to [x, y, z, qx, qy, qz, qw] (7-vector)."""
    ur6 = np.asarray(ur6, dtype=np.float64)
    return np.concatenate([ur6[:3], axisangle_to_quat(ur6[3:])])


if __name__ == "__main__":
    import requests
    url = "http://127.0.0.1:5000/"
    pose7 = requests.post(url + "getstate", timeout=2).json()["pose"]  # [x,y,z,qx,qy,qz,qw]
    print(pose7)

    pose6 = requests.post(url + "getpos_euler", timeout=2).json()["pose"]
    print(pose6)



