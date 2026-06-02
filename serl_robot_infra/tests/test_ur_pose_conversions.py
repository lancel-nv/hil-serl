"""Offline verification of quat <-> axis-angle and 7D pose <-> UR 6D pose conversions.

Run: python -m pytest serl_robot_infra/tests/test_ur_pose_conversions.py -v
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from ur_env.utils.pose_conversions import (
    quat_to_axisangle,
    axisangle_to_quat,
    pose7_to_ur6,
    ur6_to_pose7,
)


TOL = 1e-7


def _quat_close(q1, q2, tol=TOL):
    """Quaternion equality up to double-cover sign ambiguity."""
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    return min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2)) < tol


def _rotation_close(q1, q2, tol=TOL):
    m1 = R.from_quat(q1).as_matrix()
    m2 = R.from_quat(q2).as_matrix()
    return np.max(np.abs(m1 - m2)) < tol


def test_random_rotation_roundtrip_quat():
    rots = R.random(1000, random_state=42)
    for r in rots:
        q = r.as_quat()
        aa = quat_to_axisangle(q)
        q2 = axisangle_to_quat(aa)
        assert _quat_close(q, q2), f"Quat round-trip failed: {q} -> {q2}"


def test_random_rotation_roundtrip_matrix():
    rots = R.random(1000, random_state=43)
    for r in rots:
        q = r.as_quat()
        aa = quat_to_axisangle(q)
        q2 = axisangle_to_quat(aa)
        assert _rotation_close(q, q2), f"Rotation matrix round-trip failed for quat {q}"


def test_identity():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    aa = quat_to_axisangle(q)
    assert np.allclose(aa, 0.0, atol=TOL)
    q2 = axisangle_to_quat(aa)
    assert _quat_close(q, q2)


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
def test_180_deg_rotations(axis):
    aa = np.pi * np.asarray(axis, dtype=np.float64)
    q = axisangle_to_quat(aa)
    aa2 = quat_to_axisangle(q)
    # The 180-deg rotation vector is ambiguous in sign; check via matrix.
    m1 = R.from_rotvec(aa).as_matrix()
    m2 = R.from_rotvec(aa2).as_matrix()
    assert np.max(np.abs(m1 - m2)) < TOL


def test_tiny_rotation():
    aa = np.array([1e-8, 0.0, 0.0])
    q = axisangle_to_quat(aa)
    aa2 = quat_to_axisangle(q)
    assert np.allclose(aa, aa2, atol=1e-10)


def test_full_pose_roundtrip():
    rng = np.random.default_rng(7)
    for _ in range(200):
        xyz = rng.uniform(-1.0, 1.0, size=3)
        quat = R.random(random_state=rng.integers(0, 10_000_000)).as_quat()
        pose7 = np.concatenate([xyz, quat])
        ur6 = pose7_to_ur6(pose7)
        pose7_back = ur6_to_pose7(ur6)
        assert np.allclose(pose7[:3], pose7_back[:3], atol=TOL)
        assert _quat_close(pose7[3:], pose7_back[3:])


def test_pose7_to_ur6_shape():
    pose7 = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    ur6 = pose7_to_ur6(pose7)
    assert ur6.shape == (6,)
    assert np.allclose(ur6[:3], pose7[:3])
    assert np.allclose(ur6[3:], 0.0)


def test_ur6_to_pose7_shape():
    ur6 = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    pose7 = ur6_to_pose7(ur6)
    assert pose7.shape == (7,)
    assert np.allclose(pose7[:3], ur6[:3])
    assert _quat_close(pose7[3:], [0.0, 0.0, 0.0, 1.0])
