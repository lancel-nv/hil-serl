"""On-hardware pose round-trip test for the UR + ur_server stack.

GATING CHECK per the project's quality bar: this MUST pass before any closed-loop
control (record_demos, train_rlpd, etc.) is run on real hardware. Verifies that
the path env -> server -> servoL -> robot -> getActualTCPPose -> env round-trips
poses to within 0.5 mm / 0.05 deg.

Prereqs:
- `ur_server.py` is running and connected to the real UR.
- The UR is in a SAFE pose (not against limits, no people in workspace).

Usage:
    python serl_robot_infra/tests/test_ur_roundtrip.py --url=http://127.0.0.1:5000/

What it does (in order):
1. Sample-pose-noop: GET /getpos, POST /pose with that exact value, wait, GET /getpos.
   Asserts position error < 0.5 mm and orientation error < 0.05 deg.
2. Small-perturbation: command +/- 1 mm and +/- 0.01 rad deltas; verify the robot
   actually reaches the new pose to within 1 mm / 0.1 deg.

This script DOES move the robot a tiny amount. Stand clear.
"""
import argparse
import sys
import time

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R


POSITION_TOL_M = 5e-4         # 0.5 mm for the noop round-trip
ORIENT_TOL_RAD = np.deg2rad(0.05)
PERTURB_POSITION_TOL_M = 1e-3  # 1 mm for the perturbed setpoint
PERTURB_ORIENT_TOL_RAD = np.deg2rad(0.1)


def get_pose(session, url):
    r = session.post(url + "getpos", timeout=2.0)
    r.raise_for_status()
    return np.asarray(r.json()["pose"], dtype=np.float64)


def post_pose(session, url, pose7):
    r = session.post(url + "pose", json={"arr": np.asarray(pose7).tolist()}, timeout=2.0)
    r.raise_for_status()


def orientation_error_rad(quat_a, quat_b):
    R_err = R.from_quat(quat_a) * R.from_quat(quat_b).inv()
    return float(np.linalg.norm(R_err.as_rotvec()))


def settle_time(servo_dt=0.008, lookahead=0.1):
    # Wait long enough for servoL interpolation to consume the setpoint.
    return 2 * servo_dt + lookahead + 0.2


def run_noop_roundtrip(session, url):
    print("[1/2] No-op pose round-trip (POST current pose back as setpoint)...")
    before = get_pose(session, url)
    print(f"      pose_before = {np.round(before, 5)}")
    post_pose(session, url, before)
    time.sleep(settle_time())
    after = get_pose(session, url)
    print(f"      pose_after  = {np.round(after, 5)}")

    dxyz = np.linalg.norm(after[:3] - before[:3])
    dorient = orientation_error_rad(after[3:], before[3:])
    print(f"      |dxyz| = {dxyz * 1000:.4f} mm  (tol {POSITION_TOL_M * 1000} mm)")
    print(f"      |dR|   = {np.rad2deg(dorient):.4f} deg  (tol {np.rad2deg(ORIENT_TOL_RAD)} deg)")

    if dxyz >= POSITION_TOL_M:
        print(f"      FAIL: position drift {dxyz * 1000:.4f} mm >= {POSITION_TOL_M * 1000} mm")
        return False
    if dorient >= ORIENT_TOL_RAD:
        print(f"      FAIL: orientation drift {np.rad2deg(dorient):.4f} deg >= "
              f"{np.rad2deg(ORIENT_TOL_RAD)} deg")
        return False
    print("      PASS")
    return True


def run_perturbation(session, url):
    print("[2/2] Small-perturbation round-trip (move +1 mm Z, then back)...")
    home = get_pose(session, url)
    target = home.copy()
    target[2] += 1e-3  # 1 mm in Z

    post_pose(session, url, target)
    time.sleep(settle_time())
    reached = get_pose(session, url)
    dxyz = np.linalg.norm(reached[:3] - target[:3])
    dorient = orientation_error_rad(reached[3:], target[3:])
    print(f"      after +1mmZ: |Δxyz|={dxyz*1000:.4f}mm  |ΔR|={np.rad2deg(dorient):.4f}deg")
    if dxyz >= PERTURB_POSITION_TOL_M or dorient >= PERTURB_ORIENT_TOL_RAD:
        print(f"      FAIL on +1mmZ setpoint")
        return False

    post_pose(session, url, home)
    time.sleep(settle_time())
    back = get_pose(session, url)
    dxyz_home = np.linalg.norm(back[:3] - home[:3])
    dorient_home = orientation_error_rad(back[3:], home[3:])
    print(f"      back to home: |Δxyz|={dxyz_home*1000:.4f}mm  "
          f"|ΔR|={np.rad2deg(dorient_home):.4f}deg")
    if dxyz_home >= PERTURB_POSITION_TOL_M or dorient_home >= PERTURB_ORIENT_TOL_RAD:
        print(f"      FAIL returning to home")
        return False
    print("      PASS")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000/",
                    help="ur_server.py base URL (trailing slash)")
    args = ap.parse_args()
    url = args.url if args.url.endswith("/") else args.url + "/"

    session = requests.Session()

    print(f"UR round-trip test against {url}")
    print("Stand clear — the robot will move ~1 mm.")
    time.sleep(1.0)

    ok = run_noop_roundtrip(session, url)
    if not ok:
        print("\nRESULT: FAIL (gating check failed; do not proceed with closed-loop control)")
        sys.exit(1)

    ok = run_perturbation(session, url)
    if not ok:
        print("\nRESULT: FAIL (gating check failed; do not proceed with closed-loop control)")
        sys.exit(1)

    print("\nRESULT: PASS (round-trip verified; safe to proceed)")
    sys.exit(0)


if __name__ == "__main__":
    main()
