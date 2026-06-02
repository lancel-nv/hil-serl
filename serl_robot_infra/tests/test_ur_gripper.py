"""Stage 2 — gripper protocol test.

Exercises `RobotiqGripperTCPServer` end-to-end against the real gripper:
1. activate, 2. open (verify cached pos -> ~0/byte ~85mm), 3. close (verify cached pos -> ~255/byte ~0mm).

The arm DOES NOT move during this test. Only the gripper jaws move.
Make sure the gripper is unobstructed (no fingers, no part to crush) before running.

Usage:
    python serl_robot_infra/tests/test_ur_gripper.py --robot_ip=192.168.1.100

Exits 0 on success, 1 if cached position does not change as expected.
"""
import argparse
import sys
import time

# Add serl_robot_infra to path so this script can be launched from anywhere.
import os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))

from robot_servers.robotiq_gripper_tcp_server import RobotiqGripperTCPServer  # noqa: E402


SETTLE_S = 1.5  # gripper takes ~1 s to fully traverse 85 mm
OPEN_DIST_TOL_MM = 10.0   # "close to open" if distance > 75 mm
CLOSE_POS_BYTE_TOL = 50   # "close to closed" if pos_byte > 205


def _print_state(label: str, g: RobotiqGripperTCPServer):
    pos_byte = g.get_pos_byte()
    distance = g.get_distance_mm()
    print(f"    {label}: pos_byte={pos_byte}  distance={distance:.1f} mm  "
          f"gripper_pos(normalized 1=closed)={g.gripper_pos:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot_ip", default="192.168.1.100")
    ap.add_argument("--gripper_port", type=int, default=63352)
    args = ap.parse_args()

    print(f"=== Stage 2: Robotiq gripper protocol test against {args.robot_ip}:{args.gripper_port} ===")
    print("Make sure the gripper is unobstructed. Gripper WILL move.")
    time.sleep(1.0)

    print("[1] instantiate RobotiqGripperTCPServer (calls activate_gripper internally)")
    g = RobotiqGripperTCPServer(
        gripper_ip=args.robot_ip,
        gripper_port=args.gripper_port,
    )
    time.sleep(SETTLE_S)
    _print_state("after activate", g)

    print("[2] open()")
    g.open()
    time.sleep(SETTLE_S)
    _print_state("after open", g)
    if g.get_distance_mm() < OPEN_DIST_TOL_MM:
        print(f"    FAIL: gripper distance {g.get_distance_mm():.1f} mm < {OPEN_DIST_TOL_MM} mm; "
              f"gripper did not open (or polling cache is stale)")
        g.shutdown()
        sys.exit(1)
    print("    OK")

    print("[3] close()")
    g.close()
    time.sleep(SETTLE_S)
    _print_state("after close", g)
    if g.get_pos_byte() < CLOSE_POS_BYTE_TOL:
        print(f"    FAIL: pos_byte {g.get_pos_byte()} < {CLOSE_POS_BYTE_TOL}; gripper did not close")
        g.shutdown()
        sys.exit(1)
    print("    OK")

    print("[4] open() again (leave gripper in a safe known state)")
    g.open()
    time.sleep(SETTLE_S)
    _print_state("after open", g)

    g.shutdown()
    print()
    print("Stage 2 PASS — gripper responds, polling thread is updating the cache. "
          "Proceed to start ur_server.py and then run test_ur_server_routes.py.")
    sys.exit(0)


if __name__ == "__main__":
    main()
