"""Stage 1 — read-only hardware reach test.

Verifies the physical layer is alive BEFORE starting ur_server.py:
- UR is reachable over RTDE; we can read joints, TCP pose, FT.
- Robotiq gripper TCP socket on port 63352 responds to GET POS.

This script does NOT move the robot and does NOT activate the gripper.
Safe to run any time the robot is powered on.

Usage:
    python serl_robot_infra/tests/test_ur_hardware_smoke.py --robot_ip=192.168.1.100

Exits 0 on success, 1 on any failure. Print output is human-friendly.
"""
import argparse
import socket
import sys


def check_rtde(robot_ip: str) -> bool:
    print(f"[1] rtde_receive.RTDEReceiveInterface('{robot_ip}')")
    try:
        import rtde_receive
    except ImportError:
        print("    FAIL: ur-rtde not installed. Run `pip install ur-rtde`.")
        return False
    try:
        r = rtde_receive.RTDEReceiveInterface(robot_ip)
    except Exception as e:
        print(f"    FAIL: cannot connect to UR at {robot_ip}: {e}")
        print("    Hints: is the robot powered on? In remote control mode? RTDE script loaded?")
        return False

    try:
        q = r.getActualQ()
        pose = r.getActualTCPPose()
        ft = r.getActualTCPForce()
        safety = r.getSafetyMode()
    except Exception as e:
        print(f"    FAIL: connected but read failed: {e}")
        return False

    print(f"    OK")
    print(f"    q (rad):        {[round(v, 4) for v in q]}")
    print(f"    TCP pose (UR):  {[round(v, 4) for v in pose]}  (xyz + axis-angle)")
    print(f"    TCP force:      {[round(v, 3) for v in ft]}  (fx,fy,fz,tx,ty,tz)")
    print(f"    safety_mode:    {safety}  (1 = NORMAL)")
    if safety != 1:
        print(f"    WARNING: safety mode is {safety}, not NORMAL. Investigate before motion.")
    return True


def check_gripper_socket(robot_ip: str, gripper_port: int = 63352) -> bool:
    print(f"[2] gripper TCP socket {robot_ip}:{gripper_port}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((robot_ip, gripper_port))
            s.sendall(b"GET POS\n")
            data = s.recv(1024)
            response = data.decode("utf-8", errors="ignore").strip()
    except socket.timeout:
        print(f"    FAIL: timeout. Is the gripper plugged in and powered? Is the port right?")
        return False
    except ConnectionRefusedError:
        print(f"    FAIL: connection refused. Robotiq URCap may not be running on the UR.")
        return False
    except Exception as e:
        print(f"    FAIL: {e}")
        return False

    if not response:
        print("    FAIL: empty response from gripper")
        return False
    print(f"    OK")
    print(f"    raw response: '{response}'  (expect 'ACK <pos_byte>' once activated, or just 'POS <byte>')")
    try:
        if " " in response:
            pos_byte = int(response.split(" ", 1)[1])
        else:
            pos_byte = int(response)
        distance_mm = 85.0 * (1.0 - pos_byte / 255.0)
        print(f"    pos_byte={pos_byte} -> distance={distance_mm:.1f} mm  "
              f"(85=open, 0=closed; activation not required to read POS)")
    except ValueError:
        print(f"    NOTE: could not parse pos_byte from response; gripper may not be activated yet "
              f"(stage 2 test will activate it)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot_ip", default="192.168.1.100")
    ap.add_argument("--gripper_port", type=int, default=63352)
    args = ap.parse_args()

    print(f"=== Stage 1: hardware reach test against {args.robot_ip} ===")
    ok_rtde = check_rtde(args.robot_ip)
    ok_grip = check_gripper_socket(args.robot_ip, args.gripper_port)
    print()
    if ok_rtde and ok_grip:
        print("Stage 1 PASS — hardware is reachable. Proceed to test_ur_gripper.py.")
        sys.exit(0)
    print("Stage 1 FAIL — fix the failing line(s) above before proceeding.")
    sys.exit(1)


if __name__ == "__main__":
    main()
