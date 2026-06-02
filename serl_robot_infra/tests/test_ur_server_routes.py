"""Stage 3 — Flask server routes test.

With `ur_server.py` already running and connected to the real UR, POST every
endpoint that UREnv will use and verify the response shape / content. Does NOT
move the arm. Does NOT activate the gripper if not already activated (we just
read /get_gripper as a smoke check).

Use this to catch wiring errors (wrong shape, JSON missing keys, stale cache,
unreachable Flask) BEFORE the round-trip test that actually moves the robot.

Usage:
    # in terminal A
    bash serl_robot_infra/robot_servers/launch_ur_server.sh
    # in terminal B
    python serl_robot_infra/tests/test_ur_server_routes.py --url=http://127.0.0.1:5000/

Exits 0 on success, 1 on any failure.
"""
import argparse
import sys
import time

import numpy as np
import requests


def _post(session, url, route, json=None):
    full = url + route
    r = session.post(full, json=json, timeout=3.0)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return r.text  # routes like /clearerr return plain text


def check_state_routes(session, url):
    print("[1] read-only state routes")

    # /getstate has the union of everything
    state = _post(session, url, "getstate")
    required_keys = {"pose", "vel", "force", "torque", "q", "dq", "jacobian", "gripper_pos"}
    missing = required_keys - set(state.keys())
    if missing:
        print(f"    FAIL: /getstate missing keys: {missing}")
        return False
    pose = np.asarray(state["pose"])
    vel = np.asarray(state["vel"])
    force = np.asarray(state["force"])
    torque = np.asarray(state["torque"])
    q = np.asarray(state["q"])
    dq = np.asarray(state["dq"])
    jac = np.asarray(state["jacobian"])
    print(f"    /getstate shapes: pose={pose.shape} vel={vel.shape} force={force.shape} "
          f"torque={torque.shape} q={q.shape} dq={dq.shape} jacobian={jac.shape}")
    print(f"    pose = {np.round(pose, 4)}  (expect 7-vec xyz+quat)")
    print(f"    q    = {np.round(q, 4)}  (expect 6-vec)")
    print(f"    gripper_pos = {state['gripper_pos']:.3f}  (1.0 = closed, 0.0 = open)")

    checks = [
        ("pose shape", pose.shape == (7,)),
        ("quaternion is unit (||q|| ~ 1)", abs(np.linalg.norm(pose[3:]) - 1.0) < 1e-3),
        ("vel shape", vel.shape == (6,)),
        ("force shape", force.shape == (3,)),
        ("torque shape", torque.shape == (3,)),
        ("q shape (UR is 6-DoF)", q.shape == (6,)),
        ("dq shape", dq.shape == (6,)),
        ("jacobian length 36 (6x6)", jac.size == 36),
        ("gripper_pos in [0, 1]", 0.0 <= float(state["gripper_pos"]) <= 1.0),
    ]
    all_ok = True
    for name, ok in checks:
        print(f"      {'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            all_ok = False
    return all_ok


def check_individual_routes(session, url):
    print("[2] individual getter routes (each should return the same data as /getstate)")
    routes = {
        "getpos": ("pose", (7,)),
        "getpos_euler": ("pose", (6,)),
        "getvel": ("vel", (6,)),
        "getforce": ("force", (3,)),
        "gettorque": ("torque", (3,)),
        "getq": ("q", (6,)),
        "getdq": ("dq", (6,)),
        "getjacobian": ("jacobian", None),  # nested list (6, 6)
        "get_gripper": ("gripper", None),   # scalar
    }
    all_ok = True
    for route, (key, expected_shape) in routes.items():
        try:
            resp = _post(session, url, route)
            val = resp.get(key)
            if expected_shape is not None:
                arr = np.asarray(val)
                ok = arr.shape == expected_shape
                print(f"      {'OK  ' if ok else 'FAIL'} /{route} -> '{key}' shape={arr.shape} "
                      f"(expected {expected_shape})")
            else:
                ok = val is not None
                print(f"      {'OK  ' if ok else 'FAIL'} /{route} -> '{key}'={val}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"      FAIL /{route}: {e}")
            all_ok = False
    return all_ok


def check_noop_routes(session, url):
    print("[3] no-op control routes (these should accept and return OK without effect)")
    try:
        _post(session, url, "clearerr")
        print("      OK   /clearerr")
        _post(session, url, "update_param", json={})
        print("      OK   /update_param (no-op on UR)")
        _post(session, url, "startimp")
        print("      OK   /startimp (no-op on UR)")
        _post(session, url, "stopimp")
        print("      OK   /stopimp (no-op on UR)")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        return False


def check_state_cache_freshness(session, url):
    print("[4] state-cache freshness (consecutive /getstate calls; pose should be stable when arm is idle)")
    poses = []
    for _ in range(5):
        st = _post(session, url, "getstate")
        poses.append(np.asarray(st["pose"]))
        time.sleep(0.05)
    diffs = [np.linalg.norm(poses[i] - poses[0]) for i in range(1, 5)]
    print(f"      max drift across 5 reads: {max(diffs):.6f}  (expect < 1e-4 on a stationary arm)")
    if max(diffs) > 1e-3:
        print(f"      WARN: cache is updating with > 1mm drift; either the arm is moving or the "
              f"state thread is noisy. Not necessarily a failure.")
    return True


def check_gripper_routes(session, url, exercise: bool):
    if not exercise:
        print("[5] gripper routes — SKIPPED (pass --exercise_gripper to activate and move)")
        return True
    print("[5] gripper routes — gripper WILL MOVE")
    print("    /activate_gripper")
    _post(session, url, "activate_gripper")
    time.sleep(1.0)
    print("    /open_gripper")
    _post(session, url, "open_gripper")
    time.sleep(1.2)
    open_pos = _post(session, url, "get_gripper")["gripper"]
    print(f"      gripper_pos after open: {open_pos:.3f}  (expect ~0.0)")
    print("    /close_gripper")
    _post(session, url, "close_gripper")
    time.sleep(1.2)
    close_pos = _post(session, url, "get_gripper")["gripper"]
    print(f"      gripper_pos after close: {close_pos:.3f}  (expect > 0.7)")
    print("    /open_gripper (leave in known state)")
    _post(session, url, "open_gripper")
    time.sleep(1.0)
    ok = open_pos < 0.2 and close_pos > 0.7
    print(f"      {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:5000/")
    ap.add_argument("--exercise_gripper", action="store_true",
                    help="If set, also activate + open + close the gripper through HTTP routes.")
    args = ap.parse_args()
    url = args.url if args.url.endswith("/") else args.url + "/"

    print(f"=== Stage 3: Flask server routes test against {url} ===")
    session = requests.Session()

    results = []
    results.append(("state routes",       check_state_routes(session, url)))
    results.append(("individual getters", check_individual_routes(session, url)))
    results.append(("no-op routes",       check_noop_routes(session, url)))
    results.append(("cache freshness",    check_state_cache_freshness(session, url)))
    results.append(("gripper routes",     check_gripper_routes(session, url, args.exercise_gripper)))

    print()
    all_ok = all(ok for _, ok in results)
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print()
    if all_ok:
        print("Stage 3 PASS — server routes are wired correctly. "
              "Now run test_ur_roundtrip.py (THIS WILL MOVE THE ARM ~1 mm).")
        sys.exit(0)
    print("Stage 3 FAIL — fix the failing checks before running the pose round-trip test.")
    sys.exit(1)


if __name__ == "__main__":
    main()
