#!/usr/bin/env python3
"""Move the UR arm to its safe joint home with a single blocking moveJ.

This calls URExampleEnv.move_to_safe(), which reuses the server's /jointreset
route (servoStop -> blocking moveJ to the launch script's --reset_joint_target ->
settle). moveJ (joint space) is used instead of servoL/moveL because it is the
safest way to reach a known pose from an arbitrary/singular startup pose. The
target joints are configured in launch_ur_server.sh (--reset_joint_target).
Works regardless of the server's control_mode.

The env is built with `fake_env=True` so no cameras/keyboard are started; only
the HTTP control path to the running UR server is used. So you still need the
server up first:
    bash serl_robot_infra/robot_servers/launch_ur_server.sh

Usage:
    python scripts/reset_arm.py                      # exp_name defaults to example_ur
    python scripts/reset_arm.py --exp_name example_ur --yes
"""

import argparse
import os
import sys

import numpy as np

# `experiments` lives under examples/; running this file from scripts/ does not
# put examples/ on sys.path, so add it explicitly (same package used by the
# record/train scripts).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from experiments.mappings import CONFIG_MAPPING  # noqa: E402


def reset_to_default(exp_name: str):
    """Move the arm to the experiment's safe startup pose (RESET_POSE) via blocking moveL."""
    assert exp_name in CONFIG_MAPPING, f"Unknown exp_name: {exp_name}"
    config = CONFIG_MAPPING[exp_name]()
    # fake_env=True -> no cameras/keyboard; the HTTP control path still works.
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)
    base = env.unwrapped

    base._update_currpos()
    print("Before joints (q):", np.round(base.q, 4))
    print("Before TCP pose  :", np.round(base.currpos, 4))

    base.move_to_safe()

    base._update_currpos()
    print("After  joints (q):", np.round(base.q, 4))
    print("After  TCP pose  :", np.round(base.currpos, 4))
    print("Target joints    :", np.round(np.array([
        0.9547875001667508, -0.03081106084426708, 0.19998526174013018,
        -3.140317054147856, 0.007624547434513573, -1.6147417051936608,
    ]), 4))


def main():
    parser = argparse.ArgumentParser(
        description="Reset the UR arm to the record_demos initial pose (RESET_POSE), no randomization."
    )
    parser.add_argument("--exp_name", default="example_ur", help="Experiment name in CONFIG_MAPPING.")
    parser.add_argument("--yes", action="store_true", help="Skip the safety confirmation prompt.")
    args = parser.parse_args()

    print("This script will MOVE the real UR arm to the record_demos initial pose (RESET_POSE).")
    print("Start serl_robot_infra/robot_servers/launch_ur_server.sh first.")
    print("Make sure the workspace is clear and the robot is enabled.")
    if not args.yes:
        if input("Type 'yes' to start: ").strip().lower() != "yes":
            print("Aborted.")
            return

    reset_to_default(args.exp_name)
    print("Done.")


if __name__ == "__main__":
    main()
