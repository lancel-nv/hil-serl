#!/usr/bin/env bash
# Launch the UR + Robotiq Flask server. No ROS required for this path.

# Run from this script's directory so ur_server.py resolves regardless of CWD.
cd "$(dirname "$0")" || exit 1

# home choise:
# - pnp place:0.1498289853334427,-2.2200495205321253,-1.588263988494873,-0.9019187253764649,1.5800871849060059,0.1958753913640976
python ur_server.py \
    --robot_ip=192.168.1.100 \
    --gripper_type=Robotiq \
    --gripper_port=63352 \
    --reset_joint_target=0.1498289853334427,-2.2200495205321253,-1.588263988494873,-0.9019187253764649,1.5800871849060059,0.1958753913640976 \
    --control_mode=servol \
    --servo_dt=0.1 \
    --flask_url=127.0.0.1 \
    --flask_port=5000

# servo_dt MUST match the env control period (1/hz). UREnv runs at hz=10 -> 0.1s.
# If servo_dt is too small (e.g. the old 0.008 from the 125Hz gr00t path), servoL
# only controls the arm for a few ms each cycle and then times out, so keyboard
# jogging barely moves. Raise UREnv hz and lower servo_dt together for snappier
# servoing.
