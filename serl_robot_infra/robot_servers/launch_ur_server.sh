#!/usr/bin/env bash
# Launch the UR + Robotiq Flask server. No ROS required for this path.

python ur_server.py \
    --robot_ip=192.168.1.100 \
    --gripper_type=Robotiq \
    --gripper_port=63352 \
    --reset_joint_target=0.09375287592411041,-1.3284757894328614,-2.3687846660614014,-1.015773133640625,1.5728812217712402,0.1377517282962799 \
    --flask_url=127.0.0.1 \
    --flask_port=5000
