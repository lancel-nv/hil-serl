#!/usr/bin/env bash
# Launch the UR + Robotiq Flask server. No ROS required for this path.

python ur_server.py \
    --robot_ip=192.168.1.100 \
    --gripper_type=Robotiq \
    --gripper_port=63352 \
    --reset_joint_target=0.1498289853334427,-2.2200495205321253,-1.588263988494873,-0.9019187253764649,1.5800871849060059,0.1958753913640976 \
    --flask_url=127.0.0.1 \
    --flask_port=5000
