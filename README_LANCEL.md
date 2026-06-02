

机械臂控制模式支持
  默认（moveL）：
  bash serl_robot_infra/robot_servers/launch_ur_server.sh

  临时切到 servoL：
  python ur_server.py --control_mode=servol --robot_ip=192.168.1.100 ...

## 测试与真机验证流程

### 分阶段测试总览

| Stage | 命令                                                   | 动手臂？ | 动夹爪？ | 需要 server？ |
|-------|--------------------------------------------------------|:-------:|:-------:|:------------:|
| 0     | `pytest tests/test_ur_pose_conversions.py`              |   ❌    |   ❌    |      ❌      |
| 1     | `python tests/test_ur_hardware_smoke.py --robot_ip=192.168.1.100` |   ❌    |   ❌    |      ❌      |
| 2     | `python tests/test_ur_gripper.py --robot_ip=192.168.1.100`        |   ❌    |   ✅    |      ❌      |
| 3     | `python tests/test_ur_server_routes.py --url=http://127.0.0.1:5000/` |   ❌    |  可选   |      ✅      |
| 4     | `python tests/test_ur_roundtrip.py --url=http://127.0.0.1:5000/`    | ✅ (~1mm) |   ❌    |      ✅      |

---

### 各 stage 说明

- **Stage 0**：纯数学测试，已经 PASS 过了。
- **Stage 1**：只读连通性  
  使用 `rtde_receive` 读取 getActualQ / getActualTCPPose / getActualTCPForce / getSafetyMode，再开个 socket 到 `192.168.1.100:63352` 发一次 GET POS。看 Robotiq 是否响应。不动任何东西，任何时候都能跑。
- **Stage 2**：RobotiqGripperTCPServer 的完整生命周期  
  依次 activate → open → close → open，并校验缓存里的 pos_byte / distance_mm，保证每一步数据有变。夹爪会动，机械臂不动。跑之前确保夹爪夹口里没东西。
- **Stage 3**：起服务后，测试 UREnv 会调用的所有 HTTP 路由  
  包括 `/getstate`、`/getpos`、`/getq`、`/getjacobian`、`/get_gripper`、`/clearerr`、`/update_param`、`/startimp`、`/stopimp` 等，检查每个 response 的 shape 是否正确（pose 是 7-vec，q 是 6-vec，jacobian 是 36 个数，gripper 在 [0,1]，quat 是单位四元数）。并对静止机械臂连发 5 次 `/getstate` 以检查缓存是否稳定。完全不动。
- **Stage 4**：gating IK/FK 检查  
  读取当前 pose，然后 POST 回去，等待 servoL 收敛，再重新读取，要求漂移 < 0.5mm / < 0.05°；然后 +1mm Z 抖一下、复位。机械臂会动。约 ~1mm，建议站远点。
