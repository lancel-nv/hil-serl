# UR + Robotiq 真机起步 Runbook

按顺序执行 A → H。**任何一步过不去都不要跳，先解决再走下一步**。每一步都有"判断标准"，达到了才算通过。

---

## 阶段 A：第一次启动前的硬件清单（线下，没有代码）

1. **UR 控制柜上电**，示教器上把机械臂解锁、置于 Remote Control 模式
2. **示教器上加载 RTDE 外部控制程序**（External Control URCap 或 RTDE program）—— 没这个，ur_rtde 连不上
3. **示教器上确认 Robotiq URCap 已加载并 activated** —— 没这个，TCP socket port 63352 连不通
4. **网络确认**：从训练 PC `ping 192.168.1.100` 通
5. **Orbbec 摄像头插好**，udev 规则装好（参考 `ur_utils/README.md` 那段）
6. **SpaceMouse 插好**，udev 规则装好
7. **工作台清场**：机械臂大致活动半径内没有易碎物、没有人。第一次跑请站远点（≥ 2 米）

---

## 阶段 B：安装环境（一次性）

```bash
# 1) 建 conda env（hil-serl README 推荐 Python 3.10）
conda create -n hilserl python=3.10 -y
conda activate hilserl

# 2) 装 JAX（GPU 版本，跟你的 CUDA 匹配）
pip install --upgrade "jax[cuda12_pip]==0.4.35" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3) 装 serl_launcher
cd /media/data/Projects/2024-05-21-Robotics/RL/hil-serl/serl_launcher
pip install -e .
pip install -r requirements.txt

# 4) 装 serl_robot_infra（含我们新加的 ur-rtde）
cd ../serl_robot_infra
pip install -e .

# 5) 装 Orbbec SDK（pyorbbecsdk 不在 setup.py 里，按 ur_utils 的方式装）
#   参考 /media/data/Projects/2024-05-21-Robotics/2026-02-24-UR/ur_utils/README.md
pip install pyorbbecsdk
# 如果失败：按 ur_utils README 走完整 udev + 编译流程
```

**判断标准**：

```bash
python -c "import rtde_control, rtde_receive, pyorbbecsdk, pyspacemouse; print('ok')"
```

全部能 import → 阶段 B 完成。

---

## 阶段 C：跑通验证测试（按 README 顺序，不要跳）

```bash
cd /media/data/Projects/2024-05-21-Robotics/RL/hil-serl/serl_robot_infra
cat tests/README.md   # 把这个 runbook 完整读一遍
```

按顺序：

```bash
# 0. 离线数学（无硬件）
python -m pytest tests/test_ur_pose_conversions.py -v
# 期待：10 passed

# 1. 硬件可达性（无运动）
python tests/test_ur_hardware_smoke.py --robot_ip=192.168.1.100
# 期待：Stage 1 PASS

# 2. 夹爪协议（夹爪会动，机械臂不动）
python tests/test_ur_gripper.py --robot_ip=192.168.1.100
# 期待：Stage 2 PASS。看：夹爪能打开 → 闭合 → 再打开

# 3. 起服务（在 terminal A，保持运行）
bash robot_servers/launch_ur_server.sh
# 看到 "[ur_server] control_mode=movel" 和 Flask 监听 5000

# 4. 服务路由烟囱（在 terminal B；无运动）
python tests/test_ur_server_routes.py --url=http://127.0.0.1:5000/
# 期待：Stage 3 PASS

# 5. ★ 关键 gating check：pose round-trip（机械臂会动约 1mm）
python tests/test_ur_roundtrip.py --url=http://127.0.0.1:5000/
# 期待：RESULT: PASS
# 这个没过 → 不要往下走，控制层有问题

# 6. 测试键盘控制 ur 机械臂
# 窗口 1：
cd /home/lancel/hil-serl/serl_robot_infra/robot_servers
./launch_ur_server.sh
# 窗口 2：
cd /home/lancel/hil-serl
python serl_robot_infra/tests/test_arm.py
```

**判断标准**：上面 6 步**全部 PASS**。任何一步 fail：先回去解决，再前进。每个 stage fail 的原因排查在 `serl_robot_infra/tests/README.md` 里。

TODO： 图像采集是黑白的。用 ur conda 可以跑通测试，但是用hilserl conda就跑不通。

---

## 阶段 D：校准 example_ur 的工作空间（5 分钟）

打开 terminal C，做三件事。

### D.1 决定一个安全的 home 关节角

让机械臂保持 ur_server 的默认 home（这是从 ur_utils 复制过来的那组角度）。如果想换成自己的：在示教器上手动开到目标姿态 → 然后：

```bash
curl -X POST http://127.0.0.1:5000/getq | python3 -m json.tool
# 把 q 里的 6 个数复制下来
```

把这 6 个数填进 `serl_robot_infra/robot_servers/launch_ur_server.sh` 的 `--reset_joint_target=...`，然后**重启 ur_server**。

### D.2 决定 TARGET_POSE（reach 任务的目标）

让机械臂去 home（默认 reset 后会到），然后用 SpaceMouse 或示教器把它开到"训完能成功 reach 到"的位置。然后：

```bash
curl -X POST http://127.0.0.1:5000/getpos_euler | python3 -m json.tool
# 输出形如：{"pose": [0.42, -0.05, 0.31, 3.14, 0.02, -0.01]}
```

把这 6 个数填到 `examples/experiments/example_ur/config.py` 的 `TARGET_POSE`。

### D.3 决定 RESET_POSE（每个 episode 开始的复位姿态）

最简单：`RESET_POSE = TARGET_POSE + [0, 0, 0.1, 0, 0, 0]`（在目标正上方 10 cm）。代码里已经默认这样写，通常够用。

**也要确认安全盒**：`ABS_POSE_LIMIT_LOW/HIGH` 是机械臂能去的范围（相对 TARGET_POSE 的 ± 几 cm + 几 rad）。第一次跑请把数值调**严**一点：

```python
ABS_POSE_LIMIT_LOW  = TARGET_POSE - np.array([0.03, 0.03, 0.02, 0.1, 0.1, 0.2])
ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.03, 0.08, 0.2, 0.1, 0.2])
```

`ACTION_SCALE = (0.005, 0.03, 1.0)` 保持原样 —— 每 step 最多 5 mm + 0.03 rad，足够慢、足够安全。

**判断标准**：配置里 `TARGET_POSE` 和 `ABS_POSE_LIMIT_*` 已替换成跟你工作台对得上的真实值。

---

## 阶段 E：录 demo（人类用键盘教 5-10 条成功轨迹）

terminal 1:
```bash
conda activate hilserl
cd /home/lancel/hil-serl/serl_robot_infra/robot_servers
bash launch_ur_server.sh
```

terminal 2

```bash
conda activate hilserl
cd /home/lancel/hil-serl
python examples/record_demos.py --exp_name example_ur --successes_needed 5

```

会发生的事：

1. 弹出图像窗口（Orbbec 视图，已 resize 到 128×128 拼接）
2. 机械臂通过 `launch_ur_server.sh` 里的 `reset_joint_target` 执行 `/jointreset`，回到每条 episode 的固定起点，然后**等你用键盘控制**
3. 你用键盘把末端推向 TARGET_POSE。此时 policy 还没训过，脚本发的是全 0 action，只有你的键盘输入会被采纳（`KeyboardIntervention` wrapper，通过 UR server 的 `/speedl` 控制）
4. 一旦末端到 TARGET_POSE ± REWARD_THRESHOLD 容差内，`compute_reward → True`，`done=True`，本条 episode 成功（`info["succeed"]=True`），存到 `transitions`
5. 自动 reset 开下一条
6. 重复直到 5 条成功，存到 `demo_data/example_ur_5_demos_<时间戳>.pkl`

**小技巧**：

- 录之前先跑 `python serl_robot_infra/tests/test_arm.py` 确认 UR server + 键盘速度控制正常
- 键盘控制：`W/S = -X/+X`，`A/D = -Y/+Y`，`Q/E = +Z/-Z`
- 想丢弃当前 trajectory：直接让它超 `MAX_EPISODE_LENGTH=100`（约 10 秒），它会 `/jointreset` 后进入下一条
- 没按键时脚本只读状态，不会发送 zero-action 位姿命令；只有键盘介入时才通过 `/speedl` 控制机械臂
- 想中止：terminal 里 Ctrl+C，或键盘按 Esc

**判断标准**：`demo_data/` 下出现一个 `.pkl` 文件 → 阶段 E 完成。

---

## 阶段 F：训 BC（最简单的训练路径，30 分钟到 1 小时）

```bash
python examples/train_bc.py \
    --exp_name=example_ur \
    --train_steps=10000 \
    --eval_n_trajs=5 \
    --bc_checkpoint_path=./
```
TODO: 存在 bug，机械臂会突然停下无法控制，怀疑是代码哪里有冲突。

`train_bc.py` 会：

1. 加载 `demo_data/*.pkl` 里的所有 transition
2. 训 10k 步的 behavior cloning（policy 网络拟合 demo 的 obs → action）
3. 把 checkpoint 存到 `--bc_checkpoint_path`
4. 训完会自动开始 eval（在真机上跑 policy）

只想训不想 eval：`--eval_n_trajs=0`（默认就是 0）。想 eval 加 `--eval_n_trajs=5`。

**判断标准**：

- `bc_ckpt_example_ur/` 目录里出现 checkpoint 文件
- eval 时 policy 能大致朝 TARGET_POSE 移动（不一定每次都到 —— BC 极限就这样）

---

## 阶段 G：上 RLPD（完整 HIL-SERL 流程，1-2 小时）

分布式架构 —— 一个 actor 进程（机械臂端）+ 一个 learner 进程（GPU 端）。可以在同一台机器上跑。

terminal 1
```bash
conda activate hilserl
cd /home/lancel/hil-serl/serl_robot_infra/robot_servers
bash launch_ur_server.sh
```

**Terminal 2（learner）**：

```bash
python examples/train_rlpd.py     --exp_name=example_ur     --learner     --demo_path=./demo_data/example_ur_5_demos_2026-06-04_03-03-02.pkl     --checkpoint_path=./checkpoints/ur_rlpd
```

**Terminal 3（actor，机械臂端）**：

```bash
python examples/train_rlpd.py     --exp_name=example_ur     --actor     --demo_path=./demo_data/example_ur_5_demos_2026-06-04_03-03-02.pkl     --checkpoint_path=./checkpoints/ur_rlpd     --ip=localhost
```

会发生的事：

1. learner 把 demo 灌进 replay buffer + 用 demo 初始化 policy
2. actor 拿 policy 在真机上 step，收集新 transition，发给 learner
3. learner 持续训，定期把新 policy 同步给 actor
4. **你坐在机械臂旁边盯着，关键时刻用 SpaceMouse 接管** —— 这就是 HIL（human-in-the-loop）的核心
5. 一段时间后 policy 收敛，成功率接近 100%

第一次跑 RLPD 建议先用低风险的 reach 任务熟悉流程，再迁到接触类任务。

---

## 阶段 H：eval（验证训好的 policy）

```bash
python examples/train_rlpd.py \
    --exp_name=example_ur \
    --actor \
    --eval_checkpoint_step=10000 \
    --checkpoint_path=./rlpd_ckpt_example_ur \
    --eval_n_trajs=10 \
    --ip=localhost \
    --demo_path=./demo_data/example_ur_5_demos_<时间戳>.pkl
```

会跑 10 条 trajectory，统计成功率 + 平均时长。

---

## 推荐执行节奏

**第一天**（半天到一天）：

- 阶段 A → B → C（重点 C，把验证测试全过一遍）
- 阶段 D（校准）
- 阶段 E 录 1-2 条 demo 试试，**先确认 record_demos.py 能跑通**

**第二天**：

- 阶段 E 正式录 5-10 条 demo
- 阶段 F BC 训一下，eval 一下 —— 看看整条 pipeline 通不通

**第三天**：

- 阶段 G RLPD —— 这是最终目标

---

## 已知问题 / 待修复

### 1. `clip_safety_box` 在 roll≈±180° 处会让末端"突然翻转"（待根治）

**现象**：跑 RLPD/eval 时，机械臂位置基本不动，但手腕每个 step 被猛地拧 ~40°，反复横跳，看起来像"突然翻转"。

**定位**：日志里每步打印的 `nextpos before clip` vs `nextpos_clip`，前 3 个数（位置）几乎不变，后 4 个数（四元数）被改得很大。用真实数据验证：clip 前后姿态真实角距离约 **43.5°**。

**根因**：`serl_robot_infra/ur_env/envs/ur_env.py` 的 `clip_safety_box()` 用"分解欧拉角 + 逐轴 clip + abs/sign 还原 roll"的方式做姿态限位，在 roll≈±π 处有**符号二义性 bug**：

- 目标 roll = +180°，限位窗口本意是 `|roll| ∈ [168.5°, 191.5°]`（正侧）。
- 但末端在 180° 附近时，`scipy` 的 `as_euler("xyz")` 会把同一物理姿态分解成 roll≈+179° 或 -179°（甚至 -126°/yaw=-120° 这种等价但"绕远路"的分解）——这是欧拉角在 ±π 的固有二义性。
- `sign = np.sign(euler[0])` 跟着分解符号走：分解成负的时候，clip 把 roll 推到镜像侧 -168.5°，而非目标想要的 +168.5°~+180°，于是每步硬拽手腕翻转。

`config.py` 里那段 `+np.pi` 注释**只修了 TARGET 的符号**，没覆盖"实测/指令 roll 被分解成反号"的情况，所以问题仍在。

**根治方案（以后做）**：弃用逐轴欧拉角 clip，改成**相对 TARGET 姿态做限幅**——算 `R_rel = R_target⁻¹·R_next` → 转 rotvec（连续无奇异）→ 按幅度裁剪 → `R_clipped = R_target·clip(R_rel)`，彻底避开 ±π 奇异。

**当前临时规避**：见下条——已禁用 action 的旋转分量，只用 position 控制，使姿态锁在 reset 姿态附近（实测此时 clip 在 ±5° 内是 no-op，不会触发翻转）。

### 2. 临时改为"仅位置控制"（已落地）

为绕开上面的 clip bug，先把 action 的旋转 3 维（rx, ry, rz）禁用，只用 position 控制：

- 新增 `examples/experiments/example_ur/wrapper.py` 的 `PositionOnlyGripperCloseEnv`（继承 `GripperCloseEnv`，不改原代码），在最内层把 `action[3:6]` 清零。
- `examples/experiments/example_ur/config.py` 里把 `GripperCloseEnv` 换成 `PositionOnlyGripperCloseEnv`。
- 动作空间维度保持 6 维不变（旋转 3 维仍在、但无效），避免 `RelativeFrame`/`KeyboardIntervention` 的维度连锁问题。

clip bug 根治后，把 config 改回 `GripperCloseEnv` 即可恢复 6-DoF。

---

## 常见坑

1. **`record_demos.py` 报 `serl_launcher` 找不到** → 阶段 B step 3 没装好
2. **`/jointreset` 后机械臂没动** → 检查示教器是不是没在 Remote Control 模式
3. **图像窗口卡死** → Orbbec 没采上来；先单独跑
   ```bash
   python -c "from ur_env.camera.orbbec_capture import OrbbecCapture; c = OrbbecCapture('test'); print(c.read())"
   ```
4. **SpaceMouse 不动作** → udev rules 没装 / 没重新插拔 / 没把 user 加进 input group
5. **机械臂走得抖** → `--control_mode=movel` + 太小的 ACTION_SCALE = moveL 每次走得太短，电机来不及加速。把 `ACTION_SCALE[0]` 调到 0.01，或者切 `--control_mode=servol`
6. **每次 reset 都报 protective stop** → ABS_POSE_LIMIT 设得太松，跟硬件实际工作空间冲突；调严点
7. **`tests/test_ur_roundtrip.py` 失败** → **不要继续**，先调好。可能是 `--movel_speed` 太低，或者 RTDE 频率问题

---

# 附录

## 机械臂控制模式支持

默认（moveL）：

```bash
bash serl_robot_infra/robot_servers/launch_ur_server.sh
```

临时切到 servoL：

```bash
python ur_server.py --control_mode=servol --robot_ip=192.168.1.100 ...
```

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
  读取当前 pose，然后 POST 回去，等待 servoL 收敛，再重新读取，要求漂移 < 0.5mm / < 0.05°；然后 +1mm Z 抖一下、复位。机械臂会动，约 ~1mm，建议站远点。
