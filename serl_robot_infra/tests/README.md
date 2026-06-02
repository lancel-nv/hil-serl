# UR + Robotiq hardware verification runbook

Run these in strict order. Each stage assumes the previous one passed.
**Do not skip stages.** The IK/FK gating check (stage 5) only catches a narrow
class of errors; stages 1–3 catch the rest.

All commands assume CWD is the repo root (`hil-serl/`). The conda env must have
`ur-rtde`, `numpy`, `scipy`, `requests`, and (for `hil-serl` paths) the rest of
the project deps installed.

| Stage | Script | Moves arm? | Moves gripper? | Needs server? |
|------:|--------|:----------:|:--------------:|:-------------:|
| 0 | `test_ur_pose_conversions.py` (pytest) | no | no | no |
| 1 | `test_ur_hardware_smoke.py` | no | no | no |
| 2 | `test_ur_gripper.py` | no | **yes** | no |
| 3 | `test_ur_server_routes.py` | no | optional | yes |
| 4 | `test_ur_roundtrip.py` | **yes (~1 mm)** | no | yes |

---

## Stage 0 — offline pose math (no hardware)

```bash
cd serl_robot_infra
python -m pytest tests/test_ur_pose_conversions.py -v
```

Verifies quaternion ↔ axis-angle round-trips to < 1e-7 across 1000 random
rotations + edge cases. Must pass before everything else; if this fails the
math in `ur_env/utils/pose_conversions.py` is broken.

---

## Stage 1 — hardware reach (no motion)

```bash
python serl_robot_infra/tests/test_ur_hardware_smoke.py --robot_ip=192.168.1.100
```

Connects to UR via RTDE (read-only) and to the Robotiq TCP socket (read-only).
Prints joint angles, TCP pose, force, and the gripper's current `POS` byte.

**Failure modes & fixes:**
- *Cannot connect to UR*: robot off, not in remote control mode, External Control URCap not loaded, RTDE script not enabled, firewall blocking 30004.
- *Gripper connection refused*: Robotiq URCap not loaded on the UR teach pendant.
- *Safety mode != 1*: protective stop / emergency stop active — clear on the pendant before continuing.

---

## Stage 2 — gripper protocol (gripper moves, arm does NOT)

```bash
python serl_robot_infra/tests/test_ur_gripper.py --robot_ip=192.168.1.100
```

Activates the Robotiq, opens it, closes it, opens it. Verifies the polling
thread updates `gripper_pos`. **Make sure the jaws are clear of any obstacle**;
the gripper closes to ~25 N by default.

**Failure modes:**
- *Distance < 75 mm after open* or *pos_byte < 205 after close*: socket is
  responding but the cache isn't updating — check that
  `RobotiqGripperTCPServer._poll_loop` is actually running (look for errors
  printed at startup), or that the Robotiq is properly activated.

---

## Stage 3 — Flask server routes (no motion)

```bash
# terminal A
bash serl_robot_infra/robot_servers/launch_ur_server.sh

# terminal B
python serl_robot_infra/tests/test_ur_server_routes.py --url=http://127.0.0.1:5000/
```

POSTs every endpoint UREnv will use, checks each response's shape and
contents. Verifies that `/getstate` cache is stable when the arm is idle.

Optional: add `--exercise_gripper` to also exercise the gripper through HTTP
routes (re-runs stage 2 but through the server). Skip if you've already
confirmed stage 2 and the gripper is in a safe state.

**Failure modes:**
- *Pose shape != (7,)*: server's quat conversion is wrong; do not move on.
- *q shape != (6,)*: server is reading the wrong RTDE field.
- *jacobian.size != 36*: shape leak from Franka's (6,7).
- *cache drift > 1 mm with stationary arm*: state thread is hosed.

---

## Stage 4 — pose round-trip (gating IK/FK check — arm moves ~1 mm)

```bash
# server still running from stage 3
python serl_robot_infra/tests/test_ur_roundtrip.py --url=http://127.0.0.1:5000/
```

**Stand clear.** Two checks:
1. Read current pose, POST it back, re-read; assert drift < 0.5 mm / < 0.05°.
2. Command +1 mm in Z, then back; assert tracking < 1 mm / 0.1° in both
   directions.

This is the **gating check** per the project's quality bar: do not start
`record_demos.py`, `train_rlpd.py`, or any other closed-loop run until this
passes. If it fails, the quat↔axis-angle conversion in `ur_server.py.move()`
is wrong, or `servoL` parameters need tuning (`--servo_gain`,
`--servo_lookahead_time`).

---

## Stage 5 (optional, manual) — joint reset

```bash
curl -X POST http://127.0.0.1:5000/jointreset
```

Robot will `moveJ` to `--reset_joint_target` (~5 s slew). After it returns,
sanity-check:

```bash
curl -X POST http://127.0.0.1:5000/getq    # should be within 1e-2 rad of the target
curl -X POST http://127.0.0.1:5000/getpos_euler  # use this xyz+euler in EnvConfig.RESET_POSE
```

This is also how you **calibrate** `TARGET_POSE` / `RESET_POSE` for your task
config: drive the arm to the desired pose (teach pendant or SpaceMouse), POST
`/getpos_euler`, paste the 6 numbers into `examples/experiments/example_ur/config.py`.

---

## Stage 6 — end-to-end env

Only after stages 0–4 pass:

```bash
python examples/record_demos.py --exp_name example_ur --successes_needed 1
```

Confirms `UREnv` boots, cameras stream, SpaceMouse moves the arm, gripper
toggles via SpaceMouse buttons, the env reaches `TARGET_POSE`, and the demo
is written to `./demo_data/`.
