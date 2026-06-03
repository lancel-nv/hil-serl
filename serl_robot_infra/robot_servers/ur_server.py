"""Flask control server for Universal Robots + Robotiq 2F.

Mirrors `franka_server.py` route-for-route so that an env client written for the
Franka HTTP API works against UR with no client-side change beyond the URL.

Architecture:
- `ur_rtde` (rtde_control + rtde_receive) talks to the UR over the RTDE protocol.
- `RobotiqGripperTCPServer` talks to the Robotiq gripper over its TCP socket
  (port 63352). The gripper does not support RTDE.
- A background thread polls UR state at `--state_hz` (default 200 Hz) and caches
  pose / vel / force / joint / jacobian in a lock-guarded snapshot. `/getstate`
  is therefore a memory read, sub-ms.
- `/pose` dispatches according to `--control_mode`:
    - `movel` (default): `moveL(target, speed, accel, asynchronous=True)`. Each
      call plans a trapezoidal Cartesian motion to the target and returns
      immediately. If a previous async moveL is still in flight when a new pose
      arrives, the new call replaces it — so streaming small deltas at fixed hz
      can produce slightly choppy motion if `--movel_speed` is too low for the
      per-step delta to complete inside 1/hz.
    - `servol`: `servoL(target, vel, accel, dt, lookahead, gain)`. Streaming
      real-time servo; matches the Franka delta-pose feel. Tuning surface is
      larger (gain, lookahead, dt) but tracking is smoother under high-rate
      setpoint updates.
  Defaults are conservative; raise movel_speed / servo_gain if you see lag.

Quat <-> axis-angle conversion is the only kinematics math owned here; everything
else (IK, FK, trajectory interpolation) is delegated to the UR controller. The
conversion is verified offline by `tests/test_ur_pose_conversions.py` and on the
real robot by `tests/test_ur_roundtrip.py`.

Safety items deferred (see plan):
- TODO(safety): server-side workspace bounding box on /pose.
- TODO(safety): per-step force/velocity watchdog (abort + servoStop on |F| > 30 N).
- TODO(safety): emergency-stop hotkey via UR Dashboard (port 29999).
- TODO(safety): gripper-stuck detection.
- TODO(safety): RTDE protective-stop recovery via getSafetyMode().
- TODO(safety): joint-limit margin check before moveJ.
"""
import threading
import time

import numpy as np
from absl import app, flags
from flask import Flask, jsonify, request
from scipy.spatial.transform import Rotation as R

import rtde_control
import rtde_receive

from robot_servers.robotiq_gripper_tcp_server import RobotiqGripperTCPServer


FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", "192.168.1.100", "IP address of the UR robot controller")
flags.DEFINE_integer("gripper_port", 63352, "Robotiq TCP socket port")
flags.DEFINE_string("gripper_type", "Robotiq", "Gripper type: Robotiq or None")
flags.DEFINE_list(
    "reset_joint_target",
    [
        "0.09375287592411041",
        "-1.3284757894328614",
        "-2.3687846660614014",
        "-1.015773133640625",
        "1.5728812217712402",
        "0.1377517282962799",
    ],
    "Target joint angles (rad) for /jointreset",
)
flags.DEFINE_string("flask_url", "127.0.0.1", "Host to bind the Flask server on")
flags.DEFINE_integer("flask_port", 5000, "Port to bind the Flask server on")
flags.DEFINE_integer("state_hz", 200, "State cache refresh rate")
flags.DEFINE_enum(
    "control_mode", "movel", ["movel", "servol"],
    "How /pose moves the arm. movel=trapezoidal trajectory per call (conservative); "
    "servol=streaming real-time servo (matches Franka delta-pose feel).",
)
flags.DEFINE_float("movel_speed", 0.1, "moveL Cartesian speed (m/s) for /pose")
flags.DEFINE_float("movel_acceleration", 0.5, "moveL Cartesian acceleration (m/s^2) for /pose")
flags.DEFINE_float("servo_dt", 0.008, "servoL dt parameter (s)")
flags.DEFINE_float("servo_lookahead_time", 0.1, "servoL lookahead time (s)")
flags.DEFINE_float("servo_gain", 300.0, "servoL gain")
flags.DEFINE_float("servo_velocity", 0.5, "servoL velocity arg")
flags.DEFINE_float("servo_acceleration", 0.5, "servoL acceleration arg")
flags.DEFINE_float("movej_speed", 1.0, "moveJ speed (rad/s) for /jointreset")
flags.DEFINE_float("movej_acceleration", 1.0, "moveJ acceleration (rad/s^2) for /jointreset")


class URServer:
    def __init__(
        self,
        robot_ip,
        reset_joint_target,
        state_hz=200,
        control_mode="movel",
        movel_speed=0.1,
        movel_acceleration=0.5,
        servo_dt=0.008,
        servo_lookahead_time=0.1,
        servo_gain=300.0,
        servo_velocity=0.5,
        servo_acceleration=0.5,
        movej_speed=1.0,
        movej_acceleration=1.0,
    ):
        if control_mode not in ("movel", "servol"):
            raise ValueError(f"control_mode must be 'movel' or 'servol', got {control_mode!r}")
        self.robot_ip = robot_ip
        self.reset_joint_target = [float(q) for q in reset_joint_target]
        self.control_mode = control_mode
        self.movel_speed = movel_speed
        self.movel_acceleration = movel_acceleration
        self.servo_dt = servo_dt
        self.servo_lookahead_time = servo_lookahead_time
        self.servo_gain = servo_gain
        self.servo_velocity = servo_velocity
        self.servo_acceleration = servo_acceleration
        self.movej_speed = movej_speed
        self.movej_acceleration = movej_acceleration

        print(f"Connecting to UR at {robot_ip}...")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print("UR connected.")

        # State snapshot (lock-guarded).
        self._state_lock = threading.Lock()
        self.pos = np.zeros(7)         # xyz + quat-xyzw
        self.vel = np.zeros(6)         # linear + angular TCP velocity
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.q = np.zeros(6)
        self.dq = np.zeros(6)
        # Jacobian shape is (6,6) on UR vs (6,7) on Franka. UREnv._update_currpos
        # reshapes to (6,6) accordingly. v1 ships zeros (reserved); see plan
        # Section F.4 for the analytic-Jacobian upgrade path.
        self.jacobian = np.zeros((6, 6))

        self._refresh_state()  # initial fill so /getstate is valid before the thread spins

        self.state_period = 1.0 / max(1, state_hz)
        self._stop = threading.Event()
        self._poll_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._poll_thread.start()

    def _refresh_state(self):
        """One-shot read from RTDE into the cache (called by both the poll loop and at init)."""
        ur_pose = self.rtde_r.getActualTCPPose()       # [x, y, z, rx, ry, rz]
        tcp_speed = self.rtde_r.getActualTCPSpeed()    # [vx, vy, vz, wx, wy, wz]
        ft = self.rtde_r.getActualTCPForce()           # [fx, fy, fz, tx, ty, tz]
        q = self.rtde_r.getActualQ()
        dq = self.rtde_r.getActualQd()

        xyz = np.asarray(ur_pose[:3], dtype=np.float64)
        rotvec = np.asarray(ur_pose[3:], dtype=np.float64)
        quat = R.from_rotvec(rotvec).as_quat()  # xyzw

        with self._state_lock:
            self.pos = np.concatenate([xyz, quat])
            self.vel = np.asarray(tcp_speed, dtype=np.float64)
            self.force = np.asarray(ft[:3], dtype=np.float64)
            self.torque = np.asarray(ft[3:], dtype=np.float64)
            self.q = np.asarray(q, dtype=np.float64)
            self.dq = np.asarray(dq, dtype=np.float64)
            # jacobian stays zeros until an analytic implementation is wired.

    def _state_loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._refresh_state()
            except Exception as e:
                print(f"[ur_server] state refresh error: {e}")
                time.sleep(0.05)
            elapsed = time.time() - t0
            time.sleep(max(0.0, self.state_period - elapsed))

    def snapshot(self):
        """Atomic read of the full state cache; returned values are plain Python lists."""
        with self._state_lock:
            return {
                "pose": self.pos.tolist(),
                "vel": self.vel.tolist(),
                "force": self.force.tolist(),
                "torque": self.torque.tolist(),
                "q": self.q.tolist(),
                "dq": self.dq.tolist(),
                "jacobian": self.jacobian.flatten().tolist(),
            }

    def move(self, pose7):
        """Move to absolute pose [x, y, z, qx, qy, qz, qw].

        Dispatches on self.control_mode:
        - 'movel': moveL(target, speed, accel, asynchronous=True). Trapezoidal
          Cartesian motion per call. A new /pose while one is in flight replaces
          the in-flight trajectory.
        - 'servol': servoL(target, vel, accel, dt, lookahead, gain). Streaming
          real-time servo; controller interpolates between consecutive setpoints.
        """
        pose7 = np.asarray(pose7, dtype=np.float64)
        if pose7.shape != (7,):
            raise ValueError(f"/pose expects a 7-vector, got shape {pose7.shape}")
        rotvec = R.from_quat(pose7[3:]).as_rotvec()
        target = [float(pose7[0]), float(pose7[1]), float(pose7[2]),
                  float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]
        if self.control_mode == "movel":
            ok = self.rtde_c.moveL(
                target,
                self.movel_speed,
                self.movel_acceleration,
                True,  # asynchronous
            )
            if ok is False:
                print("[ur_server] WARNING: moveL command was rejected by controller.")
        else:  # servol — validated in __init__
            ok = self.rtde_c.servoL(
                target,
                self.servo_velocity,
                self.servo_acceleration,
                self.servo_dt,
                self.servo_lookahead_time,
                self.servo_gain,
            )
            if ok is False:
                print("[ur_server] WARNING: servoL command was rejected by controller.")

    def speed(self, velocity, acceleration=0.5, time_s=0.016):
        """Cartesian velocity command [vx, vy, vz, wx, wy, wz] via RTDE speedL."""
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.shape != (6,):
            raise ValueError(f"/speedl expects a 6-vector, got shape {velocity.shape}")
        ok = self.rtde_c.speedL(
            [float(v) for v in velocity],
            float(acceleration),
            float(time_s),
        )
        if ok is False:
            print("[ur_server] WARNING: speedL command was rejected by controller.")

    def speed_stop(self, acceleration=0.5):
        self.rtde_c.speedStop(float(acceleration))

    def _stop_motion(self):
        """Best-effort stop matching the active control mode. Exceptions are
        swallowed so this is cheap to call before mode transitions or shutdown."""
        try:
            if self.control_mode == "movel":
                self.rtde_c.stopL(self.movel_acceleration)
            else:
                self.rtde_c.servoStop()
        except Exception:
            pass

    def reset_joint(self):
        try:
            self.rtde_c.speedStop(self.servo_acceleration)
        except Exception:
            pass
        self._stop_motion()
        time.sleep(0.1)
        self.rtde_c.moveJ(self.reset_joint_target, self.movej_speed, self.movej_acceleration)
        # Settle + verify
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._state_lock:
                err = np.max(np.abs(np.asarray(self.q) - np.asarray(self.reset_joint_target)))
            if err < 1e-2:
                return True
            time.sleep(0.05)
        print(f"[ur_server] WARNING: joint reset did not settle within tolerance "
              f"(max err = {err:.4f} rad)")
        return False

    def clear(self):
        """`/clearerr` analog.

        FrankaEnv calls this before every pose command. Re-uploading the UR RTDE
        control script at control-loop rate interrupts motion, so this route must
        stay cheap and side-effect free during normal teleop.
        """
        return

    def set_payload(self, mass, cog):
        try:
            self.rtde_c.setPayload(float(mass), [float(c) for c in cog])
        except Exception as e:
            print(f"[ur_server] setPayload failed: {e}")

    def shutdown(self):
        self._stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
        self._stop_motion()
        try:
            self.rtde_c.stopScript()
        except Exception:
            pass


def main(_):
    webapp = Flask(__name__)

    print(f"Starting UR server: robot={FLAGS.robot_ip}, gripper_type={FLAGS.gripper_type}")

    gripper_server = None
    if FLAGS.gripper_type == "Robotiq":
        gripper_server = RobotiqGripperTCPServer(
            gripper_ip=FLAGS.robot_ip,
            gripper_port=FLAGS.gripper_port,
        )
    elif FLAGS.gripper_type == "None":
        pass
    else:
        raise NotImplementedError(f"Gripper type {FLAGS.gripper_type} not supported")

    robot_server = URServer(
        robot_ip=FLAGS.robot_ip,
        reset_joint_target=FLAGS.reset_joint_target,
        state_hz=FLAGS.state_hz,
        control_mode=FLAGS.control_mode,
        movel_speed=FLAGS.movel_speed,
        movel_acceleration=FLAGS.movel_acceleration,
        servo_dt=FLAGS.servo_dt,
        servo_lookahead_time=FLAGS.servo_lookahead_time,
        servo_gain=FLAGS.servo_gain,
        servo_velocity=FLAGS.servo_velocity,
        servo_acceleration=FLAGS.servo_acceleration,
        movej_speed=FLAGS.movej_speed,
        movej_acceleration=FLAGS.movej_acceleration,
    )
    print(f"[ur_server] control_mode={FLAGS.control_mode}")

    # ---- routes ----

    @webapp.route("/set_load", methods=["POST"])
    def set_load():
        data = request.json or {}
        mass = data.get("mass", 0.0)
        cog = data.get("F_x_center_load", [0.0, 0.0, 0.0])
        # load_inertia is silently ignored: UR's RTDE setPayload(mass, cog) does not accept a
        # full 3x3 inertia tensor.
        robot_server.set_payload(mass, cog)
        return f"Set mass to {mass}"

    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        # UR has no Franka-style Cartesian impedance controller; servoL replaces it.
        return "UR servoL active (no impedance controller)"

    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        return "no-op"

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pos_euler():
        with robot_server._state_lock:
            pose = robot_server.pos.copy()
        xyz = pose[:3]
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([xyz, euler]).tolist()})

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        with robot_server._state_lock:
            return jsonify({"pose": robot_server.pos.tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        with robot_server._state_lock:
            return jsonify({"vel": robot_server.vel.tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        with robot_server._state_lock:
            return jsonify({"force": robot_server.force.tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        with robot_server._state_lock:
            return jsonify({"torque": robot_server.torque.tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        with robot_server._state_lock:
            return jsonify({"q": robot_server.q.tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        with robot_server._state_lock:
            return jsonify({"dq": robot_server.dq.tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        with robot_server._state_lock:
            return jsonify({"jacobian": robot_server.jacobian.tolist()})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        if gripper_server is None:
            return jsonify({"gripper": 0.0})
        return jsonify({"gripper": float(gripper_server.gripper_pos)})

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        ok = robot_server.reset_joint()
        return f"Reset Joint (ok={ok})"

    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        if gripper_server is not None:
            gripper_server.activate_gripper()
        return "Activated"

    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        if gripper_server is not None:
            gripper_server.reset_gripper()
        return "Reset"

    @webapp.route("/open_gripper", methods=["POST"])
    def open_gripper():
        if gripper_server is not None:
            gripper_server.open()
        return "Opened"

    @webapp.route("/close_gripper", methods=["POST"])
    def close_gripper():
        if gripper_server is not None:
            gripper_server.close()
        return "Closed"

    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_gripper_slow():
        if gripper_server is not None:
            gripper_server.close_slow()
        return "Closed slow"

    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        if gripper_server is None:
            return "no gripper"
        gripper_pos = request.json
        pos = int(np.clip(int(gripper_pos["gripper_pos"]), 0, 255))
        gripper_server.move(pos)
        return "Moved Gripper"

    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        robot_server.clear()
        return "Clear"

    @webapp.route("/pose", methods=["POST"])
    def pose():
        pose_arr = np.array(request.json["arr"])
        robot_server.move(pose_arr)
        return "Moved"

    @webapp.route("/speedl", methods=["POST"])
    def speedl():
        data = request.json or {}
        robot_server.speed(
            data["velocity"],
            acceleration=data.get("acceleration", 0.5),
            time_s=data.get("time", 0.016),
        )
        return "SpeedL"

    @webapp.route("/speedstop", methods=["POST"])
    def speed_stop():
        data = request.json or {}
        robot_server.speed_stop(acceleration=data.get("acceleration", 0.5))
        return "SpeedStop"

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        snap = robot_server.snapshot()
        snap["gripper_pos"] = float(gripper_server.gripper_pos) if gripper_server is not None else 0.0
        return jsonify(snap)

    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        # UR has no compliance/precision params analog. Accept any payload silently
        # so FrankaEnv-style configs that POST COMPLIANCE_PARAM / PRECISION_PARAM
        # still work without modification.
        return "no-op (UR has no impedance compliance params)"

    try:
        webapp.run(host=FLAGS.flask_url, port=FLAGS.flask_port)
    finally:
        if gripper_server is not None:
            gripper_server.shutdown()
        robot_server.shutdown()


if __name__ == "__main__":
    app.run(main)
