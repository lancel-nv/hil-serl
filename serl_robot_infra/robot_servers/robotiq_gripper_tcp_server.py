"""TCP-socket Robotiq 2F gripper driver (no ROS).

UR's Robotiq 2F gripper exposes a text-based Modbus-like protocol on TCP port
63352 of the robot controller. Commands look like "SET POS 200\n" / "GET POS\n";
responses look like "ack" / "ACK 123". This module wraps that protocol and
exposes the same `GripperServer` surface used by `franka_server.py`.

Two sockets, two locks:
- Short-lived command socket (per-call connect/send/recv/close) for SET commands.
- Persistent polling socket reused by the background thread for GET POS at ~250Hz,
  with the latest value cached so `gripper_pos` reads are non-blocking.

Lifted from /media/data/Projects/2024-05-21-Robotics/2026-02-24-UR/ur_utils/workflow/
1.record_by_spacemouse.py (lines 559-679).
"""
import socket
import threading
import time

from robot_servers.gripper_server import GripperServer


GRIPPER_MAX_DISTANCE_MM = 85.0   # Robotiq 2F-85; change to 140 for 2F-140
DEFAULT_PORT = 63352
DEFAULT_SPEED = 200              # 0-255
DEFAULT_FORCE = 150              # 0-255
DEFAULT_POLL_HZ = 250
SOCKET_TIMEOUT = 2.0


class RobotiqGripperTCPServer(GripperServer):
    def __init__(
        self,
        gripper_ip: str,
        gripper_port: int = DEFAULT_PORT,
        speed: int = DEFAULT_SPEED,
        force: int = DEFAULT_FORCE,
        poll_hz: int = DEFAULT_POLL_HZ,
    ):
        super().__init__()
        self.ip = gripper_ip
        self.port = gripper_port
        self.speed = speed
        self.force = force
        self.poll_interval = 1.0 / max(1, poll_hz)

        self._cmd_lock = threading.Lock()

        self._poll_socket = None
        self._poll_socket_lock = threading.Lock()

        self._cache_lock = threading.Lock()
        self._cached_pos_byte: int = 0
        self._cached_distance_mm: float = GRIPPER_MAX_DISTANCE_MM
        # `gripper_pos` is the externally consumed normalized value (1.0 = closed,
        # 0.0 = open). Matches the convention in robotiq_gripper_server.py:61.
        self.gripper_pos: float = 0.0

        self._stop_event = threading.Event()
        self._poll_thread = None

        self.activate_gripper()
        self._start_polling()

    # ---- low-level socket I/O ----

    def _send(self, cmd: str, recv_bytes: int = 1024) -> str:
        with self._cmd_lock:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(SOCKET_TIMEOUT)
                s.connect((self.ip, self.port))
                s.sendall(f"{cmd.strip()}\n".encode("utf-8"))
                data = s.recv(recv_bytes)
                response = data.decode("utf-8", errors="ignore").strip()
                if " " in response:
                    return response.split(" ", 1)[1]
                return response

    def _set(self, var: str, value: int) -> str:
        return self._send(f"SET {var} {int(value)}")

    def _get(self, var: str) -> str:
        return self._send(f"GET {var}")

    # ---- polling socket ----

    def _ensure_poll_socket(self) -> socket.socket:
        if self._poll_socket is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(SOCKET_TIMEOUT)
            s.connect((self.ip, self.port))
            self._poll_socket = s
        return self._poll_socket

    def _close_poll_socket(self):
        if self._poll_socket is not None:
            try:
                self._poll_socket.close()
            except Exception:
                pass
            self._poll_socket = None

    def _poll_pos_byte(self) -> int:
        s = self._ensure_poll_socket()
        try:
            s.sendall(b"GET POS\n")
            data = s.recv(1024)
            response = data.decode("utf-8", errors="ignore").strip()
            if " " in response:
                return int(response.split(" ", 1)[1])
            return int(response)
        except Exception:
            self._close_poll_socket()
            raise

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                with self._poll_socket_lock:
                    pos_byte = self._poll_pos_byte()
                distance = GRIPPER_MAX_DISTANCE_MM * (1.0 - pos_byte / 255.0)
                with self._cache_lock:
                    self._cached_pos_byte = pos_byte
                    self._cached_distance_mm = distance
                    # Normalized so 1.0 = closed, 0.0 = open (matches franka_env.py:427).
                    self.gripper_pos = 1.0 - pos_byte / 255.0
            except Exception:
                time.sleep(0.1)  # back off before reconnect attempt
            time.sleep(self.poll_interval)
        with self._poll_socket_lock:
            self._close_poll_socket()

    def _start_polling(self):
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    # ---- GripperServer API ----

    def activate_gripper(self):
        self._set("ACT", 1)
        time.sleep(0.2)
        self._set("GTO", 1)
        time.sleep(0.2)
        self._set("SPE", self.speed)
        self._set("FOR", self.force)

    def reset_gripper(self):
        self._set("ACT", 0)
        time.sleep(0.2)
        self.activate_gripper()

    def open(self):
        self._set("SPE", self.speed)
        self._set("POS", 0)

    def close(self):
        self._set("SPE", self.speed)
        self._set("POS", 255)

    def close_slow(self):
        self._set("SPE", 50)
        self._set("POS", 255)
        # speed is restored on the next open()/close() call

    def move(self, position):
        # Mirrors RobotiqGripperServer.move which accepts ints or string-castable.
        try:
            pos = int(position)
        except (TypeError, ValueError):
            return
        pos = max(0, min(255, pos))
        self._set("POS", pos)

    def move_to_distance(self, distance_mm: float):
        distance_mm = max(0.0, min(GRIPPER_MAX_DISTANCE_MM, float(distance_mm)))
        pos = int(255 * (1.0 - distance_mm / GRIPPER_MAX_DISTANCE_MM))
        self._set("POS", pos)

    def get_distance_mm(self) -> float:
        with self._cache_lock:
            return self._cached_distance_mm

    def get_pos_byte(self) -> int:
        with self._cache_lock:
            return self._cached_pos_byte

    def shutdown(self):
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
