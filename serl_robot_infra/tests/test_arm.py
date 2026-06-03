#!/usr/bin/env python3
"""Minimal UR keyboard smoke test through the HIL-SERL UR server.

Run `serl_robot_infra/robot_servers/launch_ur_server.sh` first. This script only
talks to that Flask server, then the server talks to the robot through RTDE.

Controls:
    W/S: -/+X translation
    A/D: -/+Y translation
    Q/E: +/-Z translation
    ESC: stop and exit

No rotation, no gripper, no camera, no demo recording.
"""

import argparse
import threading
import time
from typing import Iterable, Set

import requests


DEFAULT_URL = "http://127.0.0.1:5000/"
DEFAULT_SPEED = 0.03
DEFAULT_ACCELERATION = 0.5
DEFAULT_PERIOD = 0.02
REQUEST_TIMEOUT = 1.0


class ServerKeyboardArmSmokeTest:
    def __init__(self, url: str, speed: float, acceleration: float, period: float):
        from pynput import keyboard

        self.keyboard = keyboard
        self.url = url if url.endswith("/") else url + "/"
        self.speed = speed
        self.acceleration = acceleration
        self.period = period
        self.pressed_keys: Set[str] = set()
        self.lock = threading.Lock()
        self.running = True
        self.was_moving = False
        self.last_pose_print = 0.0
        self.session = requests.Session()

        state = self._get_state()
        print(f"Connected to UR server at {self.url}")
        self._print_state("Initial", state)

    def _post(self, route: str, payload=None):
        response = self.session.post(
            self.url + route,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response

    def _get_state(self):
        return self._post("getstate").json()

    def _send_speed(self, velocity):
        self._post(
            "speedl",
            {
                "velocity": velocity,
                "acceleration": self.acceleration,
                "time": max(self.period * 2.0, 0.008),
            },
        )

    def _send_stop(self):
        self._post("speedstop", {"acceleration": self.acceleration})

    def _key_name(self, key) -> str:
        try:
            return key.char.lower()
        except AttributeError:
            return ""

    def _on_press(self, key):
        if key == self.keyboard.Key.esc:
            self.running = False
            return False

        key_name = self._key_name(key)
        if key_name in {"w", "a", "s", "d", "q", "e"}:
            with self.lock:
                self.pressed_keys.add(key_name)

    def _on_release(self, key):
        key_name = self._key_name(key)
        if key_name in {"w", "a", "s", "d", "q", "e"}:
            with self.lock:
                self.pressed_keys.discard(key_name)

    def _velocity_from_keys(self, keys: Iterable[str]):
        velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if "w" in keys:
            velocity[0] -= self.speed
        if "s" in keys:
            velocity[0] += self.speed
        if "a" in keys:
            velocity[1] -= self.speed
        if "d" in keys:
            velocity[1] += self.speed
        if "q" in keys:
            velocity[2] += self.speed
        if "e" in keys:
            velocity[2] -= self.speed
        return velocity

    def _get_keys_and_velocity(self):
        with self.lock:
            keys = set(self.pressed_keys)
        return keys, self._velocity_from_keys(keys)

    def _print_state(self, prefix: str, state):
        pose = state["pose"]
        q = state["q"]
        print(
            f"{prefix} TCP: X={pose[0]:+.4f}, Y={pose[1]:+.4f}, Z={pose[2]:+.4f} | "
            f"q={[round(v, 4) for v in q]}",
            flush=True,
        )

    def _control_loop(self):
        while self.running:
            keys, velocity = self._get_keys_and_velocity()
            is_moving = any(v != 0.0 for v in velocity)

            try:
                if is_moving:
                    self._send_speed(velocity)
                    self.was_moving = True
                elif self.was_moving:
                    self._send_stop()
                    self.was_moving = False

                now = time.time()
                if is_moving and now - self.last_pose_print > 0.25:
                    self.last_pose_print = now
                    key_text = "".join(sorted(keys))
                    state = self._get_state()
                    print(f"keys={key_text:<4} velocity={velocity[:3]}", end="  ", flush=True)
                    self._print_state("pose", state)
            except requests.RequestException as exc:
                print(f"\nUR server request failed: {exc}", flush=True)
                self.running = False
                break

            time.sleep(self.period)

    def run(self):
        print("\nControls:")
        print("  Hold W/S: -X / +X")
        print("  Hold A/D: -Y / +Y")
        print("  Hold Q/E: +Z / -Z")
        print("  ESC: stop and exit")
        print("\nHold a key to move. Release to stop.\n")

        control_thread = threading.Thread(target=self._control_loop, daemon=True)
        control_thread.start()

        with self.keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        ) as listener:
            listener.join()

        self.shutdown()

    def shutdown(self):
        self.running = False
        time.sleep(0.05)
        try:
            self._send_stop()
        finally:
            try:
                self._print_state("Final", self._get_state())
            finally:
                print("Stopped.")


def main():
    parser = argparse.ArgumentParser(description="Keyboard smoke test through launch_ur_server.sh.")
    parser.add_argument("--url", default=DEFAULT_URL, help="UR server URL.")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Translation speed in m/s.")
    parser.add_argument("--acceleration", type=float, default=DEFAULT_ACCELERATION, help="speedL acceleration.")
    parser.add_argument("--period", type=float, default=DEFAULT_PERIOD, help="HTTP command period in seconds.")
    parser.add_argument("--yes", action="store_true", help="Skip the safety confirmation prompt.")
    args = parser.parse_args()

    print("This script will move the real UR arm through the HIL-SERL UR server.")
    print("Start serl_robot_infra/robot_servers/launch_ur_server.sh first.")
    print("Make sure the workspace is clear and the robot is enabled.")
    if not args.yes:
        answer = input("Type 'yes' to start: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return

    tester = ServerKeyboardArmSmokeTest(
        url=args.url,
        speed=args.speed,
        acceleration=args.acceleration,
        period=args.period,
    )
    try:
        tester.run()
    except KeyboardInterrupt:
        tester.shutdown()


if __name__ == "__main__":
    main()
