import threading
from typing import Set, Tuple

import numpy as np


class KeyboardExpert:
    """
    Keyboard teleop source with the same interface as SpaceMouseExpert.

    Keyboard controls translation only:
    - W/S: -X / +X
    - A/D: -Y / +Y
    - Q/E: +Z / -Z
    """

    def __init__(self, translation_step: float = 1.0):
        from pynput import keyboard

        self.translation_step = translation_step
        self._pressed_keys: Set[str] = set()
        self._lock = threading.Lock()
        self._last_reported_keys: Set[str] = set()

        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()
        print(
            "Keyboard teleop enabled: hold W/S -> -/+X, A/D -> -/+Y, Q/E -> +/-Z.",
            flush=True,
        )

    def _normalize_key(self, key) -> str:
        try:
            return key.char.lower()
        except AttributeError:
            return ""

    def _on_press(self, key):
        key_name = self._normalize_key(key)
        if key_name in {"w", "a", "s", "d", "q", "e"}:
            with self._lock:
                self._pressed_keys.add(key_name)
                self._report_keys_locked()

    def _on_release(self, key):
        key_name = self._normalize_key(key)
        if key_name in {"w", "a", "s", "d", "q", "e"}:
            with self._lock:
                self._pressed_keys.discard(key_name)
                self._report_keys_locked()

    def _report_keys_locked(self):
        if self._pressed_keys == self._last_reported_keys:
            return
        self._last_reported_keys = set(self._pressed_keys)
        keys = "".join(sorted(self._pressed_keys)) or "none"
        print(f"Keyboard teleop active keys: {keys}", flush=True)

    def get_action(self) -> Tuple[np.ndarray, list]:
        with self._lock:
            pressed_keys = set(self._pressed_keys)

        action = np.zeros(6, dtype=np.float32)
        if "w" in pressed_keys:
            action[0] -= self.translation_step
        if "s" in pressed_keys:
            action[0] += self.translation_step
        if "a" in pressed_keys:
            action[1] -= self.translation_step
        if "d" in pressed_keys:
            action[1] += self.translation_step
        if "q" in pressed_keys:
            action[2] += self.translation_step
        if "e" in pressed_keys:
            action[2] -= self.translation_step

        return np.clip(action, -1.0, 1.0), [0, 0, 0, 0]

    def close(self):
        self.listener.stop()
