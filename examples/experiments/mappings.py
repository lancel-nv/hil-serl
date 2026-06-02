from collections.abc import Mapping
from importlib import import_module


_CONFIG_IMPORTS = {
    "ram_insertion": (
        "experiments.ram_insertion.config",
        "TrainConfig",
    ),
    "usb_pickup_insertion": (
        "experiments.usb_pickup_insertion.config",
        "TrainConfig",
    ),
    "object_handover": (
        "experiments.object_handover.config",
        "TrainConfig",
    ),
    "egg_flip": (
        "experiments.egg_flip.config",
        "TrainConfig",
    ),
    "example_ur": (
        "experiments.example_ur.config",
        "TrainConfig",
    ),
}


class LazyConfigMapping(Mapping):
    """Import only the selected experiment config.

    Some experiments have robot-specific dependencies; loading every config at
    module import time makes UR-only scripts depend on Franka-only modules.
    """

    def __init__(self, config_imports):
        self._config_imports = config_imports
        self._cache = {}

    def __getitem__(self, key):
        module_name, class_name = self._config_imports[key]
        if key not in self._cache:
            self._cache[key] = getattr(import_module(module_name), class_name)
        return self._cache[key]

    def __iter__(self):
        return iter(self._config_imports)

    def __len__(self):
        return len(self._config_imports)


CONFIG_MAPPING = LazyConfigMapping(_CONFIG_IMPORTS)