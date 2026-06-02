from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
from experiments.example_ur.config import TrainConfig as ExampleURTrainConfig

CONFIG_MAPPING = {
                "ram_insertion": RAMInsertionTrainConfig,
                "usb_pickup_insertion": USBPickupInsertionTrainConfig,
                "object_handover": ObjectHandoverTrainConfig,
                "egg_flip": EggFlipTrainConfig,
                "example_ur": ExampleURTrainConfig,
               }