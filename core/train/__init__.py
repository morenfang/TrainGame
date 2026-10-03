"""列车子系统：编组定义与纵向动力学。"""

from core.train.consist import (
    G,
    CarSpec,
    TrainCatalog,
    TrainCatalogError,
    TrainSpec,
    build_train,
)
from core.train.dynamics import (
    Train,
    TrainState,
    acceleration,
    brake_force,
    resistance_force,
    steady_state_speed,
    step,
    time_to_reach,
    tractive_force,
)

__all__ = [
    "G",
    "CarSpec",
    "TrainSpec",
    "TrainCatalog",
    "TrainCatalogError",
    "build_train",
    "Train",
    "TrainState",
    "tractive_force",
    "resistance_force",
    "brake_force",
    "acceleration",
    "step",
    "time_to_reach",
    "steady_state_speed",
]
