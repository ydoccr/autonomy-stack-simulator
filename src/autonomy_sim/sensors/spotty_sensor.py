import numpy as np

from autonomy_sim.core.types import SensorData, VehicleState


class SpottySensor:
    def __init__(
        self,
        dropout_probability=0.1,
        rng: np.random.Generator | None = None,
    ):
        if dropout_probability < 0.0 or dropout_probability > 1.0:
            raise ValueError("dropout_probability must be between zero and one")
        self.dropout_probability = float(dropout_probability)
        self.rng = rng or np.random.default_rng()

    def sense(self, true_state: VehicleState) -> SensorData | None:
        if self.rng.random() < self.dropout_probability:
            return None
        return SensorData(
            x_meas=true_state.x,
            y_meas=true_state.y,
            vx_meas=true_state.vx,
            vy_meas=true_state.vy,
        )
