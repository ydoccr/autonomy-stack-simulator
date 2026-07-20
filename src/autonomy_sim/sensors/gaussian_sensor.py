import numpy as np

from autonomy_sim.core.types import SensorData, VehicleState


class GaussianSensor:
    def __init__(
        self,
        pos_noise_std: float = 0.1,
        vel_noise_std: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        if pos_noise_std < 0 or vel_noise_std < 0:
            raise ValueError("noise standard deviations must be non-negative")
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.rng = rng or np.random.default_rng()

    def sense(self, true_state: VehicleState) -> SensorData:
        return SensorData(
            x_meas=true_state.x + self.rng.normal(0, self.pos_noise_std),
            y_meas=true_state.y + self.rng.normal(0, self.pos_noise_std),
            vx_meas=true_state.vx + self.rng.normal(0, self.vel_noise_std),
            vy_meas=true_state.vy + self.rng.normal(0, self.vel_noise_std),
        )
