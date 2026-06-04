import numpy as np
from autonomy_sim.core.types import SensorData

class GaussianSensor:
    def __init__(self, pos_noise_std: float = 0.1, vel_noise_std: float = 0.1):
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std

    def sense(self, true_state) -> SensorData:
        noisy_x = true_state.x + np.random.normal(0, self.pos_noise_std)
        noisy_y = true_state.y + np.random.normal(0, self.pos_noise_std)
        noisy_vx = true_state.vx + np.random.normal(0, self.vel_noise_std)
        noisy_vy = true_state.vy + np.random.normal(0, self.vel_noise_std)
        return SensorData(x_meas=noisy_x, y_meas=noisy_y, vx_meas=noisy_vx, vy_meas=noisy_vy)
