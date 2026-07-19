import numpy as np

from autonomy_sim.sensors.combined_error_sensor import CombinedErrorSensor
from autonomy_sim.sensors.delay_sensor import DelaySensor
from autonomy_sim.sensors.gaussian_sensor import GaussianSensor
from autonomy_sim.sensors.offset_sensor import OffsetSensor
from autonomy_sim.sensors.spotty_sensor import SpottySensor

# EVENTUALLY: want to make parameter entry happen at runtime

def create_sensor_scenario(scenario_number, random_seed=7):
    if scenario_number == 1:
        return OffsetSensor(x_offset=0.2, y_offset=-0.3)

    if scenario_number == 2:
        return DelaySensor(delay_steps=3)

    if scenario_number == 3:
        return SpottySensor(
            dropout_probability=0.4,
            rng=np.random.default_rng(random_seed),
        )

    if scenario_number == 4:
        return CombinedErrorSensor(
            gaussian_sensor=GaussianSensor(
                pos_noise_std=0.1,
                vel_noise_std=0.1,
                rng=np.random.default_rng(random_seed),
            ),
            offset_sensor=OffsetSensor(x_offset=0.2, y_offset=-0.3),
            delay_sensor=DelaySensor(delay_steps=3),
            spotty_sensor=SpottySensor(
                dropout_probability=0.4,
                rng=np.random.default_rng(random_seed + 1),
            ),
        )

    raise ValueError("scenario_number must be between one and four")
