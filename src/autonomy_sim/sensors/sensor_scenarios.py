import numpy as np

from autonomy_sim.sensors.combined_error_sensor import CombinedErrorSensor
from autonomy_sim.sensors.delay_sensor import DelaySensor
from autonomy_sim.sensors.gaussian_sensor import GaussianSensor
from autonomy_sim.sensors.offset_sensor import OffsetSensor
from autonomy_sim.sensors.spotty_sensor import SpottySensor


SENSOR_SCENARIO_NAMES = {
    0: "gaussian_nominal",
    1: "gaussian_plus_offset",
    2: "gaussian_plus_delay",
    3: "gaussian_plus_dropout",
    4: "gaussian_plus_combined_faults",
}


def sensor_scenario_name(scenario_number):
    try:
        return SENSOR_SCENARIO_NAMES[scenario_number]
    except KeyError as error:
        raise ValueError("scenario_number must be between zero and four") from error


def create_sensor_scenario(
    scenario_number,
    random_seed=7,
    pos_noise_std=0.1,
    vel_noise_std=0.1,
):
    gaussian_sensor = GaussianSensor(
        pos_noise_std=pos_noise_std,
        vel_noise_std=vel_noise_std,
        rng=np.random.default_rng(random_seed),
    )

    if scenario_number == 1:
        return CombinedErrorSensor(
            gaussian_sensor=gaussian_sensor,
            offset_sensor=OffsetSensor(x_offset=0.2, y_offset=-0.3),
        )

    if scenario_number == 2:
        return CombinedErrorSensor(
            gaussian_sensor=gaussian_sensor,
            delay_sensor=DelaySensor(delay_steps=3),
        )

    if scenario_number == 3:
        return CombinedErrorSensor(
            gaussian_sensor=gaussian_sensor,
            spotty_sensor=SpottySensor(
                dropout_probability=0.4,
                rng=np.random.default_rng(random_seed + 1),
            ),
        )

    if scenario_number == 4:
        return CombinedErrorSensor(
            gaussian_sensor=gaussian_sensor,
            offset_sensor=OffsetSensor(x_offset=0.2, y_offset=-0.3),
            delay_sensor=DelaySensor(delay_steps=3),
            spotty_sensor=SpottySensor(
                dropout_probability=0.4,
                rng=np.random.default_rng(random_seed + 1),
            ),
        )

    raise ValueError("scenario_number must be between one and four")
