import numpy as np

from autonomy_sim.core.types import VehicleState
from autonomy_sim.sensors.combined_error_sensor import CombinedErrorSensor
from autonomy_sim.sensors.delay_sensor import DelaySensor
from autonomy_sim.sensors.gaussian_sensor import GaussianSensor
from autonomy_sim.sensors.offset_sensor import OffsetSensor
from autonomy_sim.sensors.spotty_sensor import SpottySensor


def test_combined_sensor_can_use_selected_errors():
    sensor = CombinedErrorSensor(
        offset_sensor=OffsetSensor(x_offset=2.0),
        delay_sensor=DelaySensor(delay_steps=1),
    )

    first = sensor.sense(VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0))
    second = sensor.sense(VehicleState(x=1.0, y=0.0, vx=0.0, vy=0.0))

    assert first is not None
    assert second is not None
    assert first.x_meas == 2.0
    assert second.x_meas == 2.0


def test_combined_sensor_applies_all_error_types():
    gaussian_rng = np.random.default_rng(4)
    sensor = CombinedErrorSensor(
        gaussian_sensor=GaussianSensor(
            pos_noise_std=0.1,
            vel_noise_std=0.1,
            rng=gaussian_rng,
        ),
        offset_sensor=OffsetSensor(x_offset=1.0),
        delay_sensor=DelaySensor(delay_steps=1),
        spotty_sensor=SpottySensor(
            dropout_probability=0.0,
            rng=np.random.default_rng(5),
        ),
    )
    expected_rng = np.random.default_rng(4)
    expected_x = 1.0 + expected_rng.normal(0.0, 0.1)

    measurement = sensor.sense(
        VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    )

    assert measurement is not None
    assert measurement.x_meas == expected_x


def test_combined_sensor_returns_none_when_spotty_sensor_drops_data():
    sensor = CombinedErrorSensor(
        spotty_sensor=SpottySensor(
            dropout_probability=1.0,
            rng=np.random.default_rng(2),
        )
    )

    measurement = sensor.sense(
        VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    )

    assert measurement is None
