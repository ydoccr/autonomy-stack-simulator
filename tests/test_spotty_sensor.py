import numpy as np
import pytest

from autonomy_sim.core.types import VehicleState
from autonomy_sim.sensors.spotty_sensor import SpottySensor


def test_spotty_sensor_can_keep_every_measurement():
    sensor = SpottySensor(
        dropout_probability=0.0,
        rng=np.random.default_rng(1),
    )
    state = VehicleState(x=1.0, y=2.0, vx=3.0, vy=4.0)

    measurement = sensor.sense(state)

    assert measurement is not None
    assert measurement.x_meas == state.x
    assert measurement.y_meas == state.y


def test_spotty_sensor_can_drop_every_measurement():
    sensor = SpottySensor(
        dropout_probability=1.0,
        rng=np.random.default_rng(1),
    )

    measurement = sensor.sense(VehicleState(x=1.0, y=2.0, vx=3.0, vy=4.0))

    assert measurement is None


def test_dropout_probability_must_be_valid():
    with pytest.raises(ValueError):
        SpottySensor(dropout_probability=1.1)
