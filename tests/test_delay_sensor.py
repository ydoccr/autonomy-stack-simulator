import pytest

from autonomy_sim.core.types import VehicleState
from autonomy_sim.sensors.delay_sensor import DelaySensor


def state_at(x):
    return VehicleState(x=x, y=0.0, vx=1.0, vy=0.0)


def test_delay_sensor_returns_older_measurements():
    sensor = DelaySensor(delay_steps=2)

    measured_x = [sensor.sense(state_at(x)).x_meas for x in range(4)]

    assert measured_x == [0, 0, 0, 1]


def test_delay_sensor_reset_clears_measurement_history():
    sensor = DelaySensor(delay_steps=2)
    sensor.sense(state_at(0))
    sensor.sense(state_at(1))

    sensor.reset()

    assert sensor.sense(state_at(5)).x_meas == 5


def test_delay_steps_must_be_non_negative_integer():
    with pytest.raises(ValueError):
        DelaySensor(delay_steps=-1)
