from autonomy_sim.core.types import VehicleState
from autonomy_sim.sensors.gaussian_sensor import GaussianSensor


def test_zero_noise_returns_true_state():
    sensor = GaussianSensor(pos_noise_std=0.0, vel_noise_std=0.0)
    state = VehicleState(x=1.0, y=2.0, vx=3.0, vy=4.0)
    measurement = sensor.sense(state)
    assert hasattr(measurement, "x_meas")
    assert hasattr(measurement, "y_meas")
    assert hasattr(measurement, "vx_meas")
    assert hasattr(measurement, "vy_meas")
    assert measurement.x_meas == 1.0
    assert measurement.y_meas == 2.0
    assert measurement.vx_meas == 3.0
    assert measurement.vy_meas == 4.0


def test_nonzero_noise_returns_noisy_state():
    sensor = GaussianSensor(pos_noise_std=0.1, vel_noise_std=0.1)
    state = VehicleState(x=1.0, y=2.0, vx=3.0, vy=4.0)
    measurement = sensor.sense(state)
    assert hasattr(measurement, "x_meas")
    assert hasattr(measurement, "y_meas")
    assert hasattr(measurement, "vx_meas")
    assert hasattr(measurement, "vy_meas")
    assert measurement.x_meas != 1.0
    assert measurement.y_meas != 2.0
    assert measurement.vx_meas != 3.0
    assert measurement.vy_meas != 4.0
