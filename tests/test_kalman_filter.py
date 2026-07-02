import numpy as np

from autonomy_sim.core.types import ControlInput, SensorData, VehicleState
from autonomy_sim.estimation.kalman_filter import KalmanFilter


def test_reset_sets_initial_state():
    kf = KalmanFilter(dt=0.1)
    state = VehicleState(x=1.0, y=2.0, vx=3.0, vy=4.0)

    kf.reset(state)

    estimated_state = kf.current_state()

    assert estimated_state.x == 1.0
    assert estimated_state.y == 2.0
    assert estimated_state.vx == 3.0
    assert estimated_state.vy == 4.0


def test_predict_with_zero_control_keeps_constant_velocity_motion():
    kf = KalmanFilter(dt=0.1)
    initial_state = VehicleState(x=0.0, y=0.0, vx=2.0, vy=-1.0)
    kf.reset(initial_state)

    predicted_state = kf.predict(ControlInput(ax=0.0, ay=0.0))

    assert np.isclose(predicted_state.x, 0.2)
    assert np.isclose(predicted_state.y, -0.1)
    assert np.isclose(predicted_state.vx, 2.0)
    assert np.isclose(predicted_state.vy, -1.0)


def test_predict_with_acceleration_updates_position_and_velocity():
    kf = KalmanFilter(dt=0.1)
    initial_state = VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    kf.reset(initial_state)

    predicted_state = kf.predict(ControlInput(ax=2.0, ay=-4.0))

    assert np.isclose(predicted_state.x, 0.01)
    assert np.isclose(predicted_state.y, -0.02)
    assert np.isclose(predicted_state.vx, 0.2)
    assert np.isclose(predicted_state.vy, -0.4)


def test_update_moves_estimate_toward_measurement():
    kf = KalmanFilter(dt=0.1, process_var=1e-3, meas_var=1e-2)
    initial_state = VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    kf.reset(initial_state)

    measurement = SensorData(
        x_meas=10.0,
        y_meas=0.0,
        vx_meas=0.0,
        vy_meas=0.0,
    )

    updated_state = kf.update(measurement)

    assert updated_state.x > 0.0
    assert updated_state.x < 10.0


def test_step_predicts_and_updates():
    kf = KalmanFilter(dt=0.1)
    initial_state = VehicleState(x=0.0, y=0.0, vx=1.0, vy=0.0)
    kf.reset(initial_state)

    control = ControlInput(ax=0.0, ay=0.0)
    measurement = SensorData(
        x_meas=0.1,
        y_meas=0.0,
        vx_meas=1.0,
        vy_meas=0.0,
    )

    estimated_state = kf.step(control=control, measurement=measurement)

    assert isinstance(estimated_state, VehicleState)
    assert np.isclose(estimated_state.x, 0.1, atol=0.1)
    assert np.isclose(estimated_state.vx, 1.0, atol=0.1)