import numpy as np

from autonomy_sim.core.types import ControlInput, VehicleState
from autonomy_sim.dynamics.point_mass import PointMassDynamics


def test_zero_control_keeps_velocity_constant():
    dynamics = PointMassDynamics(max_speed=10.0, max_accel=5.0)

    state = VehicleState(x=0.0, y=0.0, vx=2.0, vy=0.0)
    control = ControlInput(ax=0.0, ay=0.0)

    next_state = dynamics.step(state, control, dt=1.0)

    assert next_state.x == 2.0
    assert next_state.y == 0.0
    assert next_state.vx == 2.0
    assert next_state.vy == 0.0


def test_saturation():
    dynamics = PointMassDynamics(max_speed=2.0, max_accel=100.0)

    state = VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    control = ControlInput(ax=10.0, ay=0.0)

    next_state = dynamics.step(state, control, dt=1.0)

    speed = np.hypot(next_state.vx, next_state.vy)
    assert speed <= 2.0


def test_saturation2():
    dynamics = PointMassDynamics(max_speed=100.0, max_accel=2.0)

    state = VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    control = ControlInput(ax=10.0, ay=0.0)

    next_state = dynamics.step(state, control, dt=1.0)

    velocity_change = np.hypot(next_state.vx - state.vx, next_state.vy - state.vy)
    assert velocity_change <= 2.0
