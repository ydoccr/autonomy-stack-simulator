from autonomy_sim.control.point_mass_acc_controller import PointMassAccController
from autonomy_sim.core.types import VehicleState, Waypoint


def test_controller_accelerates_toward_target_in_x():
    controller = PointMassAccController(kp=1.0, kd=0.0)

    state = VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    target = Waypoint(x=10.0, y=0.0)

    control = controller.compute_control(state, target)

    assert control.ax > 0.0
    assert control.ay == 0.0


def test_controller_accelerates_toward_target_in_y():
    controller = PointMassAccController(kp=1.0, kd=0.0)

    state = VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    target = Waypoint(x=0.0, y=10.0)

    control = controller.compute_control(state, target)

    assert control.ax == 0.0
    assert control.ay > 0.0


def test_controller_decelerates_when_moving_toward_target():
    controller = PointMassAccController(kp=0.0, kd=1.0)

    state = VehicleState(x=0.0, y=0.0, vx=5.0, vy=0.0)
    target = Waypoint(x=10.0, y=0.0)

    control = controller.compute_control(state, target)

    assert control.ax < 0.0
    assert control.ay == 0.0


def test_controller_outputs_zero_at_target_with_zero_velocity():
    controller = PointMassAccController(kp=1.0, kd=1.0)

    state = VehicleState(x=10.0, y=5.0, vx=0.0, vy=0.0)
    target = Waypoint(x=10.0, y=5.0)

    control = controller.compute_control(state, target)

    assert control.ax == 0.0
    assert control.ay == 0.0
