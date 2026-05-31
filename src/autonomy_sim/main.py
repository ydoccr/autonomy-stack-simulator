import numpy as np

from autonomy_sim.control.point_mass_acc_controller import PointMassAccController
from autonomy_sim.core.types import VehicleState, Waypoint
from autonomy_sim.dynamics.point_mass import PointMassDynamics


def distance_to_target(state: VehicleState, target: Waypoint) -> float:
    dx = target.x - state.x
    dy = target.y - state.y
    return float(np.hypot(dx, dy))


def run_simulation():
    dt = 0.1
    num_steps = 300

    state = VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    target = Waypoint(x=10.0, y=10.0)

    controller = PointMassAccController(kp=1.0, kd=2.0)
    dynamics = PointMassDynamics(max_speed=5.0, max_accel=3.0)

    trajectory = []

    for step in range(num_steps):
        time = step * dt

        control = controller.compute_control(state, target)
        state = dynamics.step(state, control, dt)

        trajectory.append(
            {
                "time": time,
                "x": state.x,
                "y": state.y,
                "vx": state.vx,
                "vy": state.vy,
                "ax_cmd": control.ax,
                "ay_cmd": control.ay,
                "distance_to_target": distance_to_target(state, target),
            }
        )

    final_distance = distance_to_target(state, target)

    print("Simulation complete.")
    print(f"Final state: {state}")
    print(f"Final distance to target: {final_distance:.3f}")

    return trajectory


if __name__ == "__main__":
    run_simulation()