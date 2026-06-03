import numpy as np

from autonomy_sim.control.point_mass_acc_controller import PointMassAccController
from autonomy_sim.core.types import VehicleState, Waypoint, SimConfig
from autonomy_sim.dynamics.point_mass import PointMassDynamics
from autonomy_sim.visualization.plot_run import plot_trajectory


def distance_to_target(state: VehicleState, target: Waypoint) -> float:
    dx = target.x - state.x
    dy = target.y - state.y
    return float(np.hypot(dx, dy))


def run_simulation():
    config = SimConfig(dt=0.1, num_steps=200)
    state = VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    controller = PointMassAccController(kp=1.0, kd=2.0)
    dynamics = PointMassDynamics(max_speed=5.0, max_accel=3.0)
    waypoints = [
        Waypoint(x=5.0, y=0.0),
        Waypoint(x=5.0, y=5.0),
        Waypoint(x=10.0, y=5.0),
        Waypoint(x=10.0, y=10.0),
    ]
    trajectory = []
    waypoint_index = 0
    waypoint_threshold = 0.5
    for step in range(config.num_steps):
        time = step * config.dt
        current_waypoint = waypoints[waypoint_index]
        if np.hypot(state.x - current_waypoint.x, state.y - current_waypoint.y) < waypoint_threshold:
            if waypoint_index < len(waypoints) - 1:
                waypoint_index += 1
                current_waypoint = waypoints[waypoint_index]
            else:
                break
        control = controller.compute_control(state, current_waypoint)
        state = dynamics.step(state, control, config.dt)
        trajectory.append(
            {
                "time": time,
                "x": state.x,
                "y": state.y,
                "vx": state.vx,
                "vy": state.vy,
                "ax_cmd": control.ax,
                "ay_cmd": control.ay,
                "current_waypoint_index": waypoint_index,
                "distance_to_waypoint": np.hypot(state.x - current_waypoint.x, state.y - current_waypoint.y),
            }
        )
    final_distance = np.hypot(state.x - waypoints[-1].x, state.y - waypoints[-1].y)
    print("Simulation complete.")
    print(f"Final state: {state}")
    print(f"Final distance to final waypoint: {final_distance:.3f}")
    print(f"Final waypoint index: {waypoint_index}")
    plot_trajectory(trajectory, waypoints)

    return trajectory


if __name__ == "__main__":
    run_simulation()