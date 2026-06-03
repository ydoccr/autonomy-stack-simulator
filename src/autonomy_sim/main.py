import numpy as np
from autonomy_sim.control.point_mass_acc_controller import PointMassAccController
from autonomy_sim.core.types import SimConfig, VehicleState, Waypoint
from autonomy_sim.dynamics.point_mass import PointMassDynamics
from autonomy_sim.guidance.waypoint_tracker import WaypointTracker
from autonomy_sim.visualization.plot_run import plot_trajectory

def run_simulation():
    config = SimConfig(dt=0.1, num_steps=600)
    state = VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    controller = PointMassAccController(kp=1.0, kd=2.0)
    dynamics = PointMassDynamics(max_speed=5.0, max_accel=3.0)
    waypoints = [
        Waypoint(x=5.0, y=0.0),
        Waypoint(x=5.0, y=5.0),
        Waypoint(x=10.0, y=5.0),
        Waypoint(x=10.0, y=10.0),
    ]
    waypoint_tracker = WaypointTracker(
        waypoints=waypoints,
        waypoint_threshold=0.5,
    )
    trajectory = []
    for step in range(config.num_steps):
        time = step * config.dt
        waypoint_tracker.update(state)
        if waypoint_tracker.complete:
            break
        current_waypoint = waypoint_tracker.current_waypoint()
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
                "current_waypoint_index": waypoint_tracker.current_index,
                "distance_to_waypoint": waypoint_tracker.distance_to_current_waypoint(
                    state
                ),
            }
        )
    final_waypoint = waypoints[-1]
    final_distance = float(
        np.hypot(state.x - final_waypoint.x, state.y - final_waypoint.y)
    )
    print("Simulation complete.")
    print(f"Final state: {state}")
    print(f"Final distance to final waypoint: {final_distance:.3f}")
    print(f"Final waypoint index: {waypoint_tracker.current_index}")
    print(f"Waypoint path complete: {waypoint_tracker.complete}")
    plot_trajectory(trajectory, waypoints)

    return trajectory


if __name__ == "__main__":
    run_simulation()