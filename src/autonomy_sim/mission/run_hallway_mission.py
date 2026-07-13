from autonomy_sim.core.types import VehicleState
from autonomy_sim.environments.example_environments import (
    create_hallway_environment,
)
from autonomy_sim.main import run_simulation
from autonomy_sim.planning.astar import astar
from autonomy_sim.planning.path_utils import grid_path_to_waypoints


def plan_hallway_mission():
    environment = create_hallway_environment()
    costmap = environment.to_costmap()

    start = (0, 0)
    goal = (100, 100)
    grid_path = astar(
        costmap,
        start,
        goal,
        allow_diagonal=True,
        fuel_rate=1.0,
        max_distance=10,
        waypoint_cost=0.02,
        turn_cost_weight=0.5,
        nominal_speed=2.0,
    )
    if not grid_path:
        raise RuntimeError("hallway mission goal is unreachable")

    waypoints = grid_path_to_waypoints(
        grid_path,
        resolution=environment.resolution,
    )
    return environment, costmap, grid_path, waypoints


def run_hallway_mission(show_plots=True):
    environment, costmap, grid_path, waypoints = plan_hallway_mission()

    initial_state = VehicleState(
        x=0.0,
        y=0.0,
        vx=0.0,
        vy=0.0,
    )
    trajectory = run_simulation(
        initial_state=initial_state,
        waypoints=waypoints,
        waypoint_threshold=0.2,
        show_plots=show_plots,
    )

    return trajectory, environment, costmap, grid_path, waypoints


if __name__ == "__main__":
    run_hallway_mission()
