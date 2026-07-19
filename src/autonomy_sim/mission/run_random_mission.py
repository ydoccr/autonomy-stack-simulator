import argparse

import numpy as np

from autonomy_sim.core.types import VehicleState
from autonomy_sim.environments.example_environments import (
    create_random_environment,
)
from autonomy_sim.main import run_simulation
from autonomy_sim.planning.astar import astar
from autonomy_sim.planning.path_utils import grid_path_to_waypoints


def plan_random_mission(random_seed=0, max_cost=np.inf):
    environment = create_random_environment(
        random_seed=random_seed,
        max_cost=max_cost,
    )
    costmap = environment.to_costmap()
    zone_costmap = environment.to_zone_costmap()

    start = (0, 0)
    goal = (environment.height - 1, environment.width - 1)
    max_waypoint_distance = 0.5
    max_distance_cells = round(
        max_waypoint_distance / environment.resolution
    )

    grid_path = astar(
        costmap,
        start,
        goal,
        allow_diagonal=True,
        fuel_rate=1.0,
        max_distance=max_distance_cells,
        waypoint_cost=0.02,
        turn_cost_weight=0.5,
        nominal_speed=2.0,
        zone_costmap=zone_costmap,
        max_cost=environment.max_cost,
    )
    if not grid_path:
        raise RuntimeError("random mission goal is unreachable within max_cost")

    waypoints = grid_path_to_waypoints(
        grid_path,
        resolution=environment.resolution,
    )
    return environment, costmap, grid_path, waypoints


def run_random_mission(
    random_seed=0,
    max_cost=np.inf,
    show_plots=True,
    show_metrics=True,
    sensor_model=None,
):
    environment, costmap, grid_path, waypoints = plan_random_mission(
        random_seed=random_seed,
        max_cost=max_cost,
    )

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
        environment=environment,
        show_plots=show_plots,
        show_metrics=show_metrics,
        sensor_model=sensor_model,
    )

    return trajectory, environment, costmap, grid_path, waypoints


def main():
    parser = argparse.ArgumentParser(description="Run a seeded random mission.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cost", type=float, default=np.inf)
    args = parser.parse_args()
    run_random_mission(
        random_seed=args.seed,
        max_cost=args.max_cost,
    )


if __name__ == "__main__":
    main()
