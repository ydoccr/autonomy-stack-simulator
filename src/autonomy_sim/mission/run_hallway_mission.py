import argparse
from pathlib import Path

from autonomy_sim.core.types import SimConfig
from autonomy_sim.environments.example_environments import (
    create_hallway_environment,
)
from autonomy_sim.main import DEFAULT_CONFIG, load_config, run_simulation
from autonomy_sim.mission.config import (
    HallwayMissionConfig,
    load_hallway_mission_config,
    world_to_grid_cell,
)
from autonomy_sim.planning.astar import astar
from autonomy_sim.planning.path_utils import grid_path_to_waypoints


DEFAULT_MISSION_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "hallway_mission.yaml"
)


def plan_hallway_mission(config: HallwayMissionConfig):
    environment = create_hallway_environment(max_cost=config.planner.max_cost)
    costmap = environment.to_costmap(
        proximity_sigma=config.planner.proximity_sigma,
        allow_disallowed=config.planner.allow_disallowed,
        minimum_clearance=config.planner.minimum_clearance,
    )
    zone_costmap = environment.to_zone_costmap()

    start = world_to_grid_cell(
        config.initial_state.x,
        config.initial_state.y,
        environment.resolution,
        environment.width,
        environment.height,
    )
    goal = world_to_grid_cell(
        config.goal.x,
        config.goal.y,
        environment.resolution,
        environment.width,
        environment.height,
    )
    max_distance_cells = round(
        config.planner.max_waypoint_distance / environment.resolution
    )
    grid_path = astar(
        costmap,
        start,
        goal,
        allow_diagonal=config.planner.allow_diagonal,
        fuel_rate=config.planner.fuel_rate,
        max_distance=max_distance_cells,
        waypoint_cost=config.planner.waypoint_cost,
        turn_cost_weight=config.planner.turn_cost_weight,
        nominal_speed=config.planner.nominal_speed,
        zone_costmap=zone_costmap,
        max_cost=config.planner.max_cost,
    )
    if not grid_path:
        raise RuntimeError("hallway mission goal is unreachable")

    waypoints = grid_path_to_waypoints(
        grid_path,
        resolution=environment.resolution,
    )
    return environment, costmap, grid_path, waypoints


def run_hallway_mission(
    simulation_config: SimConfig,
    mission_config: HallwayMissionConfig,
    *,
    show_plots=True,
    show_metrics=True,
    sensor_model=None,
    scenario=None,
):
    environment, costmap, grid_path, waypoints = plan_hallway_mission(mission_config)
    scenario_metadata = {"mission": "hallway"}
    if scenario is not None:
        scenario_metadata.update(scenario)
    result = run_simulation(
        simulation_config,
        initial_state=mission_config.initial_state,
        waypoints=waypoints,
        waypoint_threshold=mission_config.waypoint_threshold,
        true_goal_tolerance=mission_config.true_goal_tolerance,
        environment=environment,
        show_plots=show_plots,
        show_metrics=show_metrics,
        sensor_model=sensor_model,
        scenario=scenario_metadata,
    )

    return result, environment, costmap, grid_path, waypoints


def main():
    parser = argparse.ArgumentParser(description="Run the configured hallway mission.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mission-config", type=Path, default=DEFAULT_MISSION_CONFIG)
    args = parser.parse_args()
    run_hallway_mission(
        load_config(args.config),
        load_hallway_mission_config(args.mission_config),
    )


if __name__ == "__main__":
    main()
