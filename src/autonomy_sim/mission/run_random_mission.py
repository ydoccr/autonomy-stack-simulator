import argparse
from pathlib import Path

from autonomy_sim.core.types import SimConfig, SimulationResult
from autonomy_sim.environments.example_environments import (
    create_random_environment,
)
from autonomy_sim.main import DEFAULT_CONFIG, load_config, run_simulation
from autonomy_sim.metrics.metrics_run import planning_failure_metrics
from autonomy_sim.mission.config import (
    RandomMissionConfig,
    load_random_mission_config,
    world_to_grid_cell,
)
from autonomy_sim.planning.astar import astar
from autonomy_sim.planning.path_utils import grid_path_to_waypoints


DEFAULT_MISSION_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "random_mission.yaml"
)


def plan_random_mission(config: RandomMissionConfig, random_seed: int):
    environment = create_random_environment(
        random_seed=random_seed,
        width=config.width,
        height=config.height,
        resolution=config.resolution,
        max_cost=config.planner.max_cost,
        zone_probabilities=config.zone_probabilities,
    )
    costmap = environment.to_costmap(
        proximity_sigma=config.planner.proximity_sigma,
        allow_disallowed=config.planner.allow_disallowed,
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
    waypoints = []
    if grid_path:
        waypoints = grid_path_to_waypoints(
            grid_path,
            resolution=environment.resolution,
        )
    return environment, costmap, grid_path, waypoints


def run_random_mission(
    simulation_config: SimConfig,
    mission_config: RandomMissionConfig,
    random_seed: int,
    *,
    show_plots=True,
    show_metrics=True,
    sensor_model=None,
    scenario=None,
):
    environment, costmap, grid_path, waypoints = plan_random_mission(
        mission_config,
        random_seed,
    )

    scenario_metadata = {
        "mission": "random",
        "environment_seed": random_seed,
        "max_cost": mission_config.planner.max_cost,
        "zone_probabilities": environment.zone_probabilities,
    }
    if scenario is not None:
        scenario_metadata.update(scenario)
    if not grid_path:
        scenario_metadata["sensor_model"] = (
            "GaussianSensor" if sensor_model is None else type(sensor_model).__name__
        )
        result = SimulationResult(
            trajectory=[],
            metrics=planning_failure_metrics(),
            scenario=scenario_metadata,
        )
        return result, environment, costmap, grid_path, waypoints

    result = run_simulation(
        simulation_config,
        initial_state=mission_config.initial_state,
        waypoints=waypoints,
        waypoint_threshold=mission_config.waypoint_threshold,
        environment=environment,
        show_plots=show_plots,
        show_metrics=show_metrics,
        sensor_model=sensor_model,
        scenario=scenario_metadata,
    )

    return result, environment, costmap, grid_path, waypoints


def main():
    parser = argparse.ArgumentParser(description="Run a seeded random mission.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mission-config", type=Path, default=DEFAULT_MISSION_CONFIG)
    args = parser.parse_args()
    simulation_config = load_config(args.config)
    mission_config = load_random_mission_config(args.mission_config)
    run_random_mission(
        simulation_config,
        mission_config,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
