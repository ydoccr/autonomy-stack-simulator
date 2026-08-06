import argparse
from pathlib import Path

from autonomy_sim.core.types import SimConfig
from autonomy_sim.main import DEFAULT_CONFIG, load_config
from autonomy_sim.mission.config import (
    HallwayMissionConfig,
    load_hallway_mission_config,
)
from autonomy_sim.mission.run_hallway_mission import (
    DEFAULT_MISSION_CONFIG,
    run_hallway_mission,
)
from autonomy_sim.sensors.sensor_scenarios import create_sensor_scenario


def run_hallway_sensor_scenario(
    simulation_config: SimConfig,
    mission_config: HallwayMissionConfig,
    scenario_number,
    random_seed=7,
    show_plots=True,
    show_metrics=True,
):
    sensor = create_sensor_scenario(scenario_number, random_seed)
    return run_hallway_mission(
        simulation_config,
        mission_config,
        show_plots=show_plots,
        show_metrics=show_metrics,
        sensor_model=sensor,
        scenario={
            "sensor_scenario": scenario_number,
            "sensor_seed": random_seed,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a hallway mission with sensor errors."
    )
    parser.add_argument("scenario", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mission-config", type=Path, default=DEFAULT_MISSION_CONFIG)
    args = parser.parse_args()
    run_hallway_sensor_scenario(
        load_config(args.config),
        load_hallway_mission_config(args.mission_config),
        args.scenario,
        args.seed,
    )


if __name__ == "__main__":
    main()
