import argparse
from pathlib import Path

from autonomy_sim.core.types import SimConfig
from autonomy_sim.main import DEFAULT_CONFIG, load_config
from autonomy_sim.mission.config import RandomMissionConfig, load_random_mission_config
from autonomy_sim.mission.run_random_mission import (
    DEFAULT_MISSION_CONFIG,
    run_random_mission,
)
from autonomy_sim.sensors.sensor_scenarios import create_sensor_scenario


def run_random_sensor_scenario(
    simulation_config: SimConfig,
    mission_config: RandomMissionConfig,
    scenario_number,
    environment_seed=0,
    sensor_seed=7,
    show_plots=True,
    show_metrics=True,
):
    sensor = create_sensor_scenario(scenario_number, sensor_seed)
    return run_random_mission(
        simulation_config,
        mission_config,
        random_seed=environment_seed,
        show_plots=show_plots,
        show_metrics=show_metrics,
        sensor_model=sensor,
        scenario={
            "sensor_scenario": scenario_number,
            "sensor_seed": sensor_seed,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a random mission with sensor errors."
    )
    parser.add_argument("scenario", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument("--sensor-seed", type=int, default=7)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mission-config", type=Path, default=DEFAULT_MISSION_CONFIG)
    args = parser.parse_args()
    simulation_config = load_config(args.config)
    mission_config = load_random_mission_config(args.mission_config)
    run_random_sensor_scenario(
        simulation_config,
        mission_config,
        args.scenario,
        args.environment_seed,
        args.sensor_seed,
    )


if __name__ == "__main__":
    main()
