import argparse

from autonomy_sim.mission.run_random_mission import run_random_mission
from autonomy_sim.sensors.sensor_scenarios import create_sensor_scenario


def run_random_sensor_scenario(
    scenario_number,
    environment_seed=0,
    sensor_seed=7,
    show_plots=True,
    show_metrics=True,
):
    sensor = create_sensor_scenario(scenario_number, sensor_seed)
    return run_random_mission(
        random_seed=environment_seed,
        show_plots=show_plots,
        show_metrics=show_metrics,
        sensor_model=sensor,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a random mission with sensor errors."
    )
    parser.add_argument("scenario", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument("--sensor-seed", type=int, default=7)
    args = parser.parse_args()
    run_random_sensor_scenario(
        args.scenario,
        args.environment_seed,
        args.sensor_seed,
    )


if __name__ == "__main__":
    main()
