import argparse

from autonomy_sim.mission.run_hallway_mission import run_hallway_mission
from autonomy_sim.sensors.sensor_scenarios import create_sensor_scenario


def run_hallway_sensor_scenario(
    scenario_number,
    random_seed=7,
    show_plots=True,
    show_metrics=True,
):
    sensor = create_sensor_scenario(scenario_number, random_seed)
    return run_hallway_mission(
        show_plots=show_plots,
        show_metrics=show_metrics,
        sensor_model=sensor,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a hallway mission with sensor errors."
    )
    parser.add_argument("scenario", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run_hallway_sensor_scenario(args.scenario, args.seed)


if __name__ == "__main__":
    main()
