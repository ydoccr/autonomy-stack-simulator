import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from autonomy_sim.core.types import SimConfig
from autonomy_sim.main import DEFAULT_CONFIG, load_config
from autonomy_sim.mission.config import RandomMissionConfig, load_random_mission_config
from autonomy_sim.mission.run_random_mission import DEFAULT_MISSION_CONFIG
from autonomy_sim.mission.run_random_sensor_scenarios import (
    run_random_sensor_scenario,
)


TRIAL_FIELDS = (
    "trial",
    "environment_seed",
    "sensor_seed",
    "sensor_scenario",
    "sensor_model",
    "free_probability",
    "occupied_probability",
    "disallowed_probability",
    "restricted_probability",
    "planning_success",
    "onboard_completion",
    "true_goal_reached",
    "true_mission_success",
    "termination_state",
    "completion_time",
    "rmse_estimation_error",
    "mean_true_cross_track_error",
    "max_true_cross_track_error",
    "rmse_true_cross_track_error",
    "mean_estimated_cross_track_error",
    "max_estimated_cross_track_error",
    "rmse_estimated_cross_track_error",
    "minimum_true_clearance",
    "minimum_estimated_clearance",
    "commanded_control_effort",
    "applied_control_effort",
    "control_saturation_fraction",
    "mean_control_saturation_error",
    "max_control_saturation_error",
    "max_commanded_acceleration",
    "max_applied_acceleration",
    "measurement_dropout_fraction",
    "true_free_time",
    "true_occupied_time",
    "true_disallowed_time",
    "true_restricted_time",
    "true_out_of_bounds_time",
    "estimated_free_time",
    "estimated_occupied_time",
    "estimated_disallowed_time",
    "estimated_restricted_time",
    "estimated_out_of_bounds_time",
    "restricted_violation",
    "disallowed_violation",
    "out_of_bounds_violation",
)


def run_monte_carlo(
    simulation_config: SimConfig,
    mission_config: RandomMissionConfig,
    *,
    trials=100,
    base_seed=7,
    sensor_scenario=3,
    output_dir=Path("results/monte_carlo"),
    mission_runner=run_random_sensor_scenario,
):
    if trials < 1:
        raise ValueError("trials must be at least one")
    if sensor_scenario not in (1, 2, 3, 4):
        raise ValueError("sensor_scenario must be between one and four")

    seed_pairs = _trial_seed_pairs(trials, base_seed)
    rows = []
    for trial, (environment_seed, sensor_seed) in enumerate(seed_pairs):
        result, *_ = mission_runner(
            simulation_config=simulation_config,
            mission_config=mission_config,
            scenario_number=sensor_scenario,
            environment_seed=environment_seed,
            sensor_seed=sensor_seed,
            show_plots=False,
            show_metrics=False,
        )
        rows.append(
            flatten_trial_result(
                trial=trial,
                environment_seed=environment_seed,
                sensor_seed=sensor_seed,
                sensor_scenario=sensor_scenario,
                result=result,
            )
        )

    summary = summarize_trials(
        rows,
        base_seed=base_seed,
        sensor_scenario=sensor_scenario,
    )
    write_results(rows, summary, output_dir)
    return rows, summary


def flatten_trial_result(
    trial,
    environment_seed,
    sensor_seed,
    sensor_scenario,
    result,
):
    metrics = result.metrics
    scenario = result.scenario
    probabilities = scenario["zone_probabilities"]
    true_zone_time = metrics["true_zone_time_seconds"]
    estimated_zone_time = metrics["estimated_zone_time_seconds"]

    row = {
        "trial": int(trial),
        "environment_seed": int(environment_seed),
        "sensor_seed": int(sensor_seed),
        "sensor_scenario": int(sensor_scenario),
        "sensor_model": scenario["sensor_model"],
        "free_probability": probabilities["free"],
        "occupied_probability": probabilities["occupied"],
        "disallowed_probability": probabilities["disallowed"],
        "restricted_probability": probabilities["restricted"],
        "planning_success": metrics["planning_success"],
        "onboard_completion": metrics["onboard_completion"],
        "true_goal_reached": metrics["true_goal_reached"],
        "true_mission_success": metrics["true_mission_success"],
        "termination_state": metrics["termination_state"],
        "completion_time": metrics["completion_time"],
        "rmse_estimation_error": metrics["rmse_estimation_error"],
        "mean_true_cross_track_error": metrics["mean_true_cross_track_error"],
        "max_true_cross_track_error": metrics["max_true_cross_track_error"],
        "rmse_true_cross_track_error": metrics["rmse_true_cross_track_error"],
        "mean_estimated_cross_track_error": metrics["mean_estimated_cross_track_error"],
        "max_estimated_cross_track_error": metrics["max_estimated_cross_track_error"],
        "rmse_estimated_cross_track_error": metrics["rmse_estimated_cross_track_error"],
        "minimum_true_clearance": metrics["minimum_true_clearance"],
        "minimum_estimated_clearance": metrics["minimum_estimated_clearance"],
        "commanded_control_effort": metrics["commanded_control_effort"],
        "applied_control_effort": metrics["applied_control_effort"],
        "control_saturation_fraction": metrics["control_saturation_fraction"],
        "mean_control_saturation_error": metrics["mean_control_saturation_error"],
        "max_control_saturation_error": metrics["max_control_saturation_error"],
        "max_commanded_acceleration": metrics["max_commanded_acceleration"],
        "max_applied_acceleration": metrics["max_applied_acceleration"],
        "measurement_dropout_fraction": metrics["measurement_dropout_fraction"],
        "true_free_time": _zone_time_value(true_zone_time, "free"),
        "true_occupied_time": _zone_time_value(true_zone_time, "occupied"),
        "true_disallowed_time": _zone_time_value(true_zone_time, "disallowed"),
        "true_restricted_time": _zone_time_value(true_zone_time, "restricted"),
        "true_out_of_bounds_time": _zone_time_value(
            true_zone_time,
            "out_of_bounds",
        ),
        "estimated_free_time": _zone_time_value(estimated_zone_time, "free"),
        "estimated_occupied_time": _zone_time_value(
            estimated_zone_time,
            "occupied",
        ),
        "estimated_disallowed_time": _zone_time_value(
            estimated_zone_time,
            "disallowed",
        ),
        "estimated_restricted_time": _zone_time_value(
            estimated_zone_time,
            "restricted",
        ),
        "estimated_out_of_bounds_time": _zone_time_value(
            estimated_zone_time,
            "out_of_bounds",
        ),
        "restricted_violation": metrics["restricted_violation"],
        "disallowed_violation": metrics["disallowed_violation"],
        "out_of_bounds_violation": metrics["out_of_bounds_violation"],
    }
    return row


def _zone_time_value(zone_time, zone):
    if zone_time is None:
        return None
    return zone_time[zone]


def summarize_trials(rows, base_seed, sensor_scenario):
    trial_count = len(rows)
    planning_success_count = _count_true(rows, "planning_success")
    mission_success_count = _count_true(rows, "true_mission_success")
    onboard_completion_count = _count_true(rows, "onboard_completion")
    false_completion_count = sum(
        bool(row["onboard_completion"]) and not bool(row["true_goal_reached"])
        for row in rows
    )
    safety_violation_count = sum(
        any(
            row[name] is True
            for name in (
                "restricted_violation",
                "disallowed_violation",
                "out_of_bounds_violation",
            )
        )
        for row in rows
    )
    planned_rows = [row for row in rows if row["planning_success"]]

    return {
        "configuration": {
            "trials": trial_count,
            "base_seed": int(base_seed),
            "sensor_scenario": int(sensor_scenario),
        },
        "counts": {
            "planning_success": planning_success_count,
            "true_mission_success": mission_success_count,
            "onboard_completion": onboard_completion_count,
            "false_completion": false_completion_count,
            "safety_violation": safety_violation_count,
        },
        "rates": {
            "planning_success": _rate(planning_success_count, trial_count),
            "true_mission_success": _rate(mission_success_count, trial_count),
            "true_mission_success_given_plan": _rate(
                mission_success_count,
                planning_success_count,
            ),
            "onboard_completion": _rate(
                onboard_completion_count,
                trial_count,
            ),
            "safety_violation": _rate(
                safety_violation_count,
                trial_count,
            ),
        },
        "termination_counts": dict(
            sorted(Counter(row["termination_state"] for row in rows).items())
        ),
        "statistics": {
            "completion_time": _statistics(
                planned_rows,
                "completion_time",
            ),
            "rmse_estimation_error": _statistics(
                planned_rows,
                "rmse_estimation_error",
            ),
            "rmse_true_cross_track_error": _statistics(
                planned_rows,
                "rmse_true_cross_track_error",
            ),
            "minimum_true_clearance": _statistics(
                planned_rows,
                "minimum_true_clearance",
            ),
            "commanded_control_effort": _statistics(
                planned_rows,
                "commanded_control_effort",
            ),
            "applied_control_effort": _statistics(
                planned_rows,
                "applied_control_effort",
            ),
            "control_saturation_fraction": _statistics(
                planned_rows,
                "control_saturation_fraction",
            ),
            "true_occupied_time": _statistics(
                planned_rows,
                "true_occupied_time",
            ),
        },
    }


def write_results(rows, summary, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "trials.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with (output_path / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")


def _trial_seed_pairs(trials, base_seed):
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**32, size=(trials, 2), dtype=np.uint32)
    return [(int(environment), int(sensor)) for environment, sensor in seeds]


def _count_true(rows, field):
    return sum(row[field] is True for row in rows)


def _rate(count, total):
    if total == 0:
        return None
    return float(count / total)


def _statistics(rows, field):
    values = np.array(
        [row[field] for row in rows if _is_finite_number(row[field])],
        dtype=float,
    )
    if len(values) == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _is_finite_number(value):
    return value is not None and not isinstance(value, bool) and np.isfinite(value)


def main():
    parser = argparse.ArgumentParser(
        description="Run deterministic Monte Carlo random-mission trials."
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument(
        "--sensor-scenario",
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/monte_carlo"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mission-config", type=Path, default=DEFAULT_MISSION_CONFIG)
    args = parser.parse_args()
    _, summary = run_monte_carlo(
        load_config(args.config),
        load_random_mission_config(args.mission_config),
        trials=args.trials,
        base_seed=args.base_seed,
        sensor_scenario=args.sensor_scenario,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
