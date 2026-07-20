import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

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
    "control_effort",
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
    probabilities = scenario.get("zone_probabilities", {})
    true_zone_time = metrics.get("true_zone_time_seconds") or {}
    estimated_zone_time = metrics.get("estimated_zone_time_seconds") or {}

    row = {
        "trial": int(trial),
        "environment_seed": int(environment_seed),
        "sensor_seed": int(sensor_seed),
        "sensor_scenario": int(sensor_scenario),
        "sensor_model": scenario.get("sensor_model"),
        "free_probability": probabilities.get("free"),
        "occupied_probability": probabilities.get("occupied"),
        "disallowed_probability": probabilities.get("disallowed"),
        "restricted_probability": probabilities.get("restricted"),
        "planning_success": metrics.get("planning_success", False),
        "onboard_completion": metrics.get("onboard_completion", False),
        "true_goal_reached": metrics.get("true_goal_reached", False),
        "true_mission_success": metrics.get("true_mission_success", False),
        "termination_state": metrics.get("termination_state"),
        "completion_time": metrics.get("completion_time"),
        "rmse_estimation_error": metrics.get("rmse_estimation_error"),
        "control_effort": metrics.get("control_effort"),
        "measurement_dropout_fraction": metrics.get("measurement_dropout_fraction"),
        "true_free_time": true_zone_time.get("free"),
        "true_occupied_time": true_zone_time.get("occupied"),
        "true_disallowed_time": true_zone_time.get("disallowed"),
        "true_restricted_time": true_zone_time.get("restricted"),
        "true_out_of_bounds_time": true_zone_time.get("out_of_bounds"),
        "estimated_free_time": estimated_zone_time.get("free"),
        "estimated_occupied_time": estimated_zone_time.get("occupied"),
        "estimated_disallowed_time": estimated_zone_time.get("disallowed"),
        "estimated_restricted_time": estimated_zone_time.get("restricted"),
        "estimated_out_of_bounds_time": estimated_zone_time.get("out_of_bounds"),
        "restricted_violation": metrics.get("restricted_violation"),
        "disallowed_violation": metrics.get("disallowed_violation"),
        "out_of_bounds_violation": metrics.get("out_of_bounds_violation"),
    }
    return row


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
            "control_effort": _statistics(
                planned_rows,
                "control_effort",
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
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {
        "count": int(len(values)),
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
    args = parser.parse_args()
    _, summary = run_monte_carlo(
        trials=args.trials,
        base_seed=args.base_seed,
        sensor_scenario=args.sensor_scenario,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
