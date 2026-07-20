import csv
import json

import numpy as np

from autonomy_sim.core.types import SimulationResult
from autonomy_sim.experiments.run_monte_carlo import (
    flatten_trial_result,
    run_monte_carlo,
    summarize_trials,
)


ZONE_PROBABILITIES = {
    "free": 0.65,
    "occupied": 0.20,
    "disallowed": 0.10,
    "restricted": 0.05,
}


def fake_mission_runner(
    scenario_number,
    environment_seed,
    sensor_seed,
    show_plots,
    show_metrics,
):
    assert show_plots is False
    assert show_metrics is False
    planning_success = environment_seed % 2 == 0
    if planning_success:
        metrics = {
            "planning_success": True,
            "onboard_completion": True,
            "true_goal_reached": True,
            "true_mission_success": True,
            "termination_state": "goal_reached",
            "completion_time": 4.0,
            "rmse_estimation_error": 0.25,
            "control_effort": 2.0,
            "measurement_dropout_fraction": 0.1,
            "true_zone_time_seconds": _zone_times(free=3.0, occupied=1.0),
            "estimated_zone_time_seconds": _zone_times(free=4.0),
            "restricted_violation": False,
            "disallowed_violation": False,
            "out_of_bounds_violation": False,
        }
    else:
        metrics = {
            "planning_success": False,
            "onboard_completion": False,
            "true_goal_reached": False,
            "true_mission_success": False,
            "termination_state": "planning_failure",
            "completion_time": np.nan,
            "rmse_estimation_error": np.nan,
            "control_effort": np.nan,
            "true_zone_time_seconds": None,
            "estimated_zone_time_seconds": None,
            "restricted_violation": None,
            "disallowed_violation": None,
            "out_of_bounds_violation": None,
        }
    result = SimulationResult(
        trajectory=[],
        metrics=metrics,
        scenario={
            "sensor_model": "FakeSensor",
            "zone_probabilities": ZONE_PROBABILITIES,
            "sensor_scenario": scenario_number,
            "environment_seed": environment_seed,
            "sensor_seed": sensor_seed,
        },
    )
    return result, None, None, [], []


def test_monte_carlo_is_reproducible_and_writes_one_row_per_trial(tmp_path):
    first_rows, first_summary = run_monte_carlo(
        trials=5,
        base_seed=12,
        sensor_scenario=3,
        output_dir=tmp_path / "first",
        mission_runner=fake_mission_runner,
    )
    second_rows, second_summary = run_monte_carlo(
        trials=5,
        base_seed=12,
        sensor_scenario=3,
        output_dir=tmp_path / "second",
        mission_runner=fake_mission_runner,
    )

    assert first_rows == second_rows
    assert first_summary == second_summary
    assert len(first_rows) == 5
    assert any(not row["planning_success"] for row in first_rows)
    assert any(row["planning_success"] for row in first_rows)

    with (tmp_path / "first" / "trials.csv").open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        assert len(list(csv.DictReader(csv_file))) == 5
    with (tmp_path / "first" / "summary.json").open(encoding="utf-8") as summary_file:
        assert json.load(summary_file) == first_summary


def test_monte_carlo_changes_seeds_when_base_seed_changes(tmp_path):
    first_rows, _ = run_monte_carlo(
        trials=2,
        base_seed=1,
        output_dir=tmp_path / "first",
        mission_runner=fake_mission_runner,
    )
    second_rows, _ = run_monte_carlo(
        trials=2,
        base_seed=2,
        output_dir=tmp_path / "second",
        mission_runner=fake_mission_runner,
    )

    assert [row["environment_seed"] for row in first_rows] != [
        row["environment_seed"] for row in second_rows
    ]


def test_flatten_trial_result_expands_zone_times():
    result, *_ = fake_mission_runner(4, 2, 3, False, False)

    row = flatten_trial_result(0, 2, 3, 4, result)

    assert row["sensor_scenario"] == 4
    assert row["sensor_model"] == "FakeSensor"
    assert row["true_free_time"] == 3.0
    assert row["true_occupied_time"] == 1.0
    assert row["estimated_free_time"] == 4.0


def test_summary_counts_failures_and_excludes_them_from_statistics():
    success, *_ = fake_mission_runner(3, 2, 4, False, False)
    failure, *_ = fake_mission_runner(3, 3, 5, False, False)
    rows = [
        flatten_trial_result(0, 2, 4, 3, success),
        flatten_trial_result(1, 3, 5, 3, failure),
    ]

    summary = summarize_trials(rows, base_seed=7, sensor_scenario=3)

    assert summary["counts"]["planning_success"] == 1
    assert summary["counts"]["true_mission_success"] == 1
    assert summary["rates"]["planning_success"] == 0.5
    assert summary["rates"]["true_mission_success_given_plan"] == 1.0
    assert summary["termination_counts"] == {
        "goal_reached": 1,
        "planning_failure": 1,
    }
    assert summary["statistics"]["completion_time"] == {
        "count": 1,
        "mean": 4.0,
        "median": 4.0,
        "max": 4.0,
    }


def _zone_times(
    free=0.0,
    occupied=0.0,
    disallowed=0.0,
    restricted=0.0,
    out_of_bounds=0.0,
):
    return {
        "free": free,
        "occupied": occupied,
        "disallowed": disallowed,
        "restricted": restricted,
        "out_of_bounds": out_of_bounds,
    }
