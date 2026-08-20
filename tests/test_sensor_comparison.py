import csv
import json
from pathlib import Path

import yaml

from autonomy_sim.experiments.run_sensor_comparison import (
    load_sensor_comparison_config,
    run_sensor_comparison,
)


SENSOR_COMPARISON_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "sensor_robustness_v1.yaml"
)


def test_frozen_sensor_comparison_configuration_is_matched_and_dropout_focused():
    config = load_sensor_comparison_config(SENSOR_COMPARISON_CONFIG)

    assert config.frozen is True
    assert config.primary_sensor_scenario == 3
    assert [campaign.sensor_scenario for campaign in config.campaign_configs] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert {campaign.trials for campaign in config.campaign_configs} == {500}
    assert {campaign.base_seed for campaign in config.campaign_configs} == {20260809}


def test_sensor_comparison_runs_matched_scenarios_and_writes_outputs(tmp_path):
    comparison_path = _write_comparison_inputs(tmp_path)

    manifest, comparison = run_sensor_comparison(
        load_sensor_comparison_config(comparison_path)
    )

    output_dir = tmp_path / "results" / "sensor_comparison"
    assert manifest["comparison"]["total_trials"] == 5
    assert manifest["comparison"]["primary_sensor_scenario_name"] == (
        "gaussian_plus_dropout"
    )
    assert comparison["configuration"]["trials_per_scenario"] == 1
    assert set(comparison["scenarios"]) == {
        "gaussian_nominal",
        "gaussian_plus_offset",
        "gaussian_plus_delay",
        "gaussian_plus_dropout",
        "gaussian_plus_combined_faults",
    }
    assert {
        "comparison.json",
        "manifest.json",
        "paired_trials.csv",
        "sensor_comparison_summary.png",
    }.issubset(path.name for path in output_dir.iterdir())

    with (output_dir / "paired_trials.csv").open(
        newline="", encoding="utf-8"
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 5
    assert len({row["environment_seed"] for row in rows}) == 1
    assert len({row["sensor_seed"] for row in rows}) == 1

    with (output_dir / "comparison.json").open(encoding="utf-8") as json_file:
        assert json.load(json_file) == comparison


def _write_comparison_inputs(tmp_path):
    simulation = {
        "simulation": {"dt": 0.1, "num_steps": 200, "random_seed": 7},
        "initial_state": {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        "controller": {
            "type": "point_mass_acceleration",
            "kp": 0.5,
            "kd": 1.0,
        },
        "dynamics": {"type": "point_mass", "max_speed": 5.0, "max_accel": 3.0},
        "sensor": {"type": "gaussian", "pos_noise_std": 0.1, "vel_noise_std": 0.1},
        "estimator": {
            "type": "kalman_filter",
            "process_var": 0.001,
            "meas_var": 0.01,
        },
        "guidance": {"type": "waypoint_tracker", "waypoint_threshold": 0.2},
        "waypoints": [{"x": 1.0, "y": 1.0}],
    }
    mission = {
        "type": "random",
        "environment": {
            "width": 2,
            "height": 2,
            "resolution": 1.0,
            "zone_probabilities": {
                "free": 1.0,
                "occupied": 0.0,
                "disallowed": 0.0,
                "restricted": 0.0,
            },
        },
        "initial_state": {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        "goal": {"x": 1.0, "y": 1.0},
        "guidance": {"waypoint_threshold": 0.2},
        "evaluation": {"true_goal_tolerance": 0.3},
        "planner": {
            "allow_diagonal": True,
            "fuel_rate": 1.0,
            "max_waypoint_distance": 1.0,
            "waypoint_cost": 0.0,
            "turn_cost_weight": 0.0,
            "nominal_speed": 1.0,
            "proximity_sigma": 0.0,
            "minimum_clearance": 0.0,
            "allow_disallowed": False,
            "max_cost": float("inf"),
        },
    }
    (tmp_path / "simulation.yaml").write_text(
        yaml.safe_dump(simulation), encoding="utf-8"
    )
    (tmp_path / "mission.yaml").write_text(yaml.safe_dump(mission), encoding="utf-8")

    campaign_names = []
    scenario_names = (
        "gaussian_nominal",
        "gaussian_plus_offset",
        "gaussian_plus_delay",
        "gaussian_plus_dropout",
        "gaussian_plus_combined_faults",
    )
    for scenario, scenario_name in enumerate(scenario_names):
        filename = f"campaign_{scenario}.yaml"
        campaign_names.append(filename)
        campaign = {
            "type": "monte_carlo_campaign",
            "name": f"test_{scenario_name}",
            "frozen": False,
            "trials": 1,
            "base_seed": 12,
            "sensor_scenario": scenario,
            "workers": 1,
            "simulation_config": "simulation.yaml",
            "mission_config": "mission.yaml",
            "output_dir": f"results/sensor_comparison/{scenario_name}",
            "criteria": {},
        }
        (tmp_path / filename).write_text(yaml.safe_dump(campaign), encoding="utf-8")

    comparison = {
        "type": "sensor_comparison_campaign",
        "name": "test_sensor_comparison",
        "frozen": False,
        "primary_sensor_scenario": 3,
        "campaign_configs": campaign_names,
        "output_dir": "results/sensor_comparison",
    }
    comparison_path = tmp_path / "comparison.yaml"
    comparison_path.write_text(yaml.safe_dump(comparison), encoding="utf-8")
    return comparison_path
