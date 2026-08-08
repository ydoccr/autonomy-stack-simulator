import json
from pathlib import Path

import yaml

from autonomy_sim.experiments.run_campaign import (
    CRITERION_NAMES,
    evaluate_qualification,
    load_campaign_config,
    replay_trial,
    run_campaign,
)

QUALIFICATION_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "qualification_baseline.yaml"
)


def _write_campaign_inputs(tmp_path):
    simulation = {
        "simulation": {"dt": 0.1, "num_steps": 300, "random_seed": 7},
        "initial_state": {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        "controller": {"type": "point_mass_acceleration", "kp": 0.5, "kd": 1.0},
        "dynamics": {"type": "point_mass", "max_speed": 5.0, "max_accel": 3.0},
        "sensor": {
            "type": "gaussian",
            "pos_noise_std": 0.0,
            "vel_noise_std": 0.0,
        },
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
    campaign = {
        "type": "monte_carlo_campaign",
        "name": "test_nominal",
        "frozen": False,
        "trials": 2,
        "base_seed": 12,
        "sensor_scenario": 0,
        "workers": 1,
        "simulation_config": "simulation.yaml",
        "mission_config": "mission.yaml",
        "output_dir": "results",
        "criteria": {},
    }
    for name, settings in (
        ("simulation.yaml", simulation),
        ("mission.yaml", mission),
        ("campaign.yaml", campaign),
    ):
        (tmp_path / name).write_text(yaml.safe_dump(settings), encoding="utf-8")
    return tmp_path / "campaign.yaml"


def test_campaign_archives_inputs_outputs_and_replays_exactly(tmp_path):
    config_path = _write_campaign_inputs(tmp_path)

    manifest, summary, qualification = run_campaign(load_campaign_config(config_path))

    output_dir = tmp_path / "results"
    assert summary["configuration"]["sensor_scenario"] == 0
    assert qualification["status"] == "not_evaluated"
    assert manifest["campaign"]["trials"] == 2
    assert manifest["inputs"]["simulation"]["sha256"]
    assert {
        "manifest.json",
        "qualification.json",
        "qualification_summary.png",
        "summary.json",
        "trials.csv",
    }.issubset(path.name for path in output_dir.iterdir())
    assert (output_dir / "inputs" / "campaign.yaml").is_file()

    replay = replay_trial(output_dir / "manifest.json", trial=1)

    assert replay == {"trial": 1, "matched": True, "mismatches": {}}


def test_frozen_qualification_configuration_is_complete():
    config = load_campaign_config(QUALIFICATION_CONFIG)

    assert config.frozen is True
    assert config.trials == 500
    assert config.sensor_scenario == 0
    assert set(config.criteria) == set(CRITERION_NAMES)


def test_qualification_evaluates_every_frozen_gate():
    summary = {
        "rates": {
            "planning_success": 0.9,
            "true_mission_success_given_plan": 0.95,
            "false_completion_given_plan": 0.02,
            "safety_violation_given_plan": 0.01,
        },
        "statistics": {
            "rmse_true_cross_track_error": {"mean": 0.1},
            "control_saturation_fraction": {"mean": 0.05},
        },
    }
    criteria = {
        "minimum_planning_success_rate": 0.8,
        "minimum_mission_success_given_plan_rate": 0.9,
        "maximum_false_completion_given_plan_rate": 0.05,
        "maximum_safety_violation_given_plan_rate": 0.05,
        "maximum_mean_cross_track_rmse": 0.2,
        "maximum_mean_control_saturation_fraction": 0.1,
    }

    qualification = evaluate_qualification(summary, criteria)

    assert qualification["passed"] is True
    assert set(qualification["gates"]) == set(CRITERION_NAMES)


def test_manifest_is_valid_json(tmp_path):
    config_path = _write_campaign_inputs(tmp_path)
    run_campaign(load_campaign_config(config_path))

    with (tmp_path / "results" / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)

    assert manifest["schema_version"] == 1
