from pathlib import Path

import numpy as np
import yaml

from autonomy_sim.main import load_config, run_simulation


def test_headless_simulation_reaches_waypoint(tmp_path: Path):
    config = {
        "simulation": {"dt": 0.1, "num_steps": 500, "random_seed": 1},
        "initial_state": {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        "controller": {"type": "point_mass_acceleration", "kp": 1.0, "kd": 1.5},
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
        "waypoints": [{"x": 1.0, "y": 0.0}],
    }
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    simulation_config = load_config(config_path)
    result = run_simulation(
        simulation_config,
        show_metrics=False,
        scenario={"mission": "test"},
    )
    trajectory = result.trajectory

    assert len(trajectory) < config["simulation"]["num_steps"]
    assert abs(trajectory[-1]["x"] - 1.0) < 0.3
    assert trajectory[-1]["current_waypoint_index"] == 0
    assert all(sample["true_cross_track_error"] == 0.0 for sample in trajectory)
    assert all(np.isnan(sample["true_clearance"]) for sample in trajectory)
    assert result.metrics["true_mission_success"] is True
    assert result.scenario["mission"] == "test"
    assert result.scenario["simulation_seed"] == 1
    assert result.scenario["sensor_model"] == "GaussianSensor"
