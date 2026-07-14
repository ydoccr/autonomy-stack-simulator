from pathlib import Path

import yaml

from autonomy_sim.main import run_simulation


def test_headless_simulation_reaches_waypoint(tmp_path: Path):
    config = {
        "simulation": {"dt": 0.1, "num_steps": 500, "random_seed": 1},
        "initial_state": {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        "controller": {"kp": 1.0, "kd": 1.5},
        "dynamics": {"max_speed": 5.0, "max_accel": 3.0},
        "sensor": {"pos_noise_std": 0.0, "vel_noise_std": 0.0},
        "estimator": {"process_var": 0.001, "meas_var": 0.01},
        "waypoint_threshold": 0.2,
        "waypoints": [{"x": 1.0, "y": 0.0}],
    }
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    trajectory = run_simulation(config_path, show_metrics=False)

    assert len(trajectory) < config["simulation"]["num_steps"]
    assert abs(trajectory[-1]["x"] - 1.0) < 0.3
    assert trajectory[-1]["current_waypoint_index"] == 0
