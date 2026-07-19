from pathlib import Path

import numpy as np
import yaml

from autonomy_sim.main import run_simulation
from autonomy_sim.sensors.spotty_sensor import SpottySensor


def test_simulation_predicts_when_sensor_measurement_is_dropped(tmp_path: Path):
    config = {
        "simulation": {"dt": 0.1, "num_steps": 5, "random_seed": 1},
        "initial_state": {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        "controller": {"kp": 1.0, "kd": 1.5},
        "dynamics": {"max_speed": 5.0, "max_accel": 3.0},
        "sensor": {"pos_noise_std": 0.0, "vel_noise_std": 0.0},
        "estimator": {"process_var": 0.001, "meas_var": 0.01},
        "waypoint_threshold": 0.2,
        "waypoints": [{"x": 10.0, "y": 0.0}],
    }
    config_path = tmp_path / "dropout_simulation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    sensor = SpottySensor(
        dropout_probability=1.0,
        rng=np.random.default_rng(2),
    )

    trajectory = run_simulation(
        config_path,
        show_plots=False,
        show_metrics=False,
        sensor_model=sensor,
    )

    assert all(not sample["measurement_available"] for sample in trajectory)
    assert all(np.isnan(sample["x_meas"]) for sample in trajectory)
    assert all(np.isfinite(sample["x_est"]) for sample in trajectory)
