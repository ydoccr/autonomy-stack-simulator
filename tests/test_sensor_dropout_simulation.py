from pathlib import Path

import numpy as np
import yaml

from autonomy_sim.main import run_simulation
from autonomy_sim.sensors.spotty_sensor import SpottySensor


def test_simulation_predicts_when_sensor_measurement_is_dropped(tmp_path: Path):
    config = {
        "simulation": {"dt": 0.1, "num_steps": 5, "random_seed": 1},
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
        "waypoints": [{"x": 10.0, "y": 0.0}],
    }
    config_path = tmp_path / "dropout_simulation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    sensor = SpottySensor(
        dropout_probability=1.0,
        rng=np.random.default_rng(2),
    )

    result = run_simulation(
        config_path,
        show_plots=False,
        show_metrics=False,
        sensor_model=sensor,
    )
    trajectory = result.trajectory

    assert all(not sample["measurement_available"] for sample in trajectory)
    assert all(np.isnan(sample["x_meas"]) for sample in trajectory)
    assert all(np.isfinite(sample["x_est"]) for sample in trajectory)
    assert result.metrics["measurement_dropout_fraction"] == 1.0
    assert result.scenario["sensor_model"] == "SpottySensor"


def test_simulation_predicts_with_applied_acceleration_after_clamping(
    tmp_path: Path,
):
    config = {
        "simulation": {"dt": 0.1, "num_steps": 1, "random_seed": 1},
        "initial_state": {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        "controller": {"type": "point_mass_acceleration", "kp": 1.0, "kd": 0.0},
        "dynamics": {"type": "point_mass", "max_speed": 5.0, "max_accel": 0.5},
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
        "waypoints": [{"x": 10.0, "y": 0.0}],
    }
    config_path = tmp_path / "clamped_dropout_simulation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    sensor = SpottySensor(
        dropout_probability=1.0,
        rng=np.random.default_rng(2),
    )

    result = run_simulation(
        config_path,
        show_plots=False,
        show_metrics=False,
        sensor_model=sensor,
    )
    final_sample = result.trajectory[-1]

    assert final_sample["ax_cmd"] == 10.0
    assert np.isclose(final_sample["vx"], 0.05)
    assert np.isclose(final_sample["vx_est"], final_sample["vx"])
    assert np.isclose(final_sample["x_est"], final_sample["x"])
