from pathlib import Path

import pytest
import yaml

from autonomy_sim.main import load_config, run_simulation


def default_settings():
    with Path("configs/default.yaml").open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def write_config(tmp_path, settings, filename="config.yaml"):
    config_path = tmp_path / filename
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return config_path


def test_default_config_describes_complete_simulation_stack():
    config = load_config()

    assert config.simulation.dt == 0.1
    assert config.simulation.num_steps == 3000
    assert config.simulation.random_seed == 7
    assert config.controller.type == "point_mass_acceleration"
    assert config.dynamics.type == "point_mass"
    assert config.sensor.type == "gaussian"
    assert config.estimator.type == "kalman_filter"
    assert config.guidance.type == "waypoint_tracker"
    assert config.guidance.waypoint_threshold == 0.5
    assert len(config.waypoints) == 4


def test_unknown_controller_type_is_rejected(tmp_path: Path):
    settings = default_settings()
    settings["controller"]["type"] = "unknown_controller"
    config_path = write_config(tmp_path, settings)

    with pytest.raises(
        ValueError,
        match="unsupported controller type: 'unknown_controller'",
    ):
        run_simulation(load_config(config_path), show_metrics=False)


@pytest.mark.parametrize(
    ("section", "unknown_type", "message"),
    [
        ("dynamics", "unknown_dynamics", "unsupported dynamics type"),
        ("sensor", "unknown_sensor", "unsupported sensor type"),
        ("estimator", "unknown_estimator", "unsupported estimator type"),
        ("guidance", "unknown_guidance", "unsupported guidance type"),
    ],
)
def test_unknown_subsystem_types_are_rejected(
    tmp_path: Path,
    section,
    unknown_type,
    message,
):
    settings = default_settings()
    settings[section]["type"] = unknown_type
    config_path = write_config(tmp_path, settings)

    with pytest.raises(ValueError, match=message):
        run_simulation(load_config(config_path), show_metrics=False)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("simulation", "dt", 0.0, "simulation.dt must be positive"),
        ("simulation", "num_steps", 0, "simulation.num_steps must be positive"),
        (
            "simulation",
            "random_seed",
            -1,
            "simulation.random_seed must be non-negative or null",
        ),
        ("controller", "kp", -1.0, "controller gains must be non-negative"),
        ("dynamics", "max_speed", 0.0, "dynamics.max_speed must be positive"),
        ("dynamics", "max_accel", 0.0, "dynamics.max_accel must be positive"),
        (
            "sensor",
            "pos_noise_std",
            -0.1,
            "sensor noise standard deviations must be non-negative",
        ),
        (
            "estimator",
            "process_var",
            -0.1,
            "estimator.process_var must be non-negative",
        ),
        ("estimator", "meas_var", 0.0, "estimator.meas_var must be positive"),
        (
            "guidance",
            "waypoint_threshold",
            0.0,
            "guidance.waypoint_threshold must be positive",
        ),
    ],
)
def test_invalid_numeric_configuration_is_rejected(
    tmp_path: Path,
    section,
    field,
    value,
    message,
):
    settings = default_settings()
    settings[section][field] = value
    config_path = write_config(tmp_path, settings)

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_configuration_requires_waypoints(tmp_path: Path):
    settings = default_settings()
    settings["waypoints"] = []
    config_path = write_config(tmp_path, settings)

    with pytest.raises(ValueError, match="at least one waypoint"):
        load_config(config_path)


def test_configuration_root_must_be_a_mapping(tmp_path: Path):
    config_path = write_config(tmp_path, ["not", "a", "mapping"])

    with pytest.raises(ValueError, match="configuration root must be a mapping"):
        load_config(config_path)


def test_configuration_requires_each_subsystem_section(tmp_path: Path):
    settings = default_settings()
    del settings["dynamics"]
    config_path = write_config(tmp_path, settings)

    with pytest.raises(ValueError, match="section 'dynamics' is required"):
        load_config(config_path)


def test_configuration_rejects_non_finite_values(tmp_path: Path):
    settings = default_settings()
    settings["sensor"]["pos_noise_std"] = float("inf")
    config_path = write_config(tmp_path, settings)

    with pytest.raises(ValueError, match="sensor.pos_noise_std.*must be finite"):
        load_config(config_path)


def test_num_steps_must_be_an_integer(tmp_path: Path):
    settings = default_settings()
    settings["simulation"]["num_steps"] = 10.5
    config_path = write_config(tmp_path, settings)

    with pytest.raises(ValueError, match="simulation.num_steps must be an integer"):
        load_config(config_path)


def test_random_seed_field_is_required(tmp_path: Path):
    settings = default_settings()
    del settings["simulation"]["random_seed"]
    config_path = write_config(tmp_path, settings)

    with pytest.raises(
        ValueError,
        match=r"configuration field 'simulation\.random_seed' is required",
    ):
        load_config(config_path)


def test_random_seed_can_be_explicitly_null(tmp_path: Path):
    settings = default_settings()
    settings["simulation"]["random_seed"] = None
    config_path = write_config(tmp_path, settings)

    config = load_config(config_path)

    assert config.simulation.random_seed is None


def test_top_level_waypoint_threshold_is_not_accepted(tmp_path: Path):
    settings = default_settings()
    threshold = settings.pop("guidance")["waypoint_threshold"]
    settings["waypoint_threshold"] = threshold
    config_path = write_config(tmp_path, settings)

    with pytest.raises(ValueError, match="section 'guidance' is required"):
        load_config(config_path)


@pytest.mark.parametrize(
    "section",
    ["controller", "dynamics", "sensor", "estimator", "guidance"],
)
def test_subsystem_type_is_required(tmp_path: Path, section):
    settings = default_settings()
    del settings[section]["type"]
    config_path = write_config(tmp_path, settings)

    with pytest.raises(
        ValueError,
        match=rf"configuration field '{section}\.type' is required",
    ):
        load_config(config_path)
