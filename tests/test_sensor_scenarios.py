import numpy as np
import pytest

from autonomy_sim.sensors.combined_error_sensor import CombinedErrorSensor
from autonomy_sim.sensors.sensor_scenarios import (
    create_sensor_scenario,
    sensor_scenario_name,
)


def test_sensor_scenarios_have_requested_errors():
    offset = create_sensor_scenario(1)
    delay = create_sensor_scenario(2)
    spotty = create_sensor_scenario(3)
    combined = create_sensor_scenario(4)

    for sensor in (offset, delay, spotty, combined):
        assert isinstance(sensor, CombinedErrorSensor)
        assert sensor.gaussian_sensor is not None

    assert offset.offset_sensor is not None
    assert offset.offset_sensor.x_offset == 0.2
    assert offset.offset_sensor.y_offset == -0.3
    assert delay.delay_sensor is not None
    assert delay.delay_sensor.delay_steps == 3
    assert spotty.spotty_sensor is not None
    assert spotty.spotty_sensor.dropout_probability == 0.4
    assert combined.offset_sensor is not None
    assert combined.delay_sensor is not None
    assert combined.spotty_sensor is not None


def test_sensor_scenario_seed_repeats_dropouts():
    first = create_sensor_scenario(3, random_seed=12)
    second = create_sensor_scenario(3, random_seed=12)

    first_draws = [first.spotty_sensor.rng.random() for _ in range(5)]
    second_draws = [second.spotty_sensor.rng.random() for _ in range(5)]

    assert np.array_equal(first_draws, second_draws)


def test_sensor_scenario_number_must_be_valid():
    with pytest.raises(ValueError):
        create_sensor_scenario(5)


def test_sensor_scenario_names_describe_nominal_and_additive_faults():
    assert sensor_scenario_name(0) == "gaussian_nominal"
    assert sensor_scenario_name(1) == "gaussian_plus_offset"
    assert sensor_scenario_name(2) == "gaussian_plus_delay"
    assert sensor_scenario_name(3) == "gaussian_plus_dropout"
    assert sensor_scenario_name(4) == "gaussian_plus_combined_faults"

    with pytest.raises(ValueError):
        sensor_scenario_name(5)


def test_sensor_scenario_uses_configured_gaussian_noise():
    sensor = create_sensor_scenario(
        3,
        pos_noise_std=0.25,
        vel_noise_std=0.5,
    )

    assert sensor.gaussian_sensor.pos_noise_std == 0.25
    assert sensor.gaussian_sensor.vel_noise_std == 0.5
