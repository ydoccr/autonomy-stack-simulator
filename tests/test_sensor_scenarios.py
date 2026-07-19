import numpy as np
import pytest

from autonomy_sim.sensors.combined_error_sensor import CombinedErrorSensor
from autonomy_sim.sensors.delay_sensor import DelaySensor
from autonomy_sim.sensors.offset_sensor import OffsetSensor
from autonomy_sim.sensors.sensor_scenarios import create_sensor_scenario
from autonomy_sim.sensors.spotty_sensor import SpottySensor


def test_sensor_scenarios_have_requested_errors():
    offset = create_sensor_scenario(1)
    delay = create_sensor_scenario(2)
    spotty = create_sensor_scenario(3)
    combined = create_sensor_scenario(4)

    assert isinstance(offset, OffsetSensor)
    assert offset.x_offset == 0.2
    assert offset.y_offset == -0.3
    assert isinstance(delay, DelaySensor)
    assert delay.delay_steps == 3
    assert isinstance(spotty, SpottySensor)
    assert spotty.dropout_probability == 0.4
    assert isinstance(combined, CombinedErrorSensor)
    assert combined.gaussian_sensor is not None
    assert combined.offset_sensor is not None
    assert combined.delay_sensor is not None
    assert combined.spotty_sensor is not None


def test_sensor_scenario_seed_repeats_dropouts():
    first = create_sensor_scenario(3, random_seed=12)
    second = create_sensor_scenario(3, random_seed=12)

    first_draws = [first.rng.random() for _ in range(5)]
    second_draws = [second.rng.random() for _ in range(5)]

    assert np.array_equal(first_draws, second_draws)


def test_sensor_scenario_number_must_be_valid():
    with pytest.raises(ValueError):
        create_sensor_scenario(5)
