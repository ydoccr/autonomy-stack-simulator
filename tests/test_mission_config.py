from pathlib import Path

import pytest
import yaml

from autonomy_sim.mission.config import (
    load_hallway_mission_config,
    load_random_mission_config,
    world_to_grid_cell,
)
from autonomy_sim.mission.run_hallway_mission import (
    DEFAULT_MISSION_CONFIG as HALLWAY_CONFIG,
)
from autonomy_sim.mission.run_random_mission import (
    DEFAULT_MISSION_CONFIG as RANDOM_CONFIG,
)


def load_settings(path):
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def write_settings(tmp_path, settings):
    config_path = tmp_path / "mission.yaml"
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return config_path


def test_default_random_mission_configuration():
    config = load_random_mission_config(RANDOM_CONFIG)

    assert config.width == 10
    assert config.height == 10
    assert config.resolution == 0.1
    assert config.initial_state.x == 0.5
    assert config.goal.x == 9.5
    assert config.goal.y == 9.5
    assert config.waypoint_threshold == 0.2
    assert config.true_goal_tolerance == 0.3
    assert config.planner.max_waypoint_distance == 0.5
    assert config.planner.turn_cost_weight == 0.5
    assert config.planner.minimum_clearance == 0.2
    assert config.planner.allow_disallowed is False


def test_default_hallway_mission_configuration():
    config = load_hallway_mission_config(HALLWAY_CONFIG)

    assert config.initial_state.x == 0.5
    assert config.goal.x == 9.5
    assert config.goal.y == 9.5
    assert config.waypoint_threshold == 0.2
    assert config.true_goal_tolerance == 0.3
    assert config.planner.max_waypoint_distance == 1.0
    assert config.planner.minimum_clearance == 0.2


def test_mission_type_must_match_loader(tmp_path):
    settings = load_settings(RANDOM_CONFIG)
    settings["type"] = "hallway"
    config_path = write_settings(tmp_path, settings)

    with pytest.raises(ValueError, match="mission type must be 'random'"):
        load_random_mission_config(config_path)


def test_random_zone_probabilities_must_be_complete(tmp_path):
    settings = load_settings(RANDOM_CONFIG)
    del settings["environment"]["zone_probabilities"]["restricted"]
    config_path = write_settings(tmp_path, settings)

    with pytest.raises(ValueError, match="must define free, occupied"):
        load_random_mission_config(config_path)


def test_random_zone_probabilities_must_sum_to_one(tmp_path):
    settings = load_settings(RANDOM_CONFIG)
    settings["environment"]["zone_probabilities"]["free"] = 0.5
    config_path = write_settings(tmp_path, settings)

    with pytest.raises(ValueError, match="must sum to one"):
        load_random_mission_config(config_path)


def test_max_waypoint_distance_must_reach_another_cell(tmp_path):
    settings = load_settings(RANDOM_CONFIG)
    settings["planner"]["max_waypoint_distance"] = 0.01
    config_path = write_settings(tmp_path, settings)

    with pytest.raises(ValueError, match="below grid resolution"):
        load_random_mission_config(config_path)


def test_world_positions_convert_to_grid_cells():
    start = world_to_grid_cell(0.0, 0.0, 0.1, 100, 100)
    goal = world_to_grid_cell(9.9, 9.9, 0.1, 100, 100)

    assert start == (0, 0)
    assert goal == (99, 99)


def test_world_position_outside_environment_is_rejected():
    with pytest.raises(ValueError, match="outside the environment"):
        world_to_grid_cell(10.0, 10.0, 0.1, 100, 100)
