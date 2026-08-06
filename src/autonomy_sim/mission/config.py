from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from autonomy_sim.core.types import VehicleState, Waypoint


@dataclass
class PlannerConfig:
    allow_diagonal: bool
    fuel_rate: float
    max_waypoint_distance: float
    waypoint_cost: float
    turn_cost_weight: float
    nominal_speed: float
    proximity_sigma: float
    allow_disallowed: bool
    max_cost: float


@dataclass
class HallwayMissionConfig:
    initial_state: VehicleState
    goal: Waypoint
    waypoint_threshold: float
    planner: PlannerConfig


@dataclass
class RandomMissionConfig:
    width: int
    height: int
    resolution: float
    zone_probabilities: dict[str, float]
    initial_state: VehicleState
    goal: Waypoint
    waypoint_threshold: float
    planner: PlannerConfig


def load_hallway_mission_config(path) -> HallwayMissionConfig:
    settings = _load_yaml(path, "hallway")
    config = HallwayMissionConfig(
        initial_state=_vehicle_state(settings),
        goal=_goal(settings),
        waypoint_threshold=_waypoint_threshold(settings),
        planner=_planner(settings),
    )
    _validate_common(config.initial_state, config.goal, config.waypoint_threshold)
    if round(config.planner.max_waypoint_distance / 0.1) < 1:
        raise ValueError("planner.max_waypoint_distance is below hallway resolution")
    return config


def load_random_mission_config(path) -> RandomMissionConfig:
    settings = _load_yaml(path, "random")
    environment = _mapping(settings, "environment", "mission")
    width = _integer(environment, "width", "environment")
    height = _integer(environment, "height", "environment")
    resolution = _number(environment, "resolution", "environment")
    probabilities = _mapping(environment, "zone_probabilities", "environment")
    zone_names = {"free", "occupied", "disallowed", "restricted"}
    if set(probabilities) != zone_names:
        raise ValueError(
            "environment.zone_probabilities must define free, occupied, "
            "disallowed, and restricted"
        )
    zone_probabilities = {
        zone: _number(probabilities, zone, "environment.zone_probabilities")
        for zone in ("free", "occupied", "disallowed", "restricted")
    }
    config = RandomMissionConfig(
        width=width,
        height=height,
        resolution=resolution,
        zone_probabilities=zone_probabilities,
        initial_state=_vehicle_state(settings),
        goal=_goal(settings),
        waypoint_threshold=_waypoint_threshold(settings),
        planner=_planner(settings),
    )
    _validate_common(config.initial_state, config.goal, config.waypoint_threshold)
    if config.width <= 0 or config.height <= 0:
        raise ValueError("environment width and height must be positive")
    if config.resolution <= 0.0:
        raise ValueError("environment.resolution must be positive")
    cells_per_block = round(1.0 / config.resolution)
    if not np.isclose(cells_per_block * config.resolution, 1.0):
        raise ValueError("environment.resolution must divide one world unit")
    if round(config.planner.max_waypoint_distance / config.resolution) < 1:
        raise ValueError("planner.max_waypoint_distance is below grid resolution")
    probability_sum = sum(config.zone_probabilities.values())
    if any(value < 0.0 for value in config.zone_probabilities.values()):
        raise ValueError("environment zone probabilities must be non-negative")
    if not np.isclose(probability_sum, 1.0):
        raise ValueError("environment zone probabilities must sum to one")
    return config


def world_to_grid_cell(x, y, resolution, width, height):
    column = round(float(x) / resolution)
    row = round(float(y) / resolution)
    if not (0 <= row < height and 0 <= column < width):
        raise ValueError("mission position is outside the environment")
    return row, column


def _load_yaml(path, expected_type):
    with Path(path).open(encoding="utf-8") as config_file:
        settings = yaml.safe_load(config_file)
    if not isinstance(settings, dict):
        raise ValueError("mission configuration root must be a mapping")
    mission_type = settings.get("type")
    if mission_type != expected_type:
        raise ValueError(
            f"mission type must be {expected_type!r}, received {mission_type!r}"
        )
    return settings


def _vehicle_state(settings):
    state = _mapping(settings, "initial_state", "mission")
    return VehicleState(
        x=_number(state, "x", "initial_state"),
        y=_number(state, "y", "initial_state"),
        vx=_number(state, "vx", "initial_state"),
        vy=_number(state, "vy", "initial_state"),
    )


def _goal(settings):
    goal = _mapping(settings, "goal", "mission")
    return Waypoint(
        x=_number(goal, "x", "goal"),
        y=_number(goal, "y", "goal"),
    )


def _waypoint_threshold(settings):
    guidance = _mapping(settings, "guidance", "mission")
    return _number(guidance, "waypoint_threshold", "guidance")


def _planner(settings):
    planner = _mapping(settings, "planner", "mission")
    allow_diagonal = _boolean(planner, "allow_diagonal", "planner")
    allow_disallowed = _boolean(planner, "allow_disallowed", "planner")
    config = PlannerConfig(
        allow_diagonal=allow_diagonal,
        fuel_rate=_number(planner, "fuel_rate", "planner"),
        max_waypoint_distance=_number(
            planner,
            "max_waypoint_distance",
            "planner",
        ),
        waypoint_cost=_number(planner, "waypoint_cost", "planner"),
        turn_cost_weight=_number(planner, "turn_cost_weight", "planner"),
        nominal_speed=_number(planner, "nominal_speed", "planner"),
        proximity_sigma=_number(planner, "proximity_sigma", "planner"),
        allow_disallowed=allow_disallowed,
        max_cost=_number(planner, "max_cost", "planner", allow_infinite=True),
    )
    if config.fuel_rate <= 0.0:
        raise ValueError("planner.fuel_rate must be positive")
    if config.max_waypoint_distance <= 0.0:
        raise ValueError("planner.max_waypoint_distance must be positive")
    if config.waypoint_cost < 0.0 or config.turn_cost_weight < 0.0:
        raise ValueError("planner cost weights must be non-negative")
    if config.nominal_speed <= 0.0:
        raise ValueError("planner.nominal_speed must be positive")
    if config.proximity_sigma < 0.0:
        raise ValueError("planner.proximity_sigma must be non-negative")
    if config.max_cost <= 0.0:
        raise ValueError("planner.max_cost must be positive")
    return config


def _validate_common(initial_state, goal, waypoint_threshold):
    values = [*initial_state.as_array(), *goal.as_array()]
    if not np.all(np.isfinite(values)):
        raise ValueError("mission initial state and goal must be finite")
    if waypoint_threshold <= 0.0:
        raise ValueError("guidance.waypoint_threshold must be positive")


def _mapping(settings, name, section_name):
    if name not in settings:
        raise ValueError(f"mission field '{section_name}.{name}' is required")
    value = settings[name]
    if not isinstance(value, dict):
        raise ValueError(f"mission field '{section_name}.{name}' must be a mapping")
    return value


def _number(settings, name, section_name, allow_infinite=False):
    if name not in settings:
        raise ValueError(f"mission field '{section_name}.{name}' is required")
    value = settings[name]
    if isinstance(value, bool):
        raise ValueError(f"mission field '{section_name}.{name}' must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"mission field '{section_name}.{name}' must be numeric"
        ) from error
    if np.isnan(number) or (not allow_infinite and not np.isfinite(number)):
        raise ValueError(f"mission field '{section_name}.{name}' must be finite")
    return number


def _integer(settings, name, section_name):
    if name not in settings:
        raise ValueError(f"mission field '{section_name}.{name}' is required")
    value = settings[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"mission field '{section_name}.{name}' must be an integer")
    return value


def _boolean(settings, name, section_name):
    if name not in settings:
        raise ValueError(f"mission field '{section_name}.{name}' is required")
    value = settings[name]
    if not isinstance(value, bool):
        raise ValueError(f"mission field '{section_name}.{name}' must be boolean")
    return value
