from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SimulationSettings:
    dt: float
    num_steps: int
    random_seed: int | None


@dataclass
class ControllerConfig:
    type: str
    kp: float
    kd: float


@dataclass
class DynamicsConfig:
    type: str
    max_speed: float
    max_accel: float


@dataclass
class SensorConfig:
    type: str
    pos_noise_std: float
    vel_noise_std: float


@dataclass
class EstimatorConfig:
    type: str
    process_var: float
    meas_var: float


@dataclass
class GuidanceConfig:
    type: str
    waypoint_threshold: float


@dataclass
class SimulationResult:
    trajectory: list[dict[str, Any]]
    metrics: dict[str, Any]
    scenario: dict[str, Any]


@dataclass
class VehicleState:
    x: float
    y: float
    vx: float
    vy: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], dtype=float)

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)

    @staticmethod
    def from_array(arr) -> "VehicleState":
        return VehicleState(
            x=float(arr[0]),
            y=float(arr[1]),
            vx=float(arr[2]),
            vy=float(arr[3]),
        )


@dataclass
class ControlInput:
    ax: float
    ay: float

    def as_array(self) -> np.ndarray:
        return np.array([self.ax, self.ay], dtype=float)

    @staticmethod
    def from_array(arr) -> "ControlInput":
        return ControlInput(ax=float(arr[0]), ay=float(arr[1]))


@dataclass
class SensorData:
    x_meas: float
    y_meas: float
    vx_meas: float
    vy_meas: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x_meas, self.y_meas, self.vx_meas, self.vy_meas],
            dtype=float,
        )

    @staticmethod
    def from_array(arr) -> "SensorData":
        return SensorData(
            x_meas=float(arr[0]),
            y_meas=float(arr[1]),
            vx_meas=float(arr[2]),
            vy_meas=float(arr[3]),
        )


@dataclass
class Waypoint:
    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @staticmethod
    def from_array(arr) -> "Waypoint":
        return Waypoint(x=float(arr[0]), y=float(arr[1]))


@dataclass
class SimConfig:
    simulation: SimulationSettings
    initial_state: VehicleState
    controller: ControllerConfig
    dynamics: DynamicsConfig
    sensor: SensorConfig
    estimator: EstimatorConfig
    guidance: GuidanceConfig
    waypoints: list[Waypoint]

    @classmethod
    def from_dict(cls, settings: dict[str, Any]) -> "SimConfig":
        if not isinstance(settings, dict):
            raise ValueError("configuration root must be a mapping")

        simulation = _required_mapping(settings, "simulation")
        initial_state = _required_mapping(settings, "initial_state")
        controller = _required_mapping(settings, "controller")
        dynamics = _required_mapping(settings, "dynamics")
        sensor = _required_mapping(settings, "sensor")
        estimator = _required_mapping(settings, "estimator")
        guidance = _required_mapping(settings, "guidance")

        raw_waypoints = settings.get("waypoints")
        if not isinstance(raw_waypoints, list):
            raise ValueError("configuration field 'waypoints' must be a list")
        waypoints = []
        for index, waypoint in enumerate(raw_waypoints):
            if not isinstance(waypoint, dict):
                raise ValueError(f"waypoints[{index}] must be a mapping")
            waypoints.append(
                Waypoint(
                    x=_required_float(waypoint, "x", f"waypoints[{index}]"),
                    y=_required_float(waypoint, "y", f"waypoints[{index}]"),
                )
            )

        if "random_seed" not in simulation:
            raise ValueError("configuration field 'simulation.random_seed' is required")
        random_seed = simulation["random_seed"]
        if random_seed is not None and (
            isinstance(random_seed, bool) or not isinstance(random_seed, int)
        ):
            raise ValueError("simulation.random_seed must be an integer or null")

        num_steps = simulation.get("num_steps")
        if isinstance(num_steps, bool) or not isinstance(num_steps, int):
            raise ValueError("simulation.num_steps must be an integer")

        config = cls(
            simulation=SimulationSettings(
                dt=_required_float(simulation, "dt", "simulation"),
                num_steps=num_steps,
                random_seed=random_seed,
            ),
            initial_state=VehicleState(
                x=_required_float(initial_state, "x", "initial_state"),
                y=_required_float(initial_state, "y", "initial_state"),
                vx=_required_float(initial_state, "vx", "initial_state"),
                vy=_required_float(initial_state, "vy", "initial_state"),
            ),
            controller=ControllerConfig(
                type=_subsystem_type(controller, "controller"),
                kp=_required_float(controller, "kp", "controller"),
                kd=_required_float(controller, "kd", "controller"),
            ),
            dynamics=DynamicsConfig(
                type=_subsystem_type(dynamics, "dynamics"),
                max_speed=_required_float(dynamics, "max_speed", "dynamics"),
                max_accel=_required_float(dynamics, "max_accel", "dynamics"),
            ),
            sensor=SensorConfig(
                type=_subsystem_type(sensor, "sensor"),
                pos_noise_std=_required_float(sensor, "pos_noise_std", "sensor"),
                vel_noise_std=_required_float(sensor, "vel_noise_std", "sensor"),
            ),
            estimator=EstimatorConfig(
                type=_subsystem_type(estimator, "estimator"),
                process_var=_required_float(estimator, "process_var", "estimator"),
                meas_var=_required_float(estimator, "meas_var", "estimator"),
            ),
            guidance=GuidanceConfig(
                type=_subsystem_type(guidance, "guidance"),
                waypoint_threshold=_required_float(
                    guidance,
                    "waypoint_threshold",
                    "guidance",
                ),
            ),
            waypoints=waypoints,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.simulation.dt <= 0.0:
            raise ValueError("simulation.dt must be positive")
        if self.simulation.num_steps <= 0:
            raise ValueError("simulation.num_steps must be positive")
        if self.simulation.random_seed is not None and self.simulation.random_seed < 0:
            raise ValueError("simulation.random_seed must be non-negative or null")
        if self.controller.kp < 0.0 or self.controller.kd < 0.0:
            raise ValueError("controller gains must be non-negative")
        if self.dynamics.max_speed <= 0.0:
            raise ValueError("dynamics.max_speed must be positive")
        if self.dynamics.max_accel <= 0.0:
            raise ValueError("dynamics.max_accel must be positive")
        if self.sensor.pos_noise_std < 0.0 or self.sensor.vel_noise_std < 0.0:
            raise ValueError("sensor noise standard deviations must be non-negative")
        if self.estimator.process_var < 0.0:
            raise ValueError("estimator.process_var must be non-negative")
        if self.estimator.meas_var <= 0.0:
            raise ValueError("estimator.meas_var must be positive")
        if self.guidance.waypoint_threshold <= 0.0:
            raise ValueError("guidance.waypoint_threshold must be positive")
        if not self.waypoints:
            raise ValueError("waypoints must contain at least one waypoint")


def _required_mapping(settings, name):
    if name not in settings:
        raise ValueError(f"configuration section {name!r} is required")
    section = settings[name]
    if not isinstance(section, dict):
        raise ValueError(f"configuration section {name!r} must be a mapping")
    return section


def _required_float(settings, name, section_name):
    if name not in settings:
        raise ValueError(f"configuration field '{section_name}.{name}' is required")
    return _float_value(settings[name], f"{section_name}.{name}")


def _float_value(value, field_name):
    if isinstance(value, bool):
        raise ValueError(f"configuration field '{field_name}' must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"configuration field '{field_name}' must be numeric"
        ) from error
    if not np.isfinite(number):
        raise ValueError(f"configuration field '{field_name}' must be finite")
    return number


def _subsystem_type(settings, section_name):
    if "type" not in settings:
        raise ValueError(f"configuration field '{section_name}.type' is required")
    subsystem_type = settings["type"]
    if not isinstance(subsystem_type, str) or not subsystem_type:
        raise ValueError(f"configuration field '{section_name}.type' must be a name")
    return subsystem_type
