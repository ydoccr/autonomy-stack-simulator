import argparse
from pathlib import Path

import numpy as np
import yaml

from autonomy_sim.control.point_mass_acc_controller import PointMassAccController
from autonomy_sim.core.types import (
    ControllerConfig,
    DynamicsConfig,
    EstimatorConfig,
    GuidanceConfig,
    SensorConfig,
    SimConfig,
    SimulationResult,
    VehicleState,
    Waypoint,
)
from autonomy_sim.dynamics.point_mass import PointMassDynamics
from autonomy_sim.estimation.kalman_filter import KalmanFilter
from autonomy_sim.guidance.waypoint_tracker import WaypointTracker
from autonomy_sim.metrics.metrics_run import RunMetrics
from autonomy_sim.sensors.gaussian_sensor import GaussianSensor
from autonomy_sim.visualization.plot_run import plot_all, plot_metrics

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return SimConfig.from_dict(config)


def build_controller(config: ControllerConfig):
    if config.type == "point_mass_acceleration":
        return PointMassAccController(kp=config.kp, kd=config.kd)
    raise ValueError(f"unsupported controller type: {config.type!r}")


def build_dynamics(config: DynamicsConfig):
    if config.type == "point_mass":
        return PointMassDynamics(
            max_speed=config.max_speed,
            max_accel=config.max_accel,
        )
    raise ValueError(f"unsupported dynamics type: {config.type!r}")


def build_sensor(config: SensorConfig, rng: np.random.Generator):
    if config.type == "gaussian":
        return GaussianSensor(
            pos_noise_std=config.pos_noise_std,
            vel_noise_std=config.vel_noise_std,
            rng=rng,
        )
    raise ValueError(f"unsupported sensor type: {config.type!r}")


def build_estimator(config: EstimatorConfig, dt: float):
    if config.type == "kalman_filter":
        return KalmanFilter(
            dt=dt,
            process_var=config.process_var,
            meas_var=config.meas_var,
        )
    raise ValueError(f"unsupported estimator type: {config.type!r}")


def build_guidance(
    config: GuidanceConfig,
    waypoints: list[Waypoint],
):
    if config.type == "waypoint_tracker":
        return WaypointTracker(waypoints, config.waypoint_threshold)
    raise ValueError(f"unsupported guidance type: {config.type!r}")


def run_simulation(
    config: SimConfig,
    *,
    show_plots=False,
    show_metrics=True,
    initial_state=None,
    waypoints=None,
    waypoint_threshold=None,
    environment=None,
    sensor_model=None,
    scenario=None,
) -> SimulationResult:
    if initial_state is None:
        state = config.initial_state
    else:
        state = initial_state
    controller = build_controller(config.controller)
    dynamics = build_dynamics(config.dynamics)
    if waypoints is None:
        waypoints = config.waypoints
    if waypoint_threshold is None:
        waypoint_threshold = config.guidance.waypoint_threshold
    guidance_config = GuidanceConfig(
        type=config.guidance.type,
        waypoint_threshold=waypoint_threshold,
    )
    waypoint_tracker = build_guidance(guidance_config, waypoints)
    if sensor_model is None:
        rng = np.random.default_rng(config.simulation.random_seed)
        sensor = build_sensor(config.sensor, rng)
    else:
        sensor = sensor_model
    scenario_metadata = {
        "simulation_seed": config.simulation.random_seed,
        "controller_type": config.controller.type,
        "dynamics_type": config.dynamics.type,
        "sensor_type": config.sensor.type,
        "estimator_type": config.estimator.type,
        "guidance_type": config.guidance.type,
        "sensor_model": type(sensor).__name__,
    }
    if scenario is not None:
        scenario_metadata.update(scenario)
    kalman_filter = build_estimator(config.estimator, config.simulation.dt)

    sensor_data = sensor.sense(state)
    if sensor_data is None:
        kalman_filter.reset()
    else:
        kalman_filter.reset(VehicleState.from_array(sensor_data.as_array()))
    estimated_state = kalman_filter.current_state()
    waypoint_tracker.update(estimated_state)
    trajectory = []

    def record(time, ax, ay):
        measurement_available = sensor_data is not None
        if measurement_available:
            x_meas = sensor_data.x_meas
            y_meas = sensor_data.y_meas
            vx_meas = sensor_data.vx_meas
            vy_meas = sensor_data.vy_meas
        else:
            x_meas = np.nan
            y_meas = np.nan
            vx_meas = np.nan
            vy_meas = np.nan

        trajectory.append(
            {
                "time": time,
                "x": state.x,
                "y": state.y,
                "vx": state.vx,
                "vy": state.vy,
                "x_meas": x_meas,
                "y_meas": y_meas,
                "vx_meas": vx_meas,
                "vy_meas": vy_meas,
                "measurement_available": measurement_available,
                "x_est": estimated_state.x,
                "y_est": estimated_state.y,
                "vx_est": estimated_state.vx,
                "vy_est": estimated_state.vy,
                "ax_cmd": ax,
                "ay_cmd": ay,
                "current_waypoint_index": waypoint_tracker.current_index,
                "distance_to_waypoint": waypoint_tracker.distance_to_current_waypoint(
                    estimated_state
                ),
            }
        )

    record(0.0, 0.0, 0.0)
    for step in range(config.simulation.num_steps):
        if waypoint_tracker.complete:
            break
        requested_control = controller.compute_control(
            estimated_state, waypoint_tracker.current_waypoint()
        )
        state, applied_control = dynamics.step(
            state,
            requested_control,
            config.simulation.dt,
        )
        sensor_data = sensor.sense(state)
        estimated_state = kalman_filter.step(
            control=applied_control,
            measurement=sensor_data,
        )
        waypoint_tracker.update(estimated_state)
        record(
            (step + 1) * config.simulation.dt,
            requested_control.ax,
            requested_control.ay,
        )

    run_metrics = RunMetrics(
        trajectory,
        waypoints,
        waypoint_tracker.complete,
        waypoint_threshold,
        environment,
    )
    metrics = run_metrics.calculate()
    if show_plots:
        displayed_metrics = metrics if show_metrics else None
        plot_all(trajectory, waypoints, environment, displayed_metrics)
    elif show_metrics:
        plot_metrics(metrics)
    return SimulationResult(
        trajectory=trajectory,
        metrics=metrics,
        scenario=scenario_metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the waypoint simulation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    run_simulation(config, show_plots=True)


if __name__ == "__main__":
    main()
