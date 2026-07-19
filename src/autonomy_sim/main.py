import argparse
from pathlib import Path

import numpy as np
import yaml

from autonomy_sim.control.point_mass_acc_controller import PointMassAccController
from autonomy_sim.core.types import SimConfig, VehicleState, Waypoint
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
    return config


def run_simulation(
    config_path=DEFAULT_CONFIG,
    *,
    show_plots=False,
    show_metrics=True,
    initial_state=None,
    waypoints=None,
    waypoint_threshold=None,
    environment=None,
    sensor_model=None,
):
    settings = load_config(config_path)
    simulation_settings = settings["simulation"]
    config = SimConfig(
        dt=float(simulation_settings["dt"]),
        num_steps=int(simulation_settings["num_steps"]),
    )
    if initial_state is None:
        state = VehicleState(**settings["initial_state"])
    else:
        state = initial_state
    controller = PointMassAccController(**settings["controller"])
    dynamics = PointMassDynamics(**settings["dynamics"])
    if waypoints is None:
        waypoints = [
            Waypoint(**waypoint) for waypoint in settings["waypoints"]
        ]
    if waypoint_threshold is None:
        waypoint_threshold = settings["waypoint_threshold"]
    waypoint_tracker = WaypointTracker(waypoints, waypoint_threshold)
    if sensor_model is None:
        rng = np.random.default_rng(simulation_settings.get("random_seed"))
        sensor = GaussianSensor(**settings["sensor"], rng=rng)
    else:
        sensor = sensor_model
    kalman_filter = KalmanFilter(dt=config.dt, **settings["estimator"])

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
    for step in range(config.num_steps):
        if waypoint_tracker.complete:
            break
        requested_control = controller.compute_control(
            estimated_state, waypoint_tracker.current_waypoint()
        )
        state = dynamics.step(state, requested_control, config.dt)
        sensor_data = sensor.sense(state)
        estimated_state = kalman_filter.step(
            control=requested_control,
            measurement=sensor_data,
        )
        waypoint_tracker.update(estimated_state)
        record(
            (step + 1) * config.dt,
            requested_control.ax,
            requested_control.ay,
        )

    run_metrics = RunMetrics(
        trajectory,
        waypoints,
        waypoint_tracker.complete,
    )
    metrics = run_metrics.calculate()
    if show_plots:
        displayed_metrics = None
        if show_metrics:
            displayed_metrics = metrics
        plot_all(trajectory, waypoints, environment, displayed_metrics)
    elif show_metrics:
        plot_metrics(metrics)
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the waypoint simulation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run_simulation(args.config, show_plots=True)


if __name__ == "__main__":
    main()
