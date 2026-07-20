import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from autonomy_sim.environments.example_environments import (
    OCCUPIED_COST,
)

ENVIRONMENT_COLORS = [
    "#d9f2d9",  # free: light green
    "#fff4b8",  # disallowed: light yellow
    "#cfe8ff",  # occupied: light blue
    "#f7c6c7",  # restricted: light red
]


def plot_environment_overlay(costmap, resolution=1.0, ax=None):
    if ax is None:
        ax = plt.gca()

    zone_grid = np.zeros(costmap.shape, dtype=int)
    finite_positive = (costmap > 0.0) & np.isfinite(costmap)
    zone_grid[finite_positive] = 1
    zone_grid[costmap == OCCUPIED_COST] = 2
    zone_grid[np.isinf(costmap)] = 3

    height, width = costmap.shape
    extent = [
        -0.5 * resolution,
        (width - 0.5) * resolution,
        -0.5 * resolution,
        (height - 0.5) * resolution,
    ]
    ax.imshow(
        zone_grid,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(ENVIRONMENT_COLORS),
        interpolation="nearest",
        alpha=0.75,
    )


def add_environment_legend(ax=None):
    if ax is None:
        ax = plt.gca()

    handles, _ = ax.get_legend_handles_labels()
    zone_handles = [
        Patch(color=ENVIRONMENT_COLORS[0], label="Free"),
        Patch(color=ENVIRONMENT_COLORS[1], label="Disallowed"),
        Patch(color=ENVIRONMENT_COLORS[2], label="Occupied"),
        Patch(color=ENVIRONMENT_COLORS[3], label="Restricted"),
    ]
    ax.legend(handles=handles + zone_handles, fontsize="small")


def plot_spatial_tracking(ax, trajectory, waypoints, environment=None):
    true_x = [sample["x"] for sample in trajectory]
    true_y = [sample["y"] for sample in trajectory]
    measured_x = [sample["x_meas"] for sample in trajectory]
    measured_y = [sample["y_meas"] for sample in trajectory]
    estimated_x = [sample["x_est"] for sample in trajectory]
    estimated_y = [sample["y_est"] for sample in trajectory]
    waypoint_x = [true_x[0]] + [waypoint.x for waypoint in waypoints]
    waypoint_y = [true_y[0]] + [waypoint.y for waypoint in waypoints]

    if environment is not None:
        plot_environment_overlay(
            environment.to_zone_costmap(),
            environment.resolution,
            ax,
        )

    ax.plot(waypoint_x, waypoint_y, "k--", marker="x", label="Waypoint path")
    ax.scatter(
        measured_x,
        measured_y,
        s=8,
        alpha=0.35,
        label="Sensor positions",
    )
    ax.plot(estimated_x, estimated_y, "b--", label="Estimated path")
    ax.plot(true_x, true_y, "g-", label="Actual path")
    ax.scatter(true_x[0], true_y[0], marker="o", label="Start")
    ax.scatter(true_x[-1], true_y[-1], marker="*", s=80, label="Final")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Environment and Path Tracking")
    ax.grid(True)
    ax.axis("equal")
    if environment is None:
        ax.legend(fontsize="small")
    else:
        add_environment_legend(ax)


def plot_sensor_vs_estimator(ax, trajectory):
    time = np.array([sample["time"] for sample in trajectory])
    true_position = _pairs(trajectory, "x", "y")
    measured_position = _pairs(trajectory, "x_meas", "y_meas")
    estimated_position = _pairs(trajectory, "x_est", "y_est")
    sensor_error = np.linalg.norm(measured_position - true_position, axis=1)
    estimator_error = np.linalg.norm(estimated_position - true_position, axis=1)

    ax.plot(time, sensor_error, label="Sensor position error", alpha=0.7)
    ax.plot(time, estimator_error, label="Estimated position error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error")
    ax.set_title("Sensor vs. Kalman Estimate")
    ax.legend(fontsize="small")
    ax.grid(True)


def plot_speed_tracking(ax, trajectory):
    time = [sample["time"] for sample in trajectory]
    true_speed = _magnitudes(trajectory, "vx", "vy")
    measured_speed = _magnitudes(trajectory, "vx_meas", "vy_meas")
    estimated_speed = _magnitudes(trajectory, "vx_est", "vy_est")

    ax.plot(time, measured_speed, label="Sensor speed", alpha=0.6)
    ax.plot(time, estimated_speed, "b--", label="Estimated speed")
    ax.plot(time, true_speed, "g-", label="Actual speed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed")
    ax.set_title("Velocity Tracking")
    ax.legend(fontsize="small")
    ax.grid(True)


def plot_acceleration(ax, trajectory):
    time = [sample["time"] for sample in trajectory]
    acceleration = _magnitudes(trajectory, "ax_cmd", "ay_cmd")

    ax.plot(time, acceleration)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Commanded acceleration")
    ax.set_title("Control Effort")
    ax.grid(True)


def plot_distance_to_waypoint(ax, trajectory):
    time = [sample["time"] for sample in trajectory]
    distance = [sample["distance_to_waypoint"] for sample in trajectory]

    ax.plot(time, distance)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance")
    ax.set_title("Distance to Current Waypoint")
    ax.grid(True)


def plot_waypoint_index(ax, trajectory):
    time = [sample["time"] for sample in trajectory]
    waypoint_index = [sample["current_waypoint_index"] for sample in trajectory]

    ax.step(time, waypoint_index, where="post")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Waypoint index")
    ax.set_title("Guidance Progress")
    ax.grid(True)


def plot_metrics(metrics):
    figure, ax = plt.subplots(figsize=(7, 7))
    plot_metrics_panel(ax, metrics)
    figure.tight_layout()
    plt.show()


def plot_metrics_panel(ax, metrics):
    final_state = metrics["final_state"]
    lines = [
        f"Termination: {metrics['termination_state']}",
        f"Planning success: {metrics['planning_success']}",
        f"True mission success: {metrics['true_mission_success']}",
        f"True goal reached: {metrics['true_goal_reached']}",
        f"Onboard completion: {metrics['onboard_completion']}",
        f"Restricted violation: {metrics['restricted_violation']}",
        f"Disallowed violation: {metrics['disallowed_violation']}",
        f"Out-of-bounds violation: {metrics['out_of_bounds_violation']}",
        f"Completion time: {metrics['completion_time']:.2f} s",
        f"Simulation steps: {metrics['number_of_steps']}",
        "",
        f"Final position: ({final_state.x:.3f}, {final_state.y:.3f})",
        f"Final velocity: ({final_state.vx:.3f}, {final_state.vy:.3f})",
        f"Final goal distance: {metrics['final_true_distance']:.3f}",
        f"Final waypoint index: {metrics['final_waypoint_index']}",
        "",
        f"Actual path length: {metrics['actual_path_length']:.3f}",
        f"Planned path length: {metrics['planned_path_length']:.3f}",
        f"Measurement dropout: {metrics['measurement_dropout_fraction']:.3f}",
        f"Mean sensor error: {metrics['mean_measurement_error']:.3f}",
        f"Sensor RMSE: {metrics['rmse_measurement_error']:.3f}",
        f"Mean estimate error: {metrics['mean_estimation_error']:.3f}",
        f"Maximum estimate error: {metrics['max_estimation_error']:.3f}",
        f"Estimate RMSE: {metrics['rmse_estimation_error']:.3f}",
        f"Control effort: {metrics['control_effort']:.3f}",
        f"Maximum acceleration: {metrics['max_commanded_acceleration']:.3f}",
        f"Maximum speed: {metrics['max_speed']:.3f}",
    ]
    ax.set_title("Mission Metrics", loc="left")
    ax.text(
        0.0,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
    )
    ax.axis("off")


def plot_all(trajectory, waypoints, environment=None, metrics=None):
    if metrics is None:
        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        figure = plt.figure(figsize=(21, 10))
        grid = figure.add_gridspec(
            2,
            4,
            width_ratios=[1.0, 1.0, 1.0, 0.8],
        )
        axes = np.empty((2, 3), dtype=object)
        for row in range(2):
            for column in range(3):
                axes[row, column] = figure.add_subplot(grid[row, column])
        metrics_ax = figure.add_subplot(grid[:, 3])
        plot_metrics_panel(metrics_ax, metrics)

    plot_spatial_tracking(axes[0, 0], trajectory, waypoints, environment)
    plot_sensor_vs_estimator(axes[0, 1], trajectory)
    plot_speed_tracking(axes[0, 2], trajectory)
    plot_acceleration(axes[1, 0], trajectory)
    plot_distance_to_waypoint(axes[1, 1], trajectory)
    plot_waypoint_index(axes[1, 2], trajectory)
    figure.suptitle("Autonomy Mission Summary")
    figure.tight_layout()
    plt.show()


def _pairs(trajectory, first_name, second_name):
    return np.array(
        [[sample[first_name], sample[second_name]] for sample in trajectory],
        dtype=float,
    )


def _magnitudes(trajectory, first_name, second_name):
    pairs = _pairs(trajectory, first_name, second_name)
    return np.linalg.norm(pairs, axis=1)
