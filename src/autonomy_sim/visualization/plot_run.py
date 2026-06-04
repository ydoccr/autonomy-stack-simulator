import numpy as np
import matplotlib.pyplot as plt
from autonomy_sim.core.types import Waypoint

def plot_trajectory(trajectory, waypoints: list[Waypoint]):
    x_values = [sample["x"] for sample in trajectory]
    y_values = [sample["y"] for sample in trajectory]
    waypoint_x_values = [x_values[0]] + [waypoint.x for waypoint in waypoints]
    waypoint_y_values = [y_values[0]] + [waypoint.y for waypoint in waypoints]
    plt.figure(figsize=(8, 8))
    plt.plot(
        x_values,
        y_values,
        label="Vehicle Trajectory",
        marker="o",
        markersize=3,
    )
    plt.plot(
        waypoint_x_values,
        waypoint_y_values,
        label="Waypoint Path",
        marker="x",
    )
    plt.scatter(x_values[0], y_values[0], label="Start Position", marker="o")
    plt.scatter(x_values[-1], y_values[-1], label="Final Position", marker="o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Point-mass waypoint tracking")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def plot_speed(trajectory):
    time_values = [sample["time"] for sample in trajectory]
    speed_values = [
        np.hypot(sample["vx"], sample["vy"]) for sample in trajectory
    ]
    plt.figure(figsize=(8, 8))
    plt.plot(time_values, speed_values)
    plt.xlabel("Time")
    plt.ylabel("Speed")
    plt.title("Speed vs. Time")
    plt.grid(True)
    plt.show()

def plot_acceleration(trajectory):
    time_values = [sample["time"] for sample in trajectory]
    acceleration_values = [
        np.hypot(sample["ax_cmd"], sample["ay_cmd"]) for sample in trajectory
    ]
    plt.figure(figsize=(8, 8))
    plt.plot(time_values, acceleration_values)
    plt.xlabel("Time")
    plt.ylabel("Control-inflicted acceleration")
    plt.title("Control-inflicted acceleration vs. Time")
    plt.grid(True)
    plt.show()

def plot_waypoint_index(trajectory) -> None:
    time_values = [sample["time"] for sample in trajectory]
    waypoint_index_values = [
        sample["current_waypoint_index"] for sample in trajectory
    ]
    plt.figure(figsize=(8, 8))
    plt.scatter(time_values, waypoint_index_values, s=10)
    plt.xlabel("Time")
    plt.ylabel("Current waypoint index")
    plt.title("Waypoint index vs. Time")
    plt.grid(True)
    plt.show()

def plot_distance_to_waypoint(trajectory):
    plt.figure(figsize=(8, 8))
    plt.plot(
        [sample["time"] for sample in trajectory], 
        [sample["distance_to_waypoint"] for sample in trajectory]
        )
    plt.xlabel("Time")
    plt.ylabel("Distance to current waypoint")
    plt.title("Waypoint distance vs. Time")
    plt.grid(True)
    plt.show()

def plot_true_vs_measured_trajectory(trajectory, waypoints: list[Waypoint]):
    true_x_values = [sample["x"] for sample in trajectory]
    true_y_values = [sample["y"] for sample in trajectory]
    measured_x_values = [sample["x_meas"] for sample in trajectory]
    measured_y_values = [sample["y_meas"] for sample in trajectory]
    waypoint_x_values = [true_x_values[0]] + [waypoint.x for waypoint in waypoints]
    waypoint_y_values = [true_y_values[0]] + [waypoint.y for waypoint in waypoints]
    plt.figure(figsize=(8, 8))
    plt.plot(
        true_x_values,
        true_y_values,
        label="True trajectory",
    )
    plt.scatter(
        measured_x_values,
        measured_y_values,
        label="Measured positions",
        marker=".",
        s=12,
    )
    plt.plot(
        waypoint_x_values,
        waypoint_y_values,
        label="Waypoint path",
        marker="x",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True vs. Measured trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def plot_true_measured_estimated_trajectory(trajectory, waypoints):
    true_x_values = [sample["x"] for sample in trajectory]
    true_y_values = [sample["y"] for sample in trajectory]
    measured_x_values = [sample["x_meas"] for sample in trajectory]
    measured_y_values = [sample["y_meas"] for sample in trajectory]
    estimated_x_values = [sample["x_est"] for sample in trajectory]
    estimated_y_values = [sample["y_est"] for sample in trajectory]
    waypoint_x_values = [true_x_values[0]] + [waypoint.x for waypoint in waypoints]
    waypoint_y_values = [true_y_values[0]] + [waypoint.y for waypoint in waypoints]
    plt.figure(figsize=(8, 8))
    plt.plot(true_x_values, true_y_values, label="True trajectory")
    plt.scatter(
        measured_x_values,
        measured_y_values,
        label="Measured positions",
        marker=".",
        s=12,
    )
    plt.plot(
        estimated_x_values,
        estimated_y_values,
        label="Estimated trajectory",
        linestyle="--",
    )
    plt.plot(
        waypoint_x_values,
        waypoint_y_values,
        label="Waypoint path",
        marker="x",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True vs. Measured vs. Estimated trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_all(trajectory, waypoints):
    plot_trajectory(trajectory, waypoints)
    plot_speed(trajectory)
    plot_acceleration(trajectory)
    plot_waypoint_index(trajectory)
    plot_distance_to_waypoint(trajectory)
    plot_true_vs_measured_trajectory(trajectory, waypoints)
    plot_true_measured_estimated_trajectory(trajectory, waypoints)