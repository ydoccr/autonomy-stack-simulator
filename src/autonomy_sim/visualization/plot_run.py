import matplotlib.pyplot as plt
from autonomy_sim.core.types import Waypoint

def plot_trajectory(trajectory, waypoints: list[Waypoint]) -> None:
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

