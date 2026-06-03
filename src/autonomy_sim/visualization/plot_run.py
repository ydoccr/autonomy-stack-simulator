import matplotlib.pyplot as plt
from autonomy_sim.core.types import Waypoint

def plot_trajectory(trajectory, target: Waypoint) -> None:
    x_values = [sample["x"] for sample in trajectory]
    y_values = [sample["y"] for sample in trajectory]
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, label="Vehicle trajectory")
    plt.scatter(x_values[0], y_values[0], label="Start Position", marker="o")
    plt.scatter(x_values[-1], y_values[-1], label="Final Position", marker="o")
    plt.scatter(target.x, target.y, label="Target Waypoint", marker="x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Point-Mass Vehicle Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()