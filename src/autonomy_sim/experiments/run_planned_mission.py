import matplotlib.pyplot as plt
import numpy as np

from autonomy_sim.core.types import VehicleState, Waypoint
from autonomy_sim.main import run_simulation
from autonomy_sim.metrics.metrics_run import RunMetrics
from autonomy_sim.planning.astar import astar
from autonomy_sim.planning.costmap import (
    add_rectangular_cost,
    add_rectangular_obstacle,
    create_empty_costmap,
)
from autonomy_sim.planning.path_utils import grid_path_to_waypoints
from autonomy_sim.visualization.plot_run import (
    add_environment_legend,
    plot_environment_overlay,
)


def create_scenario():
    costmap = create_empty_costmap(width=20, height=16)

    add_rectangular_obstacle(costmap, 2, 12, 8, 10)
    add_rectangular_obstacle(costmap, 11, 13, 3, 9)
    add_rectangular_cost(costmap, 2, 7, 12, 18, cost=8.0)

    start = (1, 1)
    goal = (14, 18)
    return costmap, start, goal


def calculate_metrics(
    trajectory,
    goal_x,
    goal_y,
    success_threshold=0.5,
):
    final_sample = trajectory[-1]
    final_distance = float(
        np.hypot(
            final_sample["x"] - goal_x,
            final_sample["y"] - goal_y,
        )
    )
    goal = Waypoint(x=goal_x, y=goal_y)
    path_complete = final_distance < success_threshold
    return RunMetrics(trajectory, [goal], path_complete).calculate()


def plot_planned_mission(
    costmap,
    grid_path,
    trajectory,
) -> None:
    path_x = [column for row, column in grid_path]
    path_y = [row for row, column in grid_path]
    trajectory_x = [sample["x"] for sample in trajectory]
    trajectory_y = [sample["y"] for sample in trajectory]

    plt.figure(figsize=(10, 8))
    plot_environment_overlay(costmap)
    plt.plot(path_x, path_y, "b--", label="A* path")
    plt.plot(trajectory_x, trajectory_y, "g-", label="True trajectory")
    plt.scatter(path_x[0], path_y[0], marker="o", label="Start")
    plt.scatter(path_x[-1], path_y[-1], marker="*", label="Goal")
    plt.xlabel("World x / grid column")
    plt.ylabel("World y / grid row")
    plt.title("Risk-aware planned mission")
    add_environment_legend()
    plt.axis("equal")
    plt.show()


def run_planned_mission(show_plots: bool = False):
    costmap, start, goal = create_scenario()
    grid_path = astar(
        costmap,
        start,
        goal,
        allow_diagonal=True,
        fuel_rate=1.0,
    )
    if not grid_path:
        raise RuntimeError("planned mission goal is unreachable")

    waypoints = grid_path_to_waypoints(grid_path, stride=2)
    initial_state = VehicleState(
        x=float(start[1]),
        y=float(start[0]),
        vx=0.0,
        vy=0.0,
    )
    trajectory = run_simulation(
        show_plots=False,
        show_metrics=False,
        initial_state=initial_state,
        waypoints=waypoints,
    )

    metrics = calculate_metrics(
        trajectory,
        goal_x=float(goal[1]),
        goal_y=float(goal[0]),
    )
    print("Planned mission metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    if show_plots:
        plot_planned_mission(costmap, grid_path, trajectory)
    return trajectory, grid_path, metrics


if __name__ == "__main__":
    run_planned_mission(show_plots=True)
