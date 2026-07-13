import matplotlib.pyplot as plt
import numpy as np

from autonomy_sim.core.types import VehicleState
from autonomy_sim.main import run_simulation
from autonomy_sim.planning.astar import astar
from autonomy_sim.planning.costmap import (
    add_rectangular_cost,
    add_rectangular_obstacle,
    create_empty_costmap,
)
from autonomy_sim.planning.path_utils import grid_path_to_waypoints


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

    position_errors = []
    acceleration_magnitudes = []
    for sample in trajectory:
        error = np.hypot(
            sample["x"] - sample["x_est"],
            sample["y"] - sample["y_est"],
        )
        position_errors.append(error)
        acceleration = np.hypot(sample["ax_cmd"], sample["ay_cmd"])
        acceleration_magnitudes.append(acceleration)

    if len(trajectory) > 1:
        dt = float(trajectory[1]["time"] - trajectory[0]["time"])
    else:
        dt = 0.0
    squared_errors = np.square(position_errors)
    control_effort = np.sum(acceleration_magnitudes) * dt

    return {
        "mission_success": int(final_distance < success_threshold),
        "completion_time": float(final_sample["time"]),
        "final_true_distance": final_distance,
        "mean_estimation_error": float(np.mean(position_errors)),
        "max_estimation_error": float(np.max(position_errors)),
        "rmse_estimation_error": float(np.sqrt(np.mean(squared_errors))),
        "control_effort": float(control_effort),
        "max_commanded_acceleration": float(
            np.max(acceleration_magnitudes)
        ),
        "number_of_steps": len(trajectory) - 1,
    }


def plot_planned_mission(
    costmap,
    grid_path,
    trajectory,
) -> None:
    display_costmap = costmap.copy()
    finite_costs = display_costmap[np.isfinite(display_costmap)]
    obstacle_display_cost = float(np.max(finite_costs) + 5.0)
    display_costmap[np.isinf(display_costmap)] = obstacle_display_cost

    path_x = [column for row, column in grid_path]
    path_y = [row for row, column in grid_path]
    trajectory_x = [sample["x"] for sample in trajectory]
    trajectory_y = [sample["y"] for sample in trajectory]

    plt.figure(figsize=(10, 8))
    plt.imshow(display_costmap, origin="lower", cmap="YlOrRd")
    plt.colorbar(label="Environmental cost")
    plt.plot(path_x, path_y, "b--", label="A* path")
    plt.plot(trajectory_x, trajectory_y, "g-", label="True trajectory")
    plt.scatter(path_x[0], path_y[0], marker="o", label="Start")
    plt.scatter(path_x[-1], path_y[-1], marker="*", label="Goal")
    plt.xlabel("World x / grid column")
    plt.ylabel("World y / grid row")
    plt.title("Risk-aware planned mission")
    plt.legend()
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
