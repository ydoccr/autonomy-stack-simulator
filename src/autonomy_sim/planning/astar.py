import heapq
import math

import numpy as np

GridCell = tuple[int, int]


def astar(
    costmap: np.ndarray,
    start: GridCell,
    goal: GridCell,
    allow_diagonal: bool = True,
    fuel_rate: float = 1.0,
) -> list[GridCell]:
    _validate_inputs(costmap, start, goal, fuel_rate)
    if start == goal:
        return [start]

    frontier: list[tuple[float, float, GridCell]] = []
    start_priority = _heuristic(start, goal, fuel_rate)
    heapq.heappush(frontier, (start_priority, 0.0, start))

    came_from: dict[GridCell, GridCell] = {}
    g_score: dict[GridCell, float] = {start: 0.0}

    while frontier:
        _, queued_cost, current = heapq.heappop(frontier)
        if queued_cost > g_score[current]:
            continue
        if current == goal:
            return _reconstruct_path(came_from, goal)

        for neighbor in _neighbors(costmap, current, allow_diagonal):
            row_change = neighbor[0] - current[0]
            col_change = neighbor[1] - current[1]
            distance = math.hypot(row_change, col_change)
            environmental_cost = float(costmap[neighbor])
            # Same cost split as the NASA planner: fuel + environment.
            edge_cost = distance * (fuel_rate + environmental_cost)
            tentative_cost = g_score[current] + edge_cost

            if tentative_cost < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_cost
                priority = tentative_cost + _heuristic(
                    neighbor, goal, fuel_rate
                )
                heapq.heappush(
                    frontier,
                    (priority, tentative_cost, neighbor),
                )

    return []


def _neighbors(
    costmap: np.ndarray,
    cell: GridCell,
    allow_diagonal: bool,
) -> list[GridCell]:
    row, column = cell
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    neighbors = []
    height, width = costmap.shape
    for row_offset, column_offset in offsets:
        neighbor_row = row + row_offset
        neighbor_column = column + column_offset
        inside_grid = (
            0 <= neighbor_row < height and 0 <= neighbor_column < width
        )
        if not inside_grid:
            continue
        if not np.isfinite(costmap[neighbor_row, neighbor_column]):
            continue
        neighbors.append((neighbor_row, neighbor_column))
    return neighbors


def _heuristic(cell: GridCell, goal: GridCell, fuel_rate: float) -> float:
    row_distance = goal[0] - cell[0]
    column_distance = goal[1] - cell[1]
    return math.hypot(row_distance, column_distance) * fuel_rate


def _reconstruct_path(
    came_from: dict[GridCell, GridCell],
    goal: GridCell,
) -> list[GridCell]:
    path = [goal]
    current = goal
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _validate_inputs(
    costmap: np.ndarray,
    start: GridCell,
    goal: GridCell,
    fuel_rate: float,
) -> None:
    if not isinstance(costmap, np.ndarray) or costmap.ndim != 2:
        raise ValueError("costmap must be a two-dimensional NumPy array")
    if np.any(np.isnan(costmap)) or np.any(costmap < 0.0):
        raise ValueError("costmap values must be non-negative or np.inf")
    if fuel_rate <= 0.0:
        raise ValueError("fuel_rate must be positive")

    height, width = costmap.shape
    for name, cell in (("start", start), ("goal", goal)):
        if len(cell) != 2:
            raise ValueError(f"{name} must contain a row and column")
        row, column = cell
        if not (0 <= row < height and 0 <= column < width):
            raise ValueError(f"{name} is outside the costmap")
        if not np.isfinite(costmap[row, column]):
            raise ValueError(f"{name} cannot be an obstacle")
