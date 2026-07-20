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
    max_distance: int = 1,
    waypoint_cost: float = 0.0,
    turn_cost_weight: float = 0.0,
    nominal_speed: float = 1.0,
    zone_costmap: np.ndarray | None = None,
    max_cost: float = np.inf,
) -> list[GridCell]:
    _validate_inputs(
        costmap,
        start,
        goal,
        fuel_rate,
        max_distance,
        waypoint_cost,
        turn_cost_weight,
        nominal_speed,
        zone_costmap,
        max_cost,
    )
    if start == goal:
        return [start]

    start_direction = (0, 0)
    frontier = []
    push_count = 0
    start_priority = _heuristic(start, goal, fuel_rate)
    heapq.heappush(
        frontier,
        (start_priority, push_count, 0.0, start, start_direction),
    )

    came_from = {}
    g_score = {start: 0.0}

    while frontier:
        _, _, queued_cost, current, incoming_direction = heapq.heappop(frontier)
        if queued_cost > g_score[current]:
            continue
        if current == goal:
            return _reconstruct_path(came_from, current)

        neighbors = _neighbors(
            costmap,
            current,
            allow_diagonal,
            max_distance,
            zone_costmap,
        )
        for neighbor, direction, distance, environmental_cost in neighbors:
            # Same cost split as the NASA planner: fuel + environment.
            travel_cost = distance * (fuel_rate + environmental_cost)
            turn_cost = _turn_cost(
                incoming_direction,
                direction,
                nominal_speed,
                turn_cost_weight,
            )
            edge_cost = travel_cost + waypoint_cost + turn_cost
            tentative_cost = g_score[current] + edge_cost
            if tentative_cost > max_cost:
                continue

            if tentative_cost < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_cost
                priority = tentative_cost + _heuristic(
                    neighbor,
                    goal,
                    fuel_rate,
                )
                push_count += 1
                heapq.heappush(
                    frontier,
                    (
                        priority,
                        push_count,
                        tentative_cost,
                        neighbor,
                        direction,
                    ),
                )

    return []


def _neighbors(
    costmap,
    cell,
    allow_diagonal,
    max_distance,
    zone_costmap,
):
    neighbors = []
    height, width = costmap.shape
    row, column = cell

    for row_change in range(-max_distance, max_distance + 1):
        for column_change in range(-max_distance, max_distance + 1):
            if row_change == 0 and column_change == 0:
                continue
            if not allow_diagonal and row_change != 0 and column_change != 0:
                continue

            distance = math.hypot(row_change, column_change)
            if distance > max_distance:
                continue

            neighbor_row = row + row_change
            neighbor_column = column + column_change
            inside_grid = 0 <= neighbor_row < height and 0 <= neighbor_column < width
            if not inside_grid:
                continue

            crossed_cells = _cells_along_edge(
                cell,
                (neighbor_row, neighbor_column),
            )
            crossed_costs = [costmap[crossed_cell] for crossed_cell in crossed_cells]
            if not np.all(np.isfinite(crossed_costs)):
                continue

            if zone_costmap is not None and len(crossed_cells) > 1:
                crossed_zone_costs = [
                    zone_costmap[crossed_cell] for crossed_cell in crossed_cells
                ]
                if np.any(np.asarray(crossed_zone_costs) > 0.0):
                    continue

            environmental_cost = float(np.mean(crossed_costs))
            direction_divisor = math.gcd(abs(row_change), abs(column_change))
            direction = (
                row_change // direction_divisor,
                column_change // direction_divisor,
            )
            neighbors.append(
                (
                    (neighbor_row, neighbor_column),
                    direction,
                    distance,
                    environmental_cost,
                )
            )

    return neighbors


def _cells_along_edge(start, end):
    row_change = end[0] - start[0]
    column_change = end[1] - start[1]
    number_of_steps = max(abs(row_change), abs(column_change))
    cells = []

    for step in range(1, number_of_steps + 1):
        fraction = step / number_of_steps
        row = round(start[0] + fraction * row_change)
        column = round(start[1] + fraction * column_change)
        cell = (row, column)
        if not cells or cell != cells[-1]:
            cells.append(cell)

    return cells


def _turn_cost(
    incoming_direction,
    outgoing_direction,
    nominal_speed,
    turn_cost_weight,
):
    if incoming_direction == (0, 0):
        return 0.0

    incoming_length = math.hypot(*incoming_direction)
    outgoing_length = math.hypot(*outgoing_direction)
    dot_product = (
        incoming_direction[0] * outgoing_direction[0]
        + incoming_direction[1] * outgoing_direction[1]
    )
    cosine = dot_product / (incoming_length * outgoing_length)
    cosine = max(-1.0, min(1.0, cosine))
    turn_angle = math.acos(cosine)
    velocity_change = 2.0 * nominal_speed * math.sin(turn_angle / 2.0)
    return turn_cost_weight * velocity_change


def _heuristic(cell, goal, fuel_rate):
    row_distance = goal[0] - cell[0]
    column_distance = goal[1] - cell[1]
    return math.hypot(row_distance, column_distance) * fuel_rate


def _reconstruct_path(came_from, goal):
    path = [goal]
    current = goal
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _validate_inputs(
    costmap,
    start,
    goal,
    fuel_rate,
    max_distance,
    waypoint_cost,
    turn_cost_weight,
    nominal_speed,
    zone_costmap,
    max_cost,
):
    if not isinstance(costmap, np.ndarray) or costmap.ndim != 2:
        raise ValueError("costmap must be a two-dimensional NumPy array")
    if np.any(np.isnan(costmap)) or np.any(costmap < 0.0):
        raise ValueError("costmap values must be non-negative or np.inf")
    if fuel_rate <= 0.0:
        raise ValueError("fuel_rate must be positive")
    if max_distance < 1:
        raise ValueError("max_distance must be at least one")
    if waypoint_cost < 0.0 or turn_cost_weight < 0.0:
        raise ValueError("planning cost weights must be non-negative")
    if nominal_speed <= 0.0:
        raise ValueError("nominal_speed must be positive")
    if max_cost <= 0.0:
        raise ValueError("max_cost must be positive")
    if zone_costmap is not None and zone_costmap.shape != costmap.shape:
        raise ValueError("zone_costmap must match the costmap shape")

    height, width = costmap.shape
    for name, cell in (("start", start), ("goal", goal)):
        if len(cell) != 2:
            raise ValueError(f"{name} must contain a row and column")
        row, column = cell
        if not (0 <= row < height and 0 <= column < width):
            raise ValueError(f"{name} is outside the costmap")
        if not np.isfinite(costmap[row, column]):
            raise ValueError(f"{name} cannot be an obstacle")
