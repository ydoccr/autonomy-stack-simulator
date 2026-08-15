import heapq
import math

import numpy as np

from autonomy_sim.environments.grid_geometry import cells_intersected_by_segment

GridCell = tuple[int, int]
GridDirection = tuple[int, int]
SearchState = tuple[GridCell, GridDirection]


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
    start_state = (start, start_direction)
    neighbor_steps = _neighbor_steps(allow_diagonal, max_distance)
    frontier = []
    push_count = 0
    start_priority = _heuristic(start, goal, fuel_rate)
    heapq.heappush(
        frontier,
        (start_priority, push_count, 0.0, start_state),
    )

    came_from = {}
    g_score = {start_state: 0.0}
    cell_states = {start: {start_direction: 0.0}}
    neighbor_cache = {}
    turn_cost_cache = {}

    while frontier:
        _, _, queued_cost, current_state = heapq.heappop(frontier)
        if current_state not in g_score or queued_cost > g_score[current_state]:
            continue
        current, incoming_direction = current_state
        if current == goal:
            return _reconstruct_path(came_from, current_state)

        if current not in neighbor_cache:
            neighbor_cache[current] = _neighbors(
                costmap,
                current,
                allow_diagonal,
                max_distance,
                neighbor_steps,
            )
        neighbors = neighbor_cache[current]
        for neighbor, direction, distance, environmental_cost in neighbors:
            # Same cost split as the NASA planner: fuel + environment.
            travel_cost = distance * (fuel_rate + environmental_cost)
            turn = (incoming_direction, direction)
            if turn not in turn_cost_cache:
                turn_cost_cache[turn] = _turn_cost(
                    incoming_direction,
                    direction,
                    nominal_speed,
                    turn_cost_weight,
                )
            turn_cost = turn_cost_cache[turn]
            edge_cost = travel_cost + waypoint_cost + turn_cost
            tentative_cost = g_score[current_state] + edge_cost
            if tentative_cost > max_cost:
                continue

            state_direction = direction
            if turn_cost_weight == 0.0:
                state_direction = start_direction
            neighbor_state = (neighbor, state_direction)
            if tentative_cost >= g_score.get(neighbor_state, math.inf):
                continue

            arrivals = cell_states.setdefault(neighbor, {})
            if _arrival_is_dominated(
                arrivals,
                state_direction,
                tentative_cost,
                turn_cost_cache,
                nominal_speed,
                turn_cost_weight,
            ):
                continue

            dominated_directions = _dominated_arrivals(
                arrivals,
                state_direction,
                tentative_cost,
                turn_cost_cache,
                nominal_speed,
                turn_cost_weight,
            )
            for dominated_direction in dominated_directions:
                arrivals.pop(dominated_direction)
                g_score.pop((neighbor, dominated_direction), None)

            came_from[neighbor_state] = current_state
            g_score[neighbor_state] = tentative_cost
            arrivals[state_direction] = tentative_cost
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
                    neighbor_state,
                ),
            )

    return []


def _neighbors(
    costmap,
    cell,
    allow_diagonal,
    max_distance,
    neighbor_steps=None,
):
    neighbors = []
    height, width = costmap.shape
    row, column = cell

    if neighbor_steps is None:
        neighbor_steps = _neighbor_steps(allow_diagonal, max_distance)

    for (
        row_change,
        column_change,
        distance,
        direction,
        crossed_offsets,
    ) in neighbor_steps:
        neighbor_row = row + row_change
        neighbor_column = column + column_change
        inside_grid = 0 <= neighbor_row < height and 0 <= neighbor_column < width
        if not inside_grid:
            continue

        crossed_cells = [
            (row + row_offset, column + column_offset)
            for row_offset, column_offset in crossed_offsets
        ]
        crossed_costs = [costmap[crossed_cell] for crossed_cell in crossed_cells]
        if not np.all(np.isfinite(crossed_costs)):
            continue

        environmental_cost = float(np.mean(crossed_costs))
        neighbors.append(
            (
                (neighbor_row, neighbor_column),
                direction,
                distance,
                environmental_cost,
            )
        )

    return neighbors


def _neighbor_steps(allow_diagonal, max_distance):
    steps = []
    for row_change in range(-max_distance, max_distance + 1):
        for column_change in range(-max_distance, max_distance + 1):
            if row_change == 0 and column_change == 0:
                continue
            if not allow_diagonal and row_change != 0 and column_change != 0:
                continue

            distance = math.hypot(row_change, column_change)
            if distance > max_distance:
                continue

            direction_divisor = math.gcd(abs(row_change), abs(column_change))
            direction = (
                row_change // direction_divisor,
                column_change // direction_divisor,
            )
            crossed_offsets = _cells_along_edge(
                (0, 0),
                (row_change, column_change),
            )
            steps.append(
                (
                    row_change,
                    column_change,
                    distance,
                    direction,
                    crossed_offsets,
                )
            )

    return steps


def _cells_along_edge(start, end):
    return [cell for cell in cells_intersected_by_segment(start, end) if cell != start]


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


def _arrival_is_dominated(
    arrivals,
    new_direction,
    new_cost,
    turn_cost_cache,
    nominal_speed,
    turn_cost_weight,
):
    for existing_direction, existing_cost in arrivals.items():
        heading_change = _cached_turn_cost(
            existing_direction,
            new_direction,
            turn_cost_cache,
            nominal_speed,
            turn_cost_weight,
        )
        if existing_cost + heading_change <= new_cost:
            return True
    return False


def _dominated_arrivals(
    arrivals,
    new_direction,
    new_cost,
    turn_cost_cache,
    nominal_speed,
    turn_cost_weight,
):
    dominated = []
    for existing_direction, existing_cost in arrivals.items():
        heading_change = _cached_turn_cost(
            new_direction,
            existing_direction,
            turn_cost_cache,
            nominal_speed,
            turn_cost_weight,
        )
        if new_cost + heading_change <= existing_cost:
            dominated.append(existing_direction)
    return dominated


def _cached_turn_cost(
    incoming_direction,
    outgoing_direction,
    turn_cost_cache,
    nominal_speed,
    turn_cost_weight,
):
    turn = (incoming_direction, outgoing_direction)
    if turn not in turn_cost_cache:
        turn_cost_cache[turn] = _turn_cost(
            incoming_direction,
            outgoing_direction,
            nominal_speed,
            turn_cost_weight,
        )
    return turn_cost_cache[turn]


def _heuristic(cell, goal, fuel_rate):
    row_distance = goal[0] - cell[0]
    column_distance = goal[1] - cell[1]
    return math.hypot(row_distance, column_distance) * fuel_rate


def _reconstruct_path(came_from, goal_state):
    states = [goal_state]
    current_state = goal_state
    while current_state in came_from:
        current_state = came_from[current_state]
        states.append(current_state)
    states.reverse()
    return [cell for cell, _ in states]


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
