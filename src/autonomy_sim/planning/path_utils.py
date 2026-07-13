from autonomy_sim.core.types import Waypoint


def grid_path_to_waypoints(
    path: list[tuple[int, int]],
    resolution: float = 1.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    stride: int = 1,
) -> list[Waypoint]:
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not path:
        return []

    selected_cells = path[::stride]
    if selected_cells[-1] != path[-1]:
        selected_cells.append(path[-1])

    waypoints = []
    for row, column in selected_cells:
        # Grid cells are (row, column), while world points are (x, y).
        world_x = origin_x + column * resolution
        world_y = origin_y + row * resolution
        waypoints.append(Waypoint(x=world_x, y=world_y))
    return waypoints
