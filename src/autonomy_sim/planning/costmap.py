import numpy as np


def create_empty_costmap(
    width: int,
    height: int,
    default_cost: float = 0.0,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not np.isfinite(default_cost) or default_cost < 0.0:
        raise ValueError("default_cost must be finite and non-negative")

    return np.full((height, width), default_cost, dtype=float)


def add_rectangular_cost(
    costmap: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    cost: float,
) -> None:
    _validate_costmap(costmap)
    _validate_rectangle(costmap, row_min, row_max, col_min, col_max)
    if cost < 0.0 or np.isnan(cost):
        raise ValueError("cost must be non-negative or np.inf")

    costmap[row_min:row_max, col_min:col_max] = cost


def add_rectangular_obstacle(
    costmap: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> None:
    add_rectangular_cost(
        costmap,
        row_min,
        row_max,
        col_min,
        col_max,
        np.inf,
    )


def _validate_costmap(costmap: np.ndarray) -> None:
    if not isinstance(costmap, np.ndarray) or costmap.ndim != 2:
        raise ValueError("costmap must be a two-dimensional NumPy array")


def _validate_rectangle(
    costmap: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> None:
    height, width = costmap.shape
    valid_rows = 0 <= row_min < row_max <= height
    valid_columns = 0 <= col_min < col_max <= width
    if not valid_rows or not valid_columns:
        raise ValueError("rectangle must be non-empty and inside the costmap")
