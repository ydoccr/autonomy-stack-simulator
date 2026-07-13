import numpy as np

from autonomy_sim.core.types import Waypoint
from autonomy_sim.planning.path_utils import grid_path_to_waypoints


def test_grid_path_uses_column_for_x_and_row_for_y():
    waypoints = grid_path_to_waypoints(
        [(2, 3)],
        resolution=0.5,
        origin_x=10.0,
        origin_y=-4.0,
    )

    assert waypoints == [Waypoint(x=11.5, y=-3.0)]


def test_stride_preserves_final_goal():
    path = [(0, 0), (0, 1), (0, 2), (0, 3)]

    waypoints = grid_path_to_waypoints(path, stride=2)

    assert waypoints == [
        Waypoint(x=0.0, y=0.0),
        Waypoint(x=2.0, y=0.0),
        Waypoint(x=3.0, y=0.0),
    ]


def test_empty_path_returns_no_waypoints():
    assert grid_path_to_waypoints([]) == []


def test_invalid_stride_is_rejected():
    with np.testing.assert_raises(ValueError):
        grid_path_to_waypoints([(0, 0)], stride=0)
