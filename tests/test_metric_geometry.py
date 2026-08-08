import numpy as np

from autonomy_sim.core.types import Waypoint
from autonomy_sim.metrics.geometry import distances_to_polyline, planned_path_positions


def test_planned_path_starts_at_initial_position():
    path = planned_path_positions(
        [0.0, 1.0],
        [Waypoint(x=2.0, y=1.0), Waypoint(x=2.0, y=3.0)],
    )

    assert np.array_equal(path, [[0.0, 1.0], [2.0, 1.0], [2.0, 3.0]])


def test_distances_to_polyline_use_nearest_segment_or_endpoint():
    path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    positions = np.array(
        [
            [1.0, 1.0],
            [3.0, 1.0],
            [2.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    distances = distances_to_polyline(positions, path)

    assert np.array_equal(distances, [1.0, 1.0, 0.0, 1.0])
