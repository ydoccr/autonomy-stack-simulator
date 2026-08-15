import numpy as np
import pytest

from autonomy_sim.environments.grid_geometry import cells_intersected_by_segment
from autonomy_sim.planning.astar import _neighbors


@pytest.mark.parametrize(
    ("start", "end", "expected_cells"),
    [
        ((2, 1), (2, 4), {(2, 1), (2, 2), (2, 3), (2, 4)}),
        ((1, 2), (4, 2), {(1, 2), (2, 2), (3, 2), (4, 2)}),
        ((0, 0), (1, 1), {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (
            (0, 0),
            (3, 3),
            {
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 2),
                (3, 3),
            },
        ),
        (
            (0, 0),
            (2, 5),
            {
                (0, 0),
                (0, 1),
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 4),
                (2, 5),
            },
        ),
        (
            (0, 0),
            (5, 2),
            {
                (0, 0),
                (1, 0),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (4, 2),
                (5, 2),
            },
        ),
    ],
)
def test_segment_supercover_includes_every_touched_cell(
    start,
    end,
    expected_cells,
):
    cells = cells_intersected_by_segment(start, end)

    assert set(cells) == expected_cells


@pytest.mark.parametrize(
    ("shape", "start", "end", "blocked_cell", "max_distance"),
    [
        ((1, 5), (0, 0), (0, 4), (0, 2), 4),
        ((5, 1), (0, 0), (4, 0), (2, 0), 4),
        ((2, 2), (0, 0), (1, 1), (0, 1), 2),
        ((4, 4), (0, 0), (3, 3), (1, 0), 5),
        ((3, 6), (0, 0), (2, 5), (1, 1), 6),
        ((6, 3), (0, 0), (5, 2), (1, 1), 6),
    ],
)
def test_planner_transition_rejects_any_intersected_obstacle(
    shape,
    start,
    end,
    blocked_cell,
    max_distance,
):
    costmap = np.zeros(shape, dtype=float)
    costmap[blocked_cell] = np.inf

    neighbors = _neighbors(
        costmap,
        start,
        allow_diagonal=True,
        max_distance=max_distance,
    )

    neighbor_cells = {neighbor for neighbor, _, _, _ in neighbors}
    assert end not in neighbor_cells
