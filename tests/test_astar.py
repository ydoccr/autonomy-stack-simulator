import numpy as np

from autonomy_sim.planning.astar import astar
from autonomy_sim.planning.costmap import (
    add_rectangular_cost,
    add_rectangular_obstacle,
    create_empty_costmap,
)


def test_astar_returns_start_and_goal():
    costmap = create_empty_costmap(width=5, height=5)

    path = astar(costmap, start=(0, 0), goal=(4, 4))

    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)


def test_astar_routes_around_obstacle():
    costmap = create_empty_costmap(width=5, height=5)
    add_rectangular_obstacle(costmap, 0, 4, 2, 3)

    path = astar(
        costmap,
        start=(0, 0),
        goal=(0, 4),
        allow_diagonal=False,
    )

    assert path
    assert all(np.isfinite(costmap[cell]) for cell in path)
    assert any(row == 4 for row, _ in path)


def test_astar_avoids_high_environmental_cost():
    costmap = create_empty_costmap(width=5, height=3)
    add_rectangular_cost(costmap, 1, 2, 1, 4, cost=100.0)

    path = astar(
        costmap,
        start=(1, 0),
        goal=(1, 4),
        allow_diagonal=False,
    )

    expensive_cells = {(1, 1), (1, 2), (1, 3)}
    assert expensive_cells.isdisjoint(path)


def test_astar_returns_empty_path_when_goal_is_unreachable():
    costmap = create_empty_costmap(width=3, height=3)
    add_rectangular_obstacle(costmap, 1, 2, 0, 3)

    path = astar(
        costmap,
        start=(0, 0),
        goal=(2, 2),
        allow_diagonal=False,
    )

    assert path == []


def test_astar_rejects_obstacle_endpoint():
    costmap = create_empty_costmap(width=3, height=3)
    add_rectangular_obstacle(costmap, 2, 3, 2, 3)

    with np.testing.assert_raises(ValueError):
        astar(costmap, start=(0, 0), goal=(2, 2))


def test_max_distance_creates_fewer_straight_path_segments():
    costmap = create_empty_costmap(width=11, height=1)

    path = astar(
        costmap,
        start=(0, 0),
        goal=(0, 10),
        allow_diagonal=False,
        max_distance=5,
        waypoint_cost=0.1,
    )

    assert len(path) == 3
    assert path == [(0, 0), (0, 5), (0, 10)]


def test_long_edge_cannot_skip_an_obstacle():
    costmap = create_empty_costmap(width=5, height=1)
    costmap[0, 2] = np.inf

    path = astar(
        costmap,
        start=(0, 0),
        goal=(0, 4),
        allow_diagonal=False,
        max_distance=4,
    )

    assert path == []


def test_long_edge_cannot_cut_through_a_costly_zone():
    costmap = create_empty_costmap(width=5, height=1)
    zone_costmap = costmap.copy()
    costmap[0, 2] = 10.0
    zone_costmap[0, 2] = 10.0

    path = astar(
        costmap,
        start=(0, 0),
        goal=(0, 4),
        allow_diagonal=False,
        max_distance=4,
        waypoint_cost=0.1,
        zone_costmap=zone_costmap,
    )

    assert path[0] == (0, 0)
    assert path[-1] == (0, 4)
    assert (0, 2) in path
    assert path != [(0, 0), (0, 4)]


def test_astar_returns_empty_path_above_max_cost():
    costmap = create_empty_costmap(width=5, height=1)

    path = astar(
        costmap,
        start=(0, 0),
        goal=(0, 4),
        allow_diagonal=False,
        max_cost=3.0,
    )

    assert path == []
