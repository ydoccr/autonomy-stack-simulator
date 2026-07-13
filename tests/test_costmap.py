import numpy as np

from autonomy_sim.planning.costmap import (
    add_rectangular_cost,
    add_rectangular_obstacle,
    create_empty_costmap,
)


def test_create_empty_costmap_uses_height_then_width():
    costmap = create_empty_costmap(width=4, height=3, default_cost=2.0)

    assert costmap.shape == (3, 4)
    assert np.all(costmap == 2.0)


def test_add_rectangular_cost_updates_only_selected_region():
    costmap = create_empty_costmap(width=5, height=4)

    add_rectangular_cost(costmap, 1, 3, 2, 5, cost=8.0)

    assert np.all(costmap[1:3, 2:5] == 8.0)
    assert costmap[0, 0] == 0.0


def test_add_rectangular_obstacle_uses_infinite_cost():
    costmap = create_empty_costmap(width=5, height=5)

    add_rectangular_obstacle(costmap, 1, 4, 2, 3)

    assert np.all(np.isinf(costmap[1:4, 2:3]))


def test_rectangle_must_be_inside_costmap():
    costmap = create_empty_costmap(width=3, height=3)

    with np.testing.assert_raises(ValueError):
        add_rectangular_obstacle(costmap, 0, 4, 0, 1)
