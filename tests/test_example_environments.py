import numpy as np

from autonomy_sim.environments.example_environments import (
    DISALLOWED_COST,
    OCCUPIED_COST,
    RESTRICTED_COST,
    create_hallway_environment,
)


def test_hallway_environment_contains_all_zone_types():
    environment = create_hallway_environment()
    costmap = environment.to_costmap()

    assert np.any(costmap == OCCUPIED_COST)
    assert np.any(costmap == DISALLOWED_COST)
    assert np.any(costmap == RESTRICTED_COST)


def test_hallway_cells_are_free():
    costmap = create_hallway_environment().to_costmap()

    assert np.all(costmap[0:21, 0:61] == 0.0)
    assert np.all(costmap[0:101, 40:61] == 0.0)
    assert np.all(costmap[80:101, 40:101] == 0.0)


def test_hallway_environment_uses_point_one_resolution():
    environment = create_hallway_environment()

    assert environment.resolution == 0.1
    assert environment.to_costmap().shape == (101, 101)
