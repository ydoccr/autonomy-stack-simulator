import numpy as np
import pytest

from autonomy_sim.environments.example_environments import (
    DISALLOWED_COST,
    DEFAULT_RANDOM_ZONE_PROBABILITIES,
    GridEnvironment,
    OCCUPIED_COST,
    RESTRICTED_COST,
    create_hallway_environment,
    create_random_environment,
)


def test_hallway_environment_contains_all_zone_types():
    environment = create_hallway_environment()
    costmap = environment.to_zone_costmap()

    assert np.any(costmap == OCCUPIED_COST)
    assert np.any(costmap == DISALLOWED_COST)
    assert np.any(costmap == RESTRICTED_COST)


def test_normal_planning_prohibits_disallowed_and_restricted_cells():
    environment = create_hallway_environment()
    planning_costmap = environment.to_costmap(proximity_sigma=0.0)

    assert np.all(np.isfinite(planning_costmap[environment.occupied]))
    assert np.all(np.isinf(planning_costmap[environment.disallowed]))
    assert np.all(np.isinf(planning_costmap[environment.restricted]))


def test_disallowed_cells_can_be_enabled_but_restricted_cells_cannot():
    environment = create_hallway_environment()
    planning_costmap = environment.to_costmap(
        proximity_sigma=0.0,
        allow_disallowed=True,
    )

    assert np.all(np.isfinite(planning_costmap[environment.disallowed]))
    assert np.all(np.isinf(planning_costmap[environment.restricted]))


def test_hallway_cells_are_free():
    costmap = create_hallway_environment().to_zone_costmap()

    assert np.all(costmap[0:21, 0:61] == 0.0)
    assert np.all(costmap[0:101, 40:61] == 0.0)
    assert np.all(costmap[80:101, 40:101] == 0.0)


def test_environment_classifies_world_positions_by_zone():
    environment = GridEnvironment(width=4, height=1, resolution=1.0)
    environment.set_zone("occupied", 0, 1, 1, 2)
    environment.set_zone("disallowed", 0, 1, 2, 3)
    environment.set_zone("restricted", 0, 1, 3, 4)

    assert environment.zone_at_position(0.5, 0.5) == "free"
    assert environment.zone_at_position(1.5, 0.5) == "occupied"
    assert environment.zone_at_position(2.5, 0.5) == "disallowed"
    assert environment.zone_at_position(3.5, 0.5) == "restricted"
    assert environment.zone_at_position(-0.1, 0.5) == "out_of_bounds"
    assert environment.zone_at_position(4.0, 0.5) == "out_of_bounds"


def test_hallway_environment_uses_point_one_resolution():
    environment = create_hallway_environment()

    assert environment.resolution == 0.1
    assert environment.to_costmap().shape == (101, 101)


def test_gaussian_proximity_cost_reaches_nearby_free_cells():
    environment = create_hallway_environment()
    zone_costmap = environment.to_zone_costmap()
    planning_costmap = environment.to_costmap()

    assert zone_costmap[50, 40] == 0.0
    assert planning_costmap[50, 40] > 0.0


def test_default_proximity_sigma_is_point_zero_six():
    environment = create_hallway_environment()

    default_costmap = environment.to_costmap()
    explicit_costmap = environment.to_costmap(proximity_sigma=0.06)

    assert np.array_equal(default_costmap, explicit_costmap)


def test_gaussian_uses_nearest_cell_beyond_environment_edges():
    environment = GridEnvironment(width=3, height=3, resolution=0.1)
    environment.set_zone("occupied", 0, 3, 0, 3)

    planning_costmap = environment.to_costmap(proximity_sigma=0.1)

    assert np.allclose(planning_costmap, 2.0 * OCCUPIED_COST)


def test_gaussian_world_scale_is_similar_at_different_resolutions():
    coarse = GridEnvironment(width=41, height=41, resolution=0.1)
    coarse.set_zone("occupied", 10, 31, 20, 31)

    fine = GridEnvironment(width=81, height=81, resolution=0.05)
    fine.set_zone("occupied", 20, 61, 40, 61)

    coarse_cost = coarse.to_costmap(proximity_sigma=0.3)[20, 15]
    fine_cost = fine.to_costmap(proximity_sigma=0.3)[40, 30]

    assert np.isclose(coarse_cost, fine_cost, rtol=0.2)


def test_random_environment_repeats_from_seed():
    first = create_random_environment(random_seed=12)
    second = create_random_environment(random_seed=12)

    assert np.array_equal(
        first.to_zone_costmap(),
        second.to_zone_costmap(),
    )


def test_random_environment_uses_one_by_one_world_blocks():
    environment = create_random_environment(random_seed=4)
    zone_costmap = environment.to_zone_costmap()
    cells_per_block = round(1.0 / environment.resolution)

    for block_row in range(10):
        for block_column in range(10):
            row_min = block_row * cells_per_block
            row_max = row_min + cells_per_block
            col_min = block_column * cells_per_block
            col_max = col_min + cells_per_block
            block = zone_costmap[row_min:row_max, col_min:col_max]
            assert np.all(block == block[0, 0])


def test_random_environment_contains_each_zone_type():
    environment = create_random_environment(random_seed=0)
    costmap = environment.to_zone_costmap()

    assert np.any(costmap == 0.0)
    assert np.any(costmap == OCCUPIED_COST)
    assert np.any(costmap == DISALLOWED_COST)
    assert np.any(costmap == RESTRICTED_COST)


def test_random_environment_uses_biased_default_probabilities():
    environment = create_random_environment(
        random_seed=3,
        width=100,
        height=100,
        resolution=1.0,
    )
    total_cells = environment.width * environment.height
    observed = {
        "occupied": np.mean(environment.occupied),
        "disallowed": np.mean(environment.disallowed),
        "restricted": np.mean(environment.restricted),
    }
    observed["free"] = 1.0 - sum(observed.values())

    assert total_cells == 10000
    for zone, probability in DEFAULT_RANDOM_ZONE_PROBABILITIES.items():
        assert np.isclose(observed[zone], probability, atol=0.02)


def test_random_environment_accepts_custom_zone_probabilities():
    probabilities = {
        "free": 1.0,
        "occupied": 0.0,
        "disallowed": 0.0,
        "restricted": 0.0,
    }

    environment = create_random_environment(
        random_seed=4,
        zone_probabilities=probabilities,
    )

    assert np.all(environment.to_zone_costmap() == 0.0)
    assert environment.zone_probabilities == probabilities


def test_random_environment_rejects_invalid_zone_probabilities():
    with pytest.raises(ValueError, match="sum to one"):
        create_random_environment(
            zone_probabilities={
                "free": 0.5,
                "occupied": 0.5,
                "disallowed": 0.5,
                "restricted": 0.5,
            }
        )


def test_environment_stores_max_cost():
    environment = create_random_environment(max_cost=1234.0)

    assert environment.max_cost == 1234.0


def test_clearance_measures_prohibited_zones_and_environment_boundary():
    environment = GridEnvironment(width=4, height=4, resolution=1.0)
    environment.set_zone("disallowed", 1, 2, 2, 3)

    assert environment.clearance_at_position(1.0, 1.5) == 1.0
    assert environment.clearance_at_position(2.5, 1.5) == 0.0
    assert environment.clearance_at_position(0.25, 3.0) == 0.25
    assert environment.clearance_at_position(-0.1, 1.0) == 0.0


def test_clearance_cache_updates_when_zones_change():
    environment = GridEnvironment(width=4, height=4, resolution=1.0)
    assert environment.clearance_at_position(2.5, 2.5) == 1.5

    environment.set_zone("restricted", 2, 3, 3, 4)

    assert environment.clearance_at_position(2.5, 2.5) == 0.5
