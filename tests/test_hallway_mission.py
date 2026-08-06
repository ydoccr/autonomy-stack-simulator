import math

from autonomy_sim.mission.config import load_hallway_mission_config
from autonomy_sim.mission.run_hallway_mission import plan_hallway_mission
from autonomy_sim.mission.run_hallway_mission import DEFAULT_MISSION_CONFIG


def test_hallway_mission_path_stays_in_free_cells():
    mission_config = load_hallway_mission_config(DEFAULT_MISSION_CONFIG)
    environment, costmap, grid_path, waypoints = plan_hallway_mission(mission_config)

    assert grid_path[0] == (0, 0)
    assert grid_path[-1] == (100, 100)
    zone_costmap = environment.to_zone_costmap()
    assert all(zone_costmap[cell] == 0.0 for cell in grid_path)
    assert waypoints[0].x == 0.0
    assert waypoints[0].y == 0.0
    assert waypoints[-1].x == 10.0
    assert waypoints[-1].y == 10.0
    assert len(grid_path) < 30

    for start, end in zip(grid_path[:-1], grid_path[1:]):
        row_change = end[0] - start[0]
        column_change = end[1] - start[1]
        assert math.hypot(row_change, column_change) <= 10
