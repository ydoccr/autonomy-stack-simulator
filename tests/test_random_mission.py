from autonomy_sim.main import DEFAULT_CONFIG, load_config
from autonomy_sim.mission.config import load_random_mission_config
from autonomy_sim.mission.run_random_mission import DEFAULT_MISSION_CONFIG
from autonomy_sim.metrics.metrics_run import RUN_METRIC_FIELDS
from autonomy_sim.mission.run_random_mission import run_random_mission


def test_random_mission_returns_planning_failure_result_when_unreachable():
    restricted_environment = {
        "free": 0.0,
        "occupied": 0.0,
        "disallowed": 0.0,
        "restricted": 1.0,
    }

    simulation_config = load_config(DEFAULT_CONFIG)
    mission_config = load_random_mission_config(DEFAULT_MISSION_CONFIG)
    mission_config.zone_probabilities = restricted_environment
    result, environment, costmap, grid_path, waypoints = run_random_mission(
        simulation_config,
        mission_config,
        random_seed=2,
        show_plots=False,
        show_metrics=False,
    )

    assert result.trajectory == []
    assert result.metrics["planning_success"] is False
    assert result.metrics["true_mission_success"] is False
    assert result.metrics["termination_state"] == "planning_failure"
    assert result.scenario["environment_seed"] == 2
    assert result.scenario["zone_probabilities"] == restricted_environment
    assert environment.zone_probabilities == restricted_environment
    assert costmap.shape == (100, 100)
    assert grid_path == []
    assert waypoints == []

    assert tuple(result.metrics) == RUN_METRIC_FIELDS
