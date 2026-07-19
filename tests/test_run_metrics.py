import numpy as np

from autonomy_sim.core.types import Waypoint
from autonomy_sim.metrics.metrics_run import RunMetrics


def create_trajectory():
    return [
        {
            "time": 0.0,
            "x": 0.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "x_meas": 1.0,
            "y_meas": 0.0,
            "vx_meas": 0.0,
            "vy_meas": 0.0,
            "x_est": 0.0,
            "y_est": 0.0,
            "vx_est": 0.0,
            "vy_est": 0.0,
            "ax_cmd": 0.0,
            "ay_cmd": 0.0,
            "current_waypoint_index": 0,
            "distance_to_waypoint": 2.0,
        },
        {
            "time": 1.0,
            "x": 1.0,
            "y": 0.0,
            "vx": 1.0,
            "vy": 0.0,
            "x_meas": 2.0,
            "y_meas": 0.0,
            "vx_meas": 1.1,
            "vy_meas": 0.0,
            "x_est": 0.5,
            "y_est": 0.0,
            "vx_est": 0.9,
            "vy_est": 0.0,
            "ax_cmd": 1.0,
            "ay_cmd": 0.0,
            "current_waypoint_index": 0,
            "distance_to_waypoint": 1.0,
        },
        {
            "time": 2.0,
            "x": 2.0,
            "y": 0.0,
            "vx": 1.0,
            "vy": 0.0,
            "x_meas": 2.0,
            "y_meas": 0.0,
            "vx_meas": 1.0,
            "vy_meas": 0.0,
            "x_est": 2.0,
            "y_est": 0.0,
            "vx_est": 1.0,
            "vy_est": 0.0,
            "ax_cmd": 1.0,
            "ay_cmd": 0.0,
            "current_waypoint_index": 0,
            "distance_to_waypoint": 0.0,
        },
    ]


def test_run_metrics_calculates_mission_performance():
    trajectory = create_trajectory()
    metrics = RunMetrics(
        trajectory,
        [Waypoint(x=2.0, y=0.0)],
        path_complete=True,
        goal_tolerance=0.2,
    ).calculate()

    assert metrics["onboard_completion"] is True
    assert metrics["true_mission_success"] is True
    assert metrics["termination_state"] == "goal_reached"
    assert metrics["final_true_distance"] == 0.0
    assert metrics["actual_path_length"] == 2.0
    assert metrics["planned_path_length"] == 2.0
    assert metrics["control_effort"] == 2.0
    assert metrics["measurement_dropout_fraction"] == 0.0
    assert metrics["rmse_measurement_error"] == np.sqrt(2.0 / 3.0)
    assert metrics["rmse_estimation_error"] == np.sqrt(0.25 / 3.0)


def test_run_metrics_reports_time_limit_when_path_is_incomplete():
    metrics = RunMetrics(
        create_trajectory(),
        [Waypoint(x=3.0, y=0.0)],
        path_complete=False,
        goal_tolerance=0.2,
    ).calculate()

    assert metrics["onboard_completion"] is False
    assert metrics["true_mission_success"] is False
    assert metrics["termination_state"] == "time_limit"


def test_run_metrics_distinguishes_false_onboard_completion():
    metrics = RunMetrics(
        create_trajectory(),
        [Waypoint(x=3.0, y=0.0)],
        path_complete=True,
        goal_tolerance=0.2,
    ).calculate()

    assert metrics["onboard_completion"] is True
    assert metrics["true_mission_success"] is False
    assert metrics["termination_state"] == "false_completion"


def test_run_metrics_ignores_dropped_measurements():
    trajectory = create_trajectory()
    trajectory[1]["measurement_available"] = False
    trajectory[1]["x_meas"] = np.nan
    trajectory[1]["y_meas"] = np.nan

    metrics = RunMetrics(
        trajectory,
        [Waypoint(x=2.0, y=0.0)],
        path_complete=True,
        goal_tolerance=0.2,
    ).calculate()

    assert np.isclose(metrics["measurement_dropout_fraction"], 1.0 / 3.0)
    assert metrics["rmse_measurement_error"] == np.sqrt(0.5)
