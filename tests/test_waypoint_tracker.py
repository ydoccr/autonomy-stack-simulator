from autonomy_sim.core.types import VehicleState, Waypoint
from autonomy_sim.guidance.waypoint_tracker import WaypointTracker


def test_tracker_starts_at_first_waypoint():
    waypoints = [
        Waypoint(x=5.0, y=0.0),
        Waypoint(x=10.0, y=0.0),
    ]
    tracker = WaypointTracker(waypoints, waypoint_threshold=0.5)
    current = tracker.current_waypoint()
    assert tracker.current_index == 0
    assert current.x == 5.0
    assert current.y == 0.0
    assert tracker.complete is False


def test_tracker_advances_when_close_to_waypoint():
    waypoints = [
        Waypoint(x=5.0, y=0.0),
        Waypoint(x=10.0, y=0.0),
    ]
    tracker = WaypointTracker(waypoints, waypoint_threshold=0.5)
    state = VehicleState(x=5.0, y=0.0, vx=0.0, vy=0.0)
    tracker.update(state)
    assert tracker.current_index == 1
    assert tracker.complete is False


def test_tracker_does_not_advance_when_far_from_waypoint():
    waypoints = [
        Waypoint(x=5.0, y=0.0),
        Waypoint(x=10.0, y=0.0),
    ]
    tracker = WaypointTracker(waypoints, waypoint_threshold=0.5)
    state = VehicleState(x=0.0, y=0.0, vx=0.0, vy=0.0)
    tracker.update(state)
    assert tracker.current_index == 0
    assert tracker.complete is False


def test_tracker_completes_at_final_waypoint():
    waypoints = [
        Waypoint(x=5.0, y=0.0),
    ]
    tracker = WaypointTracker(waypoints, waypoint_threshold=0.5)
    state = VehicleState(x=5.0, y=0.0, vx=0.0, vy=0.0)
    tracker.update(state)
    assert tracker.complete is True


def test_distance_to_current_waypoint():
    waypoints = [
        Waypoint(x=3.0, y=4.0),
    ]
    tracker = WaypointTracker(waypoints, waypoint_threshold=0.5)
    state = VehicleState(x=0.0, y=0.0, vx=6.0, vy=7.0)
    distance = tracker.distance_to_current_waypoint(state)
    assert distance == 5.0
