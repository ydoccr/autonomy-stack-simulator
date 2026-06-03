# should track waypoints & logic, including current waypoint, threshold, iteration, relevant booleans, etc.
# warning for later: final waypoint maybe should have different threshold?

import numpy as np
from autonomy_sim.core.types import VehicleState, Waypoint


class WaypointTracker:
    def __init__(self, waypoints, waypoint_threshold: float):
        self.waypoints = waypoints
        self.waypoint_threshold = waypoint_threshold
        self.current_index = 0
        self.complete = False

    def current_waypoint(self) -> Waypoint:
        return self.waypoints[self.current_index]

    def update(self, state: VehicleState) -> None:
        if self.complete:
            return
        distance = self.distance_to_current_waypoint(state)
        if distance < self.waypoint_threshold:
            if self.current_index < len(self.waypoints) - 1:
                self.current_index += 1
        if self.is_complete(state):
            self.complete = True

    def is_complete(self, state: VehicleState) -> bool:
        at_final_waypoint = self.current_index == len(self.waypoints) - 1
        close_to_final_waypoint = (
            self.distance_to_current_waypoint(state) < self.waypoint_threshold
        )
        return at_final_waypoint and close_to_final_waypoint

    def distance_to_current_waypoint(self, state: VehicleState) -> float:
        waypoint = self.current_waypoint()
        return float(np.hypot(state.x - waypoint.x, state.y - waypoint.y))