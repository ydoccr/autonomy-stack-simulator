# takes VehicleState, Waypoint, returns ControlInput(ax, ay)
# Logic: Accelerate in direction of waypoint, with magnitude proportional to distance, 
# but capped at max_accel. Damped by current velocity to prevent overshoot.

import numpy as np
from autonomy_sim.core.types import VehicleState, ControlInput, Waypoint


class PointMassAccController:
    def __init__(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd

    def compute_control(self, state: VehicleState, waypoint: Waypoint) -> ControlInput:
        pos_error = np.array([waypoint.x - state.x, waypoint.y - state.y])
        p = self.kp * pos_error
        d = self.kd * -state.velocity()
        control_input = p + d
        return ControlInput(ax=control_input[0], ay=control_input[1])