import numpy as np

from autonomy_sim.core.types import ControlInput, VehicleState, Waypoint


class PointMassAccController:
    def __init__(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd

    def compute_control(
        self,
        state: VehicleState,
        waypoint: Waypoint,
    ) -> ControlInput:
        pos_error = np.array([waypoint.x - state.x, waypoint.y - state.y])
        p = self.kp * pos_error
        d = self.kd * -state.velocity()
        control_input = p + d
        return ControlInput(
            ax=float(control_input[0]),
            ay=float(control_input[1]),
        )
