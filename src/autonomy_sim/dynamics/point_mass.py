# takes state, controlinput, dt. returns next state.

# material limitations: max speed, max acceleration

import numpy as np
from autonomy_sim.core.types import VehicleState, ControlInput


class PointMassDynamics:
    def __init__(self, max_speed: float, max_accel: float):
        self.max_speed = max_speed
        self.max_acceleration = max_accel

    def step(self, state: VehicleState, control: ControlInput, dt: float) -> VehicleState:
        ax = control.ax
        ay = control.ay

        acceleration = np.hypot(ax, ay)
        if acceleration > self.max_acceleration:
            scale = self.max_acceleration / acceleration
            ax *= scale
            ay *= scale

        old_vx = state.vx
        old_vy = state.vy
        new_vx = old_vx + ax * dt
        new_vy = old_vy + ay * dt

        speed = np.hypot(new_vx, new_vy)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            new_vx *= scale
            new_vy *= scale

        avg_vx = 0.5 * (old_vx + new_vx)
        avg_vy = 0.5 * (old_vy + new_vy)
        new_x = state.x + avg_vx * dt
        new_y = state.y + avg_vy * dt

        return VehicleState(
            x=float(new_x), 
            y=float(new_y), 
            vx=float(new_vx), 
            vy=float(new_vy)
        )