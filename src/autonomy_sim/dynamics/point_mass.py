import numpy as np

from autonomy_sim.core.types import ControlInput, VehicleState


class PointMassDynamics:
    def __init__(self, max_speed: float, max_accel: float):
        self.max_speed = max_speed
        self.max_acceleration = max_accel

    def step(
        self,
        state: VehicleState,
        control: ControlInput,
        dt: float,
    ) -> VehicleState:
        next_state, _ = self.step_with_applied_control(state, control, dt)
        return next_state

    def step_with_applied_control(
        self,
        state: VehicleState,
        control: ControlInput,
        dt: float,
    ) -> tuple[VehicleState, ControlInput]:
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

        if dt == 0.0:
            applied_ax = ax
            applied_ay = ay
        else:
            applied_ax = (new_vx - old_vx) / dt
            applied_ay = (new_vy - old_vy) / dt

        average_vx = 0.5 * (old_vx + new_vx)
        average_vy = 0.5 * (old_vy + new_vy)
        new_x = state.x + average_vx * dt
        new_y = state.y + average_vy * dt

        next_state = VehicleState(
            x=float(new_x),
            y=float(new_y),
            vx=float(new_vx),
            vy=float(new_vy),
        )
        applied_control = ControlInput(
            ax=float(applied_ax),
            ay=float(applied_ay),
        )
        return next_state, applied_control
