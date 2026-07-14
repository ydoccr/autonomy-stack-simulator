from autonomy_sim.core.types import SensorData, VehicleState


class DelaySensor:
    def __init__(self, delay_steps=1):
        if not isinstance(delay_steps, int) or delay_steps < 0:
            raise ValueError("delay_steps must be a non-negative integer")
        self.delay_steps = delay_steps
        self._history = []

    def sense(self, true_state: VehicleState) -> SensorData:
        measurement = SensorData(
            x_meas=true_state.x,
            y_meas=true_state.y,
            vx_meas=true_state.vx,
            vy_meas=true_state.vy,
        )
        self._history.append(measurement)

        if len(self._history) <= self.delay_steps:
            return self._history[0]
        return self._history.pop(0)

    def reset(self):
        self._history = []
