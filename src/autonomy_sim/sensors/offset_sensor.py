from autonomy_sim.core.types import SensorData, VehicleState


class OffsetSensor:
    def __init__(
        self,
        x_offset=0.0,
        y_offset=0.0,
        vx_offset=0.0,
        vy_offset=0.0,
    ):
        self.x_offset = float(x_offset)
        self.y_offset = float(y_offset)
        self.vx_offset = float(vx_offset)
        self.vy_offset = float(vy_offset)

    def sense(self, true_state: VehicleState) -> SensorData:
        return SensorData(
            x_meas=true_state.x + self.x_offset,
            y_meas=true_state.y + self.y_offset,
            vx_meas=true_state.vx + self.vx_offset,
            vy_meas=true_state.vy + self.vy_offset,
        )
