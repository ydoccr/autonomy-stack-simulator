from autonomy_sim.core.types import SensorData, VehicleState
from autonomy_sim.sensors.delay_sensor import DelaySensor
from autonomy_sim.sensors.gaussian_sensor import GaussianSensor
from autonomy_sim.sensors.offset_sensor import OffsetSensor
from autonomy_sim.sensors.spotty_sensor import SpottySensor


class CombinedErrorSensor:
    def __init__(
        self,
        gaussian_sensor: GaussianSensor | None = None,
        offset_sensor: OffsetSensor | None = None,
        delay_sensor: DelaySensor | None = None,
        spotty_sensor: SpottySensor | None = None,
    ):
        self.gaussian_sensor = gaussian_sensor
        self.offset_sensor = offset_sensor
        self.delay_sensor = delay_sensor
        self.spotty_sensor = spotty_sensor

    def sense(self, true_state: VehicleState) -> SensorData | None:
        sensed_state = true_state

        if self.delay_sensor is not None:
            delayed_data = self.delay_sensor.sense(sensed_state)
            sensed_state = VehicleState.from_array(delayed_data.as_array())

        if self.offset_sensor is not None:
            offset_data = self.offset_sensor.sense(sensed_state)
            sensed_state = VehicleState.from_array(offset_data.as_array())

        if self.gaussian_sensor is not None:
            gaussian_data = self.gaussian_sensor.sense(sensed_state)
            sensed_state = VehicleState.from_array(gaussian_data.as_array())

        if self.spotty_sensor is not None:
            return self.spotty_sensor.sense(sensed_state)

        return SensorData.from_array(sensed_state.as_array())

    def reset(self):
        if self.delay_sensor is not None:
            self.delay_sensor.reset()
