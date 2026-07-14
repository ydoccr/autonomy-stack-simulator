from autonomy_sim.core.types import VehicleState
from autonomy_sim.sensors.offset_sensor import OffsetSensor


def test_offset_sensor_adds_fixed_errors():
    sensor = OffsetSensor(
        x_offset=1.0,
        y_offset=-2.0,
        vx_offset=0.5,
        vy_offset=-0.25,
    )
    state = VehicleState(x=2.0, y=3.0, vx=4.0, vy=5.0)

    measurement = sensor.sense(state)

    assert measurement.x_meas == 3.0
    assert measurement.y_meas == 1.0
    assert measurement.vx_meas == 4.5
    assert measurement.vy_meas == 4.75
