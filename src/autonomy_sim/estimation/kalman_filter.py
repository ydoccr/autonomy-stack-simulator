import numpy as np

from autonomy_sim.core.types import ControlInput, SensorData, VehicleState


class KalmanFilter:
    """Discrete linear Kalman filter for a point-mass vehicle.

    Args:
        dt: time step (s)
        process_var: scalar multiplier for process noise covariance Q
        meas_var: scalar multiplier for measurement noise covariance R
    """

    def __init__(
        self,
        dt: float,
        process_var: float = 1e-3,
        meas_var: float = 1e-2,
    ):
        self.dt = float(dt)

        # State transition (constant-velocity part) for state [x,y,vx,vy]
        self.F = np.array(
            [
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        # Control-input model from accelerations [ax, ay]
        half_dt_squared = 0.5 * self.dt**2
        self.B = np.array(
            [
                [half_dt_squared, 0],
                [0, half_dt_squared],
                [self.dt, 0],
                [0, self.dt],
            ],
            dtype=float,
        )

        # Full-state measurements (position+velocity)
        self.H = np.eye(4, dtype=float)

        # Covariances
        self.Q = process_var * np.eye(4, dtype=float)
        self.R = meas_var * np.eye(4, dtype=float)

        # State estimate and covariance
        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float)

    def reset(
        self,
        state: VehicleState | None = None,
        P: np.ndarray | None = None,
    ) -> None:
        """Reset the filter state and covariance."""
        if state is None:
            self.x = np.zeros(4, dtype=float)
        else:
            self.x = state.as_array().astype(float)

        if P is None:
            self.P = np.eye(4, dtype=float)
        else:
            self.P = P.astype(float)

    def predict(
        self,
        control: ControlInput | None = None,
    ) -> VehicleState:
        """Prediction step using optional acceleration control.

        Returns the predicted `VehicleState`.
        """
        if control is None:
            control_array = np.zeros(2, dtype=float)
        else:
            control_array = control.as_array()

        self.x = self.F @ self.x + self.B @ control_array
        self.P = self.F @ self.P @ self.F.T + self.Q

        return VehicleState.from_array(self.x)

    def update(self, measurement: SensorData) -> VehicleState:
        """Measurement update using `SensorData`.

        Returns the updated `VehicleState` estimate.
        """
        measurement_array = measurement.as_array()
        innovation = measurement_array - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation_covariance)

        self.x = self.x + kalman_gain @ innovation
        identity = np.eye(4, dtype=float)
        covariance_reduction = identity - kalman_gain @ self.H
        self.P = (
            covariance_reduction @ self.P @ covariance_reduction.T
            + kalman_gain @ self.R @ kalman_gain.T
        )

        return VehicleState.from_array(self.x)

    def step(
        self,
        control: ControlInput | None = None,
        measurement: SensorData | None = None,
    ) -> VehicleState:
        """Perform a single Kalman filter step: predict and optionally update.

        Args:
            control: optional `ControlInput` for prediction
            measurement: optional `SensorData` for update

        Returns the updated `VehicleState` estimate.
        """
        self.predict(control)
        if measurement is not None:
            self.update(measurement)
        return self.current_state()

    def current_state(self) -> VehicleState:
        """Return current state estimate as `VehicleState`."""
        return VehicleState.from_array(self.x)
