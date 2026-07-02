# discrete-time linear Kalman filter for 2D point-mass model

# State vector: [x, y, vx, vy].
# Motion model uses constant-acceleration control input u = [ax, ay].

# Provides `predict` and `update` methods that accept the project's
# `ControlInput` and `SensorData` types and return `VehicleState` estimates.


from typing import Optional
import numpy as np
from autonomy_sim.core.types import VehicleState, SensorData, ControlInput


class KalmanFilter:
	"""Discrete linear Kalman filter for a point-mass vehicle.

	Args:
		dt: time step (s)
		process_var: scalar multiplier for process noise covariance Q
		meas_var: scalar multiplier for measurement noise covariance R
	"""

	def __init__(self, dt: float, process_var: float = 1e-3, meas_var: float = 1e-2):
		self.dt = float(dt)

		# State transition (constant-velocity part) for state [x,y,vx,vy]
		self.F = np.array([
			[1, 0, self.dt, 0],
			[0, 1, 0, self.dt],
			[0, 0, 1, 0],
			[0, 0, 0, 1],
		], dtype=float)

		# Control-input model from accelerations [ax, ay]
		dt2 = 0.5 * (self.dt ** 2)
		self.B = np.array([
			[dt2, 0],
			[0, dt2],
			[self.dt, 0],
			[0, self.dt],
		], dtype=float)

		# Full-state measurements (position+velocity)
		self.H = np.eye(4, dtype=float)

		# Covariances
		self.Q = process_var * np.eye(4, dtype=float)
		self.R = meas_var * np.eye(4, dtype=float)

		# State estimate and covariance
		self.x = np.zeros(4, dtype=float)
		self.P = np.eye(4, dtype=float)

	def reset(self, state: Optional[VehicleState] = None, P: Optional[np.ndarray] = None) -> None:
		"""Reset the filter state and covariance."""
		if state is not None:
			self.x = state.as_array().astype(float)
		else:
			self.x = np.zeros(4, dtype=float)
		if P is not None:
			self.P = P.astype(float)
		else:
			self.P = np.eye(4, dtype=float)

	def predict(self, control: Optional[ControlInput] = None) -> VehicleState:
		"""Prediction step using optional acceleration control.

		Returns the predicted `VehicleState`.
		"""
		u = control.as_array() if control is not None else np.zeros(2, dtype=float)
		self.x = self.F @ self.x + self.B @ u
		self.P = self.F @ self.P @ self.F.T + self.Q
		return VehicleState.from_array(self.x)

	def update(self, measurement: SensorData) -> VehicleState:
		"""Measurement update using `SensorData`.

		Returns the updated `VehicleState` estimate.
		"""
		z = measurement.as_array()
		y = z - (self.H @ self.x)
		S = self.H @ self.P @ self.H.T + self.R
		K = self.P @ self.H.T @ np.linalg.inv(S)

		self.x = self.x + K @ y
		I = np.eye(self.P.shape[0])
		self.P = (I - K @ self.H) @ self.P
		return VehicleState.from_array(self.x)
	
	def step(self, control: Optional[ControlInput] = None, measurement: Optional[SensorData] = None) -> VehicleState:
		"""Perform a single Kalman filter step: predict and optionally update.

		Args:
			control: optional `ControlInput` for prediction
			measurement: optional `SensorData` for update

		Returns the updated `VehicleState` estimate.
		"""
		self.predict(control)
		if measurement is not None:
			self.update(measurement)
		return VehicleState.from_array(self.x)

	def current_state(self) -> VehicleState:
		"""Return current state estimate as `VehicleState`."""
		return VehicleState.from_array(self.x)

