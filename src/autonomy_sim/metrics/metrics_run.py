import numpy as np

from autonomy_sim.core.types import VehicleState


class RunMetrics:
    def __init__(self, trajectory, waypoints, path_complete):
        if not trajectory:
            raise ValueError("trajectory must contain at least one sample")
        if not waypoints:
            raise ValueError("waypoints must contain at least one waypoint")

        self.trajectory = trajectory
        self.waypoints = waypoints
        self.path_complete = bool(path_complete)

    def calculate(self):
        final_sample = self.trajectory[-1]
        final_waypoint = self.waypoints[-1]
        final_state = VehicleState(
            x=float(final_sample["x"]),
            y=float(final_sample["y"]),
            vx=float(final_sample["vx"]),
            vy=float(final_sample["vy"]),
        )

        true_positions = self._values("x", "y")
        measured_positions = self._values("x_meas", "y_meas")
        estimated_positions = self._values("x_est", "y_est")
        true_velocities = self._values("vx", "vy")
        controls = self._values("ax_cmd", "ay_cmd")
        times = np.array(
            [sample["time"] for sample in self.trajectory],
            dtype=float,
        )

        measurement_errors = np.linalg.norm(
            measured_positions - true_positions,
            axis=1,
        )
        measurement_available = np.array(
            [
                sample.get("measurement_available", True)
                for sample in self.trajectory
            ],
            dtype=bool,
        )
        measurement_available = measurement_available & np.all(
            np.isfinite(measured_positions),
            axis=1,
        )
        valid_measurement_errors = measurement_errors[
            measurement_available
        ]
        estimation_errors = np.linalg.norm(
            estimated_positions - true_positions,
            axis=1,
        )
        speeds = np.linalg.norm(true_velocities, axis=1)
        acceleration_magnitudes = np.linalg.norm(controls, axis=1)

        final_distance = float(
            np.hypot(
                final_state.x - final_waypoint.x,
                final_state.y - final_waypoint.y,
            )
        )
        actual_path_length = self._path_length(true_positions)
        planned_positions = np.array(
            [[true_positions[0, 0], true_positions[0, 1]]]
            + [[waypoint.x, waypoint.y] for waypoint in self.waypoints],
            dtype=float,
        )
        planned_path_length = self._path_length(planned_positions)

        control_effort = 0.0
        if len(times) > 1:
            time_steps = np.diff(times)
            control_effort = float(
                np.sum(acceleration_magnitudes[1:] * time_steps)
            )

        termination_state = "time_limit"
        if self.path_complete:
            termination_state = "goal_reached"

        return {
            "mission_success": self.path_complete,
            "termination_state": termination_state,
            "final_state": final_state,
            "final_true_distance": final_distance,
            "final_waypoint_index": int(
                final_sample["current_waypoint_index"]
            ),
            "waypoint_path_complete": self.path_complete,
            "completion_time": float(final_sample["time"]),
            "number_of_steps": len(self.trajectory) - 1,
            "actual_path_length": actual_path_length,
            "planned_path_length": planned_path_length,
            "measurement_dropout_fraction": float(
                1.0 - np.mean(measurement_available)
            ),
            "mean_measurement_error": self._mean(valid_measurement_errors),
            "rmse_measurement_error": self._rmse(valid_measurement_errors),
            "mean_estimation_error": float(np.mean(estimation_errors)),
            "max_estimation_error": float(np.max(estimation_errors)),
            "rmse_estimation_error": self._rmse(estimation_errors),
            "control_effort": control_effort,
            "max_commanded_acceleration": float(
                np.max(acceleration_magnitudes)
            ),
            "max_speed": float(np.max(speeds)),
        }

    def _values(self, first_name, second_name):
        return np.array(
            [
                [sample[first_name], sample[second_name]]
                for sample in self.trajectory
            ],
            dtype=float,
        )

    @staticmethod
    def _path_length(positions):
        if len(positions) < 2:
            return 0.0
        segments = np.diff(positions, axis=0)
        return float(np.sum(np.linalg.norm(segments, axis=1)))

    @staticmethod
    def _mean(values):
        if len(values) == 0:
            return np.nan
        return float(np.mean(values))

    @staticmethod
    def _rmse(errors):
        if len(errors) == 0:
            return np.nan
        return float(np.sqrt(np.mean(np.square(errors))))
