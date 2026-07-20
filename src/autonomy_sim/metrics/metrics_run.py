import numpy as np

from autonomy_sim.core.types import VehicleState
from autonomy_sim.environments.example_environments import ZONE_NAMES


class RunMetrics:
    def __init__(
        self,
        trajectory,
        waypoints,
        path_complete,
        goal_tolerance,
        environment=None,
    ):
        if not trajectory:
            raise ValueError("trajectory must contain at least one sample")
        if not waypoints:
            raise ValueError("waypoints must contain at least one waypoint")

        self.trajectory = trajectory
        self.waypoints = waypoints
        self.path_complete = bool(path_complete)
        self.goal_tolerance = float(goal_tolerance)
        self.environment = environment

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
            [sample.get("measurement_available", True) for sample in self.trajectory],
            dtype=bool,
        )
        measurement_available = measurement_available & np.all(
            np.isfinite(measured_positions),
            axis=1,
        )
        valid_measurement_errors = measurement_errors[measurement_available]
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
        true_zone_time = self._zone_time(times, true_positions)
        estimated_zone_time = self._zone_time(times, estimated_positions)

        control_effort = 0.0
        if len(times) > 1:
            time_steps = np.diff(times)
            control_effort = float(np.sum(acceleration_magnitudes[1:] * time_steps))

        onboard_completion = self.path_complete
        true_goal_reached = final_distance < self.goal_tolerance
        if true_zone_time is None:
            restricted_violation = None
            disallowed_violation = None
            out_of_bounds_violation = None
            safety_violation = False
        else:
            restricted_violation = true_zone_time["restricted"] > 0.0
            disallowed_violation = true_zone_time["disallowed"] > 0.0
            out_of_bounds_violation = true_zone_time["out_of_bounds"] > 0.0
            safety_violation = (
                restricted_violation or disallowed_violation or out_of_bounds_violation
            )
        true_mission_success = true_goal_reached and not safety_violation

        termination_state = "time_limit"
        if onboard_completion and true_mission_success:
            termination_state = "goal_reached"
        elif safety_violation and true_goal_reached:
            termination_state = "safety_violation"
        elif onboard_completion and not true_goal_reached:
            termination_state = "false_completion"
        elif true_goal_reached:
            termination_state = "unrecognized_goal_reached"

        return {
            "planning_success": True,
            "onboard_completion": onboard_completion,
            "true_goal_reached": true_goal_reached,
            "true_mission_success": true_mission_success,
            "restricted_violation": restricted_violation,
            "disallowed_violation": disallowed_violation,
            "out_of_bounds_violation": out_of_bounds_violation,
            "termination_state": termination_state,
            "final_state": final_state,
            "final_true_distance": final_distance,
            "final_waypoint_index": int(final_sample["current_waypoint_index"]),
            "completion_time": float(final_sample["time"]),
            "number_of_steps": len(self.trajectory) - 1,
            "actual_path_length": actual_path_length,
            "planned_path_length": planned_path_length,
            "true_zone_time_seconds": true_zone_time,
            "estimated_zone_time_seconds": estimated_zone_time,
            "measurement_dropout_fraction": float(1.0 - np.mean(measurement_available)),
            "mean_measurement_error": self._mean(valid_measurement_errors),
            "rmse_measurement_error": self._rmse(valid_measurement_errors),
            "mean_estimation_error": float(np.mean(estimation_errors)),
            "max_estimation_error": float(np.max(estimation_errors)),
            "rmse_estimation_error": self._rmse(estimation_errors),
            "control_effort": control_effort,
            "max_commanded_acceleration": float(np.max(acceleration_magnitudes)),
            "max_speed": float(np.max(speeds)),
        }

    def _values(self, first_name, second_name):
        return np.array(
            [[sample[first_name], sample[second_name]] for sample in self.trajectory],
            dtype=float,
        )

    def _zone_time(self, times, positions):
        if self.environment is None:
            return None

        zone_time = dict.fromkeys(ZONE_NAMES, 0.0)
        for index, time_step in enumerate(np.diff(times)):
            if time_step < 0.0:
                raise ValueError("trajectory times must be non-decreasing")
            midpoint = 0.5 * (positions[index] + positions[index + 1])
            zone = self.environment.zone_at_position(
                x=midpoint[0],
                y=midpoint[1],
            )
            zone_time[zone] += float(time_step)
        return zone_time

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
