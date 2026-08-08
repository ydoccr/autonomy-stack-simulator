import numpy as np

from autonomy_sim.core.types import VehicleState
from autonomy_sim.environments.example_environments import ZONE_NAMES


RUN_METRIC_FIELDS = (
    "planning_success",
    "onboard_completion",
    "true_goal_reached",
    "true_mission_success",
    "restricted_violation",
    "disallowed_violation",
    "out_of_bounds_violation",
    "termination_state",
    "final_state",
    "final_true_distance",
    "final_waypoint_index",
    "completion_time",
    "number_of_steps",
    "actual_path_length",
    "planned_path_length",
    "true_zone_time_seconds",
    "estimated_zone_time_seconds",
    "measurement_dropout_fraction",
    "mean_measurement_error",
    "rmse_measurement_error",
    "mean_estimation_error",
    "max_estimation_error",
    "rmse_estimation_error",
    "mean_true_cross_track_error",
    "max_true_cross_track_error",
    "rmse_true_cross_track_error",
    "mean_estimated_cross_track_error",
    "max_estimated_cross_track_error",
    "rmse_estimated_cross_track_error",
    "minimum_true_clearance",
    "minimum_estimated_clearance",
    "commanded_control_effort",
    "applied_control_effort",
    "control_saturation_fraction",
    "mean_control_saturation_error",
    "max_control_saturation_error",
    "max_commanded_acceleration",
    "max_applied_acceleration",
    "max_speed",
)


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
        commanded_controls = self._values("ax_cmd", "ay_cmd")
        applied_controls = self._values("ax_applied", "ay_applied")
        control_saturated = np.array(
            [sample["control_saturated"] for sample in self.trajectory],
            dtype=bool,
        )
        true_cross_track_errors = np.array(
            [sample["true_cross_track_error"] for sample in self.trajectory],
            dtype=float,
        )
        estimated_cross_track_errors = np.array(
            [sample["estimated_cross_track_error"] for sample in self.trajectory],
            dtype=float,
        )
        true_clearances = np.array(
            [sample["true_clearance"] for sample in self.trajectory],
            dtype=float,
        )
        estimated_clearances = np.array(
            [sample["estimated_clearance"] for sample in self.trajectory],
            dtype=float,
        )
        times = np.array(
            [sample["time"] for sample in self.trajectory],
            dtype=float,
        )

        measurement_errors = np.linalg.norm(
            measured_positions - true_positions,
            axis=1,
        )
        measurement_available = np.array(
            [sample["measurement_available"] for sample in self.trajectory],
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
        commanded_acceleration_magnitudes = np.linalg.norm(commanded_controls, axis=1)
        applied_acceleration_magnitudes = np.linalg.norm(applied_controls, axis=1)
        saturation_errors = np.linalg.norm(
            commanded_controls - applied_controls,
            axis=1,
        )

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

        commanded_control_effort = 0.0
        applied_control_effort = 0.0
        control_saturation_fraction = 0.0
        mean_control_saturation_error = 0.0
        max_control_saturation_error = 0.0
        if len(times) > 1:
            time_steps = np.diff(times)
            commanded_control_effort = float(
                np.sum(commanded_acceleration_magnitudes[1:] * time_steps)
            )
            applied_control_effort = float(
                np.sum(applied_acceleration_magnitudes[1:] * time_steps)
            )
            control_saturation_fraction = float(np.mean(control_saturated[1:]))
            mean_control_saturation_error = float(np.mean(saturation_errors[1:]))
            max_control_saturation_error = float(np.max(saturation_errors[1:]))

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
            "mean_true_cross_track_error": float(np.mean(true_cross_track_errors)),
            "max_true_cross_track_error": float(np.max(true_cross_track_errors)),
            "rmse_true_cross_track_error": self._rmse(true_cross_track_errors),
            "mean_estimated_cross_track_error": float(
                np.mean(estimated_cross_track_errors)
            ),
            "max_estimated_cross_track_error": float(
                np.max(estimated_cross_track_errors)
            ),
            "rmse_estimated_cross_track_error": self._rmse(
                estimated_cross_track_errors
            ),
            "minimum_true_clearance": self._minimum_finite(true_clearances),
            "minimum_estimated_clearance": self._minimum_finite(estimated_clearances),
            "commanded_control_effort": commanded_control_effort,
            "applied_control_effort": applied_control_effort,
            "control_saturation_fraction": control_saturation_fraction,
            "mean_control_saturation_error": mean_control_saturation_error,
            "max_control_saturation_error": max_control_saturation_error,
            "max_commanded_acceleration": float(
                np.max(commanded_acceleration_magnitudes)
            ),
            "max_applied_acceleration": float(np.max(applied_acceleration_magnitudes)),
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

    @staticmethod
    def _minimum_finite(values):
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            return np.nan
        return float(np.min(finite_values))


def planning_failure_metrics():
    return {
        "planning_success": False,
        "onboard_completion": False,
        "true_goal_reached": False,
        "true_mission_success": False,
        "restricted_violation": None,
        "disallowed_violation": None,
        "out_of_bounds_violation": None,
        "termination_state": "planning_failure",
        "final_state": None,
        "final_true_distance": np.nan,
        "final_waypoint_index": None,
        "completion_time": np.nan,
        "number_of_steps": 0,
        "actual_path_length": np.nan,
        "planned_path_length": np.nan,
        "true_zone_time_seconds": None,
        "estimated_zone_time_seconds": None,
        "measurement_dropout_fraction": np.nan,
        "mean_measurement_error": np.nan,
        "rmse_measurement_error": np.nan,
        "mean_estimation_error": np.nan,
        "max_estimation_error": np.nan,
        "rmse_estimation_error": np.nan,
        "mean_true_cross_track_error": np.nan,
        "max_true_cross_track_error": np.nan,
        "rmse_true_cross_track_error": np.nan,
        "mean_estimated_cross_track_error": np.nan,
        "max_estimated_cross_track_error": np.nan,
        "rmse_estimated_cross_track_error": np.nan,
        "minimum_true_clearance": np.nan,
        "minimum_estimated_clearance": np.nan,
        "commanded_control_effort": np.nan,
        "applied_control_effort": np.nan,
        "control_saturation_fraction": np.nan,
        "mean_control_saturation_error": np.nan,
        "max_control_saturation_error": np.nan,
        "max_commanded_acceleration": np.nan,
        "max_applied_acceleration": np.nan,
        "max_speed": np.nan,
    }
