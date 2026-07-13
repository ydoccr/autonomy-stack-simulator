# autonomy-stack-simulator

Autonomous UAV guidance, navigation, and control simulator for planning, state
estimation, control, and resilience experiments.

## Architecture

```text
Scenario -> Costmap -> A* -> Grid path -> Waypoints
                                         |
                                         v
WaypointTracker -> Controller -> Dynamics -> True state
       ^                             |          |
       |                             |          v
       +----- Estimated state <- Kalman <- Gaussian sensor

Trajectory -> Metrics and visualization
```

True state is confined to dynamics, sensor generation, logging, and metrics.
Guidance and control operate on the Kalman-filtered estimate.

## Setup

Python 3.10 or newer is required. Create a fresh environment rather than reusing
a `.venv` copied from another machine:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Run

The default run is headless and uses `configs/default.yaml`:

```powershell
python -m autonomy_sim.main
python -m autonomy_sim.main --plot
python -m autonomy_sim.main --config configs/default.yaml
```

The random seed, vehicle limits, sensor noise, estimator variances, controller
gains, and waypoint route are all configurable in YAML.

Run the risk-aware planned mission with:

```powershell
python -m autonomy_sim.experiments.run_planned_mission
```

The planner follows the cost structure used in the project's NASA precursor:
each edge combines distance-based fuel cost with the destination cell's
environmental risk cost. Infinite cost cells are obstacles.

Current simplification: dynamics clamps acceleration and speed internally,
while the Kalman filter predicts from the controller command. This small model
mismatch is intentionally left visible and is partly represented by process
noise. A future actuator interface can expose applied control explicitly.

## Verify

```powershell
ruff check .
pytest -q
```

CI runs both commands for pushes and pull requests.

## Current scope

The current implementation contains a 2-D point-mass vehicle, acceleration PD
controller, waypoint tracker, Gaussian full-state sensor, linear Kalman filter,
risk-aware grid costmap, A* path planner, path-to-waypoint conversion,
trajectory logging, metrics, and plots. See `docs/design.md` for planned mission
capabilities.
