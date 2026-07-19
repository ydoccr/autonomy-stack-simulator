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
python -m autonomy_sim.main --config configs/default.yaml
```

The random seed, vehicle limits, sensor noise, estimator variances, controller gains, and waypoint route are all configurable in YAML.

Run the risk-aware planned mission with:

```powershell
python -m autonomy_sim.experiments.run_planned_mission
```

## Sensor fault scenarios

The hallway and random missions can also run with one of four sensor fault
scenarios:

1. Fixed position offset of `(0.2, -0.3)`
2. Measurements delayed by three simulation steps
3. Each measurement has a 40% chance of being dropped
4. Offset, delay, dropout, and Gaussian noise combined

Run a hallway scenario by passing its number:

```powershell
python -m autonomy_sim.mission.run_hallway_sensor_scenarios 1
python -m autonomy_sim.mission.run_hallway_sensor_scenarios 2
python -m autonomy_sim.mission.run_hallway_sensor_scenarios 3
python -m autonomy_sim.mission.run_hallway_sensor_scenarios 4
```

Run the same scenarios in a random environment:

```powershell
python -m autonomy_sim.mission.run_random_sensor_scenarios 1
python -m autonomy_sim.mission.run_random_sensor_scenarios 2
python -m autonomy_sim.mission.run_random_sensor_scenarios 3
python -m autonomy_sim.mission.run_random_sensor_scenarios 4
```

Seeds: 

```powershell
python -m autonomy_sim.mission.run_hallway_sensor_scenarios 3 --seed 12
python -m autonomy_sim.mission.run_random_sensor_scenarios 4 --environment-seed 5 --sensor-seed 12
```

The planner follows the cost structure used in the project's NASA precursor: each edge combines distance-based fuel cost with the destination cell's environmental risk cost. Infinite cost cells are obstacles.

Current simplification: dynamics clamps acceleration and speed internally while the Kalman filter predicts from the controller command. This small model mismatch is intentionally left visible and is partly represented by process noise. A future actuator interface can expose applied control explicitly.

## Verify

```powershell
pytest -q
```

## Current scope

The current implementation contains a 2-D point-mass vehicle, acceleration PD controller, waypoint tracker, Gaussian full-state sensor, linear Kalman filter, risk-aware grid costmap, A* path planner, path-to-waypoint conversion, trajectory logging, metrics, and plots. See `docs/design.md` for planned mission capabilities.
