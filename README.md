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
       +----- Estimated state <- Kalman <- Sensor model

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

The command-line runner uses `configs/default.yaml` and displays plots:

```powershell
python -m autonomy_sim.main
python -m autonomy_sim.main --config configs/default.yaml
```

The random seed, vehicle limits, sensor noise, estimator variances, controller
gains, and waypoint route are configurable in YAML. The Python API is headless
when both display options are disabled.

Simulation configuration and mission configuration are separate. The simulation
file defines the vehicle, sensor, estimator, guidance, timing, and direct
waypoint route. `configs/random_mission.yaml` and `configs/hallway_mission.yaml`
define environment, start/goal, planner, and mission-guidance settings. Mission
commands accept `--config` and `--mission-config` when alternate files are used.

The YAML also names each configured subsystem with a `type` field. The current
supported stack is `point_mass_acceleration` control, `point_mass` dynamics,
`gaussian` sensing, `kalman_filter` estimation, and `waypoint_tracker` guidance.
Unknown types fail explicitly instead of silently running a different model.
Configuration loading also rejects missing sections, non-finite values,
non-positive timing and vehicle limits, invalid noise or covariance values,
empty waypoint routes, and non-positive waypoint thresholds. Every configuration
must use the complete schema shown in `configs/default.yaml`, including each
subsystem's `type`, the `guidance` section, and an explicit `random_seed` field.
Set `random_seed` to `null` only when nondeterministic execution is intentional.

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

Run a hallway scenario by passing a number from 1 through 4:

```powershell
python -m autonomy_sim.mission.run_hallway_sensor_scenarios 1
```

Run a scenario in a random environment the same way:

```powershell
python -m autonomy_sim.mission.run_random_sensor_scenarios 1
```

Mission and simulation files can be selected independently:

```powershell
python -m autonomy_sim.mission.run_random_mission --config configs/default.yaml --mission-config configs/random_mission.yaml
```

Specify seeds when repeatability across runs is needed:

```powershell
python -m autonomy_sim.mission.run_hallway_sensor_scenarios 3 --seed 12
python -m autonomy_sim.mission.run_random_sensor_scenarios 4 --environment-seed 5 --sensor-seed 12
```

The planner follows the cost structure used in the project's NASA precursor:
each edge combines distance-based fuel cost with the destination cell's
environmental risk cost. Infinite cost cells are obstacles.

## Monte Carlo evaluation

Run deterministic random-mission trials without opening plots:

```powershell
python -m autonomy_sim.experiments.run_monte_carlo --trials 10 --base-seed 7 --sensor-scenario 3
```

The runner writes one row per attempted trial to
`results/monte_carlo/trials.csv` and aggregate counts, rates, and statistics to
`results/monte_carlo/summary.json`. Planning failures are recorded and do not
stop later trials. Use the same base seed to reproduce the same environment and
sensor seeds.

Current simplification: dynamics clamps acceleration and speed internally while
the Kalman filter predicts from the controller command. This small model
mismatch is intentionally visible and partly represented by process noise. A
future actuator interface can expose applied control explicitly.

## Verify

```powershell
pytest -q
```

## Current scope

The project contains a 2-D point-mass vehicle, acceleration PD controller,
waypoint tracker, sensor-fault models, linear Kalman filter, risk-aware A* path
planner, trajectory logging, safety and performance metrics, visualization, and
Monte Carlo evaluation. See `docs/design.md` for design details and planned
extensions.
