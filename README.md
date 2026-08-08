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

Trajectory (including commanded/applied control) -> Metrics and visualization
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

Sensor scenario `0` is the configured nominal Gaussian sensor. Scenarios `1`
through `4` are the degradation models listed above. Independent environment
and sensor seeds are generated for every trial.

The runner writes one row per attempted trial to
`results/monte_carlo/trials.csv` and aggregate counts, rates, and statistics to
`results/monte_carlo/summary.json`. Planning failures are recorded and do not
stop later trials. Use the same base seed to reproduce the same environment and
sensor seeds.

## Qualification metrics

Each trajectory sample records true, measured, and estimated state; commanded
and applied acceleration; whether dynamics limits changed the command; true and
estimated cross-track error; true and estimated safety clearance; and true and
estimated zone classification.

Run-level metrics include:

- mean, maximum, and root-mean-square true and estimated cross-track error;
- minimum true and estimated clearance to disallowed/restricted space or the
  environment boundary;
- commanded and applied control effort, computed as the time integral of
  acceleration magnitude;
- saturation fraction plus mean and maximum command-to-applied acceleration
  mismatch; and
- maximum commanded and applied acceleration.

Cross-track error is shortest distance to the planned waypoint polyline.
Clearance excludes occupied cells because the model defines them as hazardous
but traversable; disallowed and restricted cells are prohibited. All geometric
metrics use the environment's world units. Monte Carlo trial CSVs include these
metrics, and summaries aggregate the principal cross-track, clearance, and
control quantities.

Dynamics applies acceleration and speed limits before returning the applied
control. The Kalman filter predicts from that applied value, keeping estimator
propagation consistent with the simulated vehicle response.

## Reproducible campaigns

Campaign YAML files freeze the simulation and mission inputs, trial count, seed
schedule, worker count, output location, and optional qualification gates. Run
the pilot or frozen nominal baseline with:

```powershell
python -m autonomy_sim.experiments.run_campaign run --campaign-config configs/pilot_baseline.yaml
python -m autonomy_sim.experiments.run_campaign run --campaign-config configs/qualification_baseline.yaml
```

Each campaign directory contains archived YAML inputs, SHA-256 hashes, Git and
runtime metadata in `manifest.json`, per-trial CSV data, aggregate statistics
and Wilson 95% confidence intervals, explicit gate results, and a summary plot.
Frozen campaigns require a clean Git tree and refuse to overwrite existing
results.

Any recorded trial can be regenerated from its archived inputs and seeds, then
compared field-by-field with its original row:

```powershell
python -m autonomy_sim.experiments.run_campaign replay --manifest results/qualification/nominal_baseline_v1/manifest.json --trial 17
```

The nominal baseline uses a 0.20-unit onboard waypoint threshold and a separate
0.30-unit truth-evaluation tolerance. The first triggers completion from the
estimated state; the second judges physical goal attainment with allowance for
nominal estimation error. The planner also inflates prohibited zones by the
configured minimum clearance.

### Frozen nominal baseline result

`nominal_baseline_v1` passed all six predeclared gates across 500 trials. It
achieved 92.0% planning success, 99.78% mission success given a valid plan,
0.22% false completion given a valid plan, no safety violations, 0.0374 mean
true cross-track RMSE, and no control saturation. The archived evidence is in
`results/qualification/nominal_baseline_v1/`; the separate 50-trial diagnostic
pilot is in `results/qualification/pilot_nominal_v1/`.

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
