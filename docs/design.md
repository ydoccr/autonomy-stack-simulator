# Design

The simulator models a UAV traveling through a hazardous 2-D environment. It
separates path planning, waypoint guidance, feedback control, dynamics, sensing,
state estimation, and analysis.

## Simulation flow

1. Build an environment and costmap.
2. Plan a path with A* and convert it to waypoints.
3. Use the estimated state for waypoint guidance and PD control.
4. Apply acceleration and speed limits in the point-mass dynamics.
5. Generate sensor measurements and update the Kalman filter.
6. Record the true, measured, and estimated states, commanded and applied
   control, path-tracking error, clearance, and zone classification.
7. Calculate mission, safety, estimation, path-tracking, clearance, and control
   metrics.

The vehicle state is `[x, y, vx, vy]`, and the control input is `[ax, ay]`.

Grid indices identify cell centers. Cell `(row, column)` is located at
`(column * resolution, row * resolution)`, and the physical environment extends
half a cell beyond its outermost centers. Planning, zone lookup, visualization,
and clearance calculations use this same convention.

Planner edges use a conservative grid supercover: every closed cell rectangle
touched by the line segment between cell centers is checked. This includes cells
contacted only at an edge or corner, so a multi-cell step cannot jump through
prohibited geometry. Finite occupied-zone cost remains traversable and is
averaged over the intersected cells.

## Planner state and cost

A* combines distance-based fuel and environmental cost with per-edge waypoint
cost and a heading-change penalty. When the turn penalty is enabled, internal
search state is `(cell, incoming_direction)` because future turn cost depends on
the arrival heading. The returned path remains a list of grid cells. With zero
turn-cost weight, direction is collapsed and the search is ordinary cell-based
A*. The Euclidean-distance heuristic multiplied by fuel rate remains admissible
because it ignores only nonnegative environmental, waypoint, and turn costs.

## Zone policy

- Free cells have no environmental cost.
- Occupied cells are hazardous but traversable.
- Disallowed cells are blocked during normal planning but may support a future
  emergency mode.
- Restricted cells are always prohibited.

True zone exposure determines safety outcomes. Estimated exposure records what
the onboard system believed. Zone-duration accounting retains midpoint
classification for each simulation interval, while true safety-violation flags
also evaluate the complete swept line segment between consecutive truth samples.
Thus a brief prohibited crossing is detected even when both recorded endpoints
and the interval midpoint are safe.

## State-estimation covariance

The linear Kalman filter uses the same four-state point-mass model and applied
acceleration input as the simulated vehicle. Its measurement update uses Joseph
form,
`P = (I - K H) P (I - K H)^T + K R K^T`, which is algebraically equivalent to
the simpler covariance update in exact arithmetic but better preserves symmetry
and positive semidefiniteness under finite precision.

## Qualification metric definitions

Cross-track error is the shortest Euclidean distance from a position to any
line segment in the planned waypoint path. True cross-track error measures
physical tracking performance; estimated cross-track error records the path
tracking performance perceived onboard.

Safety clearance is the shortest Euclidean distance from a position to a
disallowed cell, restricted cell, or environment boundary. Clearance is zero
inside prohibited or out-of-bounds space. Occupied cells are excluded because
they are hazardous but traversable under the zone policy. The vehicle is a
point mass, so clearance does not include a body-radius correction.

Commanded and applied control effort are the time integrals of their respective
acceleration magnitudes. Saturation metrics compare the controller request to
the acceleration actually applied by dynamics after acceleration and speed
limits. The initial zero-control sample is excluded from saturation frequency
and mismatch statistics.

## Baseline qualification

Campaign execution snapshots all YAML inputs and records their hashes, the Git
commit and cleanliness, Python and dependency versions, seed schedule, command,
and timing. Trial rows remain in deterministic seed order even when executed in
parallel. Replay uses archived inputs and recorded environment/sensor seeds,
then compares every regenerated CSV field with the original.

The pilot baseline is diagnostic and may be rerun while correcting defects. A
frozen qualification campaign requires a clean tree, complete pass criteria,
and a new output directory. It cannot silently overwrite previous evidence.
Qualification rates include Wilson 95% confidence intervals.

## Planned extensions

- Add an explicit actuator model between command and dynamics.
- Add wind and other external disturbances.
- Add emergency routing through disallowed zones.
- Port the fixed-rate guidance, estimation, control, and dynamics loop to C++.
