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

## Zone policy

- Free cells have no environmental cost.
- Occupied cells are hazardous but traversable.
- Disallowed cells are blocked during normal planning but may support a future
  emergency mode.
- Restricted cells are always prohibited.

True zone exposure determines safety outcomes. Estimated exposure records what
the onboard system believed.

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

## Planned extensions

- Add an explicit actuator model between command and dynamics.
- Add wind and other external disturbances.
- Add emergency routing through disallowed zones.
- Port the fixed-rate guidance, estimation, control, and dynamics loop to C++.
