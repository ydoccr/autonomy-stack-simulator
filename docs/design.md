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
6. Record the true, measured, and estimated states.
7. Calculate mission, safety, estimation, and control metrics.

The vehicle state is `[x, y, vx, vy]`, and the control input is `[ax, ay]`.

## Zone policy

- Free cells have no environmental cost.
- Occupied cells are hazardous but traversable.
- Disallowed cells are blocked during normal planning but may support a future
  emergency mode.
- Restricted cells are always prohibited.

True zone exposure determines safety outcomes. Estimated exposure records what
the onboard system believed.

## Planned extensions

- Separate commanded and applied control with an actuator model.
- Add wind and other external disturbances.
- Add emergency routing through disallowed zones.
- Port the fixed-rate guidance, estimation, control, and dynamics loop to C++.
