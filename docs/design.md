## Goals
UAV GNC simulation stack, combining planning, guidance, control, estimation, mission logic, failure injection, logging, visualization, and experiment analysis.

## V1 System Architecture
Mission Manager
Planner
Guidance
Controller
Vehicle Dynamics
Sensors
State Estimation
Analytic tools (simulation, data, visualization)

### Mission Manager
High-level mission logic.
- PLAN_PATH -> A*, 2d costmap
- TRACK_PATH -> list?
- REPLAN -> boolean
- SUCCESS -> boolean

### Guidance
One layer below path planning; waypoint tracking & following. Outputs desired velocity.

### Controller
One layer below guidance; desired velocity tracking & following. 
V1 controller: PD.

### Vehicle Dynamics
TRUE vehicle state, including:
- state: [x, y, vx, vy]
- control: [ax, ay]

### Sensors
For now, just generate noisy measurements from the true vehicle state. (Gaussian error maybe? Research real-world)

### State Estimator
Estimates the vehicle state from noisy measurements and control inputs. Simple Kalman filtering approach.

### Analytic Tools
Records simulation data and produces plots/metrics.
- outcome (time, elapsed, path optimality, control effort, replan #, etc)
- path created
- data table


## Workflow
Goal: UAV must travel A -> B with optimal planning strategy under hazardous conditions.

1. Build costmap, plan A*.
3. Move along the path using relevant modules.
5. Log true states, estimated states, control inputs, mission flags, etc.
6. Perform data analysis.

## Future Goals
- Maximal realism (sensor, battery, actuator inaccuracies and degradations)
- different controllers
- moving hazards
- Degradations: wrt time? inherent faultiness?