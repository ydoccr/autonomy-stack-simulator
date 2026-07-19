# necessary shared types: SimConfig, VehicleState, 
# ControlInput, SensorData, Waypoint, Path, 
# GuidanceCommand, MissionStatus, CostmapCell, ExperimentResult, etc.

# V1: only SimConfig, VehicleState, ControlInput, SensorData, Waypoint

from dataclasses import dataclass
from typing import Any
# dataclass: __init__, __repr__, __eq__, etc. automatically generated

import numpy as np


@dataclass
class SimConfig:
    dt: float  # time step
    num_steps: int  # total number of steps in the simulation


@dataclass
class SimulationResult:
    trajectory: list[dict[str, Any]]
    metrics: dict[str, Any]
    scenario: dict[str, Any]

@dataclass
class VehicleState:
    x: float  # position x
    y: float  # position y
    vx: float  # velocity in x direction
    vy: float  # velocity in y direction

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], 
                        dtype=float)
    
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], 
                        dtype=float)
    
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], 
                        dtype=float)
    
    @staticmethod
    def from_array(arr) -> 'VehicleState':
        return VehicleState(
            x=float(arr[0]), 
            y=float(arr[1]), 
            vx=float(arr[2]), 
            vy=float(arr[3]))
    
@dataclass
class ControlInput:
    ax: float  # acceleration in x direction
    ay: float  # acceleration in y direction

    def as_array(self) -> np.ndarray:
        return np.array([self.ax, self.ay], dtype=float)
    
    @staticmethod
    def from_array(arr) -> 'ControlInput':
        return ControlInput(ax=float(arr[0]), ay=float(arr[1]))
    
@dataclass
class SensorData:
    x_meas: float  # measured position x
    y_meas: float  # measured position y
    vx_meas: float  # measured velocity in x direction
    vy_meas: float  # measured velocity in y direction

    def as_array(self) -> np.ndarray:
        return np.array([self.x_meas, self.y_meas, self.vx_meas, self.vy_meas],
                        dtype=float)
    
    @staticmethod
    def from_array(arr) -> 'SensorData':
        return SensorData(
            x_meas=float(arr[0]), 
            y_meas=float(arr[1]), 
            vx_meas=float(arr[2]), 
            vy_meas=float(arr[3]))
    
@dataclass
class Waypoint:
    x: float  # waypoint position x
    y: float  # waypoint position y

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)
    
    @staticmethod
    def from_array(arr) -> 'Waypoint':
        return Waypoint(x=float(arr[0]), y=float(arr[1]))
