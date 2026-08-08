import numpy as np

from autonomy_sim.core.types import Waypoint


def planned_path_positions(start_position, waypoints: list[Waypoint]):
    return np.array(
        [start_position] + [[waypoint.x, waypoint.y] for waypoint in waypoints],
        dtype=float,
    )


def distances_to_polyline(positions, path_positions):
    positions = np.asarray(positions, dtype=float)
    path_positions = np.asarray(path_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    if path_positions.ndim != 2 or path_positions.shape[1] != 2:
        raise ValueError("path_positions must have shape (n, 2)")
    if len(path_positions) == 0:
        raise ValueError("path_positions must contain at least one point")

    if len(path_positions) == 1:
        return np.linalg.norm(positions - path_positions[0], axis=1)

    segment_starts = path_positions[:-1]
    segment_vectors = path_positions[1:] - segment_starts
    segment_lengths_squared = np.sum(segment_vectors**2, axis=1)
    distances = np.empty(len(positions), dtype=float)

    for index, position in enumerate(positions):
        offsets = position - segment_starts
        fractions = np.zeros(len(segment_vectors), dtype=float)
        nonzero_segments = segment_lengths_squared > 0.0
        fractions[nonzero_segments] = (
            np.sum(
                offsets[nonzero_segments] * segment_vectors[nonzero_segments],
                axis=1,
            )
            / segment_lengths_squared[nonzero_segments]
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        closest_points = segment_starts + fractions[:, None] * segment_vectors
        distances[index] = np.min(np.linalg.norm(position - closest_points, axis=1))

    return distances
