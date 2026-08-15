import math


GridCell = tuple[int, int]
GridPosition = tuple[float, float]
INTERSECTION_TOLERANCE = 1e-12


def cells_intersected_by_segment(
    start: GridPosition,
    end: GridPosition,
) -> list[GridCell]:
    """Return every grid cell touched by a segment in cell-center coordinates.

    Integer coordinates identify cell centers, so cell ``(row, column)`` occupies
    the closed rectangle from ``row - 0.5`` to ``row + 0.5`` and likewise for
    columns. Boundary and corner contact count as intersection to keep collision
    checks conservative.
    """
    start_row = float(start[0])
    start_column = float(start[1])
    end_row = float(end[0])
    end_column = float(end[1])

    first_row = math.ceil(min(start_row, end_row) - 0.5 - INTERSECTION_TOLERANCE)
    last_row = math.floor(max(start_row, end_row) + 0.5 + INTERSECTION_TOLERANCE)
    first_column = math.ceil(
        min(start_column, end_column) - 0.5 - INTERSECTION_TOLERANCE
    )
    last_column = math.floor(
        max(start_column, end_column) + 0.5 + INTERSECTION_TOLERANCE
    )

    intersected = []
    for row in range(first_row, last_row + 1):
        for column in range(first_column, last_column + 1):
            entry_fraction = _segment_cell_entry_fraction(
                start_row,
                start_column,
                end_row,
                end_column,
                row,
                column,
            )
            if entry_fraction is not None:
                intersected.append((entry_fraction, row, column))

    intersected.sort()
    return [(row, column) for _, row, column in intersected]


def _segment_cell_entry_fraction(
    start_row,
    start_column,
    end_row,
    end_column,
    row,
    column,
):
    """Return where a segment first enters a cell, or None if it misses."""
    minimum_fraction = 0.0
    maximum_fraction = 1.0
    coordinates = (
        (start_row, end_row - start_row, row - 0.5, row + 0.5),
        (
            start_column,
            end_column - start_column,
            column - 0.5,
            column + 0.5,
        ),
    )

    for start_value, change, lower_bound, upper_bound in coordinates:
        if change == 0.0:
            if not lower_bound <= start_value <= upper_bound:
                return None
            continue

        first_crossing = (lower_bound - start_value) / change
        second_crossing = (upper_bound - start_value) / change
        entry = min(first_crossing, second_crossing)
        exit_ = max(first_crossing, second_crossing)
        minimum_fraction = max(minimum_fraction, entry)
        maximum_fraction = min(maximum_fraction, exit_)
        if minimum_fraction - maximum_fraction > INTERSECTION_TOLERANCE:
            return None

    return minimum_fraction
