import numpy as np


OCCUPIED_COST = 10.0
DISALLOWED_COST = 20.0
RESTRICTED_COST = 10000.0


class GridEnvironment:
    def __init__(self, width, height, resolution=1.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.occupied = np.zeros((height, width), dtype=bool)
        self.disallowed = np.zeros((height, width), dtype=bool)
        self.restricted = np.zeros((height, width), dtype=bool)

    def set_zone(self, zone, row_min, row_max, col_min, col_max):
        rows = slice(row_min, row_max)
        columns = slice(col_min, col_max)

        self.occupied[rows, columns] = False
        self.disallowed[rows, columns] = False
        self.restricted[rows, columns] = False

        if zone == "occupied":
            self.occupied[rows, columns] = True
        elif zone == "disallowed":
            self.disallowed[rows, columns] = True
        elif zone == "restricted":
            self.restricted[rows, columns] = True
        elif zone != "free":
            raise ValueError("unknown zone type")

    def to_costmap(self):
        costmap = np.zeros((self.height, self.width), dtype=float)
        costmap[self.occupied] = OCCUPIED_COST
        costmap[self.disallowed] = DISALLOWED_COST
        costmap[self.restricted] = RESTRICTED_COST
        return costmap


def create_hallway_environment():
    environment = GridEnvironment(width=101, height=101, resolution=0.1)

    environment.set_zone("occupied", 0, 101, 0, 101)
    environment.set_zone("disallowed", 21, 80, 0, 40)
    environment.set_zone("restricted", 21, 80, 61, 101)

    # Two-unit-wide hallway: east, north, then east again.
    environment.set_zone("free", 0, 21, 0, 61)
    environment.set_zone("free", 0, 101, 40, 61)
    environment.set_zone("free", 80, 101, 40, 101)

    return environment
