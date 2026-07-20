import numpy as np


OCCUPIED_COST = 10.0
DISALLOWED_COST = 20.0
RESTRICTED_COST = np.inf
ZONE_NAMES = (
    "free",
    "occupied",
    "disallowed",
    "restricted",
    "out_of_bounds",
)
RANDOM_ZONE_NAMES = ("free", "occupied", "disallowed", "restricted")
DEFAULT_RANDOM_ZONE_PROBABILITIES = {
    "free": 0.65,
    "occupied": 0.20,
    "disallowed": 0.10,
    "restricted": 0.05,
}


class GridEnvironment:
    def __init__(self, width, height, resolution=1.0, max_cost=np.inf):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.max_cost = max_cost
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

    def zone_at_position(self, x, y):
        column = int(np.floor(float(x) / self.resolution))
        row = int(np.floor(float(y) / self.resolution))
        if not (0 <= row < self.height and 0 <= column < self.width):
            return "out_of_bounds"
        if self.restricted[row, column]:
            return "restricted"
        if self.disallowed[row, column]:
            return "disallowed"
        if self.occupied[row, column]:
            return "occupied"
        return "free"

    def to_zone_costmap(self):
        costmap = np.zeros((self.height, self.width), dtype=float)
        costmap[self.occupied] = OCCUPIED_COST
        costmap[self.disallowed] = DISALLOWED_COST
        costmap[self.restricted] = RESTRICTED_COST
        return costmap

    def to_costmap(
        self,
        proximity_sigma=0.06,
        allow_disallowed=False,
    ):
        zone_costmap = self.to_zone_costmap()
        proximity_source = zone_costmap.copy()
        proximity_source[self.restricted] = DISALLOWED_COST

        planning_costmap = proximity_source
        if proximity_sigma > 0.0:
            sigma_cells = proximity_sigma / self.resolution
            kernel = _gaussian_kernel(sigma_cells)
            proximity_cost = _gaussian_blur(proximity_source, kernel)
            planning_costmap = proximity_source + proximity_cost

        planning_costmap[self.restricted] = np.inf
        if not allow_disallowed:
            planning_costmap[self.disallowed] = np.inf
        return planning_costmap


def _gaussian_kernel(sigma_cells):
    radius = int(np.ceil(3.0 * sigma_cells))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    return kernel / np.sum(kernel)


def _gaussian_blur(costmap, kernel):
    radius = len(kernel) // 2

    horizontal = np.zeros_like(costmap)
    padded = np.pad(
        costmap,
        ((0, 0), (radius, radius)),
        mode="edge",
    )
    for row in range(costmap.shape[0]):
        horizontal[row] = np.convolve(padded[row], kernel, mode="valid")

    blurred = np.zeros_like(costmap)
    padded = np.pad(
        horizontal,
        ((radius, radius), (0, 0)),
        mode="edge",
    )
    for column in range(costmap.shape[1]):
        blurred[:, column] = np.convolve(
            padded[:, column],
            kernel,
            mode="valid",
        )

    return blurred


def create_hallway_environment(max_cost=np.inf):
    environment = GridEnvironment(
        width=101,
        height=101,
        resolution=0.1,
        max_cost=max_cost,
    )

    environment.set_zone("occupied", 0, 101, 0, 101)
    environment.set_zone("disallowed", 21, 80, 0, 40)
    environment.set_zone("restricted", 21, 80, 61, 101)

    # Two-unit-wide hallway: east, north, then east again.
    environment.set_zone("free", 0, 21, 0, 61)
    environment.set_zone("free", 0, 101, 40, 61)
    environment.set_zone("free", 80, 101, 40, 101)

    return environment


def create_random_environment(
    random_seed=0,
    width=10,
    height=10,
    resolution=0.1,
    max_cost=np.inf,
    zone_probabilities=None,
):
    cells_per_block = round(1.0 / resolution)
    grid_width = width * cells_per_block
    grid_height = height * cells_per_block
    environment = GridEnvironment(
        width=grid_width,
        height=grid_height,
        resolution=resolution,
        max_cost=max_cost,
    )

    probabilities = _random_zone_probabilities(zone_probabilities)
    environment.zone_probabilities = {
        name: float(probability)
        for name, probability in zip(RANDOM_ZONE_NAMES, probabilities)
    }
    rng = np.random.default_rng(random_seed)

    for block_row in range(height):
        for block_column in range(width):
            zone = rng.choice(
                RANDOM_ZONE_NAMES,
                p=probabilities,
            )

            is_start_block = block_row == 0 and block_column == 0
            is_goal_block = block_row == height - 1 and block_column == width - 1
            if is_start_block or is_goal_block:
                zone = "free"

            row_min = block_row * cells_per_block
            row_max = row_min + cells_per_block
            col_min = block_column * cells_per_block
            col_max = col_min + cells_per_block
            environment.set_zone(
                zone,
                row_min,
                row_max,
                col_min,
                col_max,
            )

    return environment


def _random_zone_probabilities(zone_probabilities):
    if zone_probabilities is None:
        zone_probabilities = DEFAULT_RANDOM_ZONE_PROBABILITIES
    if set(zone_probabilities) != set(RANDOM_ZONE_NAMES):
        raise ValueError(
            "zone_probabilities must define free, occupied, disallowed, and restricted"
        )

    probabilities = np.array(
        [zone_probabilities[name] for name in RANDOM_ZONE_NAMES],
        dtype=float,
    )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("zone probabilities must be finite and non-negative")
    if not np.isclose(np.sum(probabilities), 1.0):
        raise ValueError("zone probabilities must sum to one")
    return probabilities
