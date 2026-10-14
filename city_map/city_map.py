"""
=============================================================================
city_map.py  —  Terrain constants, cost tables, and CityMap200 builder
=============================================================================
All terrain IDs, names, colours, base comfort costs, weather modifiers, and
the 200×200 procedural city grid live here.
=============================================================================
"""

import numpy as np
import matplotlib.colors as mcolors

# =============================================================================
# TERRAIN IDs
# =============================================================================
GOOD_PAVEMENT      = 0
DEGRADED_PAVEMENT  = 1
PAVEMENT_WITH_CARS = 2
GOOD_PARK_PATH     = 3
SOILY_PARK_PATH    = 4
GOOD_PED_STREET    = 5
SOILY_PED_STREET   = 6
ROAD               = 7
HIGHWAY            = 8
CROSSWALK          = 9
PARK_GRASS         = 10
WATER_PUDDLE       = 11
STREET_CLUTTER     = 12
CONSTRUCTION       = 13
STAIRS             = 14
BUILDING           = 15

TERRAIN_NAMES = {
    0:  'Good Pavement',
    1:  'Degraded Pavement',
    2:  'Pavement (Cars)',
    3:  'Park Path (Paved)',
    4:  'Park Path (Dirt)',
    5:  'Pedestrian Street',
    6:  'Ped. Area (Dirt)',
    7:  'Road',
    8:  'Highway',
    9:  'Crosswalk',
    10: 'Park Grass',
    11: 'Water Puddle',
    12: 'Street Clutter',
    13: 'Construction Zone',
    14: 'Stairs',
    15: 'Building',
}

COLOR_SCHEME = {
    0:  '#D5E8D4',
    1:  '#FFE6CC',
    2:  '#F8CECC',
    3:  '#B3E5B3',
    4:  '#C9A76B',
    5:  '#DAE8FC',
    6:  '#D4A574',
    7:  '#505050',
    8:  '#2C3E50',
    9:  '#F39C12',
    10: '#82C682',
    11: '#3498DB',
    12: '#9B59B6',
    13: '#E74C3C',
    14: '#C0392B',
    15: '#34495E',
}

# Base comfort costs — lower = more comfortable (Table I)
BASE_COMFORT = {
    0:  1.0,
    1:  2.0,
    2:  2.5,
    3:  1.2,
    4:  2.8,
    5:  1.1,
    6:  2.6,
    7:  4.0,
    8:  6.0,
    9:  1.5,
    10: 3.0,
    11: 3.5,
    12: 3.0,
    13: 8.0,
    14: 5.0,
    15: 999,
}

# Weather multipliers (Table I rain column)
WEATHER_MODIFIERS = {
    'sunny': {
        0: 1.0,  1: 1.1,  2: 1.0,  3: 1.0,  4: 1.3,
        5: 1.0,  6: 1.2,  7: 1.0,  8: 1.0,  9: 1.0,
        10: 1.1, 11: 1.5, 12: 1.0, 13: 1.0, 14: 1.0, 15: 999,
    },
    'rainy': {
        0: 1.3,  1: 2.0,  2: 1.8,  3: 1.5,  4: 3.5,
        5: 1.3,  6: 3.0,  7: 1.5,  8: 2.0,  9: 1.8,
        10: 3.5, 11: 5.0, 12: 1.5, 13: 3.0, 14: 2.5, 15: 999,
    },
    'neutral': {t: 1.0 for t in range(16)},
}


def cell_cost(terrain_id, weather):
    """Effective cost for a single cell given terrain and weather."""
    return BASE_COMFORT[terrain_id] * WEATHER_MODIFIERS[weather][terrain_id]


# =============================================================================
# CityMap200
# =============================================================================
class CityMap200:
    """
    Procedurally generates a 200×200 pedestrian city grid.

    Design intent
    -------------
    - Direct NW→SE corridor (shortest-path route) is seeded with
      high-cost terrain: highways, construction, puddles, park grass.
    - A comfortable boulevard network (good pavement + pedestrian streets)
      runs around the hazard corridor, giving HFL-RL a genuinely better route.
    """

    SIZE = 200

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.size = self.SIZE
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self._build()

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    def _build(self):
        self._place_buildings()
        self._place_highways()
        self._place_roads()
        self._place_crosswalks()
        self._place_parks()
        self._place_pavements()
        self._place_ped_streets()
        self._place_hazard_corridor()
        self._add_pavement_details()
        self._add_water_puddles()
        self._add_clutter()
        self._add_construction()
        self._add_stairs()

    # ------------------------------------------------------------------
    # Layer builders
    # ------------------------------------------------------------------
    def _place_buildings(self):
        blocks = [
            (5,5,20,25),   (5,155,20,25),
            (175,5,20,25), (175,165,20,20),
            (40,30,18,22), (40,80,18,20), (40,140,18,22),
            (40,170,18,20),
            (90,10,15,18), (90,60,12,18), (90,110,15,18),
            (90,155,12,18),
            (130,35,16,20),(130,85,14,18),(130,135,16,20),
            (130,170,14,18),
            (60,100,10,12),(70,45,8,10),(65,155,10,12),
            (110,40,8,10), (115,85,8,10),(110,140,8,12),
            (150,60,10,14),(155,110,8,12),(150,155,10,14),
        ]
        for r, c, h, w in blocks:
            if r + h <= self.size and c + w <= self.size:
                self.grid[r:r+h, c:c+w] = BUILDING

    def _place_highways(self):
        # Horizontal highways — sit directly across SP's diagonal path
        self.grid[75:80,  :]  = HIGHWAY
        self.grid[140:145, :] = HIGHWAY
        # Vertical highways — force SP to cross them
        self.grid[:, 95:100]  = HIGHWAY
        self.grid[:, 170:175] = HIGHWAY

    def _place_roads(self):
        for r in [30, 60, 110, 165]:
            self.grid[r:r+3, :] = ROAD
        for c in [30, 65, 130, 185]:
            self.grid[:, c:c+3] = ROAD

    def _place_crosswalks(self):
        hw_rows  = [75, 140]
        road_rows = [30, 60, 110, 165]
        hw_cols  = [95, 170]
        road_cols = [30, 65, 130, 185]

        for hr in hw_rows:
            for rc in road_cols:
                r0, c0 = hr - 1, rc - 1
                if 0 <= r0 and r0+5 <= self.size and 0 <= c0 and c0+5 <= self.size:
                    self.grid[r0:r0+5, c0:c0+5] = CROSSWALK

        for rr in road_rows:
            for hc in hw_cols:
                r0, c0 = rr - 1, hc - 1
                if 0 <= r0 and r0+5 <= self.size and 0 <= c0 and c0+5 <= self.size:
                    self.grid[r0:r0+5, c0:c0+5] = CROSSWALK

        boulevard_crossings = [
            (74,28,6,5),(74,63,6,5),(74,127,6,5),
            (139,28,6,5),(139,63,6,5),(139,127,6,5),
        ]
        for r, c, h, w in boulevard_crossings:
            if r+h <= self.size and c+w <= self.size:
                for i in range(r, r+h):
                    for j in range(c, c+w):
                        if self.grid[i, j] not in [BUILDING]:
                            self.grid[i, j] = CROSSWALK

    def _place_parks(self):
        parks = [
            ((10,32,25,25),  0.85),
            ((10,120,25,30), 0.80),
            ((55,68,18,22),  0.40),
            ((105,30,20,30), 0.35),
            ((105,130,20,30),0.35),
            ((160,70,25,45), 0.70),
            ((160,120,25,40),0.75),
        ]
        for bounds, gpr in parks:
            self._create_park(bounds, gpr)

    def _create_park(self, bounds, gpr):
        r, c, h, w = bounds
        if r+h > self.size or c+w > self.size:
            return
        area = self.grid[r:r+h, c:c+w]
        mask = (area == 0) | (area == ROAD) | (area == HIGHWAY)
        self.grid[r:r+h, c:c+w] = np.where(mask, PARK_GRASS, area)

        pv = w // 3
        for i in range(r, r+h):
            for j in range(c+pv, c+pv+2):
                if j < c+w and self.grid[i, j] == PARK_GRASS:
                    self.grid[i, j] = (GOOD_PARK_PATH if np.random.random() < gpr
                                       else SOILY_PARK_PATH)

        ph = h // 3
        for i in range(r+ph, r+ph+2):
            for j in range(c, c+w):
                if i < r+h and self.grid[i, j] == PARK_GRASS:
                    self.grid[i, j] = (GOOD_PARK_PATH if np.random.random() < gpr
                                       else SOILY_PARK_PATH)

    def _place_pavements(self):
        # Adjacent to buildings
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == BUILDING:
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = r+dr, c+dc
                        if (0 <= nr < self.size and 0 <= nc < self.size
                                and self.grid[nr, nc] == 0):
                            self.grid[nr, nc] = GOOD_PAVEMENT

        for r in [29, 33, 59, 63, 109, 113, 164, 168]:
            if r < self.size:
                for c in range(self.size):
                    if self.grid[r, c] == 0:
                        self.grid[r, c] = GOOD_PAVEMENT

        for c in [29, 33, 64, 68, 129, 133, 184, 188]:
            if c < self.size:
                for r in range(self.size):
                    if self.grid[r, c] == 0:
                        self.grid[r, c] = GOOD_PAVEMENT

        # Along highways — pavement on BOTH sides to give detour options
        for r in [74, 80, 139, 145]:
            if r < self.size:
                for c in range(self.size):
                    if self.grid[r, c] == 0:
                        self.grid[r, c] = GOOD_PAVEMENT

        for c in [94, 100, 169, 175]:
            if c < self.size:
                for r in range(self.size):
                    if self.grid[r, c] == 0:
                        self.grid[r, c] = GOOD_PAVEMENT

    def _place_ped_streets(self):
        # E-W boulevard at row ~83 — HFL-RL uses this to bypass highway
        for c in range(200):
            if self.grid[81, c] == 0:
                self.grid[81, c] = GOOD_PED_STREET
            if self.grid[82, c] == 0:
                self.grid[82, c] = GOOD_PED_STREET

        # N-S boulevard at col ~25 — southern leg of HFL detour
        for r in range(200):
            if self.grid[r, 25] == 0:
                self.grid[r, 25] = GOOD_PED_STREET
            if self.grid[r, 26] == 0:
                self.grid[r, 26] = GOOD_PED_STREET

        # N-S boulevard at col ~150 — eastern leg
        for r in range(200):
            if self.grid[r, 150] == 0:
                self.grid[r, 150] = GOOD_PED_STREET
            if self.grid[r, 151] == 0:
                self.grid[r, 151] = GOOD_PED_STREET

        # E-W boulevard at row ~148 — second bypass
        for c in range(200):
            if self.grid[148, c] == 0:
                self.grid[148, c] = GOOD_PED_STREET
            if self.grid[149, c] == 0:
                self.grid[149, c] = GOOD_PED_STREET

    def _place_hazard_corridor(self):
        """
        Scatter high-cost terrain along the direct NW→SE diagonal.
        SP (shortest distance) is forced into this corridor.
        HFL-RL learns to detour around it.
        """
        np.random.seed(7)
        hazard_zones = [
            (35,  35,  20, 18, PARK_GRASS),
            (35,  55,  18, 15, PARK_GRASS),
            (50,  50,  15, 20, WATER_PUDDLE),
            (55,  70,  12, 18, PARK_GRASS),
            (80,  100, 18, 20, PARK_GRASS),
            (85,  115, 15, 18, PARK_GRASS),
            (95,  108, 12, 15, WATER_PUDDLE),
            (100, 120, 20, 18, PARK_GRASS),
            (115, 125, 15, 20, WATER_PUDDLE),
            (120, 140, 18, 18, PARK_GRASS),
            (145, 150, 15, 20, PARK_GRASS),
            (150, 160, 12, 18, WATER_PUDDLE),
        ]
        for r, c, h, w, t in hazard_zones:
            if r+h <= self.size and c+w <= self.size:
                for i in range(r, r+h):
                    for j in range(c, c+w):
                        if self.grid[i, j] not in [BUILDING, HIGHWAY, ROAD, CROSSWALK]:
                            self.grid[i, j] = t

    # ------------------------------------------------------------------
    # Detail / noise layers
    # ------------------------------------------------------------------
    def _add_pavement_details(self):
        degraded = [
            (35,8,8,10),(55,180,10,8),(100,15,8,12),
            (120,70,10,8),(160,40,8,10),(170,130,10,8),
        ]
        for r, c, h, w in degraded:
            if r+h <= self.size and c+w <= self.size:
                for i in range(r, r+h):
                    for j in range(c, c+w):
                        if self.grid[i, j] == GOOD_PAVEMENT:
                            self.grid[i, j] = DEGRADED_PAVEMENT

    def _add_water_puddles(self):
        spots = [
            (33,50,3,3),(62,100,2,3),(97,65,3,2),(133,40,2,3),
            (45,170,3,2),(88,30,2,3),(125,160,3,3),(170,80,2,3),
            (48,110,2,3),(73,140,3,2),
        ]
        for r, c, h, w in spots:
            if r+h <= self.size and c+w <= self.size:
                for i in range(r, r+h):
                    for j in range(c, c+w):
                        if self.grid[i, j] in [GOOD_PAVEMENT, DEGRADED_PAVEMENT,
                                                GOOD_PED_STREET, SOILY_PED_STREET, 0]:
                            self.grid[i, j] = WATER_PUDDLE

    def _add_clutter(self):
        spots = [
            (36,29),(36,64),(82,130),(83,131),(111,64),(112,65),
            (166,64),(167,65),(42,130),(43,131),(100,190),(101,191),
        ]
        for r, c in spots:
            if r < self.size and c < self.size:
                if self.grid[r, c] in [GOOD_PAVEMENT, DEGRADED_PAVEMENT]:
                    self.grid[r, c] = STREET_CLUTTER

    def _add_construction(self):
        zones = [
            (83,95,5,6),(83,170,5,6),
            (148,95,5,6),(148,170,5,6),
            (55,35,6,8),(120,45,5,6),(60,160,6,8),(125,155,5,6),
        ]
        for r, c, h, w in zones:
            if r+h <= self.size and c+w <= self.size:
                for i in range(r, r+h):
                    for j in range(c, c+w):
                        if self.grid[i, j] not in [BUILDING, ROAD, HIGHWAY, CROSSWALK]:
                            self.grid[i, j] = CONSTRUCTION

    def _add_stairs(self):
        zones = [
            (46,12,4,5),(70,185,4,4),(115,20,4,5),
            (160,190,4,4),(30,105,4,5),(135,105,4,5),
        ]
        for r, c, h, w in zones:
            if r+h <= self.size and c+w <= self.size:
                for i in range(r, r+h):
                    for j in range(c, c+w):
                        if self.grid[i, j] not in [BUILDING, ROAD, HIGHWAY]:
                            self.grid[i, j] = STAIRS

    # ------------------------------------------------------------------
    # Rendering helper
    # ------------------------------------------------------------------
    def create_rgb_grid(self) -> np.ndarray:
        """Return an (H, W, 3) float array of terrain colours."""
        rgb = np.zeros((self.size, self.size, 3))
        for t, hex_col in COLOR_SCHEME.items():
            rgb[self.grid == t] = mcolors.hex2color(hex_col)
        return rgb
