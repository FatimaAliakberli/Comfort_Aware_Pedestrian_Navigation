"""
=============================================================================
scenario_builder.py  —  Scenario construction and path analysis utilities
=============================================================================
build_scenario  : returns edge_nodes, λ, N_eff, σ, W_global, sudden_bad
analyze_path    : evaluates any path with a given λ (fair comparison)
=============================================================================
"""

import numpy as np

from city_map.city_map import CityMap200, TERRAIN_NAMES, cell_cost
from formulas.formulas import compute_lambda, compute_W_global, navigation_cost


# =============================================================================
# Scenario builder
# =============================================================================
def build_scenario(
    scenario_type: str,
    n_agents:      int,
    city:          CityMap200,
    weather:       str = 'neutral',
) -> tuple:
    """
    Construct edge nodes, compute λ and W_global for a given scenario.

    Parameters
    ----------
    scenario_type : 'cold_start' | 'sudden_change' | 'sunny' | 'rainy'
    n_agents      : number of agents in the scenario
    city          : CityMap200 instance
    weather       : weather condition string

    Returns
    -------
    (edge_nodes, lam, N_eff, sigma, W_global, sudden_bad_cells)
    """
    size    = city.size
    terrain = city.grid
    stride  = 3          # sample every 3rd cell to keep lists manageable

    # Four quadrant zones (each 100×100)
    zones = [
        (0,   100, 0,   100),
        (0,   100, 100, 200),
        (100, 200, 0,   100),
        (100, 200, 100, 200),
    ]
    edge_nodes:       list[dict] = []
    sudden_bad_cells: set        = set()

    # ------------------------------------------------------------------
    if scenario_type == 'cold_start':
        N_eff, sigma = 0, 1.0
        lam = compute_lambda(N_eff, sigma)
        for r0, r1, c0, c1 in zones:
            cells  = [(r, c) for r in range(r0, r1, stride)
                              for c in range(c0, c1, stride)]
            agents = [{'alpha': 0.1, 'weather': weather}
                      for _ in range(max(1, n_agents // 4))]
            edge_nodes.append({'cells': cells, 'agents': agents,
                                'p': 0.01, 'q': 0.10, 'V': 0.1})

    # ------------------------------------------------------------------
    elif scenario_type == 'sudden_change':
        np.random.seed(99)
        for _ in range(250):
            r, c = np.random.randint(40, 160), np.random.randint(40, 160)
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < size and 0 <= nc < size:
                        sudden_bad_cells.add((nr, nc))
        N_eff = n_agents * 3
        sigma = 1.2
        lam   = compute_lambda(N_eff, sigma)
        np.random.seed(None)
        for r0, r1, c0, c1 in zones:
            cells  = [(r, c) for r in range(r0, r1, stride)
                              for c in range(c0, c1, stride)]
            agents = [{'alpha': np.random.uniform(0.6, 0.9), 'weather': weather}
                      for _ in range(max(1, n_agents // 4))]
            edge_nodes.append({'cells': cells, 'agents': agents,
                                'p': 0.60, 'q': 0.70, 'V': 5.0})

    # ------------------------------------------------------------------
    elif scenario_type in ('sunny', 'rainy'):
        N_eff = n_agents * 5
        sigma = 0.3 if scenario_type == 'sunny' else 0.6
        lam   = compute_lambda(N_eff, sigma)
        qual  = 0.9 if scenario_type == 'sunny' else 0.75
        for r0, r1, c0, c1 in zones:
            cells  = [(r, c) for r in range(r0, r1, stride)
                              for c in range(c0, c1, stride)]
            agents = [{'alpha': np.random.uniform(0.7, 1.0), 'weather': scenario_type}
                      for _ in range(max(1, n_agents // 4))]
            edge_nodes.append({'cells': cells, 'agents': agents,
                                'p': 0.70, 'q': qual, 'V': 1.0})

    else:
        raise ValueError(f"Unknown scenario_type: '{scenario_type}'")

    W_global = compute_W_global(
        edge_nodes, weather, terrain,
        sudden_bad=sudden_bad_cells or None,
    )
    return edge_nodes, lam, N_eff, sigma, W_global, sudden_bad_cells


# =============================================================================
# Path analyzer — fair evaluation (both paths use the same λ)
# =============================================================================
def analyze_path(
    path:         list[tuple],
    terrain_grid: np.ndarray,
    W_global:     dict,
    lam:          float,
    weather:      str,
) -> dict:
    """
    Evaluate a path with a given λ so that HFL-RL and SP are fairly compared.

    Parameters
    ----------
    path         : list of (row, col) cells
    terrain_grid : 2-D int array
    W_global     : global experience manifold
    lam          : λ weight to apply
    weather      : current weather string

    Returns
    -------
    dict with keys: total_J, total_D, avg_W, path_length, terrain_counts
    """
    if not path:
        return {}

    total_J = total_D = total_W = 0.0
    ter_counts: dict[str, int] = {}

    for i in range(len(path) - 1):
        u, v  = path[i], path[i+1]
        D_uv  = np.sqrt((v[0]-u[0])**2 + (v[1]-u[1])**2)
        t     = terrain_grid[v]
        w_val = W_global.get(v, cell_cost(t, weather))

        total_J += lam * D_uv + (1 - lam) * w_val
        total_D += D_uv
        total_W += w_val

        name = TERRAIN_NAMES.get(t, str(t))
        ter_counts[name] = ter_counts.get(name, 0) + 1

    ns = max(len(path) - 1, 1)
    return {
        'total_J':        total_J,
        'total_D':        total_D,
        'avg_W':          total_W / ns,
        'path_length':    len(path),
        'terrain_counts': ter_counts,
    }
