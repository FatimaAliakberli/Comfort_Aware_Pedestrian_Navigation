"""
=============================================================================
formulas.py  —  HFL-RL core formula functions (Formulas 1–5)
=============================================================================
Formula 1 : compute_lambda      — λ(N_eff, σ)
Formula 2 : compute_w_edge      — per-edge experience weight w_edge
Formula 3 : compute_W_global    — global experience manifold W_global
Formula 4 : navigation_cost     — J(u→v) = λ·D + (1−λ)·W
Formula 5 : astar               — A* navigation using Formula 4 costs
=============================================================================
"""

import heapq
import numpy as np

from city_map.city_map import BASE_COMFORT, WEATHER_MODIFIERS, BUILDING, cell_cost


# =============================================================================
# Formula 1 — λ: experience-to-distance trade-off weight
# =============================================================================
def compute_lambda(N_eff: int, sigma: float, beta: float = 10.0) -> float:
    """
    λ(N_eff, σ) = 1 − [N_eff / (N_eff + β)] · exp(−σ)

    Parameters
    ----------
    N_eff  : effective number of contributing agents
    sigma  : data quality / confidence score (≥ 0)
    beta   : sensitivity constant (default 10)

    Returns
    -------
    λ ∈ [0, 1]  — closer to 0 = trust experience more; 1 = pure distance
    """
    if N_eff == 0:
        return 1.0
    return 1.0 - (N_eff / (N_eff + beta)) * np.exp(-sigma)


# =============================================================================
# Formula 2 — w_edge: weighted comfort score for a set of edge cells
# =============================================================================
def compute_w_edge(
    agents:       list[dict],
    weather:      str,
    terrain_grid: np.ndarray,
    edge_cells:   list[tuple],
    sudden_bad:   set | None = None,
) -> dict:
    """
    w_edge(C, Φ, e) = Σ_i [α_i · δ_i · w_i(e)] / Σ_i [α_i · δ_i]

    Parameters
    ----------
    agents       : list of agent dicts with keys 'alpha', 'weather'
    weather      : current weather string ('sunny'|'rainy'|'neutral')
    terrain_grid : 2-D int array of terrain IDs
    edge_cells   : list of (row, col) cells for this edge
    sudden_bad   : optional set of cells with temporary hazard

    Returns
    -------
    dict mapping (row, col) → w_edge value
    """
    wmod = WEATHER_MODIFIERS[weather]
    num  = np.zeros(len(edge_cells))
    den  = np.zeros(len(edge_cells))

    for ag in agents:
        alpha = ag['alpha']
        delta = 1.0 if ag.get('weather', weather) == weather else 0.35
        for idx, (r, c) in enumerate(edge_cells):
            t    = terrain_grid[r, c]
            w_i  = BASE_COMFORT[t] * wmod[t]
            if sudden_bad and (r, c) in sudden_bad:
                w_i *= np.random.uniform(2.5, 4.5)
            w_i *= (1.0 + np.random.uniform(-0.08, 0.08))
            num[idx] += alpha * delta * w_i
            den[idx] += alpha * delta

    with np.errstate(divide='ignore', invalid='ignore'):
        w_edge = np.where(
            den > 0,
            num / den,
            np.array([cell_cost(terrain_grid[r, c], weather)
                      for r, c in edge_cells]),
        )
    return dict(zip(edge_cells, w_edge))


# =============================================================================
# Formula 3 — W_global: crowd-weighted global experience manifold
# =============================================================================
def compute_W_global(
    edge_nodes:   list[dict],
    weather:      str,
    terrain_grid: np.ndarray,
    sudden_bad:   set | None = None,
) -> dict:
    """
    W_global(C, Φ, r, c) = Σ_e [p_e · q_e · V_e · w_edge_e(r,c)]
                           / Σ_e [p_e · q_e · V_e]

    Parameters
    ----------
    edge_nodes   : list of edge-node dicts with keys:
                     'cells'  — list of (row, col)
                     'agents' — list of agent dicts
                     'p'      — participation probability
                     'q'      — data quality
                     'V'      — volume / visit count
    weather      : current weather string
    terrain_grid : 2-D int array of terrain IDs
    sudden_bad   : optional set of cells with temporary hazard

    Returns
    -------
    dict mapping (row, col) → global experience cost
    """
    accum: dict[tuple, dict] = {}

    for node in edge_nodes:
        weight     = node['p'] * node['q'] * node['V']
        w_edge_map = compute_w_edge(
            node['agents'], weather, terrain_grid,
            node['cells'], sudden_bad,
        )
        for cell, val in w_edge_map.items():
            if cell not in accum:
                accum[cell] = {'num': 0.0, 'den': 0.0}
            accum[cell]['num'] += weight * val
            accum[cell]['den'] += weight

    W_global = {}
    for cell, vals in accum.items():
        W_global[cell] = (
            vals['num'] / vals['den']
            if vals['den'] > 0
            else cell_cost(terrain_grid[cell], weather)
        )
    return W_global


# =============================================================================
# Formula 4 — J(u→v): navigation cost for one step
# =============================================================================
def navigation_cost(
    v:            tuple,
    lam:          float,
    D_uv:         float,
    W_global:     dict,
    terrain_grid: np.ndarray,
    weather:      str,
) -> float:
    """
    J(u→v) = λ · D(u,v) + (1−λ) · W_global(v)

    Parameters
    ----------
    v            : destination cell (row, col)
    lam          : λ value (Formula 1)
    D_uv         : Euclidean distance of the step
    W_global     : global experience manifold (Formula 3)
    terrain_grid : 2-D int array
    weather      : current weather string

    Returns
    -------
    Scalar navigation cost for the step u→v.
    """
    t     = terrain_grid[v]
    w_val = W_global.get(v, cell_cost(t, weather))
    return lam * D_uv + (1.0 - lam) * w_val


# =============================================================================
# Formula 5 — A* navigation using Formula 4 cost
# =============================================================================
def astar(
    terrain_grid: np.ndarray,
    start:        tuple,
    goal:         tuple,
    W_global:     dict,
    lam:          float,
    weather:      str,
) -> list[tuple]:
    """
    A* search minimising the HFL-RL navigation cost J (Formula 4).

    Parameters
    ----------
    terrain_grid : 2-D int array of terrain IDs
    start        : (row, col) source cell
    goal         : (row, col) target cell
    W_global     : global experience manifold; pass {} for pure shortest-path
    lam          : λ weight (1.0 = pure distance)
    weather      : current weather string

    Returns
    -------
    List of (row, col) cells from start to goal (inclusive), or [] if no path.
    """
    size = terrain_grid.shape[0]

    def h(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    open_set  = [(0.0, start)]
    came_from = {}
    g_score   = {start: 0.0}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return path[::-1]

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),
                       (-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            if terrain_grid[nr, nc] == BUILDING:
                continue

            nb   = (nr, nc)
            dist = np.sqrt(dr**2 + dc**2)
            cost = navigation_cost(nb, lam, dist, W_global, terrain_grid, weather)
            tent = g_score.get(cur, 1e18) + cost

            if tent < g_score.get(nb, 1e18):
                came_from[nb] = cur
                g_score[nb]   = tent
                heapq.heappush(open_set, (tent + h(nb, goal), nb))

    return []
