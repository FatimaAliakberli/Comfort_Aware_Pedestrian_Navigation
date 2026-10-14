"""
=============================================================================
implicit_feedback_v2.py  —  Enhanced Implicit Feedback & Multi-User Simulation
=============================================================================
Conference paper extension: "Preference Inference from Behavioral Traces in
Pedestrian Navigation via Implicit Feedback and Multi-Timescale Learning"

Key contributions over v1:
  1. Richer user pool  — compliant / non-compliant / autonomous (free-roaming)
  2. Cross-user IRL    — if a non-compliant path beats the suggested one,
                          the model updates toward that path (reverse reward)
  3. Per-attempt alternatives — on a bad compliant experience the model
                          probes k alternate routes (shortest-path basis +
                          comfort perturbation) instead of repeating the same one
  4. Autonomous users  — move freely; their trajectories are observed and
                          folded into W_global via Bayesian-style EMA
  5. Temporal gating   — separate short-term (weather/events) and long-term
                          (construction) weight stores; gate updates by
                          confidence and event persistence
  6. Divergence metric — KL-style path divergence logged each round
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from copy import deepcopy
from collections import defaultdict

from city_map.city_map        import CityMap200, BUILDING, cell_cost, BASE_COMFORT
from formulas.formulas        import compute_lambda, astar
from scenarios.scenario_builder import build_scenario, analyze_path

# ── Fixed endpoints ───────────────────────────────────────────────────────────
START = (0, 0)
GOAL  = (185, 190)

# ── EMA decay rates ───────────────────────────────────────────────────────────
EMA_SHORT      = 0.40   # fast decay — weather / sudden events
EMA_LONG       = 0.05   # slow decay — construction / persistent features
EMA_AUTONOMOUS = 0.20   # moderate — free-roaming user traces

# ── Reward / threshold constants ─────────────────────────────────────────────
BAD_EXPERIENCE_THRESH = 0.10   # relative improvement needed to count as "better"
ALT_ROUTE_CANDIDATES  = 3      # number of alternative routes to probe on bad exp.
ALT_NOISE_STD         = 0.25   # std of comfort-weight noise for alt-route probing

# ── User type labels ──────────────────────────────────────────────────────────
COMPLIANT     = "compliant"
NON_COMPLIANT = "non_compliant"
AUTONOMOUS    = "autonomous"


# =============================================================================
#  User record
# =============================================================================
class User:
    """
    Represents a single simulated pedestrian.

    Attributes
    ----------
    uid          : unique integer id
    user_type    : COMPLIANT | NON_COMPLIANT | AUTONOMOUS
    alpha        : trust/experience weight (used in W_global updates)
    satisfaction : running EMA of satisfaction scores ∈ [0, 1]
    history      : list of (round, path, avg_W, total_J) records
    """

    def __init__(self, uid: int, user_type: str, alpha: float = 0.7):
        self.uid          = uid
        self.user_type    = user_type
        self.alpha        = alpha
        self.satisfaction = 0.5          # neutral prior
        self.history: list[dict] = []

    def record(self, round_no: int, path: list, stats: dict):
        self.history.append({
            'round':   round_no,
            'path':    path,
            'avg_W':   stats.get('avg_W', 1.0),
            'total_J': stats.get('total_J', float('inf')),
            'length':  stats.get('path_length', 0),
        })
        # Update satisfaction via EMA (lower cost → higher satisfaction)
        raw_sat = 1.0 / (1.0 + stats.get('avg_W', 1.0))
        self.satisfaction = (1 - EMA_SHORT) * self.satisfaction + EMA_SHORT * raw_sat

    def __repr__(self):
        return f"User(id={self.uid}, type={self.user_type}, sat={self.satisfaction:.3f})"


# =============================================================================
#  Temporal Weight Store
# =============================================================================
class TemporalWeightStore:
    """
    Maintains two weight layers:
      W_short  — fast-adapting (weather, sudden events)
      W_long   — slow-adapting (construction, permanent features)

    The effective weight is a convex combination gated by confidence:
      W_eff(cell) = γ · W_short(cell) + (1 − γ) · W_long(cell)

    γ is driven by `event_confidence`: how confident we are that the
    current conditions are SHORT-term (γ high → trust short-term more).
    """

    def __init__(self, base_W: dict, event_confidence: float = 0.5):
        self.W_short = deepcopy(base_W)
        self.W_long  = deepcopy(base_W)
        self.gamma   = event_confidence   # 0 = trust long-term, 1 = trust short-term

    def effective(self) -> dict:
        """Return gated composite weight dict."""
        cells = set(self.W_short) | set(self.W_long)
        return {
            cell: self.gamma * self.W_short.get(cell, 1.0)
                  + (1 - self.gamma) * self.W_long.get(cell, 1.0)
            for cell in cells
        }

    def update_short(self, cell: tuple, new_val: float):
        self.W_short[cell] = (1 - EMA_SHORT) * self.W_short.get(cell, new_val) \
                             + EMA_SHORT * new_val

    def update_long(self, cell: tuple, new_val: float):
        self.W_long[cell]  = (1 - EMA_LONG)  * self.W_long.get(cell,  new_val) \
                             + EMA_LONG  * new_val

    def bulk_update_from_path(self, path: list, terrain: np.ndarray,
                               weather: str, timescale: str = 'short'):
        """Update all cells along a path."""
        for cell in path:
            base = cell_cost(terrain[cell], weather)
            if timescale == 'short':
                self.update_short(cell, base)
            else:
                self.update_long(cell, base)

    def persist_construction(self, construction_cells: set,
                              terrain: np.ndarray, weather: str):
        """Fold construction-zone costs into long-term store."""
        for cell in construction_cells:
            base = cell_cost(terrain[cell], weather)
            self.update_long(cell, base * 1.5)   # persistent penalty


# =============================================================================
#  Alternative route probe
# =============================================================================
def _probe_alternative_routes(
        terrain: np.ndarray,
        W_eff:   dict,
        lam:     float,
        weather: str,
        k:       int = ALT_ROUTE_CANDIDATES,
        noise_std: float = ALT_NOISE_STD,
) -> list[list[tuple]]:
    """
    Generate k alternative routes by perturbing the experience manifold with
    Gaussian noise and re-running A*.  The baseline shortest path (W={}) is
    always included as the first candidate.

    This implements the "probe alternatives on bad experience" requirement:
    keep the shortest-path basis but explore comfort-adjusted variants.
    """
    candidates = []

    # Candidate 0 — pure shortest path
    sp = astar(terrain, START, GOAL, {}, 1.0, weather)
    if sp:
        candidates.append(sp)

    # Candidates 1..k-1 — perturbed manifold
    for _ in range(k - 1):
        W_perturbed = {
            cell: max(0.1, val * (1 + np.random.normal(0, noise_std)))
            for cell, val in W_eff.items()
        }
        path = astar(terrain, START, GOAL, W_perturbed, lam, weather)
        if path and path not in candidates:
            candidates.append(path)

    return candidates


# =============================================================================
#  Path divergence metric (KL-style)
# =============================================================================
def path_divergence(path_a: list, path_b: list) -> float:
    """
    Fraction of cells in path_a that do NOT appear in path_b.
    Returns a value in [0, 1]; 0 = identical coverage, 1 = fully disjoint.
    """
    if not path_a:
        return 0.0
    set_b = set(path_b)
    return sum(1 for c in path_a if c not in set_b) / len(path_a)


# =============================================================================
#  Main simulation class
# =============================================================================
class ImplicitFeedbackSimulationV2:
    """
    Enhanced multi-user implicit feedback simulation.

    User composition (configurable via ratios):
      - Compliant     : follow system suggestions; bad experiences trigger
                        alternative probing
      - Non-compliant : ignore suggestions, find own path; good non-compliant
                        paths are learned from (IRL cross-user update)
      - Autonomous    : purely free-roaming; their traces update W_global
                        passively via EMA

    The model maintains:
      - A TemporalWeightStore for multi-timescale adaptation
      - Per-round logs for analysis and paper figures
    """

    def __init__(
        self,
        city:              CityMap200,
        scenario_type:     str   = 'sunny',
        weather:           str   = 'neutral',
        n_agents:          int   = 20,
        compliant_ratio:   float = 0.40,
        noncompliant_ratio: float = 0.35,
        # autonomous makes up the remainder
        event_confidence:  float = 0.5,
        seed:              int   = 42,
    ):
        np.random.seed(seed)
        self.city    = city
        self.weather = weather

        # Build base scenario
        _, self.lam, self.N_eff, self.sigma, W_base, self.sudden_bad = build_scenario(
            scenario_type, n_agents, city, weather
        )

        # Temporal weight store
        self.tws = TemporalWeightStore(W_base, event_confidence)

        # Pre-compute construction cells for long-term persistence
        from city_map.city_map import CONSTRUCTION
        construction_cells = set(
            (r, c)
            for r in range(city.size)
            for c in range(city.size)
            if city.grid[r, c] == CONSTRUCTION
        )
        self.tws.persist_construction(construction_cells, city.grid, weather)

        # Build user pool
        n_compliant    = int(n_agents * compliant_ratio)
        n_noncompliant = int(n_agents * noncompliant_ratio)
        n_autonomous   = n_agents - n_compliant - n_noncompliant

        self.users: list[User] = []
        uid = 0
        for _ in range(n_compliant):
            self.users.append(User(uid, COMPLIANT,
                                   alpha=np.random.uniform(0.6, 1.0)))
            uid += 1
        for _ in range(n_noncompliant):
            self.users.append(User(uid, NON_COMPLIANT,
                                   alpha=np.random.uniform(0.3, 0.7)))
            uid += 1
        for _ in range(n_autonomous):
            self.users.append(User(uid, AUTONOMOUS,
                                   alpha=np.random.uniform(0.4, 0.8)))
            uid += 1

        self.round_logs: list[dict] = []
        self._best_known_cost = float('inf')

        print(f"\n[SimV2] Users: {n_compliant} compliant | "
              f"{n_noncompliant} non-compliant | {n_autonomous} autonomous")
        print(f"        λ={self.lam:.3f}  σ={self.sigma:.3f}  "
              f"weather={weather}  scenario={scenario_type}\n")

    # ------------------------------------------------------------------
    def run_simulation(self, total_rounds: int = 8) -> list[dict]:
        """
        Execute multi-round simulation.

        Each round:
          1. Compute effective W from temporal store
          2. Generate suggested path (HFL-RL)
          3. Each user type acts:
               Compliant    → follows suggestion; bad exp. → try alternatives
               Non-compliant→ finds own path; if better → model updates via IRL
               Autonomous   → random waypoint walk; trace folds into W
          4. Aggregate feedback, update temporal store
          5. Log round metrics
        """
        print(f"[SimV2] Running {total_rounds} rounds...\n")

        for r in range(1, total_rounds + 1):
            W_eff = self.tws.effective()

            # ── Suggested path (model recommendation) ─────────────────
            suggested = astar(self.city.grid, START, GOAL, W_eff, self.lam, self.weather)
            if not suggested:
                print(f"  Round {r}: A* failed — skipping")
                continue
            s_stats = analyze_path(suggested, self.city.grid, W_eff, self.lam, self.weather)

            # Update best known
            if s_stats['total_J'] < self._best_known_cost:
                self._best_known_cost = s_stats['total_J']

            # ── Per-user actions ───────────────────────────────────────
            compliant_results    = []
            noncompliant_results = []
            autonomous_traces    = []
            irl_updates          = []

            for user in self.users:

                # ── COMPLIANT ──────────────────────────────────────────
                if user.user_type == COMPLIANT:
                    path   = suggested
                    stats  = s_stats
                    user.record(r, path, stats)
                    compliant_results.append((user, path, stats))

                    # Bad experience? probe alternatives
                    if len(user.history) >= 2:
                        prev_J = user.history[-2]['total_J']
                        curr_J = stats['total_J']
                        if curr_J > prev_J * (1 - BAD_EXPERIENCE_THRESH):
                            # Model tests alternatives for this user next round
                            alts = _probe_alternative_routes(
                                self.city.grid, W_eff, self.lam, self.weather
                            )
                            best_alt    = None
                            best_alt_J  = curr_J
                            for alt in alts:
                                a_stats = analyze_path(alt, self.city.grid, W_eff,
                                                       self.lam, self.weather)
                                if a_stats['total_J'] < best_alt_J:
                                    best_alt   = alt
                                    best_alt_J = a_stats['total_J']
                            # If an alternative is strictly better, fold it in
                            if best_alt is not None:
                                self.tws.bulk_update_from_path(
                                    best_alt, self.city.grid,
                                    self.weather, timescale='short'
                                )

                # ── NON-COMPLIANT ─────────────────────────────────────
                elif user.user_type == NON_COMPLIANT:
                    # Ignore suggestion — use own (noise-perturbed) W
                    own_W = {
                        cell: max(0.1, val * (1 + np.random.normal(0, 0.3)))
                        for cell, val in W_eff.items()
                    }
                    own_lam = max(0.1, self.lam + np.random.uniform(-0.2, 0.2))
                    nc_path = astar(self.city.grid, START, GOAL, own_W, own_lam, self.weather)
                    if not nc_path:
                        nc_path = suggested
                    nc_stats = analyze_path(nc_path, self.city.grid, W_eff,
                                            self.lam, self.weather)
                    user.record(r, nc_path, nc_stats)
                    noncompliant_results.append((user, nc_path, nc_stats))

                    # IRL cross-user learning:
                    # If non-compliant path is better than suggestion → update model
                    improvement = (s_stats['total_J'] - nc_stats['total_J']) \
                                  / (s_stats['total_J'] + 1e-9)
                    if improvement > BAD_EXPERIENCE_THRESH:
                        irl_updates.append((user, nc_path, improvement))
                        for cell in nc_path:
                            new_val = cell_cost(self.city.grid[cell], self.weather)
                            # Weight update proportional to improvement
                            blend   = min(EMA_SHORT, improvement * 0.5)
                            self.tws.update_short(cell, new_val * (1 - blend)
                                                   + W_eff.get(cell, new_val) * blend)

                # ── AUTONOMOUS ────────────────────────────────────────
                elif user.user_type == AUTONOMOUS:
                    # Simulated free roam: random intermediate waypoints
                    auto_path = _autonomous_walk(self.city.grid, self.weather)
                    if auto_path:
                        au_stats = analyze_path(auto_path, self.city.grid, W_eff,
                                                self.lam, self.weather)
                        user.record(r, auto_path, au_stats)
                        autonomous_traces.append((user, auto_path, au_stats))
                        # Passive EMA update from observed trajectory
                        for cell in auto_path:
                            new_val = cell_cost(self.city.grid[cell], self.weather)
                            self.tws.W_short[cell] = (
                                (1 - EMA_AUTONOMOUS) * self.tws.W_short.get(cell, new_val)
                                + EMA_AUTONOMOUS * new_val
                            )

            # ── Aggregate round log ────────────────────────────────────
            rlog = _build_round_log(
                r, self.lam, self.sigma,
                suggested, s_stats,
                compliant_results,
                noncompliant_results,
                autonomous_traces,
                irl_updates,
                W_eff,
            )
            self.round_logs.append(rlog)
            _print_round(rlog)

        _print_summary(self.round_logs)
        return self.round_logs


# =============================================================================
#  Autonomous walk helper
# =============================================================================
def _autonomous_walk(terrain: np.ndarray, weather: str) -> list[tuple]:
    """
    Simulate a free-roaming user: pick 1–3 random intermediate waypoints
    between START and GOAL, then chain A* sub-paths.
    No experience manifold — just comfort-cost A*.
    """
    size = terrain.shape[0]
    n_wp = np.random.randint(1, 4)
    waypoints = [START]
    for _ in range(n_wp):
        while True:
            r = np.random.randint(0, size)
            c = np.random.randint(0, size)
            if terrain[r, c] != BUILDING:
                waypoints.append((r, c))
                break
    waypoints.append(GOAL)

    full_path = []
    W_comfort = {
        (r, c): cell_cost(terrain[r, c], weather)
        for r in range(size)
        for c in range(size)
        if terrain[r, c] != BUILDING
    }
    for i in range(len(waypoints) - 1):
        seg = astar(terrain, waypoints[i], waypoints[i+1], W_comfort, 0.5, weather)
        if not seg:
            return []
        if full_path and seg[0] == full_path[-1]:
            full_path.extend(seg[1:])
        else:
            full_path.extend(seg)

    return full_path


# =============================================================================
#  Logging helpers
# =============================================================================
def _build_round_log(
    r, lam, sigma,
    suggested, s_stats,
    compliant_results, noncompliant_results, autonomous_traces,
    irl_updates, W_eff,
) -> dict:

    def _avg_J(results):
        vals = [st['total_J'] for _, _, st in results if st]
        return float(np.mean(vals)) if vals else float('nan')

    def _avg_W(results):
        vals = [st['avg_W'] for _, _, st in results if st]
        return float(np.mean(vals)) if vals else float('nan')

    def _avg_div(results, ref):
        vals = [path_divergence(p, ref) for _, p, _ in results if p]
        return float(np.mean(vals)) if vals else float('nan')

    nc_paths = [p for _, p, _ in noncompliant_results]
    au_paths = [p for _, p, _ in autonomous_traces]

    return {
        'round':              r,
        'lambda':             lam,
        'sigma':              sigma,
        # Suggested / model
        'suggested_len':      len(suggested),
        'suggested_avgW':     s_stats.get('avg_W', 0.0),
        'suggested_totalJ':   s_stats.get('total_J', 0.0),
        # Compliant
        'n_compliant':        len(compliant_results),
        'compliant_avgJ':     _avg_J(compliant_results),
        'compliant_avgW':     _avg_W(compliant_results),
        # Non-compliant
        'n_noncompliant':     len(noncompliant_results),
        'noncompliant_avgJ':  _avg_J(noncompliant_results),
        'noncompliant_avgW':  _avg_W(noncompliant_results),
        'noncompliant_div':   _avg_div(noncompliant_results, suggested),
        # Autonomous
        'n_autonomous':       len(autonomous_traces),
        'autonomous_avgJ':    _avg_J(autonomous_traces),
        'autonomous_avgW':    _avg_W(autonomous_traces),
        # IRL
        'irl_updates':        len(irl_updates),
        'irl_avg_improvement': float(np.mean([imp for _, _, imp in irl_updates]))
                               if irl_updates else 0.0,
        # Paths for visualisation
        'suggested_path':     suggested,
        'nc_paths':           nc_paths,
        'au_paths':           au_paths,
        'W_eff':              deepcopy(W_eff),
    }


def _print_round(rlog: dict):
    r = rlog['round']
    print(f"  ── Round {r} {'─'*48}")
    print(f"     λ={rlog['lambda']:.3f}  σ={rlog['sigma']:.3f}")
    print(f"     Suggested : len={rlog['suggested_len']:4d} | "
          f"avg_W={rlog['suggested_avgW']:.3f} | total_J={rlog['suggested_totalJ']:.1f}")
    print(f"     Compliant ({rlog['n_compliant']:2d} users) : "
          f"avg_J={rlog['compliant_avgJ']:.1f}  avg_W={rlog['compliant_avgW']:.3f}")
    print(f"     Non-compliant ({rlog['n_noncompliant']:2d} users) : "
          f"avg_J={rlog['noncompliant_avgJ']:.1f}  avg_W={rlog['noncompliant_avgW']:.3f}  "
          f"div={rlog['noncompliant_div']:.3f}")
    print(f"     Autonomous ({rlog['n_autonomous']:2d} users) : "
          f"avg_J={rlog['autonomous_avgJ']:.1f}  avg_W={rlog['autonomous_avgW']:.3f}")
    if rlog['irl_updates'] > 0:
        print(f"     ✦ IRL updates: {rlog['irl_updates']} paths  "
              f"(avg improvement {rlog['irl_avg_improvement']:.3f})")


def _print_summary(logs: list[dict]):
    print("\n" + "═" * 65)
    print("  SIMULATION SUMMARY")
    print("═" * 65)
    print(f"  {'Round':<6} {'Sugg J':>8} {'Compl J':>9} {'NC J':>9} "
          f"{'Auto J':>9} {'IRL':>5} {'Div':>7}")
    print(f"  {'─'*6} {'─'*8} {'─'*9} {'─'*9} {'─'*9} {'─'*5} {'─'*7}")
    for lg in logs:
        print(f"  {lg['round']:<6} "
              f"{lg['suggested_totalJ']:>8.1f} "
              f"{lg['compliant_avgJ']:>9.1f} "
              f"{lg['noncompliant_avgJ']:>9.1f} "
              f"{lg['autonomous_avgJ']:>9.1f} "
              f"{lg['irl_updates']:>5} "
              f"{lg['noncompliant_div']:>7.3f}")
    # trend check
    if len(logs) >= 3:
        first_J = logs[0]['suggested_totalJ']
        last_J  = logs[-1]['suggested_totalJ']
        delta   = (first_J - last_J) / (first_J + 1e-9) * 100
        verdict = "✓ Model improving" if delta > 0 else "✗ Model not converging"
        print(f"\n  Suggested path cost change: {delta:+.1f}%  → {verdict}")
    print("═" * 65 + "\n")


# =============================================================================
#  Entry point
# =============================================================================
if __name__ == '__main__':
    city = CityMap200(seed=42)

    for scenario_type, weather, label in [
        ('sunny',         'sunny',   'Sunny / High data'),
        ('rainy',         'rainy',   'Rainy / Comfort matters'),
        ('sudden_change', 'neutral', 'Sudden Change / IRL stress test'),
        ('cold_start',    'neutral', 'Cold Start / Sparse data'),
    ]:
        print("\n" + "╔" + "═"*63 + "╗")
        print(f"  SCENARIO: {label}")
        print("╚" + "═"*63 + "╝")

        sim = ImplicitFeedbackSimulationV2(
            city               = city,
            scenario_type      = scenario_type,
            weather            = weather,
            n_agents           = 20,
            compliant_ratio    = 0.40,   # 8 users
            noncompliant_ratio = 0.35,   # 7 users
            # autonomous         = 0.25  # 5 users
            event_confidence   = 0.6 if weather in ('sunny', 'rainy') else 0.4,
        )
        logs = sim.run_simulation(total_rounds=8)