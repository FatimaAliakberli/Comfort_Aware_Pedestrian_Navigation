#!/usr/bin/env python3
"""
=============================================================================
hfl_rl_osm_baku.py  --  HFL-RL comfort-aware pedestrian routing on a real
                        OpenStreetMap network (Baku, Azerbaijan)
=============================================================================
Pipeline
  1. OSM extraction  : osmnx walk network -> simplified -> UTM -> undirected nx.Graph
  2. Edge weights    : OSM tags -> paper terrain class -> base_cost, rain_multiplier
  3. Hazards         : hidden hazards injected on 5-10 edges (ground truth only)
  4. Users           : N synthetic users with alpha_i, delta_i, noisy perception,
                       "turn back at point C" behaviour on hazard edges
  5. HFL-RL          : Stage 1 (w_local) -> Stage 2 (w_edge per zone)
                       -> Stage 3 (W_global with p, q, V) -> Stage 4 (lambda gate)
                       -> Stage 5 (normalised J(u,v)) -> routing
  6. Evaluation      : Shortest Path vs Centralized vs HFL-RL, several routes x seeds

Usage
  python hfl_rl_osm_baku.py                 # real Baku OSM data (needs internet)
  python hfl_rl_osm_baku.py --demo          # offline synthetic graph (smoke test)
  python hfl_rl_osm_baku.py --users 50 --seeds 5 --hazards 8 --weather rainy
=============================================================================
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field, replace
from itertools import islice

import networkx as nx
import numpy as np
import pandas as pd

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
@dataclass
class Config:
    # ---- Study area (verify coordinates on openstreetmap.org, see README notes)
    place: str | None = None                  # e.g. "Icherisheher, Baku, Azerbaijan"
    center: tuple = (40.3705, 49.8365)        # (lat, lon) used when place is None
    radius_m: int = 1500                      # bbox half-size around center
    origin: tuple = (40.3726, 49.8355)        # A: ~Fountain Square (lat, lon)
    dest: tuple = (40.3667, 49.8372)          # B: ~Maiden Tower  (lat, lon)
    n_extra_routes: int = 2                   # additional random O-D pairs
    min_od_m: float = 700.0                   # min straight-line length of extra routes

    # ---- Simulation
    n_users: int = 50
    n_seeds: int = 5
    weather: str = "rainy"                    # 'dry' | 'rainy'
    n_zones: int = 3                          # edge nodes (logical neighbourhoods)
    seed: int = 42

    # ---- Hidden hazards
    n_hazards: int = 8
    hazard_onset: int = 10                    # hazards appear at this user index
    hazard_placement: str = "corridor"        # 'corridor' | 'random'
    corridor_k: int = 8                       # k-shortest paths defining corridor
    hazard_base: float = 8.0                  # 'Construction Zone' base cost
    hazard_rain: float = 3.0                  # ...and its rain multiplier

    # ---- User behaviour
    report_prob: float = 0.8                  # P(user reports a traversed edge)
    p_context_match: float = 0.8              # P(delta_i = 1)
    delta_mismatch: float = 0.35              # delta_i when context differs
    partial_walk_frac: float = 0.25           # fraction of hazard edge walked before turning back
    max_replans: int = 30

    # ---- HFL-RL hyper-parameters
    beta: float = 10.0                        # Stage 4 sensitivity
    kappa: float = 5.0                        # Stage 3 volatility gain
    vol_ema: float = 0.3                      # EMA rate of volatility index
    gamma: float = 0.97                       # forgetting factor per user
    reset: float = 0.1                        # change-point reset on turn-back
    turnback_mult: float = 20.0               # a turn-back counts as m raw reports (explicit rejection)
    prior_neff: float = 5.0                   # OSM-tag prior as pseudo-observations on unseen edges;
                                              # 0 = strict cold start (lambda=1 on unseen edges)
    prior_sigma: float = 0.5
    length_scaled: bool = True                # comfort term scaled by D/Dmax (exposure);
                                              # False = literal J = lam*D^ + (1-lam)*W^

    outdir: str = "hfl_rl_baku_results"


# =============================================================================
# 1. TERRAIN CLASSES  (paper Table I)  -> (base_cost, rain_multiplier)
# =============================================================================
TERRAIN = {
    "Good Pavement":     (1.0, 1.3),
    "Degraded Pavement": (2.0, 2.0),
    "Pavement (Cars)":   (2.5, 1.8),
    "Park Path (Paved)": (1.2, 1.5),
    "Park Path (Dirt)":  (2.8, 3.5),
    "Pedestrian Street": (1.1, 1.3),
    "Ped. Area (Dirt)":  (2.6, 3.0),
    "Road":              (4.0, 1.5),
    "Highway":           (6.0, 2.0),
    "Crosswalk":         (1.5, 1.8),
    "Park Grass":        (3.0, 3.5),
    "Water Puddle":      (3.5, 5.0),
    "Street Clutter":    (3.0, 1.5),
    "Construction Zone": (8.0, 3.0),
    "Stairs":            (5.0, 2.5),
}

PAVED = {"asphalt", "paved", "concrete", "concrete:plates", "concrete:lanes",
         "paving_stones", "sett", "cobblestone", "bricks", "brick", "metal",
         "wood", "tartan", "unhewn_cobblestone"}
UNPAVED = {"unpaved", "dirt", "gravel", "fine_gravel", "ground", "grass", "sand",
           "earth", "mud", "compacted", "pebblestone", "woodchips", "grass_paver"}
BAD_SMOOTH = {"bad", "very_bad", "horrible", "very_horrible", "impassable"}


def _as_list(v) -> list:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def _classify_single(hw: str, surfaces: list, smooth: list, d: dict) -> str:
    """Map ONE highway value (+ surface/smoothness) to a paper terrain class."""
    unpaved = any(s in UNPAVED for s in surfaces)
    paved = any(s in PAVED for s in surfaces)
    bad = unpaved or any(s in BAD_SMOOTH for s in smooth)

    if hw == "construction" or d.get("construction") or d.get("highway") == "proposed":
        return "Construction Zone"
    if hw == "steps":
        return "Stairs"
    if hw in ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"):
        return "Highway"
    if hw in ("secondary", "secondary_link", "tertiary", "tertiary_link"):
        return "Road"
    if hw in ("residential", "unclassified", "service", "road"):
        return "Pavement (Cars)"                       # shared street, cars present
    if hw == "living_street":
        return "Ped. Area (Dirt)" if unpaved else "Pedestrian Street"
    if hw == "pedestrian":
        return "Ped. Area (Dirt)" if unpaved else "Pedestrian Street"
    if hw in ("footway", "sidewalk", "crossing", "corridor", "cycleway"):
        if hw == "crossing" or d.get("footway") == "crossing" or d.get("crossing"):
            return "Crosswalk"
        return "Degraded Pavement" if bad else "Good Pavement"
    if hw in ("path", "track", "bridleway"):
        if unpaved:
            return "Park Path (Dirt)"
        return "Park Path (Paved)" if paved else "Degraded Pavement"   # unknown surface: cautious
    return "Pavement (Cars)"


def classify_edge(d: dict) -> tuple[str, float, float]:
    """OSM edge attributes -> (terrain_name, base_cost, rain_multiplier).
    If an edge carries several highway values, the WORST class is used."""
    hws = _as_list(d.get("highway")) or ["unknown"]
    surf = [str(s) for s in _as_list(d.get("surface"))]
    smooth = [str(s) for s in _as_list(d.get("smoothness"))]
    names = [_classify_single(str(h), surf, smooth, d) for h in hws]
    name = max(names, key=lambda n: TERRAIN[n][0])
    return name, *TERRAIN[name]


# =============================================================================
# 2. GRAPH EXTRACTION  (OSM -> projected undirected simple graph)
# =============================================================================
TAG_KEYS = ("highway", "surface", "smoothness", "footway", "crossing",
            "construction", "name", "geometry")


def _collapse_to_simple_graph(Gp) -> nx.Graph:
    """MultiDiGraph (projected) -> undirected simple Graph keeping the shortest
    parallel edge; largest connected component only."""
    H = nx.Graph()
    H.graph["crs"] = Gp.graph.get("crs")
    for n, d in Gp.nodes(data=True):
        H.add_node(n, x=float(d["x"]), y=float(d["y"]))
    for u, v, d in Gp.edges(data=True):
        if u == v:
            continue
        L = max(float(d.get("length", 1.0)), 1.0)
        if H.has_edge(u, v) and H[u][v]["length"] <= L:
            continue
        H.add_edge(u, v, length=L, **{k: d[k] for k in TAG_KEYS if k in d})
    comp = max(nx.connected_components(H), key=len)
    return H.subgraph(comp).copy()


def load_osm_graph(cfg: Config) -> nx.Graph:
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.log_console = False
    if cfg.place:
        G = ox.graph_from_place(cfg.place, network_type="walk", simplify=True)
    else:
        G = ox.graph_from_point(cfg.center, dist=cfg.radius_m,
                                network_type="walk", simplify=True)
    Gp = ox.project_graph(G)                 # -> UTM (zone 39N for Baku): metres
    return _collapse_to_simple_graph(Gp)


def build_demo_graph(cfg: Config) -> nx.Graph:
    """Offline OSM-like graph (grid + random tags) for smoke-testing only."""
    rng = np.random.default_rng(cfg.seed)
    n, sp = 28, 45.0
    G = nx.grid_2d_graph(n, n)
    G = nx.convert_node_labels_to_integers(G, label_attribute="ij")
    hw_opts = ["footway", "pedestrian", "residential", "path", "steps", "tertiary", "service"]
    hw_p = [0.40, 0.10, 0.25, 0.10, 0.03, 0.07, 0.05]
    sf_opts = ["paving_stones", "asphalt", "unpaved", None]
    H = nx.Graph()
    H.graph["crs"] = None
    for nd, d in G.nodes(data=True):
        i, j = d["ij"]
        H.add_node(nd, x=j * sp + rng.normal(0, 4), y=i * sp + rng.normal(0, 4))
    for u, v in G.edges():
        if rng.random() < 0.08:
            continue
        att = dict(length=sp * rng.uniform(0.8, 1.25), highway=rng.choice(hw_opts, p=hw_p))
        s = sf_opts[rng.integers(len(sf_opts))]
        if s:
            att["surface"] = s
        H.add_edge(u, v, **att)
    comp = max(nx.connected_components(H), key=len)
    return H.subgraph(comp).copy()


# =============================================================================
# 3. NETWORK: tag-based edge weights, zones, indexing
# =============================================================================
def _kmeans(X: np.ndarray, k: int, seed: int, iters: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if np.any(lab == j):
                C[j] = X[lab == j].mean(0)
    order = np.argsort(C[:, 0])                       # name zones W -> E
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in lab])


class Network:
    """Static, tag-derived description of the walkable network."""

    def __init__(self, H: nx.Graph, cfg: Config):
        self.cfg = cfg
        self.G = H
        self.edges = list(H.edges())
        self.E = len(self.edges)
        self.eid = {}
        for i, (u, v) in enumerate(self.edges):
            self.eid[(u, v)] = i
            self.eid[(v, u)] = i
        self.length = np.array([H[u][v]["length"] for u, v in self.edges])
        self.Dmax = float(self.length.max())

        rainy = cfg.weather == "rainy"
        names, base, rain = [], [], []
        for u, v in self.edges:
            n, b, r = classify_edge(H[u][v])
            names.append(n); base.append(b); rain.append(r)
        self.terrain = np.array(names)
        self.base = np.array(base)
        self.rain = np.array(rain)
        self.prior = self.base * (self.rain if rainy else 1.0)      # tag-derived prior
        self.wmax = max(b * (r if rainy else 1.0) for b, r in TERRAIN.values())

        mid = np.array([[(H.nodes[u]["x"] + H.nodes[v]["x"]) / 2,
                         (H.nodes[u]["y"] + H.nodes[v]["y"]) / 2] for u, v in self.edges])
        self.mid = mid
        self.zone = _kmeans(mid, cfg.n_zones, cfg.seed)              # Stage 2 grouping

        for i, (u, v) in enumerate(self.edges):
            H[u][v].update(terrain=names[i], base_cost=base[i], rain_multiplier=rain[i],
                           zone=int(self.zone[i]), prior_cost=float(self.prior[i]))

        self.nodes = list(H.nodes())
        self.node_xy = np.array([[H.nodes[n]["x"], H.nodes[n]["y"]] for n in self.nodes])

    def snap(self, x: float, y: float) -> tuple:
        d = np.hypot(self.node_xy[:, 0] - x, self.node_xy[:, 1] - y)
        i = int(np.argmin(d))
        return self.nodes[i], float(d[i])


def latlon_to_xy(net: Network, lat: float, lon: float) -> tuple[float, float]:
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", net.G.graph["crs"], always_xy=True)
    return tf.transform(lon, lat)


def resolve_od_pairs(net: Network, cfg: Config, demo: bool) -> list[tuple]:
    rng = np.random.default_rng(cfg.seed + 1)
    pairs = []
    if demo:
        xy = net.node_xy
        a = net.nodes[int(np.argmin(xy[:, 0] + xy[:, 1]))]
        b = net.nodes[int(np.argmax(xy[:, 0] + xy[:, 1]))]
        pairs.append((a, b))
    else:
        a, da = net.snap(*latlon_to_xy(net, *cfg.origin))
        b, db = net.snap(*latlon_to_xy(net, *cfg.dest))
        print(f"  Origin snapped to node {a} ({da:.0f} m from requested point)")
        print(f"  Dest   snapped to node {b} ({db:.0f} m from requested point)")
        if a == b:
            raise SystemExit("Origin and destination snap to the same node - choose points further apart.")
        pairs.append((a, b))
    tries = 0
    while len(pairs) < 1 + cfg.n_extra_routes and tries < 500:
        tries += 1
        i, j = rng.choice(len(net.nodes), 2, replace=False)
        if np.hypot(*(net.node_xy[i] - net.node_xy[j])) >= cfg.min_od_m:
            pairs.append((net.nodes[i], net.nodes[j]))
    return pairs


# =============================================================================
# 4. ENVIRONMENT (ground truth) + HIDDEN HAZARD INJECTION
# =============================================================================
class Environment:
    """Ground-truth comfort per edge for one trial. Hazards are NOT in the tags."""

    def __init__(self, net: Network, cfg: Config, od: tuple, rng: np.random.Generator):
        self.net, self.cfg = net, cfg
        rainy = cfg.weather == "rainy"
        self.het = rng.uniform(0.9, 1.1, net.E)                 # hidden per-edge heterogeneity
        self.hazard_cost = cfg.hazard_base * (cfg.hazard_rain if rainy else 1.0)
        self.hazard = np.zeros(net.E, dtype=bool)
        self.inject_hazards(od, rng)

    def inject_hazards(self, od: tuple, rng: np.random.Generator) -> None:
        """Programmatically inject `n_hazards` hidden hazards (temporary construction).
        'corridor' : half on the shortest path, the rest inside the k-shortest-path
                     corridor (so the test is actually exercised);
        'random'   : uniform over all edges.
        A hazard is rejected if it would disconnect A from B (once hazard edges are avoided)."""
        net, cfg = self.net, self.cfg
        A, B = od
        H = net.G
        sp = nx.shortest_path(H, A, B, weight="length")
        sp_e = [net.eid[(sp[i], sp[i + 1])] for i in range(len(sp) - 1)]

        if cfg.hazard_placement == "random":
            pools = [list(range(net.E))]
        else:
            corridor = set(sp_e)
            for p in islice(nx.shortest_simple_paths(H, A, B, weight="length"), cfg.corridor_k):
                corridor.update(net.eid[(p[i], p[i + 1])] for i in range(len(p) - 1))
            pools = [sp_e, sorted(corridor - set(sp_e))]

        Hc = H.copy()
        chosen: list[int] = []
        quota = [math.ceil(cfg.n_hazards / 2), cfg.n_hazards] if len(pools) == 2 else [cfg.n_hazards]
        for pool, q in zip(pools, quota):
            for e in rng.permutation(pool):
                if len(chosen) >= q:
                    break
                e = int(e)
                u, v = net.edges[e]
                if e in chosen or not Hc.has_edge(u, v):
                    continue
                data = Hc[u][v]
                Hc.remove_edge(u, v)
                if nx.has_path(Hc, A, B):
                    chosen.append(e)
                else:
                    Hc.add_edge(u, v, **data)
        self.hazard[chosen] = True

    def hazard_active(self, e: int, i_user: int) -> bool:
        return bool(self.hazard[e]) and i_user >= self.cfg.hazard_onset

    def true_cost(self, e: int, i_user: int) -> float:
        if self.hazard_active(e, i_user):
            return self.hazard_cost
        return float(self.net.prior[e] * self.het[e])


# =============================================================================
# 5. SYNTHETIC USERS  (Stage 1 inputs: alpha_i, delta_i)
# =============================================================================
@dataclass
class User:
    alpha: float        # reliability
    delta: float        # contextual relevance
    sens: float         # personal sensitivity multiplier
    zone: int           # home edge node (federated cluster)


def make_users(cfg: Config, rng: np.random.Generator) -> list[User]:
    a = np.clip(rng.beta(6, 2, cfg.n_users), 0.3, 1.0)
    d = np.where(rng.random(cfg.n_users) < cfg.p_context_match, 1.0, cfg.delta_mismatch)
    s = rng.lognormal(0.0, 0.15, cfg.n_users)
    z = rng.integers(0, cfg.n_zones, cfg.n_users)
    return [User(float(a[i]), float(d[i]), float(s[i]), int(z[i])) for i in range(cfg.n_users)]


def perceive(env: Environment, user: User, e: int, i_user: int, rng, hazard_hit: bool) -> float:
    """w_i(e): the comfort cost user i reports for edge e (noise shrinks with alpha)."""
    c = env.true_cost(e, i_user)
    if hazard_hit:
        return c * user.sens * rng.uniform(1.0, 1.3)          # strong local discomfort
    sd = 0.05 + 0.35 * (1.0 - user.alpha)
    return max(0.5, c * user.sens * (1.0 + rng.normal(0.0, sd)))


# =============================================================================
# 6. ROUTERS  (Shortest Path / Centralized / HFL-RL)
# =============================================================================
class Router:
    """
    mode = 'sp'   : lambda = 1, no learning
    mode = 'cen'  : all raw reports pooled globally (1 node, no p/q/V weights)
    mode = 'hfl'  : Stage 2 per-zone aggregation, Stage 3 global with p*q*V

    Edge attributes written on self.G: W_global, lambda, J.
    """

    def __init__(self, name: str, mode: str, net: Network, cfg: Config):
        self.name, self.mode, self.net, self.cfg = name, mode, net, cfg
        self.Z = cfg.n_zones if mode == "hfl" else 1
        E = net.E
        z = lambda: np.zeros((self.Z, E))
        self.num, self.num2, self.den, self.den2 = z(), z(), z(), z()   # sums of a*d*w, a*d*w^2, a*d, (a*d)^2
        self.cnt, self.asum, self.vol = z(), z(), z()
        self.prev_w = np.full((self.Z, E), np.nan)
        self.U = np.zeros(self.Z)                                       # (decayed) users per zone
        self.G = net.G.copy()
        self.W = net.prior.copy()
        self.lam = np.ones(E)
        self.dhat = net.length / net.Dmax
        self.J = self._compute_J()
        for i, (u, v) in enumerate(net.edges):
            self.G[u][v].update(W_global=float(self.W[i]), **{"lambda": 1.0}, J=float(self.J[i]))

    # ---- Stage 5 --------------------------------------------------------
    def _compute_J(self) -> np.ndarray:
        wh = np.minimum(self.W / self.net.wmax, 1.0)
        comfort = wh * (self.dhat if self.cfg.length_scaled else 1.0)
        return self.lam * self.dhat + (1.0 - self.lam) * comfort

    # ---- Stages 2-4 (vectorised refresh of all edges) --------------------
    def refresh(self) -> None:
        cfg, net = self.cfg, self.net
        if self.mode == "sp":
            return
        ok = self.den > 1e-9
        safe = np.where(ok, self.den, 1.0)
        w_edge = np.where(ok, self.num / safe, 0.0)                     # Stage 2

        if self.mode == "hfl":                                          # Stage 3
            p = np.minimum(1.0, self.cnt / np.maximum(self.U[:, None], 1e-9))
            q = np.where(self.cnt > 1e-9, self.asum / np.maximum(self.cnt, 1e-9), 0.0)
            V = 1.0 + cfg.kappa * self.vol
            wt = np.where(ok, p * q * V, 0.0)
        else:
            wt = ok.astype(float)
        S = wt.sum(0)
        self.W = np.where(S > 0, (wt * w_edge).sum(0) / np.maximum(S, 1e-12), net.prior)

        D, D2 = self.den.sum(0), self.den2.sum(0)                       # Stage 4
        has = D > 1e-9
        Dn = np.maximum(D, 1e-12)
        neff = np.where(has, D ** 2 / np.maximum(D2, 1e-12), cfg.prior_neff)   # Kish N_eff
        mean = self.num.sum(0) / Dn
        var = np.maximum(self.num2.sum(0) / Dn - mean ** 2, 0.0)
        sigma = np.where(has, np.sqrt(var) / np.maximum(mean, 1e-6), cfg.prior_sigma)   # disagreement (CV)
        self.lam = np.where(neff > 0, 1.0 - neff / (neff + cfg.beta) * np.exp(-sigma), 1.0)

        Jn = self._compute_J()
        for i in np.nonzero(~np.isclose(Jn, self.J))[0]:                # sync nx attributes
            u, v = net.edges[i]
            self.G[u][v].update(W_global=float(self.W[i]), **{"lambda": float(self.lam[i])}, J=float(Jn[i]))
        self.J = Jn

    # ---- Stage 1 -> ingestion ------------------------------------------
    def ingest(self, user: User, reports: list[tuple]) -> None:
        """reports: list of (edge_id, w_i, is_turnback)."""
        if self.mode == "sp":
            return
        cfg = self.cfg
        z = user.zone if self.mode == "hfl" else 0
        for arr in (self.num, self.num2, self.den, self.den2, self.cnt, self.asum):
            arr *= cfg.gamma                                            # forgetting
        self.U *= cfg.gamma
        self.U[z] += 1.0

        w_edge_now = lambda zz, e: (self.num[zz, e] / self.den[zz, e]) if self.den[zz, e] > 1e-9 else np.nan
        touched = []
        for e, w_i, turnback in reports:
            if turnback:                                                # change-point reset
                for arr in (self.num, self.num2, self.den, self.den2, self.cnt, self.asum):
                    arr[z, e] *= cfg.reset
            m = cfg.turnback_mult if turnback else 1.0
            ad = user.alpha * user.delta
            w_local = ad * w_i                                          # Stage 1: w_local,i
            self.num[z, e] += m * w_local
            self.num2[z, e] += m * ad * w_i ** 2
            self.den[z, e] += m * ad
            self.den2[z, e] += m * ad ** 2
            self.cnt[z, e] += m
            self.asum[z, e] += m * user.alpha
            touched.append(e)

        for e in set(touched):                                          # volatility index
            new = w_edge_now(z, e)
            if not np.isnan(self.prev_w[z, e]) and not np.isnan(new):
                dv = abs(new - self.prev_w[z, e]) / self.net.wmax
                self.vol[z, e] = (1 - cfg.vol_ema) * self.vol[z, e] + cfg.vol_ema * dv
            self.prev_w[z, e] = new
        self.refresh()

    # ---- planning -------------------------------------------------------
    def plan(self, src, dst, blocked: set | None = None):
        eid = self.net.eid
        if blocked:
            wf = lambda u, v, d: None if eid[(u, v)] in blocked else d["J"]
        else:
            wf = "J"
        try:
            return nx.shortest_path(self.G, src, dst, weight=wf)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def route_lambda(self, od: tuple) -> float:
        if self.mode == "sp":
            return 1.0
        p = self.plan(*od)
        return float(np.mean([self.lam[self.net.eid[(p[i], p[i + 1])]] for i in range(len(p) - 1)])) if p else np.nan


# =============================================================================
# 7. TRIP SIMULATION  ("turn back at point C")
# =============================================================================
def simulate_trip(env: Environment, router: Router, user: User, i_user: int,
                  od: tuple, rng: np.random.Generator) -> dict:
    net, cfg = env.net, env.cfg
    src, dst = od
    pos, walked, comfort, devs = src, 0.0, 0.0, 0
    blocked: set[int] = set()
    reports: list[tuple] = []
    completed = True

    while pos != dst:
        path = router.plan(pos, dst, blocked)
        if path is None or devs > cfg.max_replans:
            completed = False
            break
        for a, b in zip(path[:-1], path[1:]):
            e = net.eid[(a, b)]
            L = net.length[e]
            c = env.true_cost(e, i_user)
            if env.hazard_active(e, i_user):
                # discomfort at point C: walk part of the edge, then retreat to safe node `a`
                part = cfg.partial_walk_frac * L
                walked += 2 * part
                comfort += 2 * part * c
                reports.append((e, perceive(env, user, e, i_user, rng, True), True))
                blocked.add(e)                     # this user avoids e from now on
                devs += 1
                break                              # re-plan from `a` (nearest safe node)
            walked += L
            comfort += L * c
            if rng.random() < cfg.report_prob:
                reports.append((e, perceive(env, user, e, i_user, rng, False), False))
            pos = b

    router.ingest(user, reports)                   # privacy-preserving methods learn afterwards
    return dict(walked=walked, comfort=comfort / walked if walked > 0 else np.nan,
                dev=devs, completed=completed)


# =============================================================================
# 8. EXPERIMENT DRIVER
# =============================================================================
METHODS = [("Shortest Path (lambda=1)", "sp"),
           ("Centralized Comfort", "cen"),
           ("HFL-RL (proposed)", "hfl")]


def run_trial(net: Network, cfg: Config, od: tuple, seed: int, route_id: int, keep_viz=False):
    rng_env = np.random.default_rng(seed * 7919 + route_id)
    env = Environment(net, cfg, od, rng_env)
    users = make_users(cfg, np.random.default_rng(seed * 104729 + route_id))
    routers = {n: Router(n, m, net, cfg) for n, m in METHODS}
    logs = {n: [] for n, _ in METHODS}

    for i, u in enumerate(users):
        for n, _ in METHODS:                       # identical users & environment for every method
            rng = np.random.default_rng([seed, route_id, i])
            logs[n].append(simulate_trip(env, routers[n], u, i, od, rng))

    rows = []
    half = cfg.n_users // 2
    sp_dev = sum(t["dev"] for t in logs[METHODS[0][0]])
    for n, _ in METHODS:
        df = pd.DataFrame(logs[n])
        ok = df[df.completed]
        late = ok.iloc[max(0, len(ok) - half):]
        rows.append(dict(route=route_id, seed=seed, method=n,
                         path_len=ok.walked.mean(), comfort=ok.comfort.mean(),
                         comfort_late=late.comfort.mean(), len_late=late.walked.mean(),
                         deviations=int(df.dev.sum()), dev_avoided=sp_dev - int(df.dev.sum()),
                         completion=df.completed.mean(),
                         final_lambda=routers[n].route_lambda(od)))
    viz = None
    if keep_viz:
        viz = dict(env=env, routers=routers, od=od,
                   cumdev={n: np.cumsum([t["dev"] for t in logs[n]]) for n, _ in METHODS})
    return rows, viz, {n: np.cumsum([t["dev"] for t in logs[n]]) for n, _ in METHODS}


def pm(s: pd.Series, f="{:.1f}") -> str:
    return f"{f.format(s.mean())} ± {f.format(s.std(ddof=1) if len(s) > 1 else 0.0)}"


def summarise(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = [n for n, _ in METHODS]
    main, late = [], []
    for n in order:
        g = res[res.method == n]
        main.append({"Method": n,
                     "Avg Path Length (m)": pm(g.path_len),
                     "Avg Comfort Cost": pm(g.comfort, "{:.3f}"),
                     "Deviation Events": pm(g.deviations),
                     "Deviations Avoided vs SP": pm(g.dev_avoided),
                     "Final λ": pm(g.final_lambda, "{:.3f}"),
                     "Completion": f"{100 * g.completion.mean():.0f}%"})
        late.append({"Method": n,
                     "Path Length, last 50% users (m)": pm(g.len_late),
                     "Comfort Cost, last 50% users": pm(g.comfort_late, "{:.3f}")})
    return pd.DataFrame(main), pd.DataFrame(late)


# =============================================================================
# 9. FIGURE
# =============================================================================
def make_figure(net: Network, viz: dict, curves: dict, cfg: Config, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.5, 1]})
    H = net.G
    segs = [[(H.nodes[u]["x"], H.nodes[u]["y"]), (H.nodes[v]["x"], H.nodes[v]["y"])] for u, v in net.edges]
    zc = plt.cm.Pastel1(net.zone % 9)
    ax.add_collection(LineCollection(segs, colors=zc, linewidths=1.2, zorder=1))
    haz = [segs[i] for i in np.nonzero(viz["env"].hazard)[0]]
    ax.add_collection(LineCollection(haz, colors="red", linewidths=3.2, zorder=3, label="Hidden hazard"))
    styles = {METHODS[0][0]: ("k", "--", 1.6), METHODS[1][0]: ("tab:orange", "-.", 1.8), METHODS[2][0]: ("tab:blue", "-", 2.2)}
    for n, r in viz["routers"].items():
        p = r.plan(*viz["od"])
        if p:
            c, ls, lw = styles[n]
            ax.plot([H.nodes[k]["x"] for k in p], [H.nodes[k]["y"] for k in p], color=c, ls=ls, lw=lw, zorder=4, label=n)
    for k, mk, col in ((viz["od"][0], "o", "green"), (viz["od"][1], "*", "gold")):
        ax.plot(H.nodes[k]["x"], H.nodes[k]["y"], mk, ms=13, color=col, mec="k", zorder=6)
    ax.autoscale(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Planned routes after {cfg.n_users} users (zones shaded)")
    ax.legend(fontsize=8, loc="best")

    x = np.arange(1, cfg.n_users + 1)
    for n, _ in METHODS:
        Y = np.array(curves[n])
        c, ls, _ = styles[n]
        ax2.plot(x, Y.mean(0), color=c, ls=ls, label=n)
        ax2.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=c, alpha=0.12)
    ax2.axvline(cfg.hazard_onset, color="red", ls=":", lw=1)
    ax2.set_xlabel("User index"); ax2.set_ylabel("Cumulative deviation events")
    ax2.set_title("Adaptation to hidden hazards"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="offline synthetic graph")
    ap.add_argument("--users", type=int); ap.add_argument("--seeds", type=int)
    ap.add_argument("--hazards", type=int); ap.add_argument("--weather", choices=["dry", "rainy"])
    ap.add_argument("--place", type=str); ap.add_argument("--routes", type=int, help="extra random routes")
    ap.add_argument("--prior-neff", type=float, help="0 = strict cold start")
    ap.add_argument("--placement", choices=["corridor", "random"]); ap.add_argument("--outdir", type=str)
    a = ap.parse_args()

    cfg = Config()
    upd = {k: v for k, v in dict(n_users=a.users, n_seeds=a.seeds, n_hazards=a.hazards, weather=a.weather,
                                 place=a.place, n_extra_routes=a.routes, hazard_placement=a.placement,
                                 outdir=a.outdir, prior_neff=a.prior_neff).items() if v is not None}
    cfg = replace(cfg, **upd)
    os.makedirs(cfg.outdir, exist_ok=True)

    print("[1/4] Building network ...")
    H = build_demo_graph(cfg) if a.demo else load_osm_graph(cfg)
    net = Network(H, cfg)
    print(f"  {len(net.nodes)} nodes, {net.E} edges, Dmax={net.Dmax:.0f} m, "
          f"total length={net.length.sum() / 1000:.1f} km")
    print("  Terrain mix:", {k: int(v) for k, v in pd.Series(net.terrain).value_counts().head(6).items()})

    print("[2/4] Resolving routes ...")
    ods = resolve_od_pairs(net, cfg, a.demo)

    print(f"[3/4] Simulating {len(ods)} routes x {cfg.n_seeds} seeds x {cfg.n_users} users x 3 methods ...")
    rows, curves, viz0 = [], {n: [] for n, _ in METHODS}, None
    for r, od in enumerate(ods):
        for s in range(cfg.n_seeds):
            rr, viz, cum = run_trial(net, cfg, od, cfg.seed + s, r, keep_viz=(r == 0 and s == 0))
            rows += rr
            for n in cum:
                curves[n].append(cum[n])
            viz0 = viz or viz0
        print(f"  route {r} done")
    res = pd.DataFrame(rows)

    print("[4/4] Reporting ...")
    main_tbl, late_tbl = summarise(res)
    per_route = (res.groupby(["route", "method"])[["path_len", "comfort", "deviations", "final_lambda"]]
                 .mean().round(3))
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\n=== Table: overall comparison (mean ± sd over routes x seeds) ===")
    print(main_tbl.to_string(index=False))
    print("\n=== Table: post-learning behaviour ===")
    print(late_tbl.to_string(index=False))
    print("\n=== Per-route means ===")
    print(per_route)

    res.to_csv(os.path.join(cfg.outdir, "raw_trials.csv"), index=False)
    main_tbl.to_csv(os.path.join(cfg.outdir, "table_main.csv"), index=False)
    late_tbl.to_csv(os.path.join(cfg.outdir, "table_late.csv"), index=False)
    with open(os.path.join(cfg.outdir, "table_main.tex"), "w", encoding="utf-8") as f:
        f.write(main_tbl.to_latex(index=False, escape=False))
    make_figure(net, viz0, curves, cfg, os.path.join(cfg.outdir, "fig_routes.png"))
    print(f"\nSaved tables/figures to ./{cfg.outdir}/")


if __name__ == "__main__":
    main()