#!/usr/bin/env python3
"""
=============================================================================
hfl_rl_batch_locations.py -- 40 A->B test locations, one result folder each
=============================================================================
Requires hfl_rl_osm_baku.py in the same folder (imports the core model).

Compares, for every A->B pair:
    * "OSM shortest path"  (= 'Shortest Path (lambda=1)'; distance-only routing on OSM)
    * Centralized Comfort
    * HFL-RL (proposed)

Step 1  choose locations (auto, stratified by context)  -> selected_od_pairs.csv
Step 2  simulate + zoomed per-route outputs + summaries

Examples
    python hfl_rl_batch_locations.py --select-only                     # step 1 only
    python hfl_rl_batch_locations.py --od-file results_40/selected_od_pairs.csv \
           --seeds 5 --users 50 --workers 4 --basemap                  # step 2
    python hfl_rl_batch_locations.py --n-routes 40 --workers 4         # both steps
    python hfl_rl_batch_locations.py --demo --n-routes 8 --seeds 1 --users 30   # offline test
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd

import hfl_rl_osm_baku as core
from hfl_rl_osm_baku import Config, Network, METHODS, TERRAIN, run_trial

CATEGORIES = {                       # name : (feature used for ranking, minimum value)
    "Crosswalk-heavy":   ("n_cross", 2),
    "Highway-arterial":  ("hw_m", 100.0),
    "Park":              ("park_m", 100.0),
    "Complex-junctions": ("complexity", 0.0),
}
SP_NAME, CEN_NAME, HFL_NAME = (m[0] for m in METHODS)


# =============================================================================
# 1. EXTRA NETWORK CONTEXT (parks) + PATH FEATURES
# =============================================================================
def compute_park_mask(net: Network, cfg: Config, demo: bool) -> np.ndarray:
    """True for edges whose midpoint lies inside an OSM park/green polygon."""
    fallback = np.isin(net.terrain, ["Park Path (Paved)", "Park Path (Dirt)", "Park Grass"])
    if demo:
        return fallback
    try:
        import osmnx as ox
        import shapely
        from shapely.ops import unary_union
        tags = {"leisure": ["park", "garden", "playground"],
                "landuse": ["grass", "recreation_ground"], "natural": ["wood", "grassland"]}
        gdf = (ox.features_from_place(cfg.place, tags) if cfg.place
               else ox.features_from_point(cfg.center, tags, dist=cfg.radius_m))
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(net.G.graph["crs"])
        geom = unary_union(gdf.geometry.values)
        return np.asarray(shapely.contains_xy(geom, net.mid[:, 0], net.mid[:, 1])) | fallback
    except Exception as ex:                                   # network/API problems
        warnings.warn(f"Park polygons unavailable ({ex}); using park-path terrain classes only.")
        return fallback


def path_eids(net: Network, p: list) -> np.ndarray:
    return np.array([net.eid[(p[i], p[i + 1])] for i in range(len(p) - 1)], dtype=int)


def turn_count(net: Network, p: list, thr: float = 60.0) -> int:
    xy = np.array([[net.G.nodes[n]["x"], net.G.nodes[n]["y"]] for n in p])
    d = np.diff(xy, axis=0)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return int((np.abs((np.diff(ang) + 180) % 360 - 180) > thr).sum())


def route_features(net: Network, p: list) -> dict:
    es = path_eids(net, p)
    L, ter = net.length[es], net.terrain[es]
    km = max(L.sum() / 1000.0, 0.1)
    deg5 = sum(net.G.degree(n) >= 5 for n in p[1:-1])
    turns = turn_count(net, p)
    return dict(length_m=float(L.sum()),
                n_cross=int((ter == "Crosswalk").sum()),
                hw_m=float(L[np.isin(ter, ["Highway", "Road"])].sum()),
                park_m=float(L[net.in_park[es]].sum()),
                complexity=float((deg5 + turns) / km))


# =============================================================================
# 2. LOCATION SELECTION (stratified by context) / LOADING
# =============================================================================
def _to_latlon(net: Network, x: float, y: float):
    crs = net.G.graph.get("crs")
    if crs is None:
        return np.nan, np.nan
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    return lat, lon


def select_od_pairs(net: Network, cfg: Config, n_total: int = 40, pool: int = 3000,
                    dmin: float = 450.0, dmax: float = 1300.0, sep: float = 200.0) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 77)
    G, xy = net.G, net.node_xy
    cands = []
    for _ in range(pool):
        i, j = rng.choice(len(net.nodes), 2, replace=False)
        st = float(np.hypot(*(xy[i] - xy[j])))
        if not dmin <= st <= dmax:
            continue
        a, b = net.nodes[i], net.nodes[j]
        p = nx.shortest_path(G, a, b, weight="length")
        f = route_features(net, p)
        if len(p) < 4 or f["length_m"] > 1.8 * st:
            continue
        cands.append(dict(a=a, b=b, ia=i, ib=j, **f))
    print(f"  {len(cands)} candidate routes evaluated")

    per = n_total // len(CATEGORIES)
    chosen: list[dict] = []

    def far_enough(c):
        return all(np.hypot(*(xy[c["ia"]] - xy[k["ia"]])) > sep and
                   np.hypot(*(xy[c["ib"]] - xy[k["ib"]])) > sep for k in chosen)

    for cat, (feat, thr) in CATEGORIES.items():
        got = 0
        for c in sorted(cands, key=lambda c: -c[feat]):
            if got >= per:
                break
            if c[feat] < thr or not far_enough(c):
                continue
            chosen.append(dict(c, category=cat)); got += 1
        if got < per:
            warnings.warn(f"Only {got}/{per} routes found for '{cat}' (feature {feat} >= {thr}).")
    for c in rng.permutation(cands):                          # fill any shortfall
        if len(chosen) >= n_total:
            break
        if far_enough(c):
            chosen.append(dict(c, category="Mixed"))

    rows = []
    for k, c in enumerate(chosen, 1):
        la, lo = _to_latlon(net, *xy[c["ia"]]); lb, lob = _to_latlon(net, *xy[c["ib"]])
        rows.append(dict(id=k, category=c["category"], name=f"{c['category']} #{k}",
                         origin_lat=la, origin_lon=lo, dest_lat=lb, dest_lon=lob,
                         origin_node=c["a"], dest_node=c["b"], sp_length_m=round(c["length_m"]),
                         n_crosswalks=c["n_cross"], highway_road_m=round(c["hw_m"]),
                         park_m=round(c["park_m"]), complexity=round(c["complexity"], 1)))
    return pd.DataFrame(rows)


def load_od_file(net: Network, path: str, demo: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"id", "category", "origin_lat", "origin_lon", "dest_lat", "dest_lon"}
    if not need <= set(df.columns):
        raise SystemExit(f"{path} needs columns: {sorted(need)} (optional: name)")
    if "origin_node" in df.columns and "dest_node" in df.columns and demo:
        return df
    an, bn = [], []
    for _, r in df.iterrows():
        a, da = net.snap(*core.latlon_to_xy(net, r.origin_lat, r.origin_lon))
        b, db = net.snap(*core.latlon_to_xy(net, r.dest_lat, r.dest_lon))
        if max(da, db) > 75:
            warnings.warn(f"Route {r.id}: point snapped {max(da, db):.0f} m from requested location.")
        if a == b:
            raise SystemExit(f"Route {r.id}: origin and destination snap to the same node.")
        an.append(a); bn.append(b)
    df["origin_node"], df["dest_node"] = an, bn
    if "name" not in df.columns:
        df["name"] = [f"{c} #{i}" for c, i in zip(df.category, df.id)]
    return df


# =============================================================================
# 3. PER-ROUTE ANALYSIS + CLOSE-UP OUTPUTS
# =============================================================================
def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def analyse_path(net: Network, env, p: list) -> tuple[dict, pd.Series]:
    """Ground-truth evaluation of a planned route (hazards active)."""
    es = path_eids(net, p)
    L, ter = net.length[es], net.terrain[es]
    c = np.array([env.true_cost(int(e), 10 ** 9) for e in es])
    f = route_features(net, p)
    out = dict(length_m=float(L.sum()), n_edges=len(es),
               comfort_cost=float((L * c).sum() / L.sum()),
               discomfort_exposure=float((L * c).sum()),
               hazard_edges_on_route=int(env.hazard[es].sum()),
               crosswalk_edges=f["n_cross"], highway_road_m=f["hw_m"], park_m=f["park_m"],
               turns=turn_count(net, p))
    return out, pd.Series(L).groupby(ter).sum()


def export_geojson(net: Network, env, paths: dict, detail: pd.DataFrame, fn: str) -> None:
    if net.G.graph.get("crs") is None:
        return
    from pyproj import Transformer
    tf = Transformer.from_crs(net.G.graph["crs"], "EPSG:4326", always_xy=True)
    ll = lambda n: list(tf.transform(net.G.nodes[n]["x"], net.G.nodes[n]["y"]))
    feats = []
    for name, p in paths.items():
        feats.append({"type": "Feature", "properties": {"kind": "route", "method": name,
                      **{k: float(v) for k, v in detail.loc[name].items()}},
                      "geometry": {"type": "LineString", "coordinates": [ll(n) for n in p]}})
    for e in np.nonzero(env.hazard)[0]:
        u, v = net.edges[e]
        feats.append({"type": "Feature", "properties": {"kind": "hidden_hazard"},
                      "geometry": {"type": "LineString", "coordinates": [ll(u), ll(v)]}})
    with open(fn, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)


def make_route_figure(net, cfg, env, paths, detail, terr, curves, title, base, basemap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    H = net.G
    allnodes = [n for p in paths.values() for n in p]
    px = np.array([H.nodes[n]["x"] for n in allnodes]); py = np.array([H.nodes[n]["y"] for n in allnodes])
    m = max(120.0, 0.25 * max(np.ptp(px), np.ptp(py)))
    x0, x1, y0, y1 = px.min() - m, px.max() + m, py.min() - m, py.max() + m

    tnames = list(TERRAIN.keys())
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    tcol = {t: cmap[i] for i, t in enumerate(tnames)}

    fig = plt.figure(figsize=(14, 10.5))
    gs = GridSpec(2, 2, height_ratios=[1.7, 1], hspace=0.22, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])

    inbox = (net.mid[:, 0] > x0) & (net.mid[:, 0] < x1) & (net.mid[:, 1] > y0) & (net.mid[:, 1] < y1)
    idx = np.nonzero(inbox)[0]
    segs = [[(H.nodes[net.edges[i][0]]["x"], H.nodes[net.edges[i][0]]["y"]),
             (H.nodes[net.edges[i][1]]["x"], H.nodes[net.edges[i][1]]["y"])] for i in idx]
    ax.add_collection(LineCollection(segs, colors=[tcol[net.terrain[i]] for i in idx],
                                     linewidths=1.6, alpha=0.95, zorder=2))
    hz = [s for s, i in zip(segs, idx) if env.hazard[i]]
    if hz:
        ax.add_collection(LineCollection(hz, colors="red", linewidths=5, zorder=3))

    sty = {SP_NAME: ("black", "--", 4.6), CEN_NAME: ("darkorange", "-.", 3.2), HFL_NAME: ("#1f5fbf", "-", 2.0)}
    for n in (SP_NAME, CEN_NAME, HFL_NAME):
        if n in paths:
            c, ls, lw = sty[n]
            ax.plot([H.nodes[k]["x"] for k in paths[n]], [H.nodes[k]["y"] for k in paths[n]],
                    color=c, ls=ls, lw=lw, zorder=5, solid_capstyle="round")
    a, b = paths[SP_NAME][0], paths[SP_NAME][-1]
    ax.plot(H.nodes[a]["x"], H.nodes[a]["y"], "o", ms=15, color="lime", mec="k", mew=1.5, zorder=8)
    ax.plot(H.nodes[b]["x"], H.nodes[b]["y"], "*", ms=22, color="gold", mec="k", mew=1.2, zorder=8)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if basemap and net.G.graph.get("crs") is not None:
        try:
            import contextily as cx
            cx.add_basemap(ax, crs=net.G.graph["crs"], source=cx.providers.CartoDB.PositronNoLabels,
                           zorder=0, attribution_size=6)
        except Exception as ex:
            warnings.warn(f"Basemap skipped: {ex}")
    present = sorted({net.terrain[i] for i in idx}, key=tnames.index)
    handles = [Line2D([0], [0], color=tcol[t], lw=4, label=t) for t in present]
    handles += [Line2D([0], [0], color="red", lw=5, label="Hidden hazard"),
                Line2D([0], [0], color="black", ls="--", lw=3, label="OSM shortest path"),
                Line2D([0], [0], color="darkorange", ls="-.", lw=3, label="Centralized"),
                Line2D([0], [0], color="#1f5fbf", lw=3, label="HFL-RL")]
    ax.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1.005, 0.5), title="Edges / routes")
    ax.set_title(title + "\n" + "   |   ".join(
        f"{k.split(' (')[0]}: {detail.loc[k, 'length_m']:.0f} m, comfort {detail.loc[k, 'comfort_cost']:.2f}, "
        f"hazards on route {int(detail.loc[k, 'hazard_edges_on_route'])}" for k in paths), fontsize=9)

    ax2 = fig.add_subplot(gs[1, 0])
    order = [t for t in tnames if t in terr.columns]
    left = np.zeros(len(terr))
    for t in order:
        ax2.barh(range(len(terr)), terr[t].values, left=left, color=tcol[t], label=t, edgecolor="white", lw=0.4)
        left += terr[t].values
    ax2.set_yticks(range(len(terr))); ax2.set_yticklabels([k.split(" (")[0] for k in terr.index], fontsize=8)
    ax2.invert_yaxis(); ax2.set_xlabel("Metres on route, by terrain class"); ax2.set_title("Terrain composition of planned routes", fontsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(1, cfg.n_users + 1)
    for n in (SP_NAME, CEN_NAME, HFL_NAME):
        Y = np.array(curves[n]); c, ls, _ = sty[n]
        ax3.plot(x, Y.mean(0), color=c, ls=ls, label=n.split(" (")[0])
        ax3.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=c, alpha=0.12)
    ax3.axvline(cfg.hazard_onset, color="red", ls=":", lw=1)
    ax3.set_xlabel("User index"); ax3.set_ylabel("Cumulative deviations")
    ax3.set_title(f"Adaptation (mean ± sd, {cfg.n_seeds} seeds)", fontsize=9); ax3.grid(alpha=0.3); ax3.legend(fontsize=7)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 4. WORKER (one route = one folder)
# =============================================================================
_NET = _CFG = _OUT = _BASEMAP = None


def _init(net, cfg, out, basemap):
    global _NET, _CFG, _OUT, _BASEMAP
    _NET, _CFG, _OUT, _BASEMAP = net, cfg, out, basemap


def process_route(rec: dict) -> list[dict]:
    net, cfg = _NET, _CFG
    rid, cat = int(rec["id"]), str(rec["category"])
    od = (rec["origin_node"], rec["dest_node"])
    folder = os.path.join(_OUT, f"route_{rid:02d}_{slug(cat)}")
    os.makedirs(folder, exist_ok=True)

    rows, viz = [], None
    curves = {n: [] for n, _ in METHODS}
    for s in range(cfg.n_seeds):
        rr, v, cum = run_trial(net, cfg, od, cfg.seed + s, rid, keep_viz=(s == 0))
        rows += rr
        viz = v or viz
        for n in cum:
            curves[n].append(cum[n])
    for r in rows:
        r.update(route_id=rid, category=cat, name=rec.get("name", ""))
    pd.DataFrame(rows).to_csv(os.path.join(folder, "trials.csv"), index=False)

    env = viz["env"]
    paths = {n: viz["routers"][n].plan(*od) for n, _ in METHODS}
    paths = {n: p for n, p in paths.items() if p}
    det, terr = {}, {}
    for n, p in paths.items():
        det[n], terr[n] = analyse_path(net, env, p)
    detail = pd.DataFrame(det).T
    detail["detour_vs_shortest"] = detail.length_m / detail.loc[SP_NAME, "length_m"]
    detail["mean_lambda_on_route"] = [viz["routers"][n].route_lambda(od) if n in paths else np.nan for n in detail.index]
    terr_df = pd.DataFrame(terr).T.fillna(0.0)
    detail.round(3).to_csv(os.path.join(folder, "route_detail.csv"))
    terr_df.round(1).to_csv(os.path.join(folder, "terrain_breakdown_m.csv"))
    export_geojson(net, env, paths, detail, os.path.join(folder, "routes.geojson"))
    title = (f"Route {rid} - {cat}: {rec.get('name', '')}  "
             f"(seed-0 environment, {cfg.weather}, {int(env.hazard.sum())} hidden hazards)")
    make_route_figure(net, cfg, env, paths, detail, terr_df, curves, title,
                      os.path.join(folder, "zoom_map"), _BASEMAP)
    return rows


# =============================================================================
# 5. SUMMARIES
# =============================================================================
def pm(s: pd.Series, f="{:.1f}") -> str:
    return f"{f.format(s.mean())} ± {f.format(s.std(ddof=1) if len(s) > 1 else 0.0)}"


def summarise(res: pd.DataFrame, out: str) -> None:
    order = [n for n, _ in METHODS]
    per_route = (res.groupby(["route_id", "category", "method"])
                 [["path_len", "comfort", "deviations", "dev_avoided", "final_lambda", "completion"]]
                 .mean().reset_index())
    per_route.round(3).to_csv(os.path.join(out, "summary_per_route.csv"), index=False)

    wide = per_route.pivot(index=["route_id", "category"], columns="method")
    d = pd.DataFrame(index=wide.index)
    for tag, ref in (("vs_SP", SP_NAME), ("vs_Cen", CEN_NAME)):
        d[f"comfort_change_{tag}_%"] = 100 * (wide[("comfort", HFL_NAME)] / wide[("comfort", ref)] - 1)
        d[f"length_change_{tag}_%"] = 100 * (wide[("path_len", HFL_NAME)] / wide[("path_len", ref)] - 1)
    d["deviations_avoided_vs_SP"] = wide[("dev_avoided", HFL_NAME)]
    d = d.reset_index()
    d.round(2).to_csv(os.path.join(out, "summary_hfl_deltas_per_route.csv"), index=False)

    rows = []
    for cat, g in list(res.groupby("category")) + [("ALL", res)]:
        for n in order:
            x = g[g.method == n]
            rows.append({"Category": cat, "Method": n, "Routes": x.route_id.nunique(),
                         "Avg Path Length (m)": pm(x.path_len), "Avg Comfort Cost": pm(x.comfort, "{:.3f}"),
                         "Deviation Events": pm(x.deviations), "Deviations Avoided vs SP": pm(x.dev_avoided),
                         "Final λ": pm(x.final_lambda, "{:.3f}")})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(out, "summary_per_category.csv"), index=False)
    with open(os.path.join(out, "summary_per_category.tex"), "w", encoding="utf-8") as f:
        f.write(tbl.to_latex(index=False, escape=False))

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\n=== Per-category / overall (mean ± sd over routes x seeds) ===")
    print(tbl.to_string(index=False))
    print("\n=== HFL-RL vs OSM shortest path, mean change per category (%) ===")
    print(d.groupby("category")[["comfort_change_vs_SP_%", "length_change_vs_SP_%", "deviations_avoided_vs_SP"]].mean().round(2))
    print(f"\nRoutes where HFL-RL has lower comfort cost than SP: "
          f"{int((d['comfort_change_vs_SP_%'] < 0).sum())}/{len(d)}")


def overview_map(net: Network, sel: pd.DataFrame, fn: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    H = net.G
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_collection(LineCollection([[(H.nodes[u]["x"], H.nodes[u]["y"]), (H.nodes[v]["x"], H.nodes[v]["y"])]
                                      for u, v in net.edges], colors="#cccccc", linewidths=0.6))
    cols = dict(zip(sel.category.unique(), plt.cm.Set1(range(9))))
    for _, r in sel.iterrows():
        a, b = r.origin_node, r.dest_node
        ax.plot([H.nodes[a]["x"], H.nodes[b]["x"]], [H.nodes[a]["y"], H.nodes[b]["y"]], color=cols[r.category], lw=1.5)
        ax.text(H.nodes[a]["x"], H.nodes[a]["y"], str(r.id), fontsize=7, weight="bold")
    for c, col in cols.items():
        ax.plot([], [], color=col, label=c)
    ax.legend(); ax.autoscale(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Selected A-B test pairs (straight lines; number = origin)")
    fig.savefig(fn, dpi=200, bbox_inches="tight"); plt.close(fig)


# =============================================================================
# 6. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="offline synthetic graph (test only)")
    ap.add_argument("--od-file", type=str, help="CSV of hand-picked / previously selected pairs")
    ap.add_argument("--n-routes", type=int, default=40)
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--basemap", action="store_true", help="add CARTO/OSM tiles (needs contextily + internet)")
    ap.add_argument("--users", type=int); ap.add_argument("--seeds", type=int)
    ap.add_argument("--hazards", type=int); ap.add_argument("--weather", choices=["dry", "rainy"])
    ap.add_argument("--place", type=str); ap.add_argument("--radius", type=int)
    ap.add_argument("--prior-neff", type=float); ap.add_argument("--placement", choices=["corridor", "random"])
    ap.add_argument("--outdir", type=str, default="results_40")
    a = ap.parse_args()

    cfg = Config()
    upd = dict(n_users=a.users, n_seeds=a.seeds, n_hazards=a.hazards, weather=a.weather, place=a.place,
               radius_m=a.radius, prior_neff=a.prior_neff, hazard_placement=a.placement, outdir=a.outdir)
    cfg = replace(cfg, **{k: v for k, v in upd.items() if v is not None})
    os.makedirs(cfg.outdir, exist_ok=True)

    print("[1/4] Building network ...")
    H = core.build_demo_graph(cfg) if a.demo else core.load_osm_graph(cfg)
    net = Network(H, cfg)
    net.in_park = compute_park_mask(net, cfg, a.demo)
    print(f"  {len(net.nodes)} nodes, {net.E} edges | park edges: {int(net.in_park.sum())}")
    print("  Terrain mix:", {k: int(v) for k, v in pd.Series(net.terrain).value_counts().items()})

    print("[2/4] Locations ...")
    if a.od_file:
        sel = load_od_file(net, a.od_file, a.demo)
    else:
        sel = select_od_pairs(net, cfg, a.n_routes)
        sel.to_csv(os.path.join(cfg.outdir, "selected_od_pairs.csv"), index=False)
    print(sel.groupby("category").size().to_string())
    overview_map(net, sel, os.path.join(cfg.outdir, "overview_map.png"))
    if a.select_only:
        print(f"Selection saved to {cfg.outdir}/selected_od_pairs.csv - edit it if you like, then run with --od-file.")
        return

    print(f"[3/4] Simulating {len(sel)} routes x {cfg.n_seeds} seeds x {cfg.n_users} users x 3 methods ...")
    recs = sel.to_dict("records")
    rows: list[dict] = []
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(net, cfg, cfg.outdir, a.basemap)) as ex:
            for k, r in enumerate(ex.map(process_route, recs), 1):
                rows += r; print(f"  route {k}/{len(recs)} done")
    else:
        _init(net, cfg, cfg.outdir, a.basemap)
        for k, rec in enumerate(recs, 1):
            rows += process_route(rec); print(f"  route {k}/{len(recs)} done")

    print("[4/4] Summaries ...")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(cfg.outdir, "raw_trials_all_routes.csv"), index=False)
    summarise(res, cfg.outdir)
    print(f"\nAll outputs in ./{cfg.outdir}/  (one folder per route + summary_*.csv)")


if __name__ == "__main__":
    main()