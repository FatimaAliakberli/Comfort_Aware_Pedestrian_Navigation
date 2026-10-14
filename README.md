# Comfort Aware Pedestrian Navigation System

## Overview

Pedestrian navigation systems have traditionally optimised for shortest distance while ignoring comfort, safety, and dynamic environmental conditions — a critical gap for visually impaired users who face compounding challenges: unexpected physical hazards, unfamiliar environments, and the risk of disorientation during outdoor travel.

This project addresses that gap with a **Hierarchical Federated Reinforcement Learning framework (HFL-RL)**. Rather than relying on static expert-annotated maps or personalised models that fail to generalise, the system aggregates crowd-sourced pedestrian comfort feedback across three tiers — local agents, edge nodes, and a global server — sharing only aggregated, reliability-weighted comfort statistics instead of raw traces (this is a design choice, not a formal privacy guarantee; see [Limitations](#limitations)). A **confidence-adaptive cold-start gate (λ)** schedules the influence of learned experience according to data quantity and inter-agent agreement, so the system falls back to shortest-path routing when data is scarce.

> **Terminology.** The learning is tabular and non-parametric: the quantities exchanged between tiers are per-segment comfort statistics, not neural-network weights or gradients. "Federated" therefore means *hierarchical, privacy-motivated aggregation of comfort statistics*, not federated gradient-based model training. The navigation task is formalised as an MDP (state = current road segment + context, action = next adjacent segment, reward = −local discomfort) whose value surface is the Experience Manifold `W_global`, executed with least-cost graph search (Dijkstra / A* with a null heuristic).

### Core design principles
The system is guided by three principles established in the assistive navigation literature:
1. Include users in the design process
2. Keep users in the feedback loop
3. Adapt to dynamic real-world conditions

### How it works — the five-stage pipeline

All quantities are indexed by **road segment `e`** (an edge of the walk graph). Edge nodes are indexed by `k`.

| Stage | Formula | What it does |
|---|---|---|
| **1. Local weighting** | `w_local,i(e) = αᵢ · δᵢ(C,Φ) · wᵢ(e)` | Each agent modulates their comfort signal by personal reliability αᵢ and contextual relevance δᵢ |
| **2. Edge aggregation** | `w_edge,k(e) = Σ(αᵢδᵢwᵢ(e)) / Σ(αᵢδᵢ)` | Neighbourhood node *k* forms a reliability-normalised, segment-specific comfort estimate |
| **3. Global manifold** | `W_global(e) = Σₖ Vₖ(e)·w_edge,k(e) / Σₖ Vₖ(e)` | Global server fuses node estimates weighted by `Vₖ(e) = pₖ(e)·qₖ(e)·νₖ(e)` (participation, mean reliability, volatility) |
| **4. Cold-start gate** | `λₑ = 1 − [N_eff(e)/(N_eff(e)+β)] · exp(−σ(e))` | Per-segment trust schedule; λ=1 with no data, λ→0 as data grows and agents agree |
| **5. Navigation cost** | `J(P) = Σ_{e∈P} [ λₑ·D̂(e) + (1−λₑ)·Ŵ(e)·D̂(e) ]` | Route cost with `D̂ = D/D_max` and `Ŵ = min(1, W_global/W_max)`; both components normalised to [0,1] |

> **Normalisation note.** `Ŵ(e)` is a per-metre comfort penalty; multiplying it by `D̂(e)` turns it into the comfort *exposure* of the segment so both terms are commensurable and long segments are not under-penalised. This is the default (`length_scaled=True`). The literal form `λ·D̂ + (1−λ)·Ŵ` is available via `--unscaled-comfort`. The legacy 200×200 grid code (Part B) uses the earlier un-normalised form `J = λ·D + (1−λ)·W`.
>
> **Bounded comfort term.** Because `λ + (1−λ)·Ŵ ≤ 1`, a segment never costs more than distance-only routing would charge for it. A hazard (`Ŵ = 1`) is therefore penalised at most as much as an unexplored segment of the same length, and detours around hazards are accepted only when the alternative is not much longer.

### What is evaluated

The repository contains **two evaluation tracks**:

| Track | Environment | Methods compared | Entry points |
|---|---|---|---|
| **Part A — Real-world (current)** | OpenStreetMap pedestrian network of Baku, Azerbaijan (OSMnx, UTM-projected), 40 stratified A→B routes, 50 sequential synthetic users per route, 5 seeds, hidden hazards | Shortest Path, Centralised Comfort, **HFL-RL** | `hfl_rl_osm_baku.py`, `hfl_rl_batch_locations.py` |
| **Part B — Legacy synthetic grid** | 200×200 procedural city grid, 16 terrain classes, 8 scenarios | Shortest Path vs **HFL-RL** | `main/main.py`, `scenarios/s*.py` |

---

# Part A — Real-World OpenStreetMap Evaluation (Baku)

## Project structure

```
hfl_rl/
│
├── hfl_rl_osm_baku.py            # Core real-map model: OSM graph, tag→terrain mapping, hidden hazards,
│                                 #   synthetic users, 3 routers (SP / Centralised / HFL-RL), 3-route evaluation
├── hfl_rl_batch_locations.py     # 40-location stratified batch runner: one result folder per route
│
├── hfl_rl_baku_results/          # (generated) output of hfl_rl_osm_baku.py
├── results_40/                   # (generated) output of hfl_rl_batch_locations.py
├── cache/                        # (generated by OSMnx) cached Overpass responses
│
├── city_map/  formulas/  scenarios/  main/   # Part B (legacy synthetic grid) — see below
└── implicit_feedback.py, implicit_feedback_v2.py   # "Developing" — multi-user implicit feedback
```

> The two real-map scripts live in the **repository root, side by side**: `hfl_rl_batch_locations.py` does `import hfl_rl_osm_baku`. Run both from that folder.

## Requirements

```bash
pip install osmnx networkx pandas numpy matplotlib pyproj shapely
# optional: map tiles under the close-up figures (see Known issues before using)
pip install contextily
```

`geopandas` is installed automatically as an OSMnx dependency. Python 3.10 or later is required. An internet connection is needed on the first run (Overpass API); OSMnx caches responses in `./cache`.

## How to run

### A1 — Core model on a few routes (`hfl_rl_osm_baku.py`)

Runs the fixed A→B pair from `Config` (Fountain Square → Maiden Tower, approximate) plus `--routes` extra random pairs, for all seeds, and writes aggregate tables and one figure.

```bash
python hfl_rl_osm_baku.py                       # real Baku data, defaults
python hfl_rl_osm_baku.py --demo                # offline synthetic graph (smoke test only, no internet)
python hfl_rl_osm_baku.py --users 50 --seeds 5 --hazards 8 --weather rainy --routes 2
python hfl_rl_osm_baku.py --place "Icherisheher, Baku, Azerbaijan"
```

| Flag | Meaning | Default |
|---|---|---|
| `--demo` | Offline synthetic OSM-like graph (28×28 jittered grid with random tags) — **testing only, never report these results** | off |
| `--users` | Sequential users per route | 50 |
| `--seeds` | Random seeds per route | 5 |
| `--hazards` | Hidden hazards per route | 8 |
| `--weather` | `dry` or `rainy` | `rainy` |
| `--place` | Named OSM area instead of a point + radius | none (centre `40.3705, 49.8365`, radius 1500 m) |
| `--routes` | Extra random O–D pairs | 2 |
| `--placement` | Hazard placement: `corridor` or `random` | `corridor` |
| `--prior-neff` | OSM-tag prior pseudo-count (`0` = strict cold start) | 5 |
| `--outdir` | Output folder | `hfl_rl_baku_results` |

### A2 — 40 stratified locations (`hfl_rl_batch_locations.py`)

**Step 1 — choose the locations** (no simulation):

```bash
python hfl_rl_batch_locations.py --select-only --outdir results_40
```

Writes `results_40/selected_od_pairs.csv` and `overview_map.png`. Inspect the overview map; you can edit the CSV or replace rows with your own coordinates.

**Step 2 — simulate:**

```bash
python hfl_rl_batch_locations.py --od-file results_40/selected_od_pairs.csv --seeds 5 --users 50 --workers 4
```

Both steps in one command: `python hfl_rl_batch_locations.py --n-routes 40 --workers 4`.

| Flag | Meaning | Default |
|---|---|---|
| `--od-file` | CSV of pairs (see format below); skips automatic selection | none |
| `--n-routes` | Number of routes to auto-select | 40 |
| `--select-only` | Only select locations and draw the overview map | off |
| `--workers` | Parallel processes (one route per task) | 1 |
| `--basemap` | Add map tiles under close-ups (needs `contextily`; see Known issues) | off |
| `--users`, `--seeds`, `--hazards`, `--weather` | As above | 50 / 5 / 8 / `rainy` |
| `--place`, `--radius` | Study area (named place, or radius in metres around the default centre) | centre point, 1500 m |
| `--prior-neff`, `--placement` | As above | 5 / `corridor` |
| `--beta` | Stage-4 sensitivity β | 10 |
| `--kappa` | Volatility gain (`0` disables the volatility index νₖ) | 5 |
| `--unscaled-comfort` | Literal `λ·D̂ + (1−λ)·Ŵ` (no `D̂` factor on the comfort term) | off |
| `--outdir` | Output folder | `results_40` |

**`--od-file` format** (one row per route; `name` optional):

```csv
id,category,name,origin_lat,origin_lon,dest_lat,dest_lon
1,Park,Seaside park crossing,40.3726,49.8355,40.3667,49.8372
```

Points are snapped to the nearest graph node; a warning is printed if the snap distance exceeds 75 m, and the run stops if origin and destination snap to the same node.

### Ablation and sensitivity runs

Re-run the **same routes** with one design choice changed, each in its own output folder so results are not overwritten:

```bash
# strict cold start: no OSM-tag prior (λ = 1 on every unobserved segment)
python hfl_rl_batch_locations.py --od-file results_40/selected_od_pairs.csv --prior-neff 0 --outdir results_40_strict_cold

# hazards placed uniformly at random instead of on the shortest-path corridor
python hfl_rl_batch_locations.py --od-file results_40/selected_od_pairs.csv --placement random --outdir results_40_random_hazards

# no volatility index / literal (un-scaled) cost / different β
python hfl_rl_batch_locations.py --od-file results_40/selected_od_pairs.csv --kappa 0 --outdir results_40_no_volatility
python hfl_rl_batch_locations.py --od-file results_40/selected_od_pairs.csv --unscaled-comfort --outdir results_40_unscaled
python hfl_rl_batch_locations.py --od-file results_40/selected_od_pairs.csv --beta 5 --outdir results_40_beta5
```

Per-route figures are regenerated in every run, so ablations take as long as the main run.

---

## Real-world evaluation design

### 1. Network extraction
The walk network is downloaded with OSMnx (`network_type='walk'`, simplified to intersections and street segments), projected to UTM (zone 39N for Baku) so that all lengths are in metres, collapsed to an undirected simple graph (shortest parallel edge kept) and restricted to its largest connected component.

### 2. OSM tag → terrain class mapping
`classify_edge()` assigns each segment one of the paper's terrain classes from its OSM tags (`highway`, `surface`, `smoothness`, `footway`, `crossing`, `construction`). Multi-valued tags take the **worst** class. The prior comfort cost is `base_cost × rain_multiplier` (the multiplier applies only when `weather='rainy'`).

| Terrain class | Typical OSM tags | Base | Rain × |
|---|---|---|---|
| Good Pavement | footway, paved | 1.0 | 1.3 |
| Pedestrian Street | pedestrian, living_street | 1.1 | 1.3 |
| Park Path (Paved) | path/track, paved | 1.2 | 1.5 |
| Crosswalk | footway=crossing | 1.5 | 1.8 |
| Degraded Pavement | unpaved / bad smoothness; path with unknown surface | 2.0 | 2.0 |
| Pavement (Cars) | residential, service, unclassified | 2.5 | 1.8 |
| Ped. Area (Dirt) | pedestrian, unpaved | 2.6 | 3.0 |
| Park Path (Dirt) | path/track, unpaved | 2.8 | 3.5 |
| Park Grass, Street Clutter, Water Puddle | *(no OSM tag equivalent — never assigned)* | 3.0 / 3.0 / 3.5 | 3.5 / 1.5 / 5.0 |
| Road | secondary, tertiary | 4.0 | 1.5 |
| Stairs | steps | 5.0 | 2.5 |
| Highway | motorway, trunk, primary | 6.0 | 2.0 |
| Construction Zone | highway=construction | 8.0 | 3.0 |

`W_max` (the comfort normaliser) is the largest weather-adjusted class cost: 24 under rain, 8 when dry.

### 3. Green-space overlay
`compute_park_mask()` downloads polygons tagged `leisure` (park, garden, playground), `landuse` (grass, recreation_ground) and `natural` (wood, grassland); a segment counts as green space if its midpoint lies inside their union. **This flag is used only to stratify and describe routes — it does not change comfort costs.** If the polygon download fails, the mask falls back to the Park Path / Park Grass terrain classes.

### 4. Ground truth and hidden hazards
- True comfort of a segment = its tag-derived cost × a fixed hidden factor `η_e ~ U(0.9, 1.1)`.
- For each route, **8 hidden hazards** (temporary construction / flooding absent from OSM) are injected via `Environment.inject_hazards()`. They cost `8.0 × 3.0` under rain, are excluded from the prior, and become active at the **11th user** (index 10), so the system first accumulates benign observations.
- `--placement corridor` (default): half of the hazards lie on the OSM shortest path, the rest inside the corridor of the `corridor_k = 8` shortest simple paths — a deliberate stress test that guarantees hazards are encountered. `--placement random` places them uniformly over all segments (a harder, fairer test of generalisation). Placements that would disconnect origin from destination are rejected.

### 5. Stratified route selection
`select_od_pairs()` samples 3 000 random candidate pairs (straight-line 450–1300 m; shortest-path length ≤ 1.8× straight-line; ≥ 4 nodes), scores each OSM shortest path, and selects 10 routes per stratum by ranking on the stratum's feature, with a 200 m minimum separation between origins and between destinations:

| Stratum | Ranking feature | Threshold |
|---|---|---|
| Crosswalk-heavy | number of `Crosswalk` segments | ≥ 2 |
| Highway-arterial | metres on Road/Highway-class segments | ≥ 100 m |
| Park | metres inside green-space polygons | ≥ 100 m |
| Complex-junctions | (nodes of degree ≥ 5 + turns > 60°) per km | none |

> **Mixed stratum.** If a stratum cannot be filled (a `Only 0/10 routes found for 'Crosswalk-heavy'` warning is printed) the shortfall is filled with random routes labelled **`Mixed`**. In the reported Baku run no candidate met the crossing criterion, so the 40 routes are 10 highway-arterial, 10 park, 10 complex-junction and 10 mixed.

### 6. Synthetic users and "turn back at point C"
Users traverse a route **sequentially**; each plans with the state learned from all preceding users.

| Property | Model |
|---|---|
| Reliability αᵢ | `Beta(6, 2)`, clipped to [0.3, 1] |
| Contextual relevance δᵢ | 1.0 with probability 0.8, otherwise 0.35 |
| Personal sensitivity | log-normal, σ = 0.15 |
| Perceived score `wᵢ(e)` | `c(e)·sᵢ·(1+ε)`, `ε ~ N(0, (0.05 + 0.35(1−αᵢ))²)`, floor 0.5 |
| Reporting | each traversed segment reported with probability 0.8 to the user's **home edge node** (random among 3 zones) |
| Edge nodes | 3 neighbourhood zones from k-means on segment midpoints |
| Turn-back | on reaching an active hazard the user walks 25 % of the segment, reports a high discomfort score, **retreats to the preceding (nearest safe) node**, avoids that segment for the rest of the trip, and re-plans |
| Learning from a turn-back | the report counts as `turnback_mult = 20` ordinary reports and resets that node's earlier statistics for the segment by factor `reset = 0.1`; a forgetting factor `gamma = 0.97` per user discounts older data |

### 7. Compared methods

| Method | `Router` mode | Definition |
|---|---|---|
| **Shortest Path (SP)** | `sp` | `λ = 1`, no learning. Distance-only routing on the OSM graph — the "OSM prediction" baseline. |
| **Centralised Comfort** | `cen` | The **non-private reference**: one server receives *every raw report* (user, segment, `wᵢ`, `αᵢ`, `δᵢ`) into a **single pool**. Stage 2 is applied once over all users; there are no zones and no `p·q·ν` weighting (`W_global` = the pooled `w_edge`). It uses the **same** per-segment gate `λₑ`, forgetting factor, turn-back multiplicity and normalised cost as HFL-RL, and a turn-back reset applies to the whole pool. |
| **HFL-RL (proposed)** | `hfl` | Users report to their home edge node; each node keeps per-segment sufficient statistics (`Σαδw`, `Σαδw²`, `Σαδ`, `Σ(αδ)²`, counts); the server fuses nodes with `Vₖ(e) = p·q·ν`; a turn-back reset affects only the reporting node's statistics. Only these aggregates leave a node. |

Because Centralised and HFL-RL consume **the same reports** and differ only in (i) partitioning reports across three edge nodes, (ii) the `p·q·ν` node weighting, and (iii) the scope of the turn-back reset, **near-identical results are expected**. The supported claim is *parity with centralised aggregation without pooling raw reports*, not superiority over it.

All three methods see identical users, hazards and perceptions on every route and seed (paired design). The five seeds vary hazard placement, user population and `η_e`.

### 8. Metrics

| Metric | Definition |
|---|---|
| **Avg Path Length (m)** | Metres actually walked per completed trip, *including* turn-back retreats |
| **Avg Comfort Cost** | Length-weighted mean **true** cost experienced per metre (Table-I units; 1.0 = good pavement when dry) — includes exposure while retreating from hazards |
| **Deviation Events** | Number of turn-backs per route over 50 users |
| **Deviations Avoided** | SP deviations − method deviations (paired; SP = 0 by definition) |
| **Final λ** | Mean `λₑ` over the planned route after the last user (SP = 1) |
| **Completion** | Fraction of trips that reached the destination |
| `comfort_late`, `len_late` | Comfort / length over the last 50 % of users (post-learning behaviour) |

### 9. Hyper-parameters (`Config` in `hfl_rl_osm_baku.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `beta` | 10 | Stage-4 sensitivity |
| `kappa` | 5 | Volatility gain: `νₖ = 1 + κ · EMA(|Δw_edge|/W_max)` |
| `vol_ema` | 0.3 | EMA rate of the volatility index |
| `gamma` | 0.97 | Forgetting factor per user |
| `reset` | 0.1 | Change-point reset on a turn-back |
| `turnback_mult` | 20 | A turn-back counts as this many ordinary reports |
| `prior_neff`, `prior_sigma` | 5, 0.5 | OSM-tag prior enters as `N₀` pseudo-observations with `σ₀`, so unseen segments start at `λ₀ = 1 − (5/15)·e^(−0.5) ≈ 0.80`. Set `prior_neff = 0` for strict cold start (`λ = 1`). |
| `length_scaled` | `True` | Comfort term multiplied by `D̂(e)` |
| `n_zones` | 3 | Number of edge nodes |
| `n_hazards`, `hazard_onset` | 8, 10 | Hazards per route; first active at user index 10 |
| `report_prob`, `p_context_match`, `delta_mismatch` | 0.8, 0.8, 0.35 | User reporting / context model |

---

## Output files

### `hfl_rl_baku_results/` — from `hfl_rl_osm_baku.py`

| File | Contents |
|---|---|
| `raw_trials.csv` | One row per route × seed × method: `path_len`, `comfort`, `comfort_late`, `len_late`, `deviations`, `dev_avoided`, `completion`, `final_lambda` |
| `table_main.csv` / `table_main.tex` | Overall comparison table (mean ± sd over route × seed), ready for LaTeX |
| `table_late.csv` | Post-learning table (last 50 % of users) |
| `fig_routes.png` / `.pdf` | Planned routes on the zone-shaded network (left) and cumulative deviation curves with ±sd band (right), 300 dpi |

Per-route means are also printed to the console.

### `results_40/` — from `hfl_rl_batch_locations.py`

```
results_40/
├── selected_od_pairs.csv            # the 40 A→B pairs (editable; reusable via --od-file)
├── overview_map.png                 # all pairs on the network, coloured by stratum
├── raw_trials_all_routes.csv        # every route × seed × method row
├── summary_per_route.csv            # seed-averaged metrics per route and method
├── summary_hfl_deltas_per_route.csv # HFL-RL vs SP / vs Centralised, % change per route
├── summary_per_category.csv         # mean ± sd per stratum and overall
├── summary_per_category.tex         # same, LaTeX
│
├── route_01_highway_arterial/
│   ├── zoom_map.png / zoom_map.pdf  # close-up (see below)
│   ├── route_detail.csv             # planned-route metrics per method
│   ├── terrain_breakdown_m.csv      # metres per terrain class per method
│   ├── routes.geojson               # planned routes + hazards in WGS84 (open in QGIS / geojson.io)
│   └── trials.csv                   # raw per-seed numbers for this route
├── route_02_...
└── route_40_...
```

| File | Columns / contents |
|---|---|
| `selected_od_pairs.csv` | `id, category, name, origin_lat, origin_lon, dest_lat, dest_lon, origin_node, dest_node, sp_length_m, n_crosswalks, highway_road_m, park_m, complexity` |
| `summary_per_route.csv` | `route_id, category, method, path_len, comfort, deviations, dev_avoided, final_lambda, completion` |
| `summary_hfl_deltas_per_route.csv` | `route_id, category, comfort_change_vs_SP_%, length_change_vs_SP_%, comfort_change_vs_Cen_%, length_change_vs_Cen_%, deviations_avoided_vs_SP` |
| `route_detail.csv` (index = method) | `length_m, n_edges, comfort_cost, discomfort_exposure, hazard_edges_on_route, crosswalk_edges, highway_road_m, park_m, turns, detour_vs_shortest, mean_lambda_on_route` |
| `zoom_map.png` | **Top:** terrain-coloured segments around the route, hidden hazards (red), planned routes of the three methods, A (green circle) and B (gold star). **Bottom left:** terrain composition (metres) of each planned route. **Bottom right:** cumulative deviation events vs user index (mean ± sd over seeds; dotted line = hazard onset). |

> **What the close-ups show.** `route_detail.csv`, `terrain_breakdown_m.csv`, `routes.geojson` and the map panel describe the **seed-0 environment** (seed-0 hazard placement and users) and the routes *planned after the last user*. `trials.csv` and the deviation curve aggregate **all seeds**. The Centralised and HFL-RL routes frequently coincide, so the HFL-RL line is drawn on top of the Centralised one.

---

## Results — 40 Baku routes

**Run configuration:** rainy weather · 8 hidden hazards per route (corridor placement, active from user 11) · 50 users per route · 5 seeds · 3 edge nodes · `prior_neff = 5` · `β = 10` · `length_scaled = True` (script defaults). Values are mean ± sd over route × seed runs; ten routes per stratum.

| Stratum | Method | Avg Path Length (m) | Avg Comfort Cost | Deviation Events | Avoided vs SP | Final λ |
|---|---|---|---|---|---|---|
| Complex-junctions | Shortest Path | 1078.8 ± 374.0 | 4.008 ± 1.358 | 144.0 ± 49.8 | 0.0 ± 0.0 | 1.000 ± 0.000 |
| | Centralised | 1048.6 ± 358.1 | 3.194 ± 1.167 | 71.9 ± 49.3 | 72.1 ± 52.7 | 0.492 ± 0.032 |
| | **HFL-RL** | 1048.1 ± 358.5 | 3.208 ± 1.163 | 72.8 ± 49.2 | 71.2 ± 52.4 | 0.488 ± 0.032 |
| Highway-arterial | Shortest Path | 1947.2 ± 363.5 | 7.691 ± 1.675 | 137.6 ± 53.7 | 0.0 ± 0.0 | 1.000 ± 0.000 |
| | Centralised | 1929.7 ± 337.2 | 6.903 ± 1.349 | 109.3 ± 51.6 | 28.3 ± 48.5 | 0.488 ± 0.030 |
| | **HFL-RL** | 1935.6 ± 334.3 | 6.913 ± 1.343 | 111.1 ± 50.7 | 26.5 ± 48.6 | 0.482 ± 0.033 |
| Mixed | Shortest Path | 1821.8 ± 312.0 | 5.925 ± 1.112 | 153.6 ± 51.3 | 0.0 ± 0.0 | 1.000 ± 0.000 |
| | Centralised | 1752.0 ± 302.7 | 5.209 ± 1.155 | 87.8 ± 47.1 | 65.8 ± 47.1 | 0.482 ± 0.022 |
| | **HFL-RL** | 1751.1 ± 302.7 | 5.196 ± 1.150 | 86.6 ± 47.1 | 67.0 ± 47.3 | 0.477 ± 0.021 |
| Park | Shortest Path | 1741.7 ± 274.6 | 3.603 ± 1.949 | 140.0 ± 45.2 | 0.0 ± 0.0 | 1.000 ± 0.000 |
| | Centralised | 1684.6 ± 256.3 | 2.719 ± 1.187 | 79.1 ± 46.7 | 60.9 ± 47.0 | 0.480 ± 0.025 |
| | **HFL-RL** | 1683.1 ± 258.3 | 2.717 ± 1.191 | 78.6 ± 47.2 | 61.4 ± 47.1 | 0.476 ± 0.026 |
| **ALL (40 routes)** | Shortest Path | 1647.4 ± 472.5 | 5.307 ± 2.250 | 143.8 ± 50.1 | 0.0 ± 0.0 | 1.000 ± 0.000 |
| | Centralised | 1603.7 ± 457.8 | 4.506 ± 2.064 | 87.0 ± 50.3 | 56.8 ± 51.4 | 0.485 ± 0.028 |
| | **HFL-RL** | 1604.5 ± 458.8 | 4.509 ± 2.063 | 87.3 ± 50.4 | 56.5 ± 51.7 | 0.481 ± 0.029 |

**HFL-RL vs the OSM shortest path** (mean of per-route changes, from `summary_hfl_deltas_per_route.csv`):

| Stratum | Comfort cost change | Walked-distance change | Deviations avoided (of SP's) |
|---|---|---|---|
| Park | −20.9 % | −3.2 % | 61.4 (44 %) |
| Complex-junctions | −19.0 % | −2.4 % | 71.2 (49 %) |
| Mixed | −12.6 % | −4.0 % | 67.0 (44 %) |
| Highway-arterial | −9.5 % | −0.4 % | 26.5 (19 %) |
| **Overall** | **≈ −15.5 %** | **≈ −2.5 %** | **56.5 (39 %)** |

HFL-RL had a lower comfort cost than SP on **40 / 40** routes.

### How to read these results

- **Comfort and deviations improve over distance-only routing in every stratum.** SP suffers about 2.9 turn-backs per user; HFL-RL about 1.7.
- **HFL-RL ≈ Centralised** (4.509 vs 4.506 comfort; 56.5 vs 56.8 deviations avoided). This is expected (same reports, see [Compared methods](#7-compared-methods)): hierarchical aggregation of statistics matches centralised aggregation; it does not outperform it.
- **Final λ ≈ 0.48** in all strata: confidence accumulates but never collapses to 0, because per-segment `N_eff` stays limited and disagreement is non-zero.
- **Highway-arterial routes benefit least.** Crossings and long road-side stretches leave few comfortable alternatives, and by the bounded-cost property detours are accepted only when they are short. This is consistent with — but does not by itself prove — a rerouting limit rather than an estimator limit.
- **Walked distance is slightly *lower* for HFL-RL** largely because SP users lose distance to repeated turn-backs. Compare *planned* lengths (`detour_vs_shortest` in `route_detail.csv`) to measure the detour actually accepted for comfort.
- **Residual deviations remain** (~87 per route over 50 users): a hazard must be encountered before it can be learned, and planned routes after 50 users may still cross hazard segments (`hazard_edges_on_route` in `route_detail.csv`).
- **High sd (~50 events)** reflects large between-route variation, not seed noise alone. For significance claims use paired per-route differences (Wilcoxon signed-rank / bootstrap CI on `summary_hfl_deltas_per_route.csv`) rather than the mixed route × seed sd.

---

## Known issues and caveats

| Issue | Detail / workaround |
|---|---|
| **Basemap watermark** | `--basemap` uses CARTO *Positron* tiles, which now require an API key — figures show an "API KEY REQUIRED" watermark. Run **without** `--basemap` for publication figures, or change the tile provider in `make_route_figure()` (e.g. an OpenStreetMap or Esri provider in `contextily`). The OSM attribution `© OpenStreetMap contributors` is required in any published map. |
| **Empty crosswalk stratum** | In the Baku extract no candidate route reached ≥ 2 crossing segments (crossings are often mapped as nodes or as sparse footway segments), so the stratum is filled with `Mixed` routes. Lower the threshold in `CATEGORIES` or count crossing *nodes* to restore it. |
| **Tag prior (`prior_neff = 5`)** | The OSM-tag prior counts as pseudo-observations, so unseen segments start at λ₀ ≈ 0.80 rather than 1. In early tests this was decisive for hazard avoidance; report the strict-cold-start ablation (`--prior-neff 0`) alongside the main results. |
| **Hazard placement bias** | `corridor` placement puts hazards where routes go; use `--placement random` as the robustness check. |
| **Cost bound** | `J` per segment never exceeds `D̂(e)`, so a hazard is at most as costly as distance-only routing (see Overview). |
| **Seed-0 close-ups** | Per-route maps and `route_detail.csv` describe seed 0 only; tables aggregate all seeds. |
| **Unverified default coordinates** | `origin`, `dest` and `center` in `Config` are approximate (Fountain Square, Maiden Tower, central Baku). The script prints the snap distance — check it, or verify points on openstreetmap.org. |
| **OSM data changes over time** | Re-downloading later may change the graph and therefore the routes. Record the download date and keep `cache/` for reproducibility. |
| **`--demo` results are not real** | The offline demo graph is synthetic; use it only to check that the pipeline runs. |
| **Reproducibility** | Seeds are fixed (`seed = 42`, `seed + s` per replicate); with the same OSM snapshot and flags, runs are deterministic. |

## Module reference — real-map code

### `hfl_rl_osm_baku.py`
| Component | Purpose |
|---|---|
| `Config` | Every study-area, simulation, hazard, user and HFL-RL hyper-parameter (single dataclass) |
| `TERRAIN`, `classify_edge()` | Table-I cost parameters and the OSM-tag → terrain-class mapping |
| `load_osm_graph()`, `build_demo_graph()` | OSMnx extraction + UTM projection; offline synthetic graph |
| `Network` | Segment indexing, lengths, tag prior, `D_max`, `W_max`, k-means zones, node snapping |
| `Environment` | Ground-truth comfort, hidden heterogeneity `η_e`, `inject_hazards()` |
| `make_users()`, `perceive()` | Synthetic user population (αᵢ, δᵢ, sensitivity) and noisy perception |
| `Router` | Stages 2–5 as vectorised array operations; writes `W_global`, `lambda`, `J` to networkx edge attributes; modes `sp` / `cen` / `hfl` |
| `simulate_trip()` | Walks the planned route, handles turn-back and re-planning, produces reports |
| `run_trial()` | One route × seed × all three methods on identical users and hazards |
| `summarise()`, `make_figure()`, `main()` | Tables and figure for the CLI |

### `hfl_rl_batch_locations.py`
| Component | Purpose |
|---|---|
| `compute_park_mask()` | Green-space polygons → per-segment park flag |
| `route_features()` | Crossings, arterial metres, park metres, junction complexity of a path |
| `select_od_pairs()`, `load_od_file()` | Stratified location selection; loading / snapping of a user-supplied CSV |
| `process_route()` | Runs all seeds for one route and writes its folder (parallelisable) |
| `analyse_path()`, `export_geojson()`, `make_route_figure()` | Ground-truth evaluation of planned routes, GeoJSON export, close-up figure |
| `summarise()`, `overview_map()` | Per-route / per-category summaries and the selection overview |

---

# Part B — Legacy Synthetic Grid Evaluation (8 scenarios)

> The original evaluation. It runs on a **200×200 procedural city grid** (~400 m × 400 m, each cell ≈ 2 m²) with 16 terrain classes and compares **HFL-RL** with **Shortest Path** across 8 scenarios. Both paths are evaluated with the **same λ** so that Table III reflects a fair, apples-to-apples cost comparison. It uses the un-normalised cost `J = λ·D + (1−λ)·W` and has been superseded by Part A for reported results.

## Project structure (legacy)

```
hfl_rl/
│
├── city_map/
│   ├── city_map.py              # Terrain constants, cost tables, CityMap200 class
│   └── generate_map_image.py    # Standalone script → saves city map PNG
│
├── formulas/
│   └── formulas.py              # Formulas 1–5: λ, w_edge, W_global, J, A*
│
├── scenarios/
│   ├── scenario_builder.py      # build_scenario() and analyze_path() utilities
│   ├── _scenario_runner.py      # Shared map + bar-chart visualisation helper
│   ├── s1_cold_start_3_agents.py     # Scenario 1  — Cold Start,     3 agents
│   ├── s2_cold_start_10_agents.py    # Scenario 2  — Cold Start,     10 agents
│   ├── s3_sudden_change_3_agents.py  # Scenario 3  — Sudden Change,  3 agents
│   ├── s4_sudden_change_10_agents.py # Scenario 4  — Sudden Change,  10 agents
│   ├── s5_sunny_3_agents.py          # Scenario 5  — Sunny Weather,  3 agents
│   ├── s6_sunny_10_agents.py         # Scenario 6  — Sunny Weather,  10 agents
│   ├── s7_rainy_3_agents.py          # Scenario 7  — Rainy Weather,  3 agents
│   └── s8_rainy_10_agents.py         # Scenario 8  — Rainy Weather,  10 agents
│
└── main/
    └── main.py              # Full pipeline → all 5 output figures + Table III
```

## Requirements (legacy)

```bash
pip install numpy matplotlib
```

Python 3.10 or later is recommended (uses `list[...]` type hints).

## How to run (legacy)

### Option A — Full pipeline (recommended)

Runs all 8 scenarios and produces every output figure, including Table III.

```bash
python main/main.py
```

**Outputs saved to** `output_images/`:

| File | Description |
|---|---|
| `Scenarios.png` | 8-scenario overview: maps + bar charts |
| `City_map.png` | Annotated 200×200 city map |
| `Results_table.png` | **Table III** — HFL-RL vs SP metrics |
| `Ttradeoff.png` | Trade-off bar charts (W̄, J, path length) |

### Option B — City map only

Renders and saves only the annotated city map image. Useful for inspecting terrain layout independently of any scenario.

```bash
python city_map/generate_map_image.py
```

**Output:** `City_map.png`

### Option C — Run a single scenario

Each scenario script is fully self-contained. Running one prints the console metrics report and saves a single map + bar-chart figure for that scenario only.

```bash
# S1 — Cold Start, 3 agents, neutral weather
python scenarios/s1_cold_start_3_agents.py

# S2 — Cold Start, 10 agents, neutral weather
python scenarios/s2_cold_start_10_agents.py

# S3 — Sudden Change, 3 agents, neutral weather
python scenarios/s3_sudden_change_3_agents.py

# S4 — Sudden Change, 10 agents, neutral weather
python scenarios/s4_sudden_change_10_agents.py

# S5 — Sunny Weather, 3 agents
python scenarios/s5_sunny_3_agents.py

# S6 — Sunny Weather, 10 agents
python scenarios/s6_sunny_10_agents.py

# S7 — Rainy Weather, 3 agents
python scenarios/s7_rainy_3_agents.py

# S8 — Rainy Weather, 10 agents
python scenarios/s8_rainy_10_agents.py
```

**Output per scenario:** `<scenario_type>_<N>_agents.png`
**Console output per scenario:**
```
───────────────────────────────────────────────────────
  S5: Sunny Weather (3 agents)  (sunny, weather=sunny)
───────────────────────────────────────────────────────
  λ=0.556   N_eff=15   σ=0.30
  HFL-RL : len= 228  J=   291.5  W̄=1.297
  SP     : len= 221  J=   321.8  W̄=1.677
  HFL-RL wins J? ✓ YES  ↓9.4%
  HFL top terrains: [('Good Pavement', 112), ...]
```

## Scenario reference (legacy)

| ID | Script | Type | Agents | Weather | λ | N_eff |
|---|---|---|---|---|---|---|
| S1 | `s1_cold_start_3_agents.py` | Cold Start | 3 | neutral | 1.000 | 0 |
| S2 | `s2_cold_start_10_agents.py` | Cold Start | 10 | neutral | 1.000 | 0 |
| S3 | `s3_sudden_change_3_agents.py` | Sudden Change | 3 | neutral | 0.857 | 9 |
| S4 | `s4_sudden_change_10_agents.py` | Sudden Change | 10 | neutral | 0.774 | 30 |
| S5 | `s5_sunny_3_agents.py` | Sunny | 3 | sunny | 0.556 | 15 |
| S6 | `s6_sunny_10_agents.py` | Sunny | 10 | sunny | 0.383 | 50 |
| S7 | `s7_rainy_3_agents.py` | Rainy | 3 | rainy | 0.671 | 15 |
| S8 | `s8_rainy_10_agents.py` | Rainy | 10 | rainy | 0.543 | 50 |

**λ interpretation:**
- λ = 1.0 → pure shortest distance (cold start: no experience data yet)
- λ → 0 → increasingly dominated by crowd experience W_global
- More agents + better weather = lower λ = HFL-RL trusts experience more

## Module reference (legacy)

### `city_map/city_map.py`
Defines all 16 terrain types, their base comfort costs, weather multipliers, colour scheme, and the `CityMap200` class that procedurally builds the grid. Import this module whenever you need terrain constants or want to instantiate a city.

```python
from city_map.city_map import CityMap200, cell_cost, BASE_COMFORT
city = CityMap200(seed=42)
```

### `formulas/formulas.py`
Contains the five core HFL-RL formulas as standalone functions. No side effects — pure computation only.

```python
from formulas.formulas import compute_lambda, compute_W_global, astar

lam      = compute_lambda(N_eff=15, sigma=0.3)
W_global = compute_W_global(edge_nodes, weather, terrain_grid)
path     = astar(terrain_grid, start, goal, W_global, lam, weather)
```

### `scenarios/scenario_builder.py`
Two utilities used by every scenario script and the full pipeline:

- `build_scenario(type, n_agents, city, weather)` → constructs edge nodes, computes λ and W_global
- `analyze_path(path, terrain, W_global, lam, weather)` → returns `total_J`, `avg_W`, `path_length`, `terrain_counts`

### `scenarios/_scenario_runner.py`
Internal helper imported by each `s*.py` script. Handles the visualisation (map rendering + bar chart) and console report so that individual scenario scripts stay short and readable. Not intended to be run directly.

## Results (legacy synthetic grid)

The results match the simulation output (200×200 grid). Trends and behavioural conclusions are consistent across both; absolute numbers differ because the larger grid means longer paths and proportionally higher cumulative costs.

### Table III — Navigation results by scenario

| ID | Scenario | Agents | λ | HFL Len | HFL PJ | HFL W̄ | SP Len | SP PJ | SP W̄ |
|---|---|---|---|---|---|---|---|---|---|
| S1 | Cold Start | 3 | 1.000 | 113 | 139.3 | 1.686 | 113 | 139.3 | 1.686 |
| S2 | Cold Start | 10 | 1.000 | 113 | 139.3 | 1.681 | 113 | 139.3 | 1.681 |
| S3 | Sudden Change | 3 | 0.857 | 119 | 142.5 | 1.191 | 113 | 143.1 | 1.686 |
| S4 | Sudden Change | 10 | 0.774 | 119 | 142.3 | 1.191 | 113 | 143.1 | 1.686 |
| S5 | Sunny | 3 | 0.556 | 125 | 140.5 | 1.074 | 113 | 141.2 | 1.686 |
| S6 | Sunny | 10 | 0.383 | 125 | 138.3 | 1.075 | 113 | 141.2 | 1.686 |
| S7 | Rainy | 3 | 0.671 | 125 | 157.4 | 1.451 | 113 | 159.1 | 1.686 |
| S8 | Rainy | 10 | 0.543 | 125 | 161.8 | 1.453 | 113 | 163.4 | 1.686 |

> **Columns:** Len = path length in cells · PJ = total navigation cost · W̄ = mean per-step comfort cost

### S1 & S2 — Cold Start: safe fallback without data

With no meaningful crowd data (N_eff = 0), λ = 1.000 for both scenarios regardless of agent count. The cold-start gate correctly suppresses any reliance on the uninitialised experience manifold — HFL-RL and SP produce identical paths (113 cells, PJ = 139.3, W̄ = 1.686).

This is intentional and desirable. The system does not require a minimum data threshold to operate; it guarantees a geometrically valid route from day one, with improvement accruing gradually and transparently as the manifold is populated. New deployments work out of the box with zero degradation relative to a conventional navigation system.

### S3 & S4 — Sudden Change: conservative rerouting under conflict

A sudden environmental event (flooding, unexpected construction) is simulated across ~150 randomly distributed cell clusters in the central map region. Edge volatility spikes to Vₑ = 5.0 and inter-agent variance is elevated to σ = 1.2, reflecting genuine disagreement between agents encountering the event at different times.

- λ = 0.857 (3 agents) and λ = 0.774 (10 agents) — a cautious, partial transition away from shortest-path
- HFL-RL paths extend to 119 cells (+6 vs baseline) but achieve W̄ = 1.191, a **29.4% reduction** in per-step comfort cost
- Increasing agents from 3 to 10 reduces λ by only 0.083 — far less than in the sunny case — because the `exp(−σ)` term treats a large but divided crowd with the same caution as a small one

The system does not overreact to the high-volatility signal. The balance between responsiveness and conservatism is automatically regulated by the interaction between Vₑ (urgency) and σ (disagreement), with no manually tuned switching threshold.

### S5 & S6 — Sunny weather: maximum learning signal

Stable, low-variance conditions (σ = 0.30) produce the strongest learning signal of any scenario:

- λ = 0.556 (3 agents) → λ = 0.383 (10 agents) — the lowest values observed across all tests
- The λ drop from adding agents (0.173) is the largest of any paired comparison, because consistent observations correctly earn greater system trust
- HFL-RL achieves W̄ = 1.074, approaching the theoretical minimum of 1.0
- Terrain composition shifts almost entirely to good pavement (115/125 cells) and crosswalks (7 cells), eliminating highway crossings and grass shortcuts present in the cold-start route
- **S6 achieves PJ = 138.3 — the lowest total cost of all eight scenarios, below even the cold-start shortest path (139.3)**

The HFL-RL route is 10.6% longer in path length but **36.2% cheaper per step**, confirming that the additional distance is more than compensated by surface quality. This trade-off is especially relevant for visually impaired users, older adults, and wheelchair users who prioritise stability and safety over raw speed.

### S7 & S8 — Rainy weather: cost inflation without topological change

Rain multipliers inflate costs substantially across the map — dirt paths (3.5×), water puddles (5.0×), unpaved areas (3.0×) — while good pavement receives only a 1.3× multiplier. The system responds not by changing which route it takes, but by confirming that the pavement-prioritised route chosen under sunny conditions is also the least penalised under rain.

- HFL-RL path length matches the sunny case at 125 cells; terrain composition is likewise dominated by good pavement and crosswalks
- PJ rises to 157.4–161.8 vs 138.3–140.5 under sunny conditions — an increase of 14–21 units attributable entirely to the uniform weather multipliers, not to any topological change
- λ sits between 0.543 and 0.671, held higher than the sunny case by moderate inter-agent disagreement (σ = 0.60)

This confirms that the experience manifold captures **durable surface quality** rather than condition-specific artefacts — the routing strategy it induces is weatherproof at the topological level.

### Summary of key behavioural properties

The eight scenarios collectively confirm four properties that make HFL-RL viable for real-world pedestrian navigation:

| Property | Evidence |
|---|---|
| **Safe initialisation** | S1/S2: λ = 1.0, identical to SP — system is always deployable with no degradation |
| **Adaptive trust** | S3/S4 vs S5/S6: λ responds to both data quantity and inter-agent agreement, not just agent count |
| **Comfort-optimal routing** | S6: 36.2% per-step cost reduction at only 10.6% path length overhead vs SP |
| **Weather robustness** | S7/S8: same route topology as sunny case — manifold captures surface quality, not weather noise |

<br>
<br>
<br>

---

# Developing
---


# Multi-User Implicit Feedback System

Instead of relying on explicit ratings, the system infers route comfort preferences from trajectory deviations, temporal weight updates, and cross-user imitation learning (IRL). It features dynamic trust scheduling ($\lambda$), short/long-term memory gating, and fair baseline comparisons.

## 🔍 Interpreting Your Console Output

Based on the logs you shared, here's what each metric means and why it looks the way it does:

| Metric | Meaning | Your Output Pattern | Expected? |
| :--- | :--- | :--- | :--- |
| **$\lambda$ decay** | Trust balance: $1.0$ = pure distance, lower = more trust in learned comfort weights | Drops each round (e.g., Cold Start: $1.000 \rightarrow 0.729$) | ✅ **Yes.** Driven by $\text{decay} = 0.85 + 0.05 \times (1 - \text{success\_ratio})$ |
| **total_J change** | Navigation cost improvement over rounds | Decreases steadily ($+1.3\%$ to $+6.4\%$) | ✅ **Yes.** Lower $J$ = better route |
| **avg_W drop** | Average per-step comfort weight | Falls from $\sim1.6 \rightarrow \sim1.19$ | ✅ **Yes.** System learns smoother/safer cells |
| **IRL = 0** | Cross-user imitation updates triggered | Always $0$ across all 4 scenarios | ✅ **Yes.** `BAD_EXPERIENCE_THRESH=0.10` is intentionally strict to prevent overfitting to noisy deviations |
| **div (0.3–0.9)** | Non-compliant divergence from suggestion | Varies by scenario | ✅ **Normal.** Higher div in rainy/sudden = healthy exploration under uncertainty |
| **✓ Model improving** | $(Round1\_J - Final\_J) / Round1\_J > 0$ | All scenarios pass | ✅ Confirms dynamic $\lambda$ scheduling fixed the cold-start stagnation |

---
<br>

## ⚙️ Configuration & Tuning Guide

| Parameter | Location | Effect | Recommended Range |
| :--- | :--- | :--- | :--- |
| **decay base** | `implicit_feedback_v2.py` $\rightarrow$ `run_simulation()` | How fast $\lambda$ trusts learned weights | $0.80$ (fast) $\leftrightarrow$ $0.90$ (conservative) |
| **$\lambda$ floor** | `self.lam = max(0.10, ...)` | Minimum distance trust to avoid overfitting | $0.05 - 0.15$ |
| **BAD_EXPERIENCE_THRESH** | Module level constant | IRL trigger sensitivity | $0.08$ (more IRL) $\leftrightarrow$ $0.12$ (stricter) |
| **EMA_SHORT / EMA_LONG** | Module level constants | Memory update speed | $0.30 - 0.50$ (short), $0.02 - 0.08$ (long) |
| **User ratios** | `ImplicitFeedbackSimulationV2.__init__()` | Compliant / NC / Autonomous split | Keep $\text{NC} \ge 0.25$ for meaningful divergence data |

---

## 🧠 Why IRL=0 Everywhere? (Not a Bug)

The system uses a $10\%$ relative improvement threshold to trigger cross-user IRL updates:

```python
if improvement > BAD_EXPERIENCE_THRESH:  # > 0.10
    irl_updates.append(...)
```
<br>
<br>
<br>


---
## Limitations

The current implementation is a **simulation-based proof of concept**. The real-map evaluation (Part A) removes the synthetic-grid limitation, but several practical dimensions remain open before real-world deployment.

**Synthetic users and injected hazards.** Part A runs on a real OpenStreetMap network, but users are simulated: reliability, perception noise and turn-back behaviour are *modelled*, not observed, and hazards are injected (a fixed number per route, by default partly on the shortest-path corridor). Real visually impaired users may have idiosyncratic comfort preferences — reactions to specific textures, slopes, noise levels or crowding — that a tag-derived cost table cannot capture.

**One city, one weather condition per run.** The reported results cover central Baku under rain only. Tag coverage and quality in OpenStreetMap vary by city, and the tag → terrain mapping is a modelling assumption (for example, paths with unknown surface are assumed degraded, and some terrain classes cannot be derived from tags at all).

**Hazards are learned only after first encounters.** A hazard must be met by at least one user before it can be learned, leaving residual deviations, and the bounded cost (`J` per segment ≤ `D̂`) means a hazard is never costlier than distance-only routing — detours are accepted only when alternatives are not much longer. Comfort estimates are advisory and **do not replace safety-critical assistive aids**.

**Privacy is by design, not by proof.** Only aggregated per-segment statistics leave a user or edge node, but the framework assumes honest-but-curious nodes and offers no formal guarantee (no differential privacy or secure aggregation); aggregates over very few users could still leak information about individuals.

**Centralised baseline is a reference, not a competing algorithm.** It uses the same estimator and gate on pooled data, so parity with HFL-RL is expected. Comparisons with independent non-federated comfort-aware routers (for example static tag-only comfort routing) are not yet included.

**No live data pipeline.** The simulation generates reports sequentially in memory. A real deployment would require a continuous ingestion pipeline where agents stream comfort feedback asynchronously, edge nodes re-aggregate on rolling time windows, and the global manifold updates incrementally.

**Computational and energy footprint.** On mobile hardware with battery constraints, aggregation and route optimisation would need to be offloaded to edge and global servers, with user devices limited to transmitting and receiving route-relevant data. This infrastructure dependency has not been benchmarked.

**Cold-start duration is scenario-dependent.** The speed at which λ transitions from 1.0 to experience-driven values depends on agent density, weather stability and terrain diversity. In sparse or low-footfall zones the cold-start period could be prolonged indefinitely. In Part A the OSM-tag prior (`prior_neff = 5`) softens cold start; the strict variant (`--prior-neff 0`) should be reported alongside.

**Fixed hyper-parameters.** β = 10 (and the other values in `Config`) are used throughout the main experiments. `--beta` and `--kappa` allow sensitivity runs, but optimal values are likely context-dependent and have not been tuned against real data.

---

## Future Directions

**Validation with visually impaired participants.** The most pressing next step is replacing simulated users with genuine comfort feedback from visually impaired participants on the real network, and extending the evaluation to further cities, weather regimes and network densities.

**Longitudinal user studies.** Simulation results must ultimately be validated in naturalistic settings over extended periods. Such studies would reveal whether the manifold's learned surface preferences align with real user experience, and at what data volume the system transitions from cold start to meaningfully personalised routing.

**Stronger baselines and ablations.** Add independent non-federated comfort-aware routers (static tag-only comfort routing, simple moving-average comfort estimates) and report the full ablation grid (`--prior-neff 0`, `--placement random`, `--kappa 0`, `--unscaled-comfort`, `--beta` sweep) with paired significance tests.

**Passive feedback inference.** Requiring explicit post-trip ratings imposes burden on users. A more scalable direction is passive inference from navigation behaviour — for example, detecting unexpected traversal latency relative to predicted duration, or identifying route deviations — and presenting an optional lightweight voice-feedback prompt only when anomalous behaviour is detected.

**Incremental manifold updates.** A production system would benefit from incremental update mechanisms — maintaining running weighted statistics at the edge level and propagating only deltas to the global server — to reduce bandwidth and latency.

**Dynamic β calibration.** Rather than fixing β = 10, future work could learn the optimal β for a given deployment area from historical feedback patterns, or adapt it dynamically based on observed agent agreement trends over time.

**Multi-modal and indoor extension.** The current model is strictly outdoor and pedestrian. Extending to mixed indoor-outdoor journeys (shopping centres, transit stations, underpasses) and to multi-modal trips that include public transport segments would substantially broaden the system's practical utility for visually impaired users.

**Social infrastructure vision.** Sighted pedestrians navigating the same urban environment naturally generate comfort observations as a by-product of daily movement. Future work could make this contribution mechanism explicit — allowing any pedestrian app to act as a passive contributor to the experience manifold — creating a self-sustaining feedback loop in which the broader pedestrian community continuously curates safer, more comfortable paths for visually impaired users without any additional individual effort.

---

## Data attribution

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), available under the Open Database Licence (ODbL). Network extraction uses [OSMnx](https://github.com/gboeing/osmnx) (Boeing, 2017).