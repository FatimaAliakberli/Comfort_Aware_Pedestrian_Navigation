"""
=============================================================================
_scenario_runner.py  —  Shared visualisation helper for individual scenarios
=============================================================================
Imported by each s*.py script; not meant to be run directly.
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from city_map.city_map import CityMap200
from formulas.formulas import astar
from scenarios.scenario_builder import build_scenario, analyze_path

OUTPUT_DIR = 'output_images'


def run_single_scenario(
    scenario_type: str,
    n_agents:      int,
    weather:       str,
    label:         str,
    start:         tuple,
    goal:          tuple,
    city:          CityMap200,
    save:          bool = True,
    output_dir:    str  = OUTPUT_DIR,
) -> dict:
    """
    Run one scenario: compute paths, print metrics, optionally save figure.

    Returns
    -------
    dict with keys: hfl_stats, sp_stats, lam, N_eff, sigma
    """
    terrain = city.grid

    _, lam, N_eff, sigma, W_global, sudden_bad = build_scenario(
        scenario_type, n_agents, city, weather,
    )

    path_hfl   = astar(terrain, start, goal, W_global, lam, weather)
    path_short = astar(terrain, start, goal, {},       1.0, weather)

    # Re-evaluate SP with the same λ as HFL-RL (fair comparison)
    sh = analyze_path(path_hfl,   terrain, W_global, lam, weather)
    ss = analyze_path(path_short, terrain, W_global, lam, weather)

    # ── Console report ──────────────────────────────────────────────────
    sep = '─' * 55
    print(f"\n{sep}")
    print(f"  {label}  ({scenario_type}, weather={weather})")
    print(sep)
    print(f"  λ={lam:.3f}   N_eff={N_eff}   σ={sigma:.2f}")
    print(f"  HFL-RL : len={sh.get('path_length',0):4d}  "
          f"J={sh.get('total_J',0):8.1f}  W̄={sh.get('avg_W',0):.3f}")
    print(f"  SP     : len={ss.get('path_length',0):4d}  "
          f"J={ss.get('total_J',0):8.1f}  W̄={ss.get('avg_W',0):.3f}")

    hfl_wins_J = sh.get('total_J', 999) < ss.get('total_J', 0)
    pct_J = (ss.get('total_J',0) - sh.get('total_J',0)) / max(ss.get('total_J',1), 1) * 100
    print(f"  HFL-RL wins J? {'✓ YES  ↓'+f'{pct_J:.1f}%' if hfl_wins_J else '✗ no'}")

    top3 = sorted(sh.get('terrain_counts',{}).items(), key=lambda x: -x[1])[:3]
    print(f"  HFL top terrains: {top3}")

    # ── Figure ───────────────────────────────────────────────────────────
    if save:
        _save_figure(city, path_hfl, path_short, sudden_bad,
                     sh, ss, lam, N_eff, sigma, label,
                     scenario_type, n_agents, output_dir)

    return {'hfl_stats': sh, 'sp_stats': ss,
            'lam': lam, 'N_eff': N_eff, 'sigma': sigma}


def _save_figure(city, path_hfl, path_short, sudden_bad,
                 sh, ss, lam, N_eff, sigma, label,
                 scenario_type, n_agents, output_dir):
    """Render and save the map + bar-chart figure for one scenario."""
    fig, (ax_map, ax_cost) = plt.subplots(
        1, 2, figsize=(18, 9),
        gridspec_kw={'width_ratios': [1.55, 1]},
    )
    fig.patch.set_facecolor('#0d0d1a')

    # Map ----------------------------------------------------------------
    rgb = city.create_rgb_grid()
    ax_map.imshow(rgb, origin='lower', interpolation='nearest', aspect='equal')
    if sudden_bad:
        br = [r for r, c in sudden_bad]
        bc = [c for r, c in sudden_bad]
        ax_map.scatter(bc, br, c='red', s=0.5, alpha=0.15, zorder=2)
    if path_short:
        ax_map.plot([p[1] for p in path_short], [p[0] for p in path_short],
                    'w--', lw=1.0, alpha=0.55, label='Shortest', zorder=4)
    if path_hfl:
        ax_map.plot([p[1] for p in path_hfl], [p[0] for p in path_hfl],
                    color='#00FFCC', lw=1.8, alpha=0.95, label='HFL-RL', zorder=5)

    ax_map.set_xticks([]); ax_map.set_yticks([])
    ax_map.set_title(f'{label}\nλ={lam:.3f}  N_eff={N_eff}  σ={sigma:.2f}',
                     fontsize=11, fontweight='bold', color='white', pad=6)
    ax_map.set_facecolor('#1a1a2e')
    ax_map.legend(fontsize=8, loc='lower right',
                  facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')

    # Bar chart ----------------------------------------------------------
    blabels = ['Path Len\n(cells)', 'Total J\n(fair λ)',
               'Avg W̄\n(per step)', 'Total D\n(Euclidean)']
    hfl_v = [sh.get('path_length',0), sh.get('total_J',0),
              sh.get('avg_W',0),       sh.get('total_D',0)]
    sp_v  = [ss.get('path_length',0), ss.get('total_J',0),
              ss.get('avg_W',0),       ss.get('total_D',0)]
    x = np.arange(len(blabels)); bw = 0.35
    b1 = ax_cost.bar(x-bw/2, hfl_v, bw, label='HFL-RL',
                     color='#00FFCC', alpha=0.85, edgecolor='#00aa88')
    b2 = ax_cost.bar(x+bw/2, sp_v,  bw, label='Shortest (fair λ)',
                     color='#FF6B6B', alpha=0.85, edgecolor='#cc4444')
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(blabels, fontsize=8, color='#cccccc')
    ax_cost.set_facecolor('#12122a')
    ax_cost.tick_params(colors='#cccccc', labelsize=8)
    ax_cost.spines[:].set_color('#333355')
    ax_cost.set_title('Cost Comparison (SP re-evaluated with same λ)',
                      fontsize=9, color='#aaaacc', pad=6)
    ax_cost.legend(fontsize=8, facecolor='#12122a',
                   edgecolor='#444', labelcolor='white')
    for bar in b1:
        h = bar.get_height()
        ax_cost.text(bar.get_x()+bar.get_width()/2, h+1,
                     f'{h:.1f}', ha='center', va='bottom',
                     fontsize=7, color='#00FFCC')
    for bar in b2:
        h = bar.get_height()
        ax_cost.text(bar.get_x()+bar.get_width()/2, h+1,
                     f'{h:.1f}', ha='center', va='bottom',
                     fontsize=7, color='#FF6B6B')

    slug = f"{scenario_type}_{n_agents}_agents"
    out  = os.path.join(output_dir, f'{slug}.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"  ✓ Figure saved: {out}")
