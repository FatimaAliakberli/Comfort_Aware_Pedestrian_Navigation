"""
=============================================================================
run_full.py  —  Full pipeline: all 8 scenarios → Table III + all figures
=============================================================================
This is the primary entry point.  Running it reproduces every output from
the original HFL-RL v3 script:

  hfl_rl_v3_scenarios.png      — 8-scenario overview map
  hfl_rl_v3_city_map.png       — annotated city map
  hfl_rl_v3_results_table.png  — Table III (fair λ comparison)
  hfl_rl_v3_tradeoff.png       — trade-off bar charts
Usage:
    python main/run_full.py
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from city_map.city_map  import CityMap200, BUILDING
from city_map.generate_map_image import make_city_map
from formulas.formulas  import astar
from scenarios.scenario_builder import build_scenario, analyze_path
from implicit_feedback_v2 import ImplicitFeedbackSimulationV2

OUTPUT_DIR = 'output_images'

START = (0, 0)
GOAL  = (185, 190)

ALL_SCENARIOS = [
    ('cold_start',    3,  'neutral', 'Cold Start\n(3 agents)'),
    ('cold_start',    10, 'neutral', 'Cold Start\n(10 agents)'),
    ('sudden_change', 3,  'neutral', 'Sudden Change\n(3 agents)'),
    ('sudden_change', 10, 'neutral', 'Sudden Change\n(10 agents)'),
    ('sunny',         3,  'sunny',   'Sunny Weather\n(3 agents)'),
    ('sunny',         10, 'sunny',   'Sunny Weather\n(10 agents)'),
    ('rainy',         3,  'rainy',   'Rainy Weather\n(3 agents)'),
    ('rainy',         10, 'rainy',   'Rainy Weather\n(10 agents)'),
]


# =============================================================================
# Figure 1 — 8-scenario overview
# =============================================================================
def _render_one_scenario(city, stype, nagents, weather, label,
                          ax_map, ax_cost):
    terrain = city.grid
    _, lam, N_eff, sigma, W_global, sudden_bad = build_scenario(
        stype, nagents, city, weather)

    path_hfl   = astar(terrain, START, GOAL, W_global, lam, weather)
    path_short = astar(terrain, START, GOAL, {},       1.0, weather)

    sh = analyze_path(path_hfl,   terrain, W_global, lam, weather)
    ss = analyze_path(path_short, terrain, W_global, lam, weather)

    # Map
    rgb = city.create_rgb_grid()
    ax_map.imshow(rgb, origin='lower', interpolation='nearest', aspect='equal')
    if sudden_bad:
        br = [r for r, c in sudden_bad]; bc = [c for r, c in sudden_bad]
        ax_map.scatter(bc, br, c='red', s=0.5, alpha=0.15, zorder=2)
    if path_short:
        ax_map.plot([p[1] for p in path_short], [p[0] for p in path_short],
                    'w--', lw=1.0, alpha=0.55, label='Shortest', zorder=4)
    if path_hfl:
        ax_map.plot([p[1] for p in path_hfl], [p[0] for p in path_hfl],
                    color='#00FFCC', lw=1.8, alpha=0.95, label='HFL-RL', zorder=5)
    ax_map.plot(START[1], START[0], 'o', color='#39FF14', ms=9,
                markeredgecolor='darkgreen', markeredgewidth=1.5, zorder=10)
    ax_map.plot(GOAL[1],  GOAL[0],  '*', color='#FFD700', ms=13,
                markeredgecolor='darkred',  markeredgewidth=1.5, zorder=10)
    ax_map.text(START[1]+2, START[0]+2, 'A', color='white',
                fontsize=6, fontweight='bold', zorder=11)
    ax_map.text(GOAL[1]-6,  GOAL[0]+2,  'B', color='white',
                fontsize=6, fontweight='bold', zorder=11)
    ax_map.set_xticks([]); ax_map.set_yticks([])
    ax_map.set_title(f'{label}\nλ={lam:.3f}  N_eff={N_eff}  σ={sigma:.2f}',
                     fontsize=8, fontweight='bold', color='white', pad=4)
    ax_map.set_facecolor('#1a1a2e')
    ax_map.legend(fontsize=6, loc='lower right',
                  facecolor='#1a1a2e', edgecolor='#444', labelcolor='white',
                  handlelength=1.5)

    # Bar chart
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
    ax_cost.set_xticklabels(blabels, fontsize=6, color='#cccccc')
    ax_cost.set_facecolor('#12122a')
    ax_cost.tick_params(colors='#cccccc', labelsize=6)
    ax_cost.spines[:].set_color('#333355')
    ax_cost.set_title('Cost Comparison (SP re-evaluated w/ same λ)',
                      fontsize=6.5, color='#aaaacc', pad=3)
    ax_cost.legend(fontsize=5.5, facecolor='#12122a',
                   edgecolor='#444', labelcolor='white')
    for bar in b1:
        h = bar.get_height()
        ax_cost.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}',
                     ha='center', va='bottom', fontsize=5, color='#00FFCC')
    for bar in b2:
        h = bar.get_height()
        ax_cost.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}',
                     ha='center', va='bottom', fontsize=5, color='#FF6B6B')
    return sh, ss, lam, N_eff, sigma


def run_all_scenarios(city):
    print("=" * 70)
    print("  HFL-RL v3 — 200×200 MAP — FAIR J COMPARISON")
    print("=" * 70)

    fig = plt.figure(figsize=(22, 64))
    fig.patch.set_facecolor('#0d0d1a')
    fig.text(0.5, 0.998,
             'HFL-RL Navigation System  —  200×200 City Map',
             ha='center', va='top', fontsize=18, fontweight='bold', color='white')
    fig.text(0.5, 0.993,
             'Cyan = HFL-RL  |  White dashed = Shortest path  |  '
             'Both evaluated with same λ (fair comparison)',
             ha='center', va='top', fontsize=10, color='#8899bb')

    summary = []
    for idx, (stype, nagents, weather, label) in enumerate(ALL_SCENARIOS):
        ax_map  = fig.add_axes([0.04, 0.879 - idx*0.122, 0.55, 0.106])
        ax_cost = fig.add_axes([0.62, 0.879 - idx*0.122, 0.36, 0.096])
        print(f"\n{'─'*50}\n  S{idx+1}: {label.replace(chr(10),' ')}")
        sh, ss, lam, N_eff, sigma = _render_one_scenario(
            city, stype, nagents, weather, label, ax_map, ax_cost)

        hfl_wins_J = sh.get('total_J', 999) < ss.get('total_J', 0)
        pct_J = (ss.get('total_J',0)-sh.get('total_J',0)) / max(ss.get('total_J',1),1)*100
        pct_W = (ss.get('avg_W',0) -sh.get('avg_W',0))  / max(ss.get('avg_W',1),1)*100
        print(f"  λ={lam:.3f}  N_eff={N_eff}  σ={sigma:.2f}")
        print(f"  HFL-RL: len={sh.get('path_length',0)}  J={sh.get('total_J',0):.1f}  W̄={sh.get('avg_W',0):.3f}")
        print(f"  SP    : len={ss.get('path_length',0)}  J={ss.get('total_J',0):.1f}  W̄={ss.get('avg_W',0):.3f}")
        print(f"  HFL-RL wins J? {'YES ↓'+f'{pct_J:.1f}%' if hfl_wins_J else 'tie/no'}")
        summary.append({
            'label': label.replace('\n',' '),
            'lambda': lam, 'N_eff': N_eff, 'sigma': sigma,
            'hfl_len': sh.get('path_length',0), 'hfl_J': sh.get('total_J',0),
            'hfl_W':  sh.get('avg_W',0),
            'sp_len':  ss.get('path_length',0), 'sp_J':  ss.get('total_J',0),
            'sp_W':   ss.get('avg_W',0),
        })

    # Summary table inside figure
    ax_t = fig.add_axes([0.01, 0.001, 0.98, 0.050])
    ax_t.set_facecolor('#0d0d1a'); ax_t.axis('off')
    cols = ['ID','Scenario','λ','N_eff','σ',
            'HFL Len','HFL PJ','HFL W̄',
            'SP Len','SP PJ (fair λ)','SP W̄','HFL wins J?']
    ids  = [f'S{i+1}' for i in range(8)]
    data = []
    for i, r in enumerate(summary):
        wins = '✓ YES' if r['hfl_J']<r['sp_J'] else ('= TIE' if r['lambda']==1.0 else '✗ NO')
        data.append([ids[i], r['label'],
                     f"{r['lambda']:.3f}", str(r['N_eff']), f"{r['sigma']:.2f}",
                     str(r['hfl_len']), f"{r['hfl_J']:.1f}", f"{r['hfl_W']:.3f}",
                     str(r['sp_len']),  f"{r['sp_J']:.1f}",  f"{r['sp_W']:.3f}",
                     wins])
    tbl = ax_t.table(cellText=data, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.7)
    grp = {0:'#1a2a3a',1:'#1a2a3a',2:'#2a1a1a',3:'#2a1a1a',
           4:'#1a2a1a',5:'#1a2a1a',6:'#1a1a2e',7:'#1a1a2e'}
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_edgecolor('#334466')
        if ri == 0:
            cell.set_facecolor('#0d1a3a')
            cell.set_text_props(color='#88ccff', fontweight='bold')
        else:
            cell.set_facecolor(grp.get(ri-1, '#101028'))
            if ci == 11:
                col = '#44ff88' if '✓' in data[ri-1][11] else \
                      ('#aaaaaa' if '=' in data[ri-1][11] else '#ff6666')
                cell.set_text_props(color=col, fontweight='bold')
            else:
                cell.set_text_props(color='#ccddee')

    out = os.path.join(OUTPUT_DIR, 'Scenarios.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"\n✓ Saved: {out}")
    return summary


# =============================================================================
# Figure 3 — Table III
# =============================================================================
def make_results_table(summary):
    fig, ax = plt.subplots(figsize=(20, 5))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a'); ax.axis('off')
    ax.set_title(
        'Table III — Navigation Results  '
        '(SP re-evaluated with HFL-RL λ for fair comparison)',
        fontsize=13, fontweight='bold', color='white', pad=16)

    cols = ['ID','Scenario','λ','N_eff','σ',
            'HFL Len','HFL PJ','HFL W̄',
            'SP Len','SP PJ','SP W̄',
            'J saved','W̄ saved']
    ids  = [f'S{i+1}' for i in range(8)]
    data = []
    for i, r in enumerate(summary):
        j_saved = r['sp_J'] - r['hfl_J']
        w_saved = r['sp_W'] - r['hfl_W']
        j_pct   = j_saved / max(r['sp_J'], 1) * 100
        w_pct   = w_saved / max(r['sp_W'], 1) * 100
        data.append([
            ids[i], r['label'],
            f"{r['lambda']:.3f}", str(r['N_eff']), f"{r['sigma']:.2f}",
            str(r['hfl_len']), f"{r['hfl_J']:.1f}", f"{r['hfl_W']:.3f}",
            str(r['sp_len']),  f"{r['sp_J']:.1f}",  f"{r['sp_W']:.3f}",
            f"↓{j_pct:.1f}%",  f"↓{w_pct:.1f}%",
        ])

    tbl = ax.table(cellText=data, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 2.2)
    grp = {0:'#1a2a3a',1:'#1a2a3a',2:'#2a1a1a',3:'#2a1a1a',
           4:'#1a2a1a',5:'#1a2a1a',6:'#1a1a2e',7:'#1a1a2e'}
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_edgecolor('#334466')
        if ri == 0:
            cell.set_facecolor('#0d1a3a')
            cell.set_text_props(color='#88ccff', fontweight='bold')
        else:
            cell.set_facecolor(grp.get(ri-1, '#101028'))
            if ci in [11, 12]:
                cell.set_text_props(color='#44ff88', fontweight='bold')
            elif ci in [6, 7]:
                cell.set_text_props(color='#00FFCC')
            elif ci in [9, 10]:
                cell.set_text_props(color='#FF6B6B')
            else:
                cell.set_text_props(color='#ccddee')

    out = os.path.join(OUTPUT_DIR, 'Results_table.png')
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"✓ Saved: {out}")


# =============================================================================
# Figure 4 — Trade-off chart
# =============================================================================
def make_tradeoff_chart(summary):
    ids   = [f'S{i+1}' for i in range(8)]
    hfl_W = [r['hfl_W']   for r in summary]
    sp_W  = [r['sp_W']    for r in summary]
    hfl_J = [r['hfl_J']   for r in summary]
    sp_J  = [r['sp_J']    for r in summary]
    hfl_L = [r['hfl_len'] for r in summary]
    sp_L  = [r['sp_len']  for r in summary]
    lams  = [r['lambda']  for r in summary]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.patch.set_facecolor('#0d0d1a')
    fig.suptitle(
        'HFL-RL vs Shortest Path — Key Metrics  '
        '(SP fairly re-evaluated with HFL-RL λ)',
        fontsize=14, fontweight='bold', color='white')

    panels = [
        (axes[0], hfl_W, sp_W,
         'Avg W̄  (per-step comfort cost)',
         'Lower = smoother, safer route — HFL-RL wins all non-cold-start scenarios'),
        (axes[1], hfl_J, sp_J,
         'Total PJ  (navigation cost, fair λ)',
         'HFL-RL total cost beats SP once both use same λ'),
        (axes[2], hfl_L, sp_L,
         'Path length (cells)',
         'HFL-RL takes longer detour — but comfort gain outweighs it'),
    ]
    x = np.arange(len(ids)); bw = 0.35

    for ax, hfl_v, sp_v, ylabel, subtitle in panels:
        ax.set_facecolor('#12122a')
        for x0, x1, col in [(-0.5,1.5,'#1a3a1a'),(1.5,3.5,'#3a1a1a'),
                              (3.5,5.5,'#2a3a0a'),(5.5,7.5,'#0a1a3a')]:
            ax.axvspan(x0, x1, alpha=0.18, color=col, zorder=0)

        b1 = ax.bar(x-bw/2, hfl_v, bw, label='HFL-RL',
                    color='#00FFCC', alpha=0.85, edgecolor='#00aa88')
        b2 = ax.bar(x+bw/2, sp_v,  bw, label='Shortest (fair λ)',
                    color='#FF6B6B', alpha=0.85, edgecolor='#cc4444')
        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.0f}',
                    ha='center', va='bottom', fontsize=6.5, color='#00FFCC')
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.0f}',
                    ha='center', va='bottom', fontsize=6.5, color='#FF6B6B')
        for i, lam in enumerate(lams):
            ax.text(i, ax.get_ylim()[0], f'λ={lam:.2f}',
                    ha='center', va='bottom', fontsize=5.5, color='#8899bb')
        ax.set_xticks(x)
        ax.set_xticklabels(ids, color='#cccccc', fontsize=9)
        ax.tick_params(colors='#aaaaaa')
        ax.spines[:].set_color('#334466')
        ax.set_ylabel(ylabel, color='#cccccc', fontsize=9)
        ax.set_title(subtitle, color='#aaaacc', fontsize=8, pad=6)
        ax.legend(fontsize=8, facecolor='#12122a',
                  edgecolor='#445566', labelcolor='white')

    grp_labels = [('Cold\nStart','#448844'),('Sudden\nChange','#884444'),
                  ('Sunny','#888833'),('Rainy','#334488')]
    for axi in axes:
        for i, (lbl, col) in enumerate(grp_labels):
            axi.text(i*2+0.5, axi.get_ylim()[1]*0.97, lbl,
                     ha='center', va='top', fontsize=7, color=col, style='italic')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Tradeoff.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"✓ Saved: {out}")


# =============================================================================
# Figure 5 — Implicit Feedback simulation results
# =============================================================================
def make_feedback_figure(city: CityMap200):
    """
    Run ImplicitFeedbackSimulationV2 across four representative scenarios and
    produce a single multi-panel figure that shows:

      Col 1 — Suggested path evolution on the city map (first vs last round)
      Col 2 — Per-round total_J for compliant / non-compliant / autonomous users
      Col 3 — λ decay and IRL update count over rounds
      Col 4 — Path divergence (NC vs suggestion) over rounds

    The figure is saved as Implicit_Feedback.png.
    """

    # Four scenarios to compare — (scenario_type, weather, event_confidence, rounds, label)
    FEEDBACK_RUNS = [
        ('sunny',         'sunny',   0.6, 6, 'Sunny / High data'),
        ('rainy',         'rainy',   0.6, 6, 'Rainy / Weather stress'),
        ('sudden_change', 'neutral', 0.4, 5, 'Sudden Change / IRL stress'),
        ('cold_start',    'neutral', 0.4, 5, 'Cold Start / Sparse data'),
    ]

    N_ROWS = len(FEEDBACK_RUNS)
    fig = plt.figure(figsize=(26, 7 * N_ROWS))
    fig.patch.set_facecolor('#0d0d1a')
    fig.text(
        0.5, 0.998,
        'Implicit Feedback System (v2) — Compliant · Non-Compliant · Autonomous Users',
        ha='center', va='top', fontsize=15, fontweight='bold', color='white',
    )
    fig.text(
        0.5, 0.991,
        'Cyan = suggested path  |  Orange = non-compliant paths  |  '
        'Purple = autonomous walk  |  Gold star = IRL cross-user update',
        ha='center', va='top', fontsize=9, color='#8899bb',
    )

    rgb = city.create_rgb_grid()

    def _dark(ax):
        ax.set_facecolor('#12122a')
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        ax.spines[:].set_color('#334466')

    row_h = 0.88 / N_ROWS

    for ridx, (stype, weather, ec, n_rounds, label) in enumerate(FEEDBACK_RUNS):
        print(f"  [Feedback] {label} ...")

        sim = ImplicitFeedbackSimulationV2(
            city               = city,
            scenario_type      = stype,
            weather            = weather,
            n_agents           = 20,
            compliant_ratio    = 0.40,
            noncompliant_ratio = 0.35,
            event_confidence   = ec,
            seed               = 42 + ridx,
        )
        logs = sim.run_simulation(total_rounds=n_rounds)

        top    = 0.975 - (ridx + 1) * row_h
        rounds = [lg['round'] for lg in logs]

        # ── Panel A: city map with first and last suggested path ──────────
        ax_map = fig.add_axes([0.01, top, 0.21, row_h * 0.87])
        ax_map.imshow(rgb, origin='lower', interpolation='nearest', aspect='equal')

        # First-round suggested path (white dashed — starting point)
        path_r1 = logs[0]['suggested_path']
        if path_r1:
            ax_map.plot([p[1] for p in path_r1], [p[0] for p in path_r1],
                        'w--', lw=0.9, alpha=0.45, label=f'Suggested R1')

        # Last-round suggested path (cyan — after learning)
        path_rN = logs[-1]['suggested_path']
        if path_rN:
            ax_map.plot([p[1] for p in path_rN], [p[0] for p in path_rN],
                        color='#00FFCC', lw=1.8, alpha=0.95, label=f'Suggested R{n_rounds}')

        # One NC path from the last round (orange, faint)
        nc_last = logs[-1]['nc_paths']
        if nc_last:
            p = nc_last[0]
            ax_map.plot([c[1] for c in p], [c[0] for c in p],
                        color='#FF9944', lw=1.1, alpha=0.60, label='NC path')

        # One autonomous walk from the last round (purple, faint)
        au_last = logs[-1]['au_paths']
        if au_last:
            p = au_last[0]
            ax_map.plot([c[1] for c in p], [c[0] for c in p],
                        color='#CC99FF', lw=0.9, alpha=0.55, label='Autonomous')

        ax_map.plot(START[1], START[0], 'o', color='#39FF14', ms=8,
                    markeredgecolor='darkgreen', zorder=10)
        ax_map.plot(GOAL[1],  GOAL[0],  '*', color='#FFD700', ms=12,
                    markeredgecolor='darkred',  zorder=10)
        ax_map.set_xticks([]); ax_map.set_yticks([])
        ax_map.set_facecolor('#1a1a2e')
        ax_map.set_title(
            f'{label}\nλ₀={logs[0]["lambda"]:.3f} → λ{n_rounds}={logs[-1]["lambda"]:.3f}',
            fontsize=8, fontweight='bold', color='white', pad=3,
        )
        ax_map.legend(fontsize=5, loc='lower right',
                      facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

        # ── Panel B: per-round total_J by user type ───────────────────────
        ax_J = fig.add_axes([0.245, top, 0.165, row_h * 0.80])
        _dark(ax_J)
        ax_J.plot(rounds, [lg['compliant_avgJ']    for lg in logs],
                  'o-', color='#00FFCC', lw=1.6, ms=5, label='Compliant')
        ax_J.plot(rounds, [lg['noncompliant_avgJ'] for lg in logs],
                  's--', color='#FF9944', lw=1.6, ms=5, label='Non-compliant')
        ax_J.plot(rounds, [lg['autonomous_avgJ']   for lg in logs],
                  '^:', color='#CC99FF', lw=1.4, ms=5, label='Autonomous')
        ax_J.plot(rounds, [lg['suggested_totalJ']  for lg in logs],
                  'D-', color='#FFD700', lw=1.2, ms=4, alpha=0.7, label='Suggested (model)')
        ax_J.set_xlabel('Round', color='#cccccc', fontsize=7)
        ax_J.set_ylabel('Avg total J', color='#cccccc', fontsize=7)
        ax_J.set_title('Navigation cost per user type', fontsize=7.5, color='#aaaacc')
        ax_J.legend(fontsize=5.5, facecolor='#12122a', edgecolor='#444', labelcolor='white')

        # ── Panel C: avg_W (comfort cost) by user type ────────────────────
        ax_W = fig.add_axes([0.425, top, 0.150, row_h * 0.80])
        _dark(ax_W)
        ax_W.plot(rounds, [lg['compliant_avgW']    for lg in logs],
                  'o-', color='#00FFCC', lw=1.6, ms=5, label='Compliant W̄')
        ax_W.plot(rounds, [lg['noncompliant_avgW'] for lg in logs],
                  's--', color='#FF9944', lw=1.6, ms=5, label='NC W̄')
        ax_W.plot(rounds, [lg['suggested_avgW']    for lg in logs],
                  'D-', color='#FFD700', lw=1.2, ms=4, alpha=0.7, label='Suggested W̄')
        ax_W.set_xlabel('Round', color='#cccccc', fontsize=7)
        ax_W.set_ylabel('Avg W̄ (comfort)', color='#cccccc', fontsize=7)
        ax_W.set_title('Per-step comfort cost\n(lower = better surface)', fontsize=7.5, color='#aaaacc')
        ax_W.legend(fontsize=5.5, facecolor='#12122a', edgecolor='#444', labelcolor='white')

        # ── Panel D: λ decay + IRL updates ───────────────────────────────
        ax_lam = fig.add_axes([0.590, top, 0.130, row_h * 0.80])
        _dark(ax_lam)
        ax_lam.plot(rounds, [lg['lambda'] for lg in logs],
                    'D-', color='#BB88FF', lw=1.8, ms=5, label='λ')
        ax_lam.set_ylim(-0.05, 1.05)
        ax_lam.set_xlabel('Round', color='#cccccc', fontsize=7)
        ax_lam.set_ylabel('λ (trust gate)', color='#BB88FF', fontsize=7)
        ax_lam.set_title('λ decay & IRL updates\nover rounds', fontsize=7.5, color='#aaaacc')

        ax_irl = ax_lam.twinx()
        ax_irl.bar(rounds, [lg['irl_updates'] for lg in logs],
                   color='gold', alpha=0.55, edgecolor='#998800', width=0.5,
                   label='IRL updates')
        ax_irl.set_ylabel('# IRL updates', color='gold', fontsize=7)
        ax_irl.tick_params(axis='y', colors='gold', labelsize=7)
        ax_irl.spines[:].set_color('#334466')

        lines1, lab1 = ax_lam.get_legend_handles_labels()
        lines2, lab2 = ax_irl.get_legend_handles_labels()
        ax_lam.legend(lines1 + lines2, lab1 + lab2,
                      fontsize=5.5, facecolor='#12122a', edgecolor='#444', labelcolor='white')

        # ── Panel E: path divergence (NC vs suggestion) ───────────────────
        ax_div = fig.add_axes([0.735, top, 0.125, row_h * 0.80])
        _dark(ax_div)
        divs = [lg['noncompliant_div'] for lg in logs]
        ax_div.fill_between(rounds, divs, alpha=0.25, color='#FF9944')
        ax_div.plot(rounds, divs, 'o-', color='#FF9944', lw=1.6, ms=5)
        ax_div.set_ylim(0, 1)
        ax_div.set_xlabel('Round', color='#cccccc', fontsize=7)
        ax_div.set_ylabel('Path divergence', color='#cccccc', fontsize=7)
        ax_div.set_title('NC vs suggested path\ndivergence (0=same, 1=disjoint)',
                         fontsize=7.5, color='#aaaacc')
        ax_div.axhline(0.5, color='white', lw=0.7, linestyle=':', alpha=0.4)

        # ── Panel F: final metric bars ────────────────────────────────────
        ax_bar = fig.add_axes([0.873, top, 0.12, row_h * 0.80])
        _dark(ax_bar)
        cats  = ['Compl\nJ', 'NC\nJ', 'Auto\nJ', 'Sugg\nJ']
        first = [logs[0]['compliant_avgJ'],    logs[0]['noncompliant_avgJ'],
                 logs[0]['autonomous_avgJ'],   logs[0]['suggested_totalJ']]
        last  = [logs[-1]['compliant_avgJ'],   logs[-1]['noncompliant_avgJ'],
                 logs[-1]['autonomous_avgJ'],  logs[-1]['suggested_totalJ']]
        x_pos  = np.arange(len(cats))
        bwidth = 0.35
        b1 = ax_bar.bar(x_pos - bwidth/2, first, bwidth,
                        color='#556688', alpha=0.85, edgecolor='#334466', label='Round 1')
        b2 = ax_bar.bar(x_pos + bwidth/2, last,  bwidth,
                        color='#00FFCC', alpha=0.85, edgecolor='#009988', label=f'Round {n_rounds}')
        for bar in b2:
            h = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2, h + 1,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=5.5, color='#00FFCC')
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(cats, fontsize=6, color='#cccccc')
        ax_bar.set_ylabel('Total J', color='#cccccc', fontsize=7)
        ax_bar.set_title(f'R1 vs R{n_rounds} cost\n(lower = model improved)',
                         fontsize=7.5, color='#aaaacc')
        ax_bar.legend(fontsize=5.5, facecolor='#12122a', edgecolor='#444', labelcolor='white')

    out = os.path.join(OUTPUT_DIR, 'Implicit_Feedback.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"\n✓ Saved: {out}")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    import warnings
    warnings.filterwarnings('ignore')

    print("\n" + "="*70)
    print("  HFL-RL v3 — FULL PIPELINE")
    print("="*70)

    city = CityMap200(seed=42)

    print("\n[1/5] Running all 8 scenarios...")
    summary = run_all_scenarios(city)

    print("\n[2/5] Generating annotated city map...")
    make_city_map(city, START, GOAL)

    print("\n[3/5] Generating Table III...")
    make_results_table(summary)

    print("\n[4/5] Generating trade-off chart...")
    make_tradeoff_chart(summary)

    print("\n[5/5] Running implicit feedback simulation...")
    make_feedback_figure(city)

    print("\n" + "="*70)
    print("  ✅  All outputs saved to", OUTPUT_DIR)
    print("="*70)