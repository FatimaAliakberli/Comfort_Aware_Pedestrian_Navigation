"""
=============================================================================
generate_map_image.py  —  Render and save the annotated 200×200 city map
=============================================================================
Run this script independently to produce hfl_rl_v3_city_map.png.

Usage:
    python city_map/generate_map_image.py
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from city_map.city_map import (
    CityMap200, COLOR_SCHEME, TERRAIN_NAMES, BASE_COMFORT,
)

OUTPUT_DIR = 'output_images'


def make_city_map(city: CityMap200, start: tuple, goal: tuple,
                  output_dir: str = OUTPUT_DIR) -> str:
    """
    Render the annotated city map and save it as a PNG.

    Parameters
    ----------
    city       : CityMap200 instance
    start      : (row, col) of point A
    goal       : (row, col) of point B
    output_dir : directory where the image is saved

    Returns
    -------
    Path to the saved file.
    """

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 16))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    rgb = city.create_rgb_grid()
    ax.imshow(rgb, origin='lower', interpolation='nearest', aspect='equal')

    # Quadrant boundaries
    ax.axvline(100, color='white', lw=0.5, alpha=0.2, ls=':')
    ax.axhline(100, color='white', lw=0.5, alpha=0.2, ls=':')

    # # Quadrant labels
    # for yr, xc, txt in [(150,50,'NW'),(150,150,'NE'),(50,50,'SW'),(50,150,'SE')]:
    #     ax.text(xc, yr, txt, color='white', fontsize=14, alpha=0.2,
    #             fontweight='bold', ha='center', va='center')

    # Start / Goal markers
    ax.plot(start[1], start[0], 'o', color='#39FF14', ms=14,
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=10)
    ax.plot(goal[1],  goal[0],  '*', color='#FFD700', ms=18,
            markeredgecolor='darkred',  markeredgewidth=2, zorder=10)
    ax.text(start[1]+3, start[0]+3, 'A (0,0)',
            color='#39FF14', fontsize=9, fontweight='bold')
    ax.text(goal[1]-18, goal[0]+3, f'B ({goal[0]},{goal[1]})',
            color='#FFD700', fontsize=9, fontweight='bold')

    # # Annotate hazard corridor
    # ax.annotate('', xy=(110, 55), xytext=(40, 10),
    #             arrowprops=dict(arrowstyle='->', color='#FF6666', lw=1.5))
    # ax.text(5, 5,
    #         'SP direct-path\nhazard corridor\n(grass, puddles,\nhighways)',
    #         color='#FF9999', fontsize=7, va='bottom',
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='#330000', alpha=0.7))

    # # Annotate boulevard (HFL-RL preferred path)
    # ax.text(170, 82, '← HFL-RL boulevard\n   (good pavement)',
    #         color='#00FFCC', fontsize=7,
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='#003333', alpha=0.7))

    ax.set_xticks(np.arange(0, 201, 20))
    ax.set_yticks(np.arange(0, 201, 20))
    ax.tick_params(colors='#666688', labelsize=7)
    ax.set_xlim(-1, 200)
    ax.set_ylim(-1, 200)
    ax.set_title(
        'City Map — 200×200 Pedestrian Grid  (~400m×400m)\n'
        'SP direct path loaded with hazard terrain  |  '
        'HFL-RL boulevard network bypasses hazards',
        fontsize=12, fontweight='bold', color='white', pad=10,
    )
    ax.set_xlabel('Column →', color='#aaaacc', fontsize=9)
    ax.set_ylabel('Row →',    color='#aaaacc', fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334466')

    # Terrain legend
    patches = [
        mpatches.Patch(
            color=mcolors.hex2color(COLOR_SCHEME[t]),
            label=f"{TERRAIN_NAMES[t]}  (base {BASE_COMFORT[t]:.1f})",
        )
        for t in sorted(COLOR_SCHEME.keys())
    ]
    leg = ax.legend(
        handles=patches, loc='upper right', bbox_to_anchor=(1.36, 1.0),
        fontsize=7, framealpha=0.9, facecolor='#12122a',
        edgecolor='#445566', labelcolor='white',
        title='Terrain Classes', title_fontsize=9,
        handlelength=1.2, handleheight=1.0,
    )
    leg.get_title().set_color('#88ccff')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'City_map.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"✓ City map saved: {out_path}")
    return out_path


# =============================================================================
# Entry point — run standalone
# =============================================================================
if __name__ == '__main__':
    city  = CityMap200(seed=42)
    START = (0, 0)
    GOAL  = (185, 190)
    make_city_map(city, START, GOAL)
