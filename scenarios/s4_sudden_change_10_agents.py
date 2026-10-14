"""
=============================================================================
s4_sudden_change_10_agents.py  —  S4: Sudden Change (10 agents)
=============================================================================
Run standalone to execute this single scenario and save its figure.

Usage:
    python scenarios/s4_sudden_change_10_agents.py
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from city_map.city_map import CityMap200
from scenarios._scenario_runner import run_single_scenario

SCENARIO_TYPE = 'sudden_change'
N_AGENTS      = 10
WEATHER       = 'neutral'
LABEL         = 'S4: Sudden Change (10 agents)'
START         = (0, 0)
GOAL          = (185, 190)

if __name__ == '__main__':
    city = CityMap200(seed=42)
    run_single_scenario(
        scenario_type = SCENARIO_TYPE,
        n_agents      = N_AGENTS,
        weather       = WEATHER,
        label         = LABEL,
        start         = START,
        goal          = GOAL,
        city          = city,
        save          = True,
    )
