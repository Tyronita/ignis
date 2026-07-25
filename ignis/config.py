"""Central configuration for the Ignis fire simulator.

Grid size is no longer hardcoded: instantiate SimConfig(n=...) and call .apply()
to reconfigure the 2D engine at any size. The indoor voxel sim (ignis.indoor)
takes independent (nx, ny, nz) dimensions.
"""

from dataclasses import dataclass


@dataclass
class SimConfig:
    # --- grid ---
    n: int = 48              # 2D grid is n x n  (configurable — was a hardcoded constant)
    # --- fire dynamics ---
    p_base: float = 0.27     # base per-burning-neighbor ignition probability
    wind_k: float = 0.7      # wind bias on spread direction
    slope_k: float = 0.55    # terrain-slope bias (fire climbs uphill)
    slope_gain: float = 22.0
    burn_time: int = 4       # steps a cell burns before turning to ash
    moist_thresh: float = 0.5
    # --- agent / water ---
    agent_speed: int = 3
    water_radius: int = 3
    water_budget: int = 48
    steps: int = 64          # steps per episode

    def apply(self):
        """Push this config into the fire_env module (rebuilds coordinate grids)."""
        from . import fire_env as fe
        fe.P_BASE = self.p_base
        fe.WIND_K = self.wind_k
        fe.SLOPE_K = self.slope_k
        fe.SLOPE_GAIN = self.slope_gain
        fe.BURN_TIME = self.burn_time
        fe.MOIST_THRESH = self.moist_thresh
        fe.AGENT_SPEED = self.agent_speed
        fe.WATER_RADIUS = self.water_radius
        fe.WATER_BUDGET = self.water_budget
        fe.T = self.steps
        fe.set_grid(self.n)   # rebuilds N, _YS, _XS and clears terrain
        return self


DEFAULT = SimConfig()
