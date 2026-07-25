"""
Material table for indoor fire.

We deliberately do NOT run a temperature/CFD field (per project directive — a heavy
fluid solver would slow us down). Instead, each material modulates the fast
probabilistic cellular-automaton spread through a few parameters:

  obstacle    : non-combustible AND blocks spread/flow (walls, glass, steel)
  p_mult      : multiplier on ignition probability (how readily it catches)
  burn_time   : how many steps it burns before becoming ash
  fuel_load   : relative fuel mass (weights burned-volume + damage evals)
  value       : $ per voxel (weights the material-damage eval)

Fire physics stays cheap; materials give the scene meaning.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    id: int
    name: str
    obstacle: bool     # blocks fire spread and air flow (structural / non-combustible)
    p_mult: float      # ignition-probability multiplier
    burn_time: int     # burn duration in steps
    fuel_load: float   # relative fuel mass
    value: float       # $ per voxel (for damage eval)


# id 0 is reserved for AIR: carries a short-lived flame front (so fire climbs, fills
# rooms top-down and crosses doorways) but holds no fuel and is never consumed.
AIR = Material(0, "air", obstacle=False, p_mult=0.7, burn_time=2, fuel_load=0.0, value=0.0)

_LIST = [
    AIR,
    Material(1, "concrete", obstacle=True,  p_mult=0.0, burn_time=0,  fuel_load=0.0, value=8.0),
    Material(2, "glass",    obstacle=True,  p_mult=0.0, burn_time=0,  fuel_load=0.0, value=6.0),
    Material(3, "steel",    obstacle=True,  p_mult=0.0, burn_time=0,  fuel_load=0.0, value=12.0),
    Material(4, "drywall",  obstacle=False, p_mult=0.35, burn_time=3, fuel_load=0.3, value=3.0),
    Material(5, "wood",     obstacle=False, p_mult=0.9,  burn_time=6, fuel_load=1.0, value=5.0),
    Material(6, "fabric",   obstacle=False, p_mult=1.5,  burn_time=3, fuel_load=0.8, value=4.0),
    Material(7, "foam",     obstacle=False, p_mult=1.8,  burn_time=2, fuel_load=0.9, value=4.0),
    Material(8, "paper",    obstacle=False, p_mult=1.6,  burn_time=1, fuel_load=0.4, value=1.0),
]

BY_NAME = {m.name: m for m in _LIST}
BY_ID = {m.id: m for m in _LIST}
N_MATERIALS = len(_LIST)


def id_of(name):
    if name not in BY_NAME:
        raise KeyError(f"unknown material '{name}'. known: {sorted(BY_NAME)}")
    return BY_NAME[name].id


def _column(attr, dtype):
    """Lookup array indexed by material id, e.g. p_mult[material_grid]."""
    import numpy as np
    arr = np.zeros(N_MATERIALS, dtype)
    for m in _LIST:
        arr[m.id] = getattr(m, attr)
    return arr
