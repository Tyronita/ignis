"""
Indoor 3D voxel fire — the 2D wildfire CA lifted to (nx,ny,nz) with geometry + materials.

No fluid solver (project directive). Instead a cheap *buoyancy-biased* spread kernel:
flame prefers to climb (+z), spreads along the ceiling, and crosses open air (doorways)
as a short-lived flame front, igniting combustible materials it reaches. Walls (solid
voxels) block spread entirely. This captures the qualitative indoor behavior — fire goes
up first, then fills the room top-down and pushes through doorways — at CA speed.

Materials modulate receptivity (p_mult), burn duration, fuel load and $-value.
"""

import numpy as np
from . import materials as M

P_BASE = 0.34          # base per-effective-neighbor ignition probability
W_UP = 2.2             # buoyancy: spread from a burning voxel BELOW (fire climbs)
W_DOWN = 0.4           # weak downward spread
W_LAT = 1.1            # lateral spread
FLASHOVER = 0.35       # room burning-fraction that counts as flashover


def init_from_scene(scene):
    """Build a fresh sim state from a VoxelScene."""
    burnable = ~scene.solid & (M._column("p_mult", np.float32)[scene.material] > 0)
    st = dict(
        burning=np.zeros(scene.dims, np.float32),
        timer=np.zeros(scene.dims, np.int32),
        burned=np.zeros(scene.dims, bool),          # consumed fuel voxels (permanent)
        flamed_air=np.zeros(scene.dims, bool),      # air ever touched by flame (smoke proxy)
        fuel=scene.fuel.copy(),
        material=scene.material,
        solid=scene.solid,
        is_air=(scene.material == 0),
        burnable=burnable,
        solid_fuel=burnable & (scene.material != 0),   # real combustibles (exclude air)
        room_id=scene.room_id,
        p_mult=M._column("p_mult", np.float32)[scene.material],
        burn_time=M._column("burn_time", np.int32)[scene.material],
        value=M._column("value", np.float32)[scene.material],
        t=0,
        flashover_t=-1,
    )
    for (x, y, z) in scene.ignition:
        st["burning"][x, y, z] = 1.0
        st["timer"][x, y, z] = max(1, int(st["burn_time"][x, y, z]))
    st["fuel0"] = float((scene.fuel > 0).sum())
    return st


def step(st, rng):
    """Advance the voxel fire one step."""
    b = st["burning"]
    below = np.roll(b, 1, axis=2)     # burning voxel below -> climbs up into this cell
    above = np.roll(b, -1, axis=2)
    xm, xp = np.roll(b, 1, axis=0), np.roll(b, -1, axis=0)
    ym, yp = np.roll(b, 1, axis=1), np.roll(b, -1, axis=1)
    eff = W_UP * below + W_DOWN * above + W_LAT * (xm + xp + ym + yp)

    p = (1.0 - (1.0 - P_BASE) ** eff) * st["p_mult"]
    can = st["burnable"] & (b == 0) & (~st["burned"])
    ignite = can & (rng.random(b.shape) < p)
    b[ignite] = 1.0
    st["timer"][ignite] = np.maximum(1, st["burn_time"][ignite])

    # burnout
    alight = b > 0
    st["flamed_air"] |= alight & st["is_air"]
    st["timer"][alight] -= 1
    done = alight & (st["timer"] <= 0)
    # fuel voxels are consumed permanently; air just stops flaming (can re-flash)
    consumed = done & ~st["is_air"]
    st["burned"][consumed] = True
    st["fuel"][consumed] = 0.0
    b[done] = 0.0

    # flashover: first room where >FLASHOVER of its combustibles are involved (burning or burned)
    if st["flashover_t"] < 0:
        rid = st["room_id"]
        involved = (b > 0) | st["burned"]
        for r in range(rid.max() + 1):
            m = (rid == r) & st["solid_fuel"]
            if m.sum() > 0 and involved[m].mean() > FLASHOVER:
                st["flashover_t"] = st["t"]
                break
    st["t"] += 1
    return st


def metrics(st):
    """Evals based on spread + materials."""
    burned = st["burned"]
    burned_frac = burned.sum() / max(st["fuel0"], 1)
    rid = st["room_id"]
    rooms_lost = 0
    for r in range(rid.max() + 1):
        m = (rid == r) & st["solid_fuel"]
        if m.sum() > 0 and (burned[m].mean() > 0.5):
            rooms_lost += 1
    n_rooms = int(rid.max() + 1) if rid.max() >= 0 else 0
    damage = float((st["value"] * burned).sum())
    air = st["is_air"]
    smoke_frac = st["flamed_air"].sum() / max(air.sum(), 1)
    active = (st["burning"] > 0).sum()
    return dict(
        step=st["t"],
        fuel_saved_pct=100 * (1 - burned_frac),
        burned_pct=100 * burned_frac,
        rooms_lost=rooms_lost,
        n_rooms=n_rooms,
        flashover_step=st["flashover_t"],
        damage_usd=damage,
        smoke_pct=100 * smoke_frac,
        active_flame=int(active),
        contained=bool(active == 0 and burned_frac < 0.999),
    )
