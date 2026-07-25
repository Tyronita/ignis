"""
2D wildfire spread + air-tanker suppression environment (pure NumPy, fully batched).

State is a batch of B independent grids. Fire spreads via a wind- and slope-weighted
cellular automaton; an air-tanker agent moves over the grid and drops water, wetting
cells (raising moisture so they resist ignition) and extinguishing burning cells.

Grid size is configurable via set_grid(n) / ignis.config.SimConfig — it is NOT a
hardcoded constant. This is the fast, abstract "brain" simulator the RL policy trains
against; JaxWildfire / PyTorchFire are drop-in fidelity upgrades behind the same API.
"""

import numpy as np

# --- dynamics constants (overridable via ignis.config.SimConfig.apply) ---
BURN_TIME = 4
P_BASE = 0.27
WIND_K = 0.7
SLOPE_K = 0.55
SLOPE_GAIN = 22.0
MOIST_THRESH = 0.5
AGENT_SPEED = 3
WATER_RADIUS = 3
WATER_BUDGET = 48
T = 64

# action deltas: 0=stay 1=up 2=down 3=left 4=right   (axis1=y, axis2=x)
_DELTAS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)
N_ACTIONS = 5
N_FEAT = 11

# --- grid state (set by set_grid) ---
N = None
_YS = _XS = None
TERRAIN = None
_SN = _SS = _SW = _SE = None


def set_grid(n):
    """(Re)configure the grid to n x n. Rebuilds coordinate grids, clears terrain."""
    global N, _YS, _XS
    N = int(n)
    _YS, _XS = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    set_terrain(None)


def make_terrain(seed=0):
    """Smooth random hills, normalized to [0,1]. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    t = np.zeros((N, N), np.float32)
    yy = _YS / N * 2 * np.pi
    xx = _XS / N * 2 * np.pi
    for k in range(1, 5):
        py, px = rng.uniform(0, 2 * np.pi, 2)
        a = rng.uniform(0.4, 1.0) / k
        t += a * np.sin(k * yy + py) * np.cos(k * xx + px)
    t += 0.6 * np.exp(-(((_YS - N*0.35)**2 + (_XS - N*0.6)**2) / (2*(N*0.28)**2)))
    t -= t.min(); t /= max(t.max(), 1e-6)
    return t.astype(np.float32)


def set_terrain(t):
    """Install terrain and precompute static uphill-spread multipliers."""
    global TERRAIN, _SN, _SS, _SW, _SE
    if t is None:
        TERRAIN = _SN = _SS = _SW = _SE = None
        return
    TERRAIN = t.astype(np.float32)
    def fac(dz):
        return (1 + SLOPE_K * np.clip(dz * SLOPE_GAIN, -1, 1)).astype(np.float32)
    _SN = fac(TERRAIN - np.roll(TERRAIN, 1, axis=0))
    _SS = fac(TERRAIN - np.roll(TERRAIN, -1, axis=0))
    _SW = fac(TERRAIN - np.roll(TERRAIN, 1, axis=1))
    _SE = fac(TERRAIN - np.roll(TERRAIN, -1, axis=1))


def init_state(B, rng):
    """Create a batch of B fresh episodes with random ignition, wind, agent pos."""
    fuel = np.ones((B, N, N), np.float32)
    burning = np.zeros((B, N, N), np.float32)
    timer = np.zeros((B, N, N), np.int32)
    moisture = np.zeros((B, N, N), np.float32)

    iy = rng.integers(8, N - 8, B)
    ix = rng.integers(8, N - 8, B)
    for b in range(B):
        burning[b, iy[b]-1:iy[b]+2, ix[b]-1:ix[b]+2] = 1.0
        timer[b, iy[b]-1:iy[b]+2, ix[b]-1:ix[b]+2] = BURN_TIME

    ang = rng.uniform(0, 2 * np.pi, B).astype(np.float32)
    wind = np.stack([np.sin(ang), np.cos(ang)], axis=1)

    agent = rng.integers(4, N - 4, size=(B, 2)).astype(np.int32)
    budget = np.full(B, WATER_BUDGET, np.int32)
    fuel0 = fuel.sum(axis=(1, 2))
    return dict(fuel=fuel, burning=burning, timer=timer, moisture=moisture,
                wind=wind, agent=agent, budget=budget, fuel0=fuel0)


def features(s):
    """Per-env policy observation, shape (B, N_FEAT)."""
    burning = s["burning"]
    B = burning.shape[0]
    ay = s["agent"][:, 0].astype(np.float32)
    ax = s["agent"][:, 1].astype(np.float32)
    tot = burning.sum(axis=(1, 2))
    safe = np.maximum(tot, 1.0)
    cy = (burning * _YS[None]).sum(axis=(1, 2)) / safe
    cx = (burning * _XS[None]).sum(axis=(1, 2)) / safe
    top = (_YS[None] < ay[:, None, None])
    left = (_XS[None] < ax[:, None, None])
    def frac(mask):
        return (burning * mask).sum(axis=(1, 2)) / safe
    feat = np.stack([
        np.ones(B, np.float32),
        ay / N, ax / N,
        (cy - ay) / N, (cx - ax) / N,
        np.minimum(tot / (N * N), 1.0),
        frac(top & left), frac(top & ~left), frac(~top & left), frac(~top & ~left),
        s["budget"] / WATER_BUDGET,
    ], axis=1).astype(np.float32)
    return feat


def policy_action(W, feat):
    """W: (B, N_ACTIONS, N_FEAT) batched linear policy -> action per env (B,)."""
    return np.argmax(np.einsum("baf,bf->ba", W, feat), axis=1)


def step(s, actions, rng):
    """Advance every env one step given a per-env action. Mutates & returns s."""
    B = s["fuel"].shape[0]
    fuel, burning, timer, moisture = s["fuel"], s["burning"], s["timer"], s["moisture"]

    wy = s["wind"][:, 0][:, None, None]
    wx = s["wind"][:, 1][:, None, None]
    north = np.roll(burning, 1, axis=1)
    south = np.roll(burning, -1, axis=1)
    west = np.roll(burning, 1, axis=2)
    east = np.roll(burning, -1, axis=2)
    if TERRAIN is None:
        sN = sS = sW = sE = 1.0
    else:
        sN, sS, sW, sE = _SN[None], _SS[None], _SW[None], _SE[None]
    eff = (np.clip(1 + WIND_K * wy, 0, None) * sN * north +
           np.clip(1 - WIND_K * wy, 0, None) * sS * south +
           np.clip(1 + WIND_K * wx, 0, None) * sW * west +
           np.clip(1 - WIND_K * wx, 0, None) * sE * east)
    p_ign = 1.0 - (1.0 - P_BASE) ** eff
    can = (fuel > 0) & (burning == 0) & (moisture < MOIST_THRESH)
    ignite = can & (rng.random((B, N, N)) < p_ign)
    burning[ignite] = 1.0
    timer[ignite] = BURN_TIME

    alight = burning > 0
    timer[alight] -= 1
    ash = alight & (timer <= 0)
    burning[ash] = 0.0
    fuel[ash] = 0.0
    moisture *= 0.985

    d = _DELTAS[actions]
    s["agent"] = np.clip(s["agent"] + d * AGENT_SPEED, 0, N - 1)

    has_water = s["budget"] > 0
    ay = s["agent"][:, 0]; ax = s["agent"][:, 1]
    dist2 = (_YS[None] - ay[:, None, None]) ** 2 + (_XS[None] - ax[:, None, None]) ** 2
    disk = (dist2 <= WATER_RADIUS ** 2) & has_water[:, None, None]
    moisture[disk] = 1.0
    doused = disk & (burning > 0)
    burning[doused] = 0.0
    timer[doused] = 0
    s["budget"] = np.where(has_water, s["budget"] - 1, s["budget"])
    return s


def fuel_saved(s):
    """Fraction of initial fuel still unburned (the return we maximize)."""
    return s["fuel"].sum(axis=(1, 2)) / s["fuel0"]


def cell_rgb(s, b=0):
    """Per-cell RGB float image (N,N,3) in [0,1] — shared by 2D and 3D renderers."""
    fuel = s["fuel"][b]; burning = s["burning"][b]
    moisture = s["moisture"][b]; timer = s["timer"][b]
    img = np.zeros((N, N, 3), np.float32)
    img[..., 1] = 0.15 + 0.55 * fuel
    wet = (moisture > MOIST_THRESH) & (burning == 0)
    img[wet] = np.array([0.15, 0.45, 0.85], np.float32)
    ash = (fuel <= 0) & (burning == 0)
    img[ash] = np.array([0.08, 0.06, 0.06], np.float32)
    fire = burning > 0
    hot = np.clip(timer / BURN_TIME, 0, 1)
    fr = np.zeros_like(img)
    fr[..., 0] = 1.0
    fr[..., 1] = 0.35 + 0.55 * hot
    img[fire] = fr[fire]
    return np.clip(img, 0, 1)


def render(s, b=0, scale=10):
    """RGB image (uint8) of env b, upscaled by `scale`, with agent marker."""
    big = np.kron(cell_rgb(s, b), np.ones((scale, scale, 1)))
    ay = int(s["agent"][b, 0]) * scale
    ax = int(s["agent"][b, 1]) * scale
    r = scale
    y0, y1 = max(0, ay - r), min(big.shape[0], ay + r + scale)
    x0, x1 = max(0, ax - r), min(big.shape[1], ax + r + scale)
    big[y0:y1, x0:x0 + 2] = 1.0; big[y0:y1, x1 - 2:x1] = 1.0
    big[y0:y0 + 2, x0:x1] = 1.0; big[y1 - 2:y1, x0:x1] = 1.0
    return (big * 255).astype(np.uint8)


def rollout(W, rng, record=False):
    """Run one episode with policy matrix W (N_ACTIONS,N_FEAT). W=None -> do-nothing."""
    s = init_state(1, rng)
    frames = [] if record else None
    Wb = None if W is None else W[None]
    for _ in range(T):
        if record:
            frames.append(render(s))
        act = np.zeros(1, np.int32) if W is None else policy_action(Wb, features(s))
        step(s, act, rng)
    if record:
        frames.append(render(s))
    return float(fuel_saved(s)[0]), frames


# initialize to the default grid on import
set_grid(48)
