# Ignis — physics & equations (every equation currently used)

This documents the **current** model exactly as implemented, with real spatial units, so every
assumption can be challenged. Notation: cells/voxels indexed on a grid; `U(0,1)` is a uniform draw.

## 0. Spatial units (real dimensions)
Each voxel is a cube of edge **`cell_size_m = 0.25 m`** (`schema.VoxelScene.cell_size_m`).
So the example apartment grid `40 × 28 × 16` voxels = **10.0 m (x) × 7.0 m (y) × 4.0 m (z)** — a realistic
flat with a 4 m floor-to-ceiling stack. Time advances in discrete steps (dimensionless; ~1 s/step is the
intended calibration target, see Assumptions).

## 1. Outdoor wildfire — ignition (2D cellular automaton)
A non-burning fuel cell ignites with probability
```
p_ignite = 1 − (1 − p_base) ^ E
```
- `p_base = 0.27` — base per-effective-neighbor ignition probability.
- `E` = **effective burning-neighbor count** =
  `Σ_d  w_wind,d · w_slope,d · b_d`, over the 4 neighbours `d ∈ {N,S,W,E}`, `b_d ∈ {0,1}` = neighbour burning.

**Wind weighting** (fire spreads faster downwind), with wind unit vector `ŵ=(w_y,w_x)`, `k_wind = 0.7`:
```
w_wind,N = clip(1 + k_wind·w_y, 0, ∞)      # spread southward
w_wind,S = clip(1 − k_wind·w_y, 0, ∞)
w_wind,W = clip(1 + k_wind·w_x, 0, ∞)
w_wind,E = clip(1 − k_wind·w_x, 0, ∞)
```
**Slope weighting** (fire climbs uphill), terrain `T∈[0,1]`, `k_slope = 0.55`, gain `G = 22`:
```
w_slope,d = 1 + k_slope · clip( (T − shift_d T) · G , −1, 1 )
```
**Ignition rule:** ignite iff `fuel>0 ∧ not burning ∧ moisture<0.5 ∧ U(0,1) < p_ignite`.

## 2. Burning, burnout, drying
```
burning cell: timer ← timer − 1;   if timer ≤ 0 → ash (fuel ← 0, burning ← 0)   # BURN_TIME = 4
moisture:     m ← 0.985 · m                                                     # slow drying
```

## 3. Suppression (air-tanker water drop)
Agent at `(a_y,a_x)` moves by `AGENT_SPEED = 3` cells/step. A drop wets a disk of radius `R = 3`:
```
for cells with (y−a_y)² + (x−a_x)² ≤ R² and budget>0:
    moisture ← 1     (blocks ignition, since moisture ≥ 0.5)
    if burning: extinguish (burning ← 0)
budget ← budget − 1                          # WATER_BUDGET = 48 drops
```

## 4. Indoor 3D structural fire — buoyancy-biased ignition
Same probabilistic form, but `E` is a **buoyancy-weighted** 6-neighbour sum and is scaled by the cell's
**material receptivity** `p_mult(material)`:
```
E = w_up·b_below + w_down·b_above + w_lat·(b_x+ + b_x− + b_y+ + b_y−)
p_ignite = (1 − (1 − P_BASE) ^ E) · p_mult(material)
```
- `P_BASE = 0.34`, `w_up = 2.2` (fire climbs), `w_down = 0.4`, `w_lat = 1.1`.
- **Air carries a short-lived flame front** (`p_mult=0.7`, `burn_time=2`, fuel_load=0, never consumed) so fire
  crosses open air, fills rooms top-down, and passes through doorways. Walls (`solid`) block spread entirely.
- Wetted voxels (`wet ≥ 0.5`) cannot ignite; `wet ← 0.9·wet` each step (drying).

## 5. Materials (modulate the cheap CA; no temperature field yet)
| id | material | obstacle | p_mult | burn_time | fuel_load | $/voxel |
|---|---|---|---|---|---|---|
|0|air|no|0.70|2|0.0|0|
|1|concrete|**yes**|0|0|0|8|
|2|glass|**yes**|0|0|0|6|
|3|steel|**yes**|0|0|0|12|
|4|drywall|no|0.35|3|0.3|3|
|5|wood|no|0.90|6|1.0|5|
|6|fabric|no|1.50|3|0.8|4|
|7|foam|no|1.80|2|0.9|4|
|8|paper|no|1.60|1|0.4|1|

## 6. Suppression reward + zoned-sprinkler action (indoor RL)
Ceiling split into `GX×GY = 4×3` zones; action = spray one zone (wet its column, extinguish flame) or no-op:
```
observation s = [ per-zone burning fraction (12) , water_budget/water_max ]      ∈ [0,1]^13
action a ∈ {0..11 (zones), 12 (no-op)}      policy: a = argmax(W · s)
reward R = fuel_saved = 1 − burned / fuel0        (Δ per step in the Gym env)
```

## 7. Evaluation metrics (spread + materials)
```
burned%        = 100 · Σ burned(solid_fuel) / |solid_fuel|
rooms_lost     = #{ rooms r : mean(burned | room r ∩ solid_fuel) > 0.5 }
flashover_step = first step where any room has involved-fraction > 0.35
$ damage       = Σ  value(material) · 1[burned]
smoke%         = 100 · |air ever flamed| / |air|
contained      = (no active flame) ∧ (burned < 99.9%)
```

## 8. Optimizer — Cross-Entropy Method (HPO of the policy)
Gradient-free; population `P=48` (wildfire) / `24` (indoor), elites `K=10/6`, generations `30/14`:
```
θ_i ~ N(μ, diag(σ²)),  i = 1..P
score_i = mean over S random scenes of  fuel_saved(rollout(θ_i))
elites  = top-K by score
μ ← mean(elites);   σ ← std(elites) + ε        (ε = 0.02 noise floor keeps exploration alive)
```

## Assumptions to challenge (explicit)
1. **Step ≈ 1 s** is an intended calibration, not yet fitted to real burn rates (e.g. wood HRR, ISO 834 curve).
2. **No temperature/heat field yet** — materials modulate a probabilistic CA; conduction (κ∇²T) is the next PR.
3. **No smoke transport as a fluid** — "smoke" = air voxels ever flamed; real smoke advection is future.
4. **Water is a wet-mask**, not a 3D droplet spray with a parabolic beam and collisions (planned).
5. **$ values are placeholders**, not California-code-calibrated; safety scoring (time-to-untenable, egress)
   is the next reward direction.
6. **`cell_size_m` and `BURN_TIME` set the effective spread rate** — calibrate against a known compartment fire.

References for calibration targets: Rothermel ROS; NIST FDS; ISO 834 / ASTM E119 time-temperature curve;
California Building Code (CBC) Ch.7 fire-resistance, NFPA 101 egress. See [REFERENCES.md](REFERENCES.md).
