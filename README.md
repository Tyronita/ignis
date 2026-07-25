# 🔥 Ignis

**A fast fire-spread simulator and reinforcement-learning suppression sandbox — wildfire today, indoor next.**

Ignis simulates fire spreading through an environment and trains an agent to put it out.
It runs in pure NumPy (no GPU, no heavy deps), so it clones and runs anywhere in seconds,
yet it already shows the full loop: **simulate spread → train a suppression policy → measure the outcome.**

> Hackathon MVP (v0.1). Built to be built upon — see the [roadmap](#roadmap).

---

## What it does

| | |
|---|---|
| **Outdoor wildfire (2D)** | Wind-driven cellular-automaton fire; an air-tanker RL agent learns *where* to drop water to save the most fuel. |
| **Outdoor wildfire (3D terrain)** | Same fire lifted onto an elevation heightmap — **fire climbs uphill** (slope-coupled spread) and the tanker is retrained on the hills. |
| **Indoor building (3D voxel)** | A JSON floorplan is converted to a voxel scene with **materials**; fire **climbs, fills rooms top-down, and crosses doorways**, scored by an **eval harness** (damage, rooms lost, flashover). |

### Results

| Scenario | No-agent baseline | Trained agent |
|---|---|---|
| Wildfire, flat | **41.7%** fuel saved | **~98%** |
| Wildfire, hilly terrain | **63.9%** fuel saved | **99.6%** |
| Indoor apartment (no suppression) | 3/3 rooms lost, $1384 damage, flashover @ step 4 | *suppression = next build* |

<p align="center">
  <img src="assets/compare2d.gif" width="80%" alt="2D wildfire: no tanker vs trained tanker"/><br/>
  <em>Same fire, no tanker (left) vs. the trained air-tanker (right).</em>
</p>

<p align="center">
  <img src="assets/rollout.gif" width="45%" alt="2D trained rollout"/>
  <img src="assets/scene3d.gif" width="45%" alt="3D terrain scene"/><br/>
  <em>Left: trained tanker containing a fire. Right: 3D terrain scene — fire climbs uphill, tanker drops water from above.</em>
</p>

<p align="center">
  <img src="assets/indoor.gif" width="70%" alt="Indoor apartment fire"/><br/>
  <em>Indoor apartment: top view (room-to-room spread) + side view (plume climbs, ceiling fills). No suppression = total loss — the baseline the future policy is scored against.</em>
</p>

<p align="center"><img src="assets/curve.png" width="60%" alt="learning curve"/></p>

---

## How the current model works

The spread model is a **probabilistic cellular automaton** (fast, RL-friendly) — **not** a fluid solver.
Each step, a non-burning fuel cell ignites with probability

```
p_ignite = 1 − (1 − p_base) ^ (effective burning neighbors)
```

where *effective neighbors* are weighted by **wind** and **terrain slope** (outdoor) or by
**buoyancy** — fire prefers to climb, then spreads along the ceiling — and **materials** (indoor).
Moisture/water and walls gate ignition. Burning cells count down, then become ash.

The agent is a tiny linear policy trained with the **Cross-Entropy Method (CEM)** — gradient-free,
no tuning, converges in seconds. Reward = fraction of fuel saved.

> **Note:** Ignis does **not** use JaxWildfire — this is an independent NumPy implementation of the
> same *model class* (probabilistic CA). JaxWildfire is our documented GPU scale-up path
> (see [REFERENCES.md](REFERENCES.md)). No real CFD/fluid solver runs in the loop, by design.

---

## Quickstart

```bash
pip install -r requirements.txt

# reproduce every artifact (train + all gifs + indoor eval)
python scripts/regenerate.py          # or:  ./run.ps1   /   ./run.sh

# live windows
python -m ignis.viz2d                  # 2D side-by-side, live
python -m ignis.scene3d                # 3D terrain scene, live rotating

# indoor pipeline
python -m ignis.indoor.importer ignis/indoor/examples/apartment.json
python -m ignis.indoor.evals    ignis/indoor/examples/apartment.json
```

Grid size is configurable (no hardcoded `N`):

```python
from ignis import SimConfig
SimConfig(n=64).apply()   # any size; indoor uses independent (nx, ny, nz)
```

---

## Architecture

```
scene (JSON floorplan / USD* / mesh*)  ──importer──►  "our format" (VoxelScene)
                                                          │
   fire_env (2D CA) ── config(n) ──┐                      ▼
                                   ├─► spread model ──►  eval harness (damage, rooms, flashover)
   indoor_env (3D voxel CA) ───────┘                      │
                                                          ▼
                          CEM-trained suppression policy (air-tanker; indoor agent = next)
   *USD / mesh voxelizer = experimental preset, not default
```

The sim consumes only the **VoxelScene** format, so any front-end (JSON, USD, voxelized mesh)
plugs in without touching the physics — and JaxWildfire/PyTorchFire can replace the spread core
behind the same interface.

## Repo layout

```
ignis/            engine + training + viz (fire_env, train, viz2d, scene3d, config)
ignis/indoor/     schema, materials, importer, indoor_env, evals, examples/apartment.json
assets/           all GIFs + curves
scripts/          regenerate.py — reproduces everything
presentation/     React (Vite) pitch deck
```

## Roadmap

- [ ] Indoor **suppression** agent (sprinkler activation / firefighter) trained against the eval harness
- [ ] **Budgets** as hard constraints in the RL loop (water / time / compute)
- [ ] **USD / mesh voxelizer** importer (experimental preset → default-quality)
- [ ] **JAX** backend swap (JaxWildfire/PyTorchFire) for GPU-scale training
- [ ] Photoreal render layer (PyVista → Unreal Niagara / FieryGS) consuming the same state

See [REFERENCES.md](REFERENCES.md) for sources and design rationale, and [CHANGELOG.md](CHANGELOG.md) for build history.

## License
MIT (see repo).
