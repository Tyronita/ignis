# Changelog

## v0.2.0 — 3D indoor sim + suppression RL + Gym env (2026-07-25)

- **Procedural 3D indoor generator** (`ignis/indoor/generate.py`) — seeded buildings (rooms/walls/doors/
  furniture/materials), doorway-per-split so they're connected by construction (BFS-verified).
- **3D voxel renderer** (`ignis/indoor/render3d.py`) — matplotlib `ax.voxels`, orbiting camera → `indoor3d.gif`.
- **Indoor suppression RL** (`ignis/indoor/suppress.py`) — 4×3 zoned ceiling sprinklers, `wet` mechanic in
  `indoor_env`, CEM over a generated distribution with a held-out validation re-rank; `indoor_compare.gif`.
- **Gymnasium envs** (`ignis/gym_env.py`) — `Ignis-Indoor-v0` / `Ignis-Wildfire-v0`, registered as a gymnasium
  plugin via `pyproject.toml`. A greedy oracle solves the indoor env (100% fuel saved) → it's solvable.
- **Spatial fire-dataset export** (`ignis/dataset.py`) — per-step voxel fire-state + actions + a provenance manifest.
- **Docs**: `PHYSICS.md` (every equation, real 0.25 m voxels), `SUBMISSION.md` (hackathon form fields),
  `pyproject.toml` (pip-installable). Presentation rebuilt as a 10-page deck.

## v0.1.0 — Hackathon MVP (2026-07-25)

First submittable build. The full loop works end-to-end: **simulate → train → evaluate.**

**Outdoor wildfire**
- Batched NumPy cellular-automaton fire model (wind-weighted spread, moisture, fuel, burnout).
- Air-tanker suppression agent trained with the Cross-Entropy Method.
  - Flat terrain: **41.7% → ~98%** fuel saved vs. no-agent baseline.
- 3D terrain scene: elevation heightmap with **slope-coupled spread** (fire climbs uphill); tanker retrained.
  - Hilly terrain: **63.9% → 99.6%** fuel saved.
- Live GUIs (2D side-by-side, 3D rotating) + saved GIFs.

**Indoor foundation (new)**
- `VoxelScene` — the canonical voxel format all front-ends convert to.
- JSON floorplan schema + importer (rooms / walls / doors / furniture → voxels); USD/mesh voxelizer stubbed as experimental preset.
- Material table (concrete/glass/steel obstacles; wood/drywall/fabric/foam/paper combustibles) modulating a cheap buoyancy-biased 3D CA — fire climbs, fills rooms top-down, crosses doorways. No fluid solver.
- Eval harness: % burned, rooms lost, time-to-flashover, $ material damage, smoke-filled %, containment.
  - Apartment baseline (no suppression): **3/3 rooms lost, $1384, flashover @ step 4, 96% smoke.**

**Infra**
- Grid size configurable via `SimConfig` (removed hardcoded `N`).
- `scripts/regenerate.py` reproduces every artifact and syncs media into the React deck.
- README, REFERENCES (sources + rationale), React (Vite) presentation.

### Next
- Indoor suppression agent; budgets as RL constraints; USD/mesh importer; JAX backend; photoreal render layer.
