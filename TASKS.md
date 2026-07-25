# Ignis — tasks, rewards, RL algorithm, hyperparameters

Two agent tasks share one simulator and one `VoxelScene`. Real constants below; physics in
[PHYSICS.md](PHYSICS.md) (which also lists **every assumption** — read it and challenge them).

## Real constants used
| Constant | Value | Note |
|---|---|---|
| Voxel size | **0.25 m** | house = 72×52×22 voxels = **18 × 13 × 5.5 m** |
| Time step `dt` | **1.0 s/step** | calibration target (fire-rate calibration is an open assumption) |
| Walking speed | **1.2 m/s** (×0.5 in smoke) | NFPA-typical; → ~5 cells/step at 0.25 m |
| Water budget | 90 sprays | one ceiling-zone spray per step |

---

## Task 1 — Fire suppression (reinforcement learning)
**Goal:** put the fire out / minimise loss by operating ceiling sprinklers under a water budget.

| | |
|---|---|
| **Observation** | `[per-zone burning fraction (4×3=12), water_budget_fraction]` ∈ [0,1]¹³ |
| **Action** | `Discrete(13)` — spray one of 12 zones, or no-op |
| **Reward** | `R = fuel_saved = 1 − burned/fuel₀` (Δ per step in the Gym env) |
| **Algorithm** | **Cross-Entropy Method (CEM)** — gradient-free; linear policy `a = argmax(W·obs)` |
| **Hyperparameters** | population **32**, elite **8**, generations **18**, **6** scenes/eval, σ-floor 0.03, held-out validation re-rank on 10 scenes |
| **Result (held-out)** | **43.9% → 55.2%** fuel saved · $698 → $573 · smoke 56% → 43%. Greedy oracle = 100% (headroom). |
| **Gym id** | `Ignis-Indoor-v0` (also `Ignis-Wildfire-v0` for the 2D baseline) |

## Task 2 — Evacuation (escape against the clock)
**Goal:** occupants reach an exit before fire/smoke traps them.

| | |
|---|---|
| **Agents** | N occupants spawned in rooms |
| **Policy** | **fire-aware shortest path** — a distance field to exits re-solved each step (BFS over free floor; burning cells are lethal, walls & furniture block; smoke halves speed) |
| **Objective / metrics** | maximise **% escaped**, minimise **casualties** and **mean escape time (s)** |
| **Result (house)** | **8/8 escaped, 0 casualties, mean 10.6 s, all out by 17 s** |
| **RL upgrade** | multi-agent RL (PettingZoo) to learn routing under congestion/uncertainty — documented next step |

## Meta-task — Safe building design (next)
Optimise the **building** (materials, compartmentation, exits, sprinkler/detector placement, smoke
vents/baffles, exit lighting) to **maximise safety beyond code minimums**: maximise time-to-untenability
and % escaped, minimise casualties and $-damage — scored against California Building Code Ch.7, NFPA 101,
ASTM E119 / ISO 834. This turns the two agent tasks into the *inner loop* of a design optimiser.

---

## Playthrough
`python -m ignis.indoor.playthrough` runs fire + suppression + evacuation on the house and renders a
**synchronised plan-view + 3D-view** GIF (`assets/playthrough.gif`); occupant **trajectories** are recorded
and overlaid (blue = escaping, green = out, red = casualty).
