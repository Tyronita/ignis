# References & design rationale

Sources that informed Ignis, and — for each — **what we borrowed and why**. Ignis is an
independent implementation; these are intellectual sources, not dependencies.

## Fire-spread simulators (the model class)

- **JaxWildfire — A GPU-Accelerated Wildfire Simulator for Reinforcement Learning** (NeurIPS ML4PS 2025, arXiv:2512.06102).
  *Borrowed:* the core idea of a **probabilistic cellular-automaton fire model** made fast enough for RL,
  the **air-tanker suppression** scenario, and terrain/DEM-driven spread. *Why:* it is purpose-built for
  training RL agents (6–35× faster than prior tools) — exactly our goal. *Our difference:* we implement the
  same model class in **NumPy** (the machine's JAX was unusable); JaxWildfire is our GPU **scale-up path**.
  https://arxiv.org/abs/2512.06102

- **PyTorchFire — A Differentiable Cellular-Automata Wildfire Simulator** (Environmental Modelling & Software, 2025).
  *Borrowed:* the differentiable-CA framing. *Why:* it shows the same CA can be made GPU-fast **and**
  differentiable, enabling gradient-based control later. An alternative backend behind our interface.

- **Cell2Fire — a cell-based forest fire growth model.**
  *Borrowed:* per-cell fuel/weather/topography characterization. *Why:* validates the cell-based approach
  to landscape fire and informs our fuel/moisture/slope cell attributes. https://github.com/cell2fire/Cell2Fire

- **SimFire + SimHarness (MITRE Fireline).**
  *Borrowed:* the pattern of an agent placing **mitigations** (firelines, **wetlines**) in a fire sim, wired to RL.
  *Why:* confirms the "move agent + apply suppression" action design we use for the air-tanker.
  https://github.com/mitrefireline/simfire · https://mitrefireline.github.io/simharness/

## Fire physics (fidelity ladder we deliberately did NOT climb)

- **NIST Fire Dynamics Simulator (FDS).** CFD/LES combustion with real sprinkler-droplet suppression.
  *Why cited:* the gold standard for **indoor** fire and water suppression — but minutes-to-hours per run,
  so it is an **offline validation** target, never in our training loop (per project directive to avoid heavy fluid).

- **Rothermel surface rate-of-spread.** The empirical ROS model underlying most operational tools;
  our wind/slope weighting is a lightweight nod to it.

- **J. Stam, "Stable Fluids" (SIGGRAPH 1999).** The real-time Navier-Stokes approach we *considered* for
  buoyant indoor plumes and consciously **deferred** — we use a cheap buoyancy-biased CA instead to stay fast.

## Visualization / 3D scenes (future render layer)

- **FieryGS — In-the-Wild Fire Synthesis with Physics-Integrated Gaussian Splatting** (ICLR 2026).
  *Why:* the path to putting our simulated fire into **photographed real scenes**. https://pku-vcl-geometry.github.io/FieryGS/
- **Unreal Niagara Fluids / EmberGen.** Real-time volumetric flame/smoke for a game-engine render layer
  consuming our voxel state.
- **OpenUSD (Pixar).** Target format for the experimental indoor-scene importer (voxelize a USD building).

## Design decisions (the "why")

1. **CA, not CFD.** RL needs millions of cheap steps; CFD gives tens of slow ones. We simulate spread with a
   probabilistic CA and reserve CFD (FDS) for offline validation. (Directive: don't let fluid slow us down.)
2. **NumPy, not JAX.** The dev machine's JAX checkout was incompatible with its NumPy; NumPy keeps the MVP
   runnable everywhere. JAX (JaxWildfire/PyTorchFire) is the documented scale-up.
3. **CEM, not PPO.** A tiny linear policy + gradient-free CEM converges in seconds with zero tuning — right for
   an MVP. PPO on a neural policy is the upgrade once the sim moves to JAX.
4. **One voxel format, swappable front-ends.** JSON floorplan today; USD / mesh voxelizer as experimental
   presets — all emit the same `VoxelScene`, so the physics never changes when the input does.
5. **Materials modulate a cheap model.** No temperature field; materials set ignition receptivity, burn time,
   fuel load and $-value — enough for meaningful **evals** without a solver.
