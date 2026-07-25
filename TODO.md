# Ignis — TODO & parallelization

_From branch `feat/indoor-3d-sim` ([PR #1](https://github.com/Tyronita/ignis/pull/1)). Goal: split work so two people run in parallel with minimal merge conflicts._

**Legend:** 🖥️ = best on the current machine (Windows · CPU · Python/NumPy/matplotlib · Node · `gh`). ☁️ = better elsewhere (GPU/JAX, OpenUSD, VLGE account, web-dev focus, or domain research). Each stream lists the files it owns — **streams that own different files run fully in parallel.**

---

## Decoupling map (who can touch what without colliding)
| Stream | Owns (files) | Parallel-safe? |
|---|---|---|
| A. Physics core | `indoor_env.py`, `materials.py` | ⚠️ **serialize** — A & B both edit the sim core; one owner |
| B. Water beam | `indoor/waterbeam.py` (+ small hook in env) | ⚠️ coordinate with A |
| C. MARL agents | `indoor/marl.py` (reads env, no core edits) | ✅ fully parallel |
| D. Analytics | `indoor/analytics.py` | ✅ fully parallel |
| E. Safety + standards | `SAFETY.md`, `indoor/safety.py` | ✅ fully parallel |
| F. Scene realism & views | `indoor/generate.py`, `indoor/render_rooms.py` | ✅ parallel (own render file) |
| G. Web explorable world | `web/` (new dir, three.js) | ✅ fully isolated |
| H. RL scale-up | `ignis_jax/` (new pkg) | ✅ isolated |
| I. USD import | `indoor/usd_import.py` | ✅ isolated |
| J. VLGE integration | `indoor/vlge_import.py` | ✅ isolated |

---

## 🖥️ Makes sense on THIS machine (CPU, NumPy, no external accounts)

### D. Performance analytics — `indoor/analytics.py`  ✅ decoupled, quick win
- [ ] Compare modes over N scenes: **no-suppression · suppression-only · evacuation-only · dual** → `analytics.png` + table + JSON
- [ ] Add **ASET vs RSET** (Available/Required Safe Egress Time) as the headline safety metric
- [ ] Report how CEM-suppression changes *both* fire outcomes and evacuation casualties

### E. Safety, California standards, materials grading — `SAFETY.md` + `indoor/safety.py`  ✅ decoupled
- [x] `SAFETY.md`: CBC/CRC (Ch.7 fire-resistance, Ch.9 sprinklers/NFPA 13, Ch.10 egress), NFPA 72/101, ASTM **E84** Class A/B/C, **E119**, ISO 834
- [x] **Materials grading** table (non-combustible / FSI class / fire role)
- [x] **Full safety-variable list** (the optimiser's objective space)
- [x] `safety.py`: safe-building **score** + **ASET−RSET**, code-min vs optimised comparison → `assets/safety.png`
- [ ] **Safe-building-design meta-task**: CEM over building parameters (add/relocate exits, finishes, sprinklers) — NEXT

### E2. VLGE-shaped trajectories + HPO  ✅ done this pass
- [x] `vlge_export.py`: occupant paths → VLGE `characterSnapshots` shape (pc/velocity/movement_state/templateSize) → `datasets/vlge/`
- [x] `hpo_compare.py`: suppression CEM under small/medium/large hyperparameters → `assets/hpo.png`

### A. Physics realism — `indoor_env.py`, `materials.py`  ⚠️ one owner
- [ ] **Smoke transport** as an advected buoyant field + tenability layer height (replaces "air ever flamed")
- [ ] **Heat/conductivity** field (κ∇²T + buoyancy), ignition on accumulated heat; per-material conductivity
- [ ] Calibrate `dt`→seconds & burn rates to HRR / ISO-834

### B. Water as a real 3D beam — `indoor/waterbeam.py`  ⚠️ coordinate with A
- [ ] Parabolic hose beam (ballistic arc), **object collision**, **movable hose agent**; impact removes heat/fuel

### C. MARL — **two agent classes** — `indoor/marl.py`  🟡 in progress
- [x] **Class 1 · Fire-engine responders** = vehicle **+** operators: arrive after a response delay, advance to
  the fire, hose it (extinguish + wet radius) under a water budget — heuristic policy — **A** (sim) / **B** (3D vehicle anim)
- [x] **Class 2 · Civilians** = occupant evacuation + pre-movement delay — **A**
- [x] Two-class **playthrough** (plan + 3D) → `assets/marl.gif`; responders ~halve fire damage
- [ ] **Cooperative *learned* MARL**: replace the responder heuristic with CEM/PPO, per-class rewards,
  PettingZoo-style API — NEXT — **A**

### E3. Deep RL (PPO) + comparison — `indoor/train_ppo.py`  ✅ done this pass
- [x] PPO (SB3 `MlpPolicy`) on `Ignis-Indoor-v0`; **CEM-vs-PPO-vs-baseline** → `assets/cem_vs_ppo.png` (CEM edges PPO — honest)

### F. Scene realism & views — `indoor/generate.py`, `indoor/render_rooms.py`  ✅ decoupled
- [ ] **Per-room views** (grid of room close-ups) + unified **indoor / outdoor / plan / room** multi-view
- [ ] Detailed procedural house generator (furniture templates: bed/table/chairs/sofa)

### Submission-critical 🖥️ (do first)
- [ ] **60–90 s demo video** (imageio-ffmpeg) — still pending
- [ ] Keep deck + GitHub Pages current

---

## ☁️ Doesn't make sense on this machine (teammate / other box / accounts)

### G. Explorable web world — `web/` (three.js)  ☁️ web-dev focus
- [ ] Load a `VoxelScene` JSON in-browser, orbit/zoom/**walk**, fire playback → **the URL world**
- Runs anywhere with Node; best owned by whoever's frontend-focused. *(Could be 🖥️ but naturally its own stream.)*

### H. RL scale-up — `ignis_jax/`  ⏸ **PARKED** (JAX broken on this machine; not on anyone's plate for now)
- [ ] _Deferred:_ JAX/Gymnax port, **PPO** neural policy, vectorized envs. Revisit only if a GPU box appears.
- CEM on CPU is sufficient for current scale — no JAX dependency.

### I. USD import — `indoor/usd_import.py`  ☁️ needs OpenUSD (`usd-core`)
- [ ] Sample `.usda` house → voxelize → `VoxelScene` (same downstream)

### J. VLGE integration — `indoor/vlge_import.py`  ✅ **DONE**
- [x] Export a VLGE world → voxelize; pair with VLGE behavioural data _(+ VLGE-shaped trajectory export via `vlge_export.py`)_

### Domain research  ☁️ (no coding — validate E's numbers)
- [ ] Confirm California code specifics (CRC sprinkler mandate, egress distances, FSI classes per finish)

---

## Priority lists

### 👤 You (this machine) — order
1. **Demo video** (submission-critical, 15 min)
2. **D. Analytics** — fast, strong for judges (baseline/suppress/evac/dual + ASET/RSET)
3. **E. Safety + standards + materials grading** (+ safe-building score)
4. **A. Physics realism** (smoke → heat/conductivity)
5. **B. Water beam** → then **C. MARL**
6. **F. Per-room views / richer generator**

### 🤝 Teammate (B) — parallel, zero conflict with A (all consume the stable `VoxelScene` / exported JSON)

**Web / explorable world (`web/`) — the flagship → the URL**
1. three.js viewer: load a `VoxelScene` JSON, **orbit / zoom / walk**, fire playback
2. **3D meshes**: trees & bushes, higher-fidelity furniture, and the **fire-engine vehicle + moving animation**
3. Overlay **trajectories** in the viewer — civilians *and* responders (VLGE-shaped data already exported by `vlge_export.py`)
4. Viewer UX: timeline scrub, plan / 3D / per-room / exterior camera presets, then **deploy the viewer URL**

**Asset & scene pipeline**
5. **USD import** (`usd-core`) → `VoxelScene`; and **export scenes + vegetation to glTF/USD** for the viewer
6. ~~**VLGE integration**~~ — ✅ **DONE** (VLGE world → voxelize + VLGE-shaped trajectories exported)

**Domain & data (research — no sim code)**
7. Validate **California code** numbers + **materials grading** (FSI/SDI, HRR, ignition temp) for `SAFETY.md` / materials DB
8. **Occupancy & fuel-load presets** per room type (code fuel-load tables) → hand to A as data

- ~~JAX / GPU scale-up~~ — **parked** (no GPU; CEM on CPU is enough).

**Rule of thumb:** anything creating a *new file/dir* (C, D, E, F-render, G, H, I, J) is parallel-safe. Only **A + B** touch the sim core — assign both to one person and land them in sequence.

---

## Backlog v2 — mixed (tag = who can do it) · **A = this machine** (NumPy/CPU/matplotlib/gh) · **B = teammate** (web/GPU/USD/domain)

**Vegetation & wildland–urban interface (WUI)**
- [ ] Add `bush` and `tree` materials (fuel load, ignition prob, moisture, ember behaviour) — **A**
- [ ] Place exterior vegetation around the house; ignite the structure from a nearby burning tree — **A**
- [ ] WUI scenario: outdoor wildfire → vegetation → structure (couple `fire_env` ↔ indoor voxel) — **A**
- [ ] Ember / firebrand spotting (wind-carried ignition jumps) — **A**
- [ ] Tree/bush 3D models in the explorable viewer — **B**

**Furniture & materials analysis**
- [ ] Parametric furniture library (bed/sofa/table/chairs/wardrobe templates, real sizes) — **A**
- [ ] Real materials database: HRR curve, ignition temp, FSI/SDI, char rate per material — **A** (values validated by **B**)
- [ ] Per-material fire-behaviour report (burn time, peak HRR, smoke) → chart — **A**
- [ ] Higher-fidelity furniture meshes for rendering — **B**

**Two agent classes: responders vs civilians**
- [ ] **Fire-engine responders** (vehicle + operators): engine drives up the road & parks; operators dismount,
  position, operate hose / truck water-cannon; realistic response-time delay; pump/tank water supply — **A** (sim) / **B** (3D vehicle anim)
- [ ] **Civilians**: evacuation agents (baseline done) + pre-movement / exit-knowledge / panic / group behaviour — **A**
- [ ] Cooperative objective: responders **suppress *and* protect civilians** → score lives saved + damage — **A**
- [ ] Parked **car as high-hazard fuel** (fuel tank) in garage/driveway; garage ignition → house spread — **A**

**Mixed (viz / data / infra / standards)**
- [ ] Unified multi-view: exterior + plan + 3D + per-room, synchronised — **A**
- [ ] Export scenes + vegetation to glTF/USD for the web viewer — **B**
- [ ] Occupancy / fuel-load presets per room type (code fuel-load tables) — **A** (data by **B**)
- [ ] Demo video stitching all scenarios — **A**
- [ ] CI to run `regenerate.py` + publish Pages on merge — **A**
