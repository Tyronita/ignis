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
- [ ] `SAFETY.md`: CBC/CRC (Title 24 Ch.7 fire-resistance, Ch.9 sprinklers/NFPA 13, Ch.10 egress), NFPA 72 alarms, NFPA 101, ASTM **E84** flame-spread Class A/B/C, **E119** hourly ratings, ISO 834
- [ ] **Materials grading** table: map our materials → non-combustible / FSI / SDI / fire-rating
- [ ] **Full safety-variable list** (the optimizer's objective space) — draft in `SAFETY.md`
- [ ] `safety.py`: a safe-building **score** (ASET−RSET, compartmentation, egress distance, sprinkler coverage, alarm time)
- [ ] **Safe-building-design meta-task**: CEM over building parameters to maximise the safety score

### A. Physics realism — `indoor_env.py`, `materials.py`  ⚠️ one owner
- [ ] **Smoke transport** as an advected buoyant field + tenability layer height (replaces "air ever flamed")
- [ ] **Heat/conductivity** field (κ∇²T + buoyancy), ignition on accumulated heat; per-material conductivity
- [ ] Calibrate `dt`→seconds & burn rates to HRR / ISO-834

### B. Water as a real 3D beam — `indoor/waterbeam.py`  ⚠️ coordinate with A
- [ ] Parabolic hose beam (ballistic arc), **object collision**, **movable hose agent**; impact removes heat/fuel

### C. MARL agents — `indoor/marl.py`  ✅ decoupled
- [ ] Multi-agent **suppression** (N hose/sprinkler agents) + multi-agent **evacuation**
- [ ] **Dual-task** coordination; PettingZoo-style API; CEM multi-agent training (CPU ok)

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

### H. RL scale-up — `ignis_jax/`  ☁️ needs GPU + working JAX (broken here)
- [ ] JAX/Gymnax port, **PPO** neural policy, vectorized envs, wandb runs

### I. USD import — `indoor/usd_import.py`  ☁️ needs OpenUSD (`usd-core`)
- [ ] Sample `.usda` house → voxelize → `VoxelScene` (same downstream)

### J. VLGE integration — `indoor/vlge_import.py`  ☁️ needs VLGE account/export
- [ ] Export a VLGE world (USD) → voxelize; pair with VLGE behavioral data

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

### 🤝 Teammate (parallel, zero conflict with the above)
1. **G. three.js explorable web world** — biggest visible win, fully isolated (`web/`) → the URL
2. **H. RL scale-up** (JAX/Gymnax/PPO) — if they have a GPU
3. **I. USD import** → **J. VLGE integration**
4. **Domain research** to validate the California/materials numbers in E

**Rule of thumb:** anything creating a *new file/dir* (C, D, E, F-render, G, H, I, J) is parallel-safe. Only **A + B** touch the sim core — assign both to one person and land them in sequence.
