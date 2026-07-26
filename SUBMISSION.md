# Ignis — Hackathon Submission

> Open World Hackathon: Building the Future of Physical AI · **Track 3 (Free)**
> Fill the two `⟨…⟩` placeholders (team + participant names) before submitting.

## Google-Form fields

| Field | Answer |
|---|---|
| **Team name** | ⟨TEAM NAME⟩ |
| **Participant names** | ⟨PARTICIPANT NAMES⟩ |
| **Project title** | Ignis |
| **Track** | Track 3 — Free |
| **Engine** | Custom Python/NumPy physical-AI fire simulator, packaged as a **Gymnasium** environment (CPU-only) |
| **One-sentence pitch** | *Ignis is a physical-AI fire sandbox: generate a 3D building, simulate structural fire spreading through it, train a reinforcement-learning agent to put it out, and export spatial fire datasets with full provenance.* |
| **Judge-accessible build link** | **https://tyronita.github.io/ignis/** (repo: https://github.com/Tyronita/ignis) |
| **60–90s backup demo video** | [`assets/demo.mp4`](assets/demo.mp4) |
| **Three strongest screenshots** | [`assets/shots/`](assets/shots/) (1: 3D indoor fire · 2: baseline vs trained suppression · 3: learning curve) |
| **Setup / controls / expected outcome** | See below |
| **Track-2: future data-collection plan** | See below |
| **External assets / datasets / AI tools** | See below |
| **Repo / notes + device requirements** | https://github.com/Tyronita/ignis · Python 3.12 + Node/npm · **CPU-only, no GPU** |

## What it is
A fast, dependency-light **fire-spread simulator + RL suppression** environment. The differentiator is the
**3D indoor / structural-fire** setting: procedurally generated buildings (rooms, walls, doors, materials)
where fire climbs, fills rooms top-down and crosses doorways, and an RL agent operates ceiling sprinklers to
contain it. It ships as a **Gym environment** so anyone can train their own agents, and exports **spatial
fire datasets with provenance**.

## Setup / controls / expected outcome
```bash
pip install -r requirements.txt
python scripts/regenerate.py              # reproduces every GIF + trains the agents

# use it as a Gym environment
python -c "import gymnasium as gym, ignis.gym_env; e=gym.make('Ignis-Indoor-v0'); print(e.reset())"

# generate + view a 3D building on fire
python -m ignis.indoor.generate --seed 3 --rooms 5
python -m ignis.indoor.render3d           # -> assets/indoor3d.gif

# train indoor suppression + compare to no-suppression baseline
python -m ignis.indoor.suppress
python -m ignis.indoor.evals --pool 8 --policy policy_indoor.npy
```
**Controls:** the agent's action each step is *which ceiling sprinkler zone to spray* (or no-op), under a
water budget. **Expected outcome:** the trained policy measurably reduces burned volume, rooms lost, and
dollar-damage versus doing nothing (numbers on the build link / in the deck).

## Track-2: future data-collection plan
Ignis is also a **data generator**: every rollout emits per-voxel fire-state, agent actions and outcome
metrics on a configurable spatial grid, with a **provenance manifest** (seed, config, scene source, git SHA,
timestamp, license). Roadmap: (1) release a labelled spatial fire-spread benchmark; (2) pair synthetic fire
scenarios with **human firefighting/evacuation behavior** captured on the VLGE platform (spatial + behavioral
data with transparent provenance); (3) ingest real building geometry via **USD** for grounded scenarios.

## External assets / datasets / AI tools used
- **Libraries:** NumPy, Matplotlib, imageio (sim + viz); React + Vite (presentation); Gymnasium (RL API).
- **Datasets:** none used for training — all scenes are procedurally generated; no external data dependency.
- **AI tools:** built with **Claude Code** (pair-programming the simulator, RL, and docs).
- **Prior art cited (not used as code):** JaxWildfire (arXiv:2512.06102 — 2D outdoor; we extend to 3D indoor),
  PyTorchFire, SimFire/SimHarness, NIST FDS, FieryGS. See [REFERENCES.md](REFERENCES.md).

## Device requirements
CPU-only, no GPU. Python 3.12, Node 18+/npm for the presentation. Runs on a laptop in minutes.
