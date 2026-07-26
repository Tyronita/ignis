# Ignis — 60-second demo video plan

Target: the hackathon "60–90s backup demo video". ~150 words of VO at ~2.5 words/s.
Source clips are all in `assets/` (open `scratchpad/gifs.html` to preview). Record by screen-capturing
the live windows + the deck, or stitch the GIFs with title cards (I can auto-build this — see bottom).

## Storyboard (7 beats = 60 s)

| # | Time | Visual (asset) | On-screen caption | Voiceover |
|---|------|----------------|-------------------|-----------|
| 1 | 0–5 s | title card over `indoor3d.gif` | **Ignis — a physical-AI fire sandbox** | "This is Ignis — a physical-AI fire sandbox: we simulate fire, train agents to fight it, and design safer buildings." |
| 2 | 5–13 s | `rollout.gif` → `compare2d.gif` | Fast fire sim for RL | "Real fire simulators are too slow for reinforcement learning. Ours is a fast cellular automaton — wind, slope, buoyancy, materials — thousands of steps a second." |
| 3 | 13–24 s | `indoor3d.gif` → `scene3d.gif` | Real 3D scenes — indoor & outdoor | "From a floor plan we build a real 3-D house — rooms, furniture, doors — and outdoor terrain with trees and bushes. Fire climbs, fills rooms, and jumps to vegetation." |
| 4 | 24–38 s | **`marl.gif`** (hero, let it play) | Two agent classes: responders + civilians | "Two classes of agents live here. Civilians evacuate along the fastest safe path; fire-engine responders arrive, advance, and hose the fire — a genuine multi-agent problem." |
| 5 | 38–48 s | `cem_vs_ppo.png` → `marl_rl.png` | Real RL — and it's a Gym environment | "It's a standard Gym environment. We train with CEM and PPO — and a learned cooperative responder that puts lives over property, saving more occupants than a greedy baseline." |
| 6 | 48–56 s | `safety.png` (+ flash `datasets/vlge`) | Safe building design + open data | "We score buildings against California code — available versus required escape time — and export every run as spatial data with provenance." |
| 7 | 56–60 s | title card | github.com/Tyronita/ignis · `Ignis-Indoor-v0` | "Ignis — open source, and ready to build worlds on." |

## Direction notes
- **Lead with the wow, keep beat 4 longest.** `marl.gif` (two synced views + both agent classes) is the money shot — give it ~14 s and don't cut away.
- **Captions big & short** (≤6 words); VO carries the detail. Fire-orange accent (#ff6a2b) on black.
- **Pace:** beats 1–3 quick (world-building), beat 4 breathe, beats 5–6 crisp (proof), beat 7 hard cut to repo.
- **Music:** low tension bed, subtle. **No jargon** the judges won't parse in one pass.
- One honest line is a plus: in beat 5 the caption can read "CEM ≈ PPO; learned responders save +0.9 lives" — reviewers respect the candor.

## Two ways to produce it
1. **Screen-record** the live 3D window + the deck slides (OBS / Xbox Game Bar) → edit in Clipchamp/CapCut with the VO above.
2. **Auto-stitch** (I can build `scripts/make_demo_video.py`): concatenate the GIFs above with title cards + captions into `assets/demo.mp4` via imageio-ffmpeg — no editor needed, deterministic, regenerates with the assets. Add a VO track after.

Recommended: auto-stitch a silent cut first (fast, gives the exact 60 s timing), then drop VO on top.

---

## Loom walkthrough — MUSTS + per-slide speaker notes
Deck (13 slides): https://tyronita.github.io/ignis/ · drive with ← → · ~4–5 s per slide for a 60–75 s Loom.

**MUSTS (do not skip):**
1. The **two agent classes** (responders + civilians) — this is the hero and the novelty. Show `marl.gif` moving.
2. It's a **Gym environment** + our **PR #1** (open-source contribution) — say the env id `Ignis-Indoor-v0`.
3. **Real RL**: CEM *and* PPO — and be honest ("CEM ≈ PPO; learned responders trade property for **+lives**").
4. **Novelty vs JaxWildfire**: 3D indoor/structural fire (they're 2D outdoor-only) — say it once.
5. **Safe building design** (ASET−RSET, California code) — the societal-impact hook for the judges.
6. End on the **repo link**.

**Per-slide notes:**
- **0 Hero:** "Ignis — a physical-AI fire sandbox: simulate fire, train agents to fight it, design safer buildings." Point at the legend (yellow = responders, blue = civilians; plan left, 3D right, synced).
- **1 Problem:** "CFD is too slow for RL; we use a fast cellular automaton — seconds, on CPU."
- **2 Architecture:** "Any scene becomes one VoxelScene the physics and agents run on."
- **3 3D scenes:** "A real furnished house, and outdoor terrain with vegetation — fire climbs and jumps to trees."
- **4 Two classes:** *(slow down)* "Two agent classes — civilians evacuate, fire-engine responders advance and hose the fire."
- **5 Suppression:** "An RL agent contains the fire; trajectories show everyone getting out."
- **6 Learners:** "Same Gym env, two learners — CEM and PPO. Honestly, CEM edges PPO here."
- **7 Cooperative MARL:** "A learned responder puts lives first — cuts casualties 3.1 → 2.2 vs a greedy baseline."
- **8 Physics:** "Probabilistic CA — wind, slope, buoyancy, materials; real 0.25 m voxels."
- **9 Gym env:** "It's `gym.make('Ignis-Indoor-v0')` — PPO drops in. Open-sourced in PR #1."
- **10 Safety:** "We score buildings against California code — available vs required escape time."
- **11 Results:** "CPU-only, reproducible with one script."
- **12 Links:** "Repo, the Gym-env PR, and the roadmap — thanks."

**What was missing before (now added to the deck):** hero/title with the annotated MARL gif; a visual on every slide;
learned-responders result; the Gym-env + **PR link**; vegetation/outdoor; safe-building design; a reproduce/links slide.
