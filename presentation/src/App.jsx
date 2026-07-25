import React, { useState, useEffect, useCallback } from "react";

const asset = (f) => `${import.meta.env.BASE_URL}${f}`;

// ---- data ----
const RESULTS = [
  ["Wildfire — flat", "41.7% saved", "~98%"],
  ["Wildfire — hilly terrain", "63.9% saved", "99.6%"],
  ["Indoor — RL sprinklers (held-out)", "43.9% saved", "55.2%"],
  ["Indoor — $ damage / smoke", "$698 · 56% smoke", "$573 · 43%"],
];
const HYPERPARAMS = [
  ["Grid / voxel", "40×28×16 @ 0.25 m = 10×7×4 m"],
  ["Action space", "Discrete(13) — 4×3 sprinkler zones + no-op"],
  ["Observation", "per-zone burning frac (12) + water budget (1)"],
  ["Reward", "fuel saved = 1 − burned/fuel₀"],
  ["Optimizer", "CEM — pop 24, elite 6, gens 14, 3 scenes/eval"],
  ["Spread", "P_base 0.34 · w_up 2.2 · w_lat 1.1 · w_down 0.4"],
];
const TREE = [
  ["ignis/fire_env.py", "2D wildfire cellular automaton (wind + slope)"],
  ["ignis/config.py", "SimConfig — configurable grid, no hardcoded N"],
  ["ignis/train*.py", "CEM training (flat + terrain)"],
  ["ignis/gym_env.py", "Gymnasium envs: Ignis-Wildfire-v0 / Ignis-Indoor-v0"],
  ["ignis/indoor/schema.py", "VoxelScene — the shared 'our format'"],
  ["ignis/indoor/materials.py", "material table → spread params + $ value"],
  ["ignis/indoor/importer.py", "floorplan JSON → voxels"],
  ["ignis/indoor/generate.py", "procedural building generator (seeded, connected)"],
  ["ignis/indoor/indoor_env.py", "3D voxel fire (buoyancy) + wet mechanic"],
  ["ignis/indoor/suppress.py", "zoned sprinklers + CEM suppression policy"],
  ["ignis/indoor/render3d.py", "matplotlib voxel 3D → indoor3d.gif"],
  ["ignis/indoor/evals.py", "metrics + baseline-vs-trained compare"],
  ["ignis/dataset.py", "spatial fire dataset export + provenance"],
];

// ---- slides ----
const slides = [
  {
    kicker: "01 · Problem", title: "A fast fire sim to train firefighting agents",
    body: (
      <>
        <p>High-fidelity fire (CFD) is minutes–hours per run — far too slow for the millions of
        interactions RL needs. <b>Ignis</b> simulates fire spread with a cheap probabilistic
        <b> cellular automaton</b>, so we can train a suppression policy in <b>seconds</b> and still
        capture wind, slope, buoyancy and materials.</p>
        <p className="fine">Built for the Open World Hackathon (Physical AI, Track 3 — Free). Pure NumPy,
        CPU-only, and shipped as a reusable Gym environment. Hero = <b>3D indoor / structural fire</b>,
        which JaxWildfire (2D, outdoor-only) explicitly excludes.</p>
      </>
    ),
  },
  {
    kicker: "02 · Approach", title: "One voxel format, swappable front-ends & backends",
    body: (
      <>
        <pre className="arch">{`scene (floorplan JSON / USD* / mesh*) ──importer──► VoxelScene ("our format")
                                                      │
  fire_env (2D CA) ─ config(n) ─┐                     ▼
                                ├─► spread model ──► evals + spatial dataset (provenance)
  indoor_env (3D voxel CA) ─────┘                     │
                                                      ▼
                       CEM-trained suppression policy  ·  Gymnasium env
  *USD / mesh = future`}</pre>
        <p className="fine">The physics never changes when the input does — any scene source emits the same
        <code> VoxelScene</code>. Equations: <a href="https://github.com/Tyronita/ignis/blob/main/PHYSICS.md" target="_blank" rel="noreferrer">PHYSICS.md</a>.</p>
      </>
    ),
  },
  {
    kicker: "03 · Demo", title: "3D structural fire — see it burn",
    body: (
      <>
        <figure><img src={asset("indoor3d.gif")} alt="3D house fire" />
          <figcaption>A realistic 15×11×5 m house (5 rooms, real furniture, 0.25 m voxels) — fire ignites in the kitchen and climbs. Slow 3D orbit.</figcaption></figure>
        <div className="two">
          <figure><img src={asset("indoor.gif")} alt="indoor top/side" /><figcaption>Top view (room-to-room) + side view (plume climbs, ceiling fills).</figcaption></figure>
          <figure><img src={asset("scene3d.gif")} alt="outdoor terrain" /><figcaption>Outdoor 3D — fire climbs uphill, tanker drops from above.</figcaption></figure>
        </div>
      </>
    ),
  },
  {
    kicker: "04 · Suppression", title: "An RL agent contains the fire",
    body: (
      <>
        <figure><img src={asset("indoor_compare.gif")} alt="baseline vs trained sprinklers" />
          <figcaption>Same fire — no suppression (left) vs. trained zoned sprinklers (right).</figcaption></figure>
        <figure><img src={asset("compare2d.gif")} alt="wildfire compare" />
          <figcaption>Wildfire: no tanker vs. trained air-tanker — an honest counterfactual.</figcaption></figure>
      </>
    ),
  },
  {
    kicker: "05 · Training", title: "Learning curves",
    body: (
      <>
        <div className="two">
          <figure><img src={asset("curve.png")} alt="wildfire curve" /><figcaption>Wildfire CEM: 41.7% → ~98% fuel saved.</figcaption></figure>
          <figure><img src={asset("curve3d.png")} alt="terrain curve" /><figcaption>Hilly terrain CEM: 63.9% → 99.6%.</figcaption></figure>
        </div>
        <p>Indoor suppression (CEM over a distribution of generated buildings), measured on a
        <b>held-out</b> set: <b>43.9% → 55.2%</b> fuel saved, $698 → $573 damage. A greedy oracle reaches
        100%, so there is clear headroom — the linear policy, not the scenes, is the current ceiling.</p>
      </>
    ),
  },
  {
    kicker: "06 · HPO", title: "Cross-Entropy Method (gradient-free)",
    body: (
      <>
        <p>A tiny linear policy trained with CEM — no gradients, no tuning, converges in seconds:</p>
        <pre className="arch">{`θ_i ~ N(μ, σ²),  i = 1..P                       # sample a population
score_i = mean over S scenes of fuel_saved(θ_i)  # evaluate
elites  = top-K by score
μ ← mean(elites);  σ ← std(elites) + ε           # refit, ε keeps exploration`}</pre>
        <p className="fine">P=24, K=6, gens=14. CEM is robust for small policies; PPO on a neural policy is the
        upgrade once the sim moves to JAX/Gymnax.</p>
      </>
    ),
  },
  {
    kicker: "07 · Physics", title: "Cellular-automaton design & inspiration",
    body: (
      <>
        <div className="formula">p<sub>ignite</sub> = 1 − (1 − p<sub>base</sub>)<sup>&nbsp;E</sup> · p<sub>mult</sub>(material)</div>
        <p><b>E</b> = effective burning-neighbour count, weighted by <b>wind</b> &amp; <b>slope</b> (outdoor) or
        <b> buoyancy</b> (indoor: fire climbs, w<sub>up</sub>=2.2). Walls block; air carries a short-lived flame
        front so fire crosses doorways. Materials set receptivity, burn time, fuel &amp; $-value.</p>
        <p className="fine">Inspiration: Rothermel rate-of-spread, Karafyllidis CA, JaxWildfire (cited prior art).
        Full derivation in <a href="https://github.com/Tyronita/ignis/blob/main/PHYSICS.md" target="_blank" rel="noreferrer">PHYSICS.md</a>.</p>
        <figure><img src={asset("rollout.gif")} alt="CA rollout" /><figcaption>The CA in action — a trained tanker chasing the front.</figcaption></figure>
      </>
    ),
  },
  {
    kicker: "08 · RL env", title: "Ignis-Indoor-v0 — a solvable Gym env",
    body: (
      <>
        <pre className="arch">{`import gymnasium as gym, ignis.gym_env
env = gym.make("Ignis-Indoor-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)`}</pre>
        <table className="results"><tbody>
          {HYPERPARAMS.map(([k, v]) => <tr key={k}><td className="base">{k}</td><td>{v}</td></tr>)}
        </tbody></table>
      </>
    ),
  },
  {
    kicker: "09 · Software", title: "Repo file tree & design",
    body: (
      <table className="results"><tbody>
        {TREE.map(([k, v]) => <tr key={k}><td><code>{k}</code></td><td className="base">{v}</td></tr>)}
      </tbody></table>
    ),
  },
  {
    kicker: "10 · Results & next", title: "Results, and where this goes",
    body: (
      <>
        <table className="results">
          <thead><tr><th>Scenario</th><th>Baseline</th><th>Trained</th></tr></thead>
          <tbody>{RESULTS.map((r) => <tr key={r[0]}><td>{r[0]}</td><td className="base">{r[1]}</td><td className="good">{r[2]}</td></tr>)}</tbody>
        </table>
        <p><b>Next:</b> reward = <b>safe building design</b> (California-code + protocols, time-to-untenable),
        realistic physics (smoke transport, 3D water beam + collisions, movable hose), USD real-scene import,
        and contributing the env back to open source.</p>
        <p className="fine">Prior art &amp; sources: <a href="https://github.com/Tyronita/ignis/blob/main/REFERENCES.md" target="_blank" rel="noreferrer">REFERENCES.md</a> · Repo: <a href="https://github.com/Tyronita/ignis" target="_blank" rel="noreferrer">github.com/Tyronita/ignis</a></p>
      </>
    ),
  },
];

export default function App() {
  const [i, setI] = useState(0);
  const go = useCallback((d) => setI((p) => Math.max(0, Math.min(slides.length - 1, p + d))), []);
  useEffect(() => {
    const h = (e) => { if (e.key === "ArrowRight") go(1); if (e.key === "ArrowLeft") go(-1); };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, [go]);
  const s = slides[i];
  return (
    <div className="deck">
      <header className="topbar">
        <span className="brand">🔥 Ignis</span>
        <span className="count">{String(i + 1).padStart(2, "0")} / {slides.length}</span>
      </header>
      <section className="slide">
        <div className="kicker">{s.kicker}</div>
        <h2>{s.title}</h2>
        {s.body}
      </section>
      <nav className="nav">
        <button onClick={() => go(-1)} disabled={i === 0}>← Prev</button>
        <div className="dots">{slides.map((_, k) => (
          <span key={k} className={k === i ? "dot on" : "dot"} onClick={() => setI(k)} />))}</div>
        <button onClick={() => go(1)} disabled={i === slides.length - 1}>Next →</button>
      </nav>
      <footer>Ignis v0.2 · Open World Hackathon (Physical AI, Track 3) · use ← → to navigate</footer>
    </div>
  );
}
