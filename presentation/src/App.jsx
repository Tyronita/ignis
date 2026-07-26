import React, { useState, useEffect, useCallback } from "react";

const asset = (f) => `${import.meta.env.BASE_URL}${f}`;
const REPO = "https://github.com/Tyronita/ignis";
const PR = "https://github.com/Tyronita/ignis/pull/1";
const GIST = "https://gist.github.com/Tyronita/72afe8efea415627d6d4be1eef7179f3";

const Fig = ({ src, cap, w }) => (
  <figure style={w ? { width: w, margin: "12px auto" } : {}}>
    <img src={asset(src)} alt={cap} />
    <figcaption>{cap}</figcaption>
  </figure>
);

const RESULTS = [
  ["Wildfire — flat", "41.7% saved", "~98%"],
  ["Wildfire — hilly terrain", "63.9% saved", "99.6%"],
  ["Indoor suppression — CEM / PPO", "47%", "60% / 56%"],
  ["Indoor damage / smoke", "$698 · 56%", "$573 · 43%"],
  ["Responders (learned vs none)", "3.1 casualties", "2.2 · $ halved"],
];
const HYPER = [
  ["Grid / voxel", "72×52×22 @ 0.25 m = 18×13×5.5 m"],
  ["Action", "Discrete(13) — 4×3 sprinkler zones + no-op"],
  ["Reward", "fuel saved = 1 − burned/fuel₀"],
  ["Learners", "CEM (policy search) · PPO (deep RL, MLP)"],
  ["CEM", "pop 32 · elite 8 · gens 18 · held-out re-rank"],
  ["PPO", "MlpPolicy · 150k steps · 6 vec-envs"],
];
const TREE = [
  ["ignis/fire_env.py", "2D wildfire CA (wind + slope)"],
  ["ignis/indoor/generate.py", "procedural building generator"],
  ["ignis/indoor/indoor_env.py", "3D voxel fire (buoyancy) + water"],
  ["ignis/indoor/suppress.py", "zoned sprinklers + CEM"],
  ["ignis/indoor/marl.py", "two-class responders vs civilians"],
  ["ignis/gym_env.py", "Ignis-Indoor-v0 / -Wildfire-v0"],
  ["ignis/indoor/safety.py", "ASET−RSET safe-building score"],
];

const slides = [
  // 0 — HERO
  { hero: true },

  {
    kicker: "01 · Problem", title: "Fire is a physical-AI problem — and sims are too slow for RL",
    body: (<>
      <p>High-fidelity fire (CFD) is minutes–hours per run — far too slow for the millions of interactions RL
      needs. Ignis simulates spread with a fast <b>cellular automaton</b> (wind, slope, buoyancy, materials),
      so we train in <b>seconds</b> on CPU.</p>
      <div className="two">
        <Fig src="rollout.gif" cap="Trained air-tanker containing a fire (2D)" />
        <Fig src="compare2d.gif" cap="Same fire — no tanker vs. trained tanker" />
      </div>
    </>),
  },
  {
    kicker: "02 · Architecture", title: "One voxel format, swappable front-ends & learners",
    body: (<>
      <pre className="arch">{`scene (floor-plan JSON / USD*) ─importer─► VoxelScene ("our format")
                                             │
  fire_env (2D CA) ─┐                        ▼
                    ├─► spread ─► evals · safety · spatial dataset (provenance)
  indoor_env (3D) ──┘                        │
                                             ▼
              agents: RL suppression · responders + civilians (MARL)`}</pre>
      <Fig src="indoor3d.gif" cap="Any scene → the same VoxelScene the physics + agents run on" w="62%" />
    </>),
  },
  {
    kicker: "03 · 3D scenes", title: "Real 3D worlds — indoor & outdoor",
    body: (<div className="two">
      <Fig src="indoor3d.gif" cap="18×13×5.5 m house, real furniture (brown wood / blue upholstery), fire climbs" />
      <Fig src="scene3d.gif" cap="Terrain with trees & bushes; fire climbs uphill, tanker drops water" />
    </div>),
  },
  {
    kicker: "04 · Two agent classes", title: "Responders + civilians — a multi-agent problem",
    body: (<>
      <Fig src="marl.gif" cap="Plan + 3D synced · yellow □ = fire-engine responders · blue → green = civilians escaping" />
      <p className="fine">Civilians evacuate the fastest safe path; fire-engine responders arrive, advance, and
      hose the fire. Full task/reward/hyperparams in the repo (TASKS.md).</p>
    </>),
  },
  {
    kicker: "05 · Suppression RL", title: "An agent learns to contain the fire",
    body: (<div className="two">
      <Fig src="indoor_compare.gif" cap="Indoor: no sprinklers vs. trained zoned sprinklers" />
      <Fig src="playthrough.gif" cap="Plan + 3D playthrough with occupant trajectories" />
    </div>),
  },
  {
    kicker: "06 · Learners", title: "CEM (policy search) vs PPO (deep RL) — same Gym env",
    body: (<>
      <div className="two">
        <Fig src="cem_vs_ppo.png" cap="Held-out: no-agent 47% · PPO 56% · CEM 60%" />
        <Fig src="hpo.png" cap="CEM hyperparameter sensitivity" />
      </div>
      <p className="fine">Honest finding: the simple CEM baseline edges PPO — common on low-dimensional control.
      CEM is <i>policy search</i>, not a gradient RL algorithm; PPO makes the deep-RL claim literal.</p>
    </>),
  },
  {
    kicker: "07 · Cooperative MARL", title: "Learned responders put lives over property",
    body: (<>
      <Fig src="marl_rl.png" cap="none vs heuristic vs LEARNED (CEM) — learned cuts casualties 3.1 → 2.2" w="72%" />
      <p className="fine">With a lives-weighted reward, the learned policy prioritises the egress route over the
      biggest fire — saving ~0.9 more occupants per run than a greedy nearest-fire baseline.</p>
    </>),
  },
  {
    kicker: "08 · Physics", title: "The cellular-automaton model (real 0.25 m voxels)",
    body: (<>
      <div className="formula">p<sub>ignite</sub> = 1 − (1 − p<sub>base</sub>)<sup>&nbsp;E</sup> · p<sub>mult</sub>(material)</div>
      <p><b>E</b> = burning-neighbour count weighted by <b>wind</b> &amp; <b>slope</b> (outdoor) or <b>buoyancy</b>
      (indoor: fire climbs). Walls block; materials set receptivity, burn time, fuel &amp; $-value.</p>
      <Fig src="curve_indoor.png" cap="Indoor suppression learning curve (CEM)" w="60%" />
    </>),
  },
  {
    kicker: "09 · Gym environment", title: "Ignis-Indoor-v0 — a solvable, reusable Gym env",
    body: (<>
      <pre className="arch">{`import gymnasium as gym, ignis.gym_env
env = gym.make("Ignis-Indoor-v0")            # deep RL drops in:
from stable_baselines3 import PPO
PPO("MlpPolicy", "Ignis-Indoor-v0").learn(150_000)`}</pre>
      <table className="results"><tbody>
        {HYPER.map(([k, v]) => <tr key={k}><td className="base">{k}</td><td>{v}</td></tr>)}
      </tbody></table>
      <p className="fine">Open-source contribution — the env + all code: <a href={PR} target="_blank" rel="noreferrer">PR&nbsp;#1</a> · <a href={REPO} target="_blank" rel="noreferrer">repo</a></p>
    </>),
  },
  {
    kicker: "10 · Safety", title: "Safe building design — ASET − RSET",
    body: (<>
      <Fig src="safety.png" cap="Code-minimum (1 exit) vs optimised (2 exits + Class-A): margin, casualties, score" w="72%" />
      <p className="fine">Available vs Required Safe Egress Time; materials graded to ASTM E84 (Class A/B/C);
      checks CBC/CRC egress + sprinklers. Occupant paths export in the VLGE trajectory shape.</p>
    </>),
  },
  {
    kicker: "11 · Results", title: "Results",
    body: (<>
      <table className="results">
        <thead><tr><th>Scenario</th><th>Baseline</th><th>Trained / better</th></tr></thead>
        <tbody>{RESULTS.map((r) => <tr key={r[0]}><td>{r[0]}</td><td className="base">{r[1]}</td><td className="good">{r[2]}</td></tr>)}</tbody>
      </table>
      <Fig src="compare2d.gif" cap="Everything runs on CPU, NumPy — reproducible via scripts/regenerate.py" w="72%" />
    </>),
  },
  {
    kicker: "12 · Software & links", title: "Repo, contribution & reproduce",
    body: (<>
      <table className="results"><tbody>
        {TREE.map(([k, v]) => <tr key={k}><td><code>{k}</code></td><td className="base">{v}</td></tr>)}
      </tbody></table>
      <p className="links">
        <a href={REPO} target="_blank" rel="noreferrer">▸ github.com/Tyronita/ignis</a>
        <a href={PR} target="_blank" rel="noreferrer">▸ Gym-env PR #1</a>
        <a href={GIST} target="_blank" rel="noreferrer">▸ roadmap gist</a>
      </p>
      <p className="fine">Physics: PHYSICS.md · Tasks: TASKS.md · Safety: SAFETY.md · CPU-only, no GPU.</p>
    </>),
  },
];

function Hero() {
  return (
    <section className="slide hero">
      <div className="heroTop">
        <div>
          <div className="brandBig">🔥 Ignis</div>
          <p className="tag">A physical-AI fire sandbox — simulate fire, train agents to fight it, design safer buildings.</p>
          <div className="pills">
            <span>3D structural fire</span><span>Two-class MARL</span><span>Gym env</span>
            <span>Safe-building design</span>
          </div>
          <p className="fine">Open World Hackathon · Physical AI · Track 3 (Free) ·
            {" "}<a href={REPO} target="_blank" rel="noreferrer">github.com/Tyronita/ignis</a></p>
        </div>
      </div>
      <figure className="heroFig">
        <img src={asset("marl.gif")} alt="two-class MARL" />
      </figure>
      <div className="legend">
        <span><b style={{ color: "#f1c40f" }}>■</b> fire-engine responders</span>
        <span><b style={{ color: "#1f6feb" }}>●</b> civilians escaping</span>
        <span><b style={{ color: "#2ecc71" }}>●</b> escaped</span>
        <span><b style={{ color: "#e74c3c" }}>●</b> casualty</span>
        <span className="sub">left = architectural PLAN · right = 3D voxel · time-synced</span>
      </div>
    </section>
  );
}

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
      <div className="progress"><div style={{ width: `${(i / (slides.length - 1)) * 100}%` }} /></div>
      <header className="topbar">
        <span className="brand">🔥 Ignis</span>
        <span className="count">{String(i + 1).padStart(2, "0")} / {slides.length}</span>
      </header>
      {s.hero ? <Hero /> : (
        <section className="slide">
          <div className="kicker">{s.kicker}</div>
          <h2>{s.title}</h2>
          {s.body}
        </section>
      )}
      <nav className="nav">
        <button onClick={() => go(-1)} disabled={i === 0}>← Prev</button>
        <div className="dots">{slides.map((_, k) => (
          <span key={k} className={k === i ? "dot on" : "dot"} onClick={() => setI(k)} />))}</div>
        <button onClick={() => go(1)} disabled={i === slides.length - 1}>Next →</button>
      </nav>
      <footer>Ignis · Open World Hackathon (Physical AI) · ← → to navigate</footer>
    </div>
  );
}
