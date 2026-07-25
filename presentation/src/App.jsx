import React from "react";

// media lives in /public and is copied there by scripts/regenerate.py
const asset = (f) => `${import.meta.env.BASE_URL}${f}`;

const RESULTS = [
  { scenario: "Wildfire — flat", base: "41.7%", trained: "~98%" },
  { scenario: "Wildfire — hilly terrain", base: "63.9%", trained: "99.6%" },
  { scenario: "Indoor apartment (no suppression)", base: "3/3 rooms lost · $1384", trained: "next build" },
];

const ROADMAP = [
  "Indoor suppression agent (sprinklers / firefighter) vs. the eval harness",
  "Budgets as hard RL constraints (water / time / compute)",
  "USD / mesh voxelizer importer (experimental → default-quality)",
  "JAX backend swap (JaxWildfire / PyTorchFire) for GPU-scale training",
  "Photoreal render layer (PyVista → Unreal Niagara / FieryGS)",
];

const REFS = [
  ["JaxWildfire (NeurIPS ML4PS 2025)", "https://arxiv.org/abs/2512.06102", "probabilistic CA + air-tanker RL — our model class & scale-up path"],
  ["PyTorchFire (EMS 2025)", "https://www.sciencedirect.com/science/article/pii/S1364815225000854", "differentiable CA — alt GPU backend"],
  ["SimFire / SimHarness (MITRE)", "https://mitrefireline.github.io/simharness/", "agent places mitigations (wetlines) — action design"],
  ["NIST FDS", "https://pages.nist.gov/fds-smv/", "CFD gold standard — offline validation only"],
  ["FieryGS (ICLR 2026)", "https://pku-vcl-geometry.github.io/FieryGS/", "fire into real photographed scenes — future render"],
];

function Slide({ id, kicker, title, children }) {
  return (
    <section className="slide" id={id}>
      {kicker && <div className="kicker">{kicker}</div>}
      {title && <h2>{title}</h2>}
      {children}
    </section>
  );
}

export default function App() {
  return (
    <div className="deck">
      <header className="hero">
        <div className="flame">🔥</div>
        <h1>Ignis</h1>
        <p className="tag">A fast fire-spread simulator &amp; RL suppression sandbox — <b>wildfire today, indoor next.</b></p>
        <p className="sub">Simulate spread → train an agent to put it out → measure the outcome. Pure NumPy, runs anywhere.</p>
        <div className="pills">
          <span>Cellular-automaton physics</span>
          <span>Cross-Entropy RL</span>
          <span>Materials &amp; evals</span>
          <span>v0.1 MVP</span>
        </div>
      </header>

      <Slide id="problem" kicker="The problem" title="Training a firefighting agent needs a fast fire sim">
        <p>High-fidelity fire (CFD) is minutes-to-hours per run — far too slow for the millions of
        interactions RL needs. Ignis simulates spread with a cheap probabilistic <b>cellular automaton</b>,
        so we can train a suppression policy in <b>seconds</b> and still capture wind, slope, buoyancy and materials.</p>
        <div className="formula">p<sub>ignite</sub> = 1 − (1 − p<sub>base</sub>)<sup>&nbsp;effective burning neighbors</sup></div>
        <p className="fine">Effective neighbors weighted by wind &amp; slope (outdoor) or buoyancy &amp; materials (indoor). No fluid solver in the loop, by design.</p>
      </Slide>

      <Slide id="results" kicker="Results" title="A trained agent dramatically contains the fire">
        <table className="results">
          <thead><tr><th>Scenario</th><th>No-agent baseline</th><th>Trained agent</th></tr></thead>
          <tbody>
            {RESULTS.map((r) => (
              <tr key={r.scenario}><td>{r.scenario}</td><td className="base">{r.base}</td><td className="good">{r.trained}</td></tr>
            ))}
          </tbody>
        </table>
        <figure>
          <img src={asset("compare2d.gif")} alt="no tanker vs trained tanker" />
          <figcaption>Same fire — no tanker (left) vs. the trained air-tanker (right). An honest counterfactual.</figcaption>
        </figure>
        <div className="two">
          <figure><img src={asset("rollout.gif")} alt="trained rollout" /><figcaption>Trained tanker containing a fire.</figcaption></figure>
          <figure><img src={asset("scene3d.gif")} alt="3D terrain" /><figcaption>3D terrain — fire climbs uphill; tanker drops from above.</figcaption></figure>
        </div>
        <figure><img src={asset("curve.png")} alt="learning curve" /><figcaption>CEM learning curve — fuel saved vs. the do-nothing baseline.</figcaption></figure>
      </Slide>

      <Slide id="indoor" kicker="New foundation" title="Indoor: a floorplan becomes a burning building">
        <p>A <b>JSON floorplan</b> (rooms, walls, doors, furniture) is converted to a voxel <b>“our format”</b> with
        <b> materials</b>. Fire <b>climbs</b>, fills rooms <b>top-down</b>, and crosses <b>doorways</b> — scored by an
        eval harness. Any front-end (JSON today, USD / voxelized mesh next) emits the same format, so the physics never changes.</p>
        <figure><img src={asset("indoor.gif")} alt="indoor apartment fire" />
          <figcaption>Top view (room-to-room spread) + side view (plume climbs, ceiling fills). No suppression = total loss — the baseline the future policy is scored against.</figcaption></figure>
        <div className="metrics">
          {["% burned / fuel saved", "rooms lost", "time-to-flashover", "$ material damage", "smoke-filled %", "contained?"].map((m) => (
            <span key={m}>{m}</span>
          ))}
        </div>
      </Slide>

      <Slide id="arch" kicker="Architecture" title="One format, swappable front-ends & backends">
        <pre className="arch">{`scene (JSON floorplan / USD* / mesh*) ──importer──► VoxelScene ("our format")
                                                       │
   fire_env (2D CA) ── config(n) ──┐                   ▼
                                   ├─► spread model ─► eval harness (damage, rooms, flashover)
   indoor_env (3D voxel CA) ───────┘                   │
                                                       ▼
                 CEM-trained suppression policy (air-tanker; indoor agent = next)
   *USD / mesh voxelizer = experimental preset, not default`}</pre>
        <p className="fine">Grid size is configurable (<code>SimConfig(n=…)</code>, indoor <code>(nx,ny,nz)</code>) — nothing is hardcoded. JaxWildfire / PyTorchFire can replace the spread core behind the same interface.</p>
      </Slide>

      <Slide id="roadmap" kicker="Roadmap" title="Built to be built upon">
        <ul className="roadmap">{ROADMAP.map((r) => <li key={r}>{r}</li>)}</ul>
      </Slide>

      <Slide id="refs" kicker="Sources & rationale" title="What we borrowed, and why">
        <ul className="refs">
          {REFS.map(([name, url, why]) => (
            <li key={name}><a href={url} target="_blank" rel="noreferrer">{name}</a> — {why}</li>
          ))}
        </ul>
        <p className="fine">Ignis is an independent NumPy implementation — these are intellectual sources, not dependencies. It does <b>not</b> use JaxWildfire; that is the documented GPU scale-up path.</p>
      </Slide>

      <footer>Ignis v0.1 · hackathon MVP · maintained deck — regenerated by <code>scripts/regenerate.py</code></footer>
    </div>
  );
}
