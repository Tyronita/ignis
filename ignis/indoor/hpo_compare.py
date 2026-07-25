"""
Hyperparameter comparison for the suppression policy (CEM).

Trains the sprinkler policy under several CEM hyperparameter settings and reports
held-out fuel-saved, so we can see how the choice of population / generations /
scenes-per-eval affects performance.

  python -m ignis.indoor.hpo_compare
"""

import os
import numpy as np
from . import suppress, generate

DIM = suppress.DIM
CONFIGS = [
    {"name": "small",   "pop": 12, "elite": 3, "gens": 6,  "scen": 4},
    {"name": "medium",  "pop": 24, "elite": 6, "gens": 8,  "scen": 6},
    {"name": "large",   "pop": 40, "elite": 8, "gens": 8,  "scen": 6},
]


def _cem(cfg, pool, val, rng):
    mean = np.zeros(DIM, np.float32); std = np.ones(DIM, np.float32) * 0.8
    best_flat, best = None, -1.0
    for g in range(cfg["gens"]):
        pop = mean[None] + std[None] * rng.standard_normal((cfg["pop"], DIM)).astype(np.float32)
        scenes = [pool[int(rng.integers(len(pool)))] for _ in range(cfg["scen"])]
        scores = np.array([suppress._score(w, scenes, np.random.default_rng(g * 997 + i))
                           for i, w in enumerate(pop)])
        order = np.argsort(scores)[::-1]
        elites = pop[order[:cfg["elite"]]]
        mean, std = elites.mean(0), elites.std(0) + 0.03
        if scores[order[0]] > best:
            best, best_flat = scores[order[0]], pop[order[0]].copy()
    # held-out validation
    W = best_flat.reshape(suppress.N_ACT, suppress.N_ACT)
    val_saved = np.mean([suppress.rollout(W, sc, np.random.default_rng(j))[0]
                         for j, sc in enumerate(val)])
    return float(val_saved)


def main():
    rng = np.random.default_rng(0)
    pool = generate.sample_pool(12, seed0=100)
    val = generate.sample_pool(10, seed0=500)
    base = np.mean([suppress.rollout(None, sc, np.random.default_rng(j))[0]
                    for j, sc in enumerate(val)])
    print(f"held-out baseline (no suppression): {base*100:.1f}% fuel saved\n")

    rows = []
    for cfg in CONFIGS:
        v = _cem(cfg, pool, val, np.random.default_rng(1))
        rows.append((cfg, v))
        print(f"{cfg['name']:8s} pop={cfg['pop']:2d} gens={cfg['gens']:2d} scen={cfg['scen']} "
              f"-> held-out {v*100:5.1f}%  (+{(v-base)*100:.1f} pts)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = [c["name"] for c, _ in rows]
        vals = [v * 100 for _, v in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, vals, color="#e67e22")
        ax.axhline(base * 100, ls="--", c="crimson", label="no-suppression")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_ylabel("% fuel saved (held-out)")
        ax.set_title("Suppression CEM — hyperparameter comparison")
        ax.legend(); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "hpo.png")
        fig.savefig(out, dpi=120); print("\nsaved assets/hpo.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
