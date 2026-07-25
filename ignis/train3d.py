"""Retrain the air-tanker on hilly terrain (slope-biased spread).

Run:  python -m ignis.train3d
Writes ../policy3d.npy and assets/curve3d.png.
"""

import os
import numpy as np
from . import fire_env as fe
from .train import evaluate, baseline, POP, ELITE, GENS, DIM, ASSETS

TERRAIN_SEED = 3


def main():
    os.makedirs(ASSETS, exist_ok=True)
    fe.set_terrain(fe.make_terrain(TERRAIN_SEED))   # fire now climbs uphill
    rng = np.random.default_rng(0)
    mean = np.zeros(DIM, np.float32)
    std = np.ones(DIM, np.float32) * 0.8
    base = baseline(rng)
    print(f"terrain baseline (no tanker): {base*100:5.1f}% fuel saved")

    history, best_flat, best_score = [], None, -1.0
    for g in range(GENS):
        pop = mean[None] + std[None] * rng.standard_normal((POP, DIM)).astype(np.float32)
        scores = evaluate(pop, rng)
        order = np.argsort(scores)[::-1]
        elites = pop[order[:ELITE]]
        mean, std = elites.mean(0), elites.std(0) + 0.02
        if scores[order[0]] > best_score:
            best_score, best_flat = scores[order[0]], pop[order[0]].copy()
        history.append((scores[order[:ELITE]].mean(), scores[order[0]]))
        if g % 5 == 0 or g == GENS - 1:
            print(f"gen {g:2d}  elite {history[-1][0]*100:5.1f}%  best {history[-1][1]*100:5.1f}%")

    np.save(os.path.join(ASSETS, "..", "policy3d.npy"),
            best_flat.reshape(fe.N_ACTIONS, fe.N_FEAT))
    print(f"\nsaved policy3d.npy  best {best_score*100:.1f}%  (+{(best_score-base)*100:.1f} pts)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hist = np.array(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist[:, 0] * 100, label="elite mean", lw=2)
    ax.plot(hist[:, 1] * 100, label="gen best", lw=1, alpha=0.6)
    ax.axhline(base * 100, ls="--", c="crimson", label="no-tanker baseline")
    ax.set_xlabel("generation"); ax.set_ylabel("% fuel saved")
    ax.set_title("Air-tanker on hilly terrain (slope-coupled CEM)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "curve3d.png"), dpi=110)
    print("saved assets/curve3d.png")


if __name__ == "__main__":
    main()
