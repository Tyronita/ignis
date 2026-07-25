"""
Train the air-tanker policy with the Cross-Entropy Method (CEM).

CEM is gradient-free and rock-solid for tiny policies: sample a population of
linear policies, score each on several random fires, keep the elites, refit the
sampling Gaussian, repeat. Run as a module:  python -m ignis.train

Artifacts (written under assets/):
  ../policy.npy     trained policy matrix   (repo root, loaded by viz)
  assets/curve.png  learning curve
  assets/rollout.gif trained agent fighting a fresh fire
"""

import os
import numpy as np
from . import fire_env as fe

POP, ELITE, GENS, SCEN, SEED = 48, 10, 30, 6, 0
DIM = fe.N_ACTIONS * fe.N_FEAT
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")


def evaluate(pop_flat, rng):
    """Score every policy on SCEN shared random fires -> mean fuel saved (POP,)."""
    P = pop_flat.shape[0]
    W = pop_flat.reshape(P, fe.N_ACTIONS, fe.N_FEAT)
    Wb = np.repeat(W, SCEN, axis=0)
    s = fe.init_state(P * SCEN, rng)
    for _ in range(fe.T):
        fe.step(s, fe.policy_action(Wb, fe.features(s)), rng)
    return fe.fuel_saved(s).reshape(P, SCEN).mean(axis=1)


def baseline(rng, n=64):
    """Mean fuel saved by doing nothing."""
    s = fe.init_state(n, rng)
    for _ in range(fe.T):
        fe.step(s, np.zeros(n, np.int32), rng)
    return float(fe.fuel_saved(s).mean())


def main():
    os.makedirs(ASSETS, exist_ok=True)
    rng = np.random.default_rng(SEED)
    mean = np.zeros(DIM, np.float32)
    std = np.ones(DIM, np.float32) * 0.8
    base = baseline(rng)
    print(f"no-agent baseline: {base*100:5.1f}% fuel saved")

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
        print(f"gen {g:2d}  elite {history[-1][0]*100:5.1f}%  best {history[-1][1]*100:5.1f}%")

    W = best_flat.reshape(fe.N_ACTIONS, fe.N_FEAT)
    np.save(os.path.join(ASSETS, "..", "policy.npy"), W)
    print(f"\nsaved policy.npy  best {best_score*100:.1f}%  (+{(best_score-base)*100:.1f} pts)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hist = np.array(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist[:, 0] * 100, label="elite mean", lw=2)
    ax.plot(hist[:, 1] * 100, label="gen best", lw=1, alpha=0.6)
    ax.axhline(base * 100, ls="--", c="crimson", label="no-agent baseline")
    ax.set_xlabel("generation"); ax.set_ylabel("% fuel saved")
    ax.set_title("Air-tanker learning to contain the fire (CEM)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "curve.png"), dpi=110)
    print("saved assets/curve.png")

    import imageio.v2 as imageio
    _, frames = fe.rollout(W, np.random.default_rng(123), record=True)
    imageio.mimsave(os.path.join(ASSETS, "rollout.gif"), frames, fps=12, loop=0)
    print("saved assets/rollout.gif")


if __name__ == "__main__":
    main()
