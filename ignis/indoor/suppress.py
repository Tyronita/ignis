"""
Indoor suppression: zoned ceiling sprinklers + a CEM-trained policy.

The ceiling is partitioned into a fixed GX×GY grid of sprinkler zones (layout-agnostic,
so one policy generalizes across generated buildings). Each step the agent sprays one
zone (or no-op), wetting that footprint and extinguishing flame there, under a water
budget. Reward = fraction of fuel saved.

  python -m ignis.indoor.suppress            # train across generated scenes, save policy

Shared with ignis.gym_env (Ignis-Indoor-v0) so training and the Gym env use one code path.
"""

import os
import numpy as np
from . import indoor_env as ie
from . import generate

ZONES = (4, 3)                      # GX × GY ceiling sprinkler zones
N_ACT = ZONES[0] * ZONES[1] + 1     # + no-op
ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")


def _zone_bounds(nx, ny):
    gx, gy = ZONES
    xb = np.linspace(0, nx, gx + 1).astype(int)
    yb = np.linspace(0, ny, gy + 1).astype(int)
    return xb, yb


def init(scene):
    st = ie.init_from_scene(scene)
    st["wet"] = np.zeros(scene.dims, np.float32)
    st["water"] = int(scene.budgets.get("water", 60))
    st["_wmax"] = st["water"] or 1
    st["_xb"], st["_yb"] = _zone_bounds(scene.nx, scene.ny)
    return st


def observe(st):
    """Per-zone burning fraction + budget fraction → (N_ACT,) in [0,1]."""
    gx, gy = ZONES
    xb, yb = st["_xb"], st["_yb"]
    b = st["burning"] > 0
    obs = np.zeros(N_ACT, np.float32)
    for zx in range(gx):
        for zy in range(gy):
            sub = b[xb[zx]:xb[zx+1], yb[zy]:yb[zy+1], :]
            obs[zx * gy + zy] = sub.mean() if sub.size else 0.0
    obs[-1] = st["water"] / max(1, int(st.get("_wmax", st["water"] or 1)))
    return obs


def apply_action(st, a):
    """Spray zone a (0..GX*GY-1), or no-op (a == last). Costs 1 water."""
    gx, gy = ZONES
    if a >= gx * gy or st["water"] <= 0:
        return
    zx, zy = divmod(a, gy)
    xb, yb = st["_xb"], st["_yb"]
    xs, xe = xb[zx], xb[zx+1]; ys, ye = yb[zy], yb[zy+1]
    st["wet"][xs:xe, ys:ye, :] = 1.0                 # water curtain over the zone
    doused = (st["burning"][xs:xe, ys:ye, :] > 0)
    st["burning"][xs:xe, ys:ye, :][doused] = 0.0
    st["timer"][xs:xe, ys:ye, :][doused] = 0
    st["water"] -= 1


def sim_step(st, rng):
    ie.step(st, rng)


def fuel_saved(st):
    return float(1.0 - st["burned"].sum() / max(st["fuel0"], 1))


def policy_action(W, obs):
    return int(np.argmax(W @ obs))


def rollout(W, scene, rng, steps=90):
    st = init(scene)
    st["_wmax"] = st["water"] or 1
    for _ in range(steps):
        a = policy_action(W, observe(st)) if W is not None else N_ACT - 1
        apply_action(st, a)
        sim_step(st, rng)
        if (st["burning"] > 0).sum() == 0 and st["t"] > 5:
            break
    return fuel_saved(st), st


# ---------------- CEM training across a generated distribution ----------------
POP, ELITE, GENS, SCEN = 32, 8, 18, 6
DIM = N_ACT * N_ACT


def _score(Wflat, scenes, rng):
    W = Wflat.reshape(N_ACT, N_ACT)
    return np.mean([rollout(W, sc, rng)[0] for sc in scenes])


def train_indoor(pool_k=14, seed=0):
    rng = np.random.default_rng(seed)
    pool = generate.sample_pool(pool_k, seed0=100)
    base = np.mean([rollout(None, sc, np.random.default_rng(i))[0]
                    for i, sc in enumerate(pool)])
    print(f"no-suppression baseline: {base*100:.1f}% fuel saved")

    mean = np.zeros(DIM, np.float32); std = np.ones(DIM, np.float32) * 0.8
    candidates, history = [], []
    for g in range(GENS):
        popW = mean[None] + std[None] * rng.standard_normal((POP, DIM)).astype(np.float32)
        scenes = [pool[int(rng.integers(len(pool)))] for _ in range(SCEN)]
        scores = np.array([_score(w, scenes, np.random.default_rng(g * 1000 + i))
                           for i, w in enumerate(popW)])
        order = np.argsort(scores)[::-1]
        elites = popW[order[:ELITE]]
        mean, std = elites.mean(0), elites.std(0) + 0.03
        candidates.append((scores[order[0]], popW[order[0]].copy()))
        candidates.append((scores[order[:ELITE]].mean(), mean.copy()))  # the robust mean too
        history.append(scores[order[:ELITE]].mean())
        print(f"gen {g:2d}  elite {history[-1]*100:5.1f}%  best {scores[order[0]]*100:5.1f}%")

    # validation re-rank on a HELD-OUT set → pick a policy that generalizes, not a lucky one
    val = generate.sample_pool(10, seed0=500)
    cand = sorted(candidates, key=lambda c: -c[0])[:8]
    val_scores = [(np.mean([rollout(w.reshape(N_ACT, N_ACT), sc, np.random.default_rng(j))[0]
                            for j, sc in enumerate(val)]), w) for _, w in cand]
    best_val, best_flat = max(val_scores, key=lambda c: c[0])
    val_base = np.mean([rollout(None, sc, np.random.default_rng(j))[0] for j, sc in enumerate(val)])

    os.makedirs(ASSETS, exist_ok=True)
    np.save(os.path.join(ASSETS, "..", "policy_indoor.npy"), best_flat.reshape(N_ACT, N_ACT))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.array(history) * 100, lw=2, label="elite mean (train)")
        ax.axhline(val_base * 100, ls="--", c="crimson", label="no-suppression (val)")
        ax.axhline(best_val * 100, ls="--", c="seagreen", label="chosen policy (val)")
        ax.set_xlabel("generation"); ax.set_ylabel("% fuel saved")
        ax.set_title("Indoor sprinkler policy — CEM"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "curve_indoor.png"), dpi=110)
        print("saved assets/curve_indoor.png")
    except Exception as e:
        print("curve skipped:", e)
    print(f"\nsaved policy_indoor.npy  validation: {val_base*100:.1f}% -> {best_val*100:.1f}% "
          f"fuel saved  (+{(best_val-val_base)*100:.1f} pts, held-out)")
    return best_flat


if __name__ == "__main__":
    train_indoor()
