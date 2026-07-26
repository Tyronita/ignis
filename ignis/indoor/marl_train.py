"""
Learned cooperative responder policy (CEM) vs the nearest-fire heuristic.

The responder team's *targeting* is a 3-weight linear score over each fire zone —
`[burning fraction, −distance to responder, −distance to exit]` — so it can learn to
prioritise fires that both burn hot and threaten the egress route. CEM optimises those
weights for a cooperative reward (fuel saved + civilians out − casualties). We then
compare no-responders vs heuristic vs learned on held-out seeds.

  python -m ignis.indoor.marl_train
"""

import os
import numpy as np
from . import importer, marl

HOUSE = os.path.join(os.path.dirname(__file__), "examples", "house.json")
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CIV_DELAY = 34            # hard scenario: slow alarm, so fire genuinely threatens occupants


def reward(scene, w, seed):
    m, _ = marl.run(scene, responders=True, civ_delay_s=CIV_DELAY, seed=seed, resp_policy=w)
    return (m["fuel_saved"] / 100 + 0.2 * m["escaped"] / m["occupants"]
            - 0.5 * m["casualties"] / m["occupants"])


def train(scene, gens=8, pop=12, elite=3, scen=3, seed=0):
    rng = np.random.default_rng(seed)
    mean = np.array([1.0, 0.5, 0.5]); std = np.ones(3) * 0.9
    best, best_w = -1e9, mean.copy()
    for g in range(gens):
        pw = mean[None] + std[None] * rng.standard_normal((pop, 3))
        seeds = rng.integers(0, 9999, scen)
        scores = np.array([np.mean([reward(scene, w, int(s)) for s in seeds]) for w in pw])
        order = np.argsort(scores)[::-1]; elites = pw[order[:elite]]
        mean, std = elites.mean(0), elites.std(0) + 0.05
        if scores[order[0]] > best:
            best, best_w = scores[order[0]], pw[order[0]].copy()
        print(f"gen {g}  best reward {best:.3f}  w={np.round(best_w,2)}")
    return best_w


def _agg(scene, policy, responders, seeds):
    ms = [marl.run(scene, responders=responders, civ_delay_s=CIV_DELAY, seed=s,
                   resp_policy=policy)[0] for s in seeds]
    f = lambda k: float(np.mean([m[k] for m in ms]))
    return dict(casualties=f("casualties"), burned=f("burned_pct"), damage=f("damage_usd"))


def main():
    scene = importer.load(HOUSE)
    w = train(scene)
    np.save(os.path.join(ROOT, "resp_policy.npy"), w)
    seeds = list(range(300, 312))
    none = _agg(scene, None, False, seeds)
    heur = _agg(scene, None, True, seeds)
    learn = _agg(scene, w, True, seeds)
    print("\n=== held-out: no-responders vs heuristic vs learned responders ===")
    print(f"{'metric':12s}{'none':>12s}{'heuristic':>12s}{'learned':>12s}")
    for k in ("casualties", "burned", "damage"):
        print(f"{k:12s}{none[k]:>12.1f}{heur[k]:>12.1f}{learn[k]:>12.1f}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        labels = ["casualties", "burned %", "$ damage/100"]
        vals = lambda d: [d["casualties"], d["burned"], d["damage"] / 100]
        x = np.arange(3); w3 = 0.26
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        ax.bar(x - w3, vals(none), w3, label="no responders", color="#7f8c8d")
        ax.bar(x, vals(heur), w3, label="heuristic", color="#e67e22")
        ax.bar(x + w3, vals(learn), w3, label="learned (CEM)", color="#2980b9")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend()
        ax.set_title("Cooperative responders — none vs heuristic vs learned")
        ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "assets", "marl_rl.png"), dpi=120)
        print("saved assets/marl_rl.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
