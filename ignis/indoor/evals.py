"""
Indoor eval harness: importer/generator -> sim (+optional suppression policy) -> metrics -> gif.

  python -m ignis.indoor.evals                         # single generated scene, no suppression
  python -m ignis.indoor.evals --pool 8 --policy policy_indoor.npy   # baseline vs trained table + compare gif

Metrics (spread + materials): % fuel saved/burned, rooms lost, time-to-flashover, $ material damage,
smoke-filled %, containment. The no-suppression run is the baseline the trained policy is scored against.
"""

import argparse
import os
import numpy as np
from . import indoor_env as ie
from . import suppress, generate, importer

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")


def _volume_rgb(st):
    nx, ny, nz = st["burning"].shape
    rgb = np.zeros((nx, ny, nz, 3), np.float32)
    pri = np.zeros((nx, ny, nz), np.float32)
    rgb[st["is_air"]] = np.array([0.93, 0.94, 0.97]); pri[st["is_air"]] = 0.1
    wall = st["solid"]; rgb[wall] = np.array([0.55, 0.55, 0.58]); pri[wall] = 1.0
    fuelv = (st["fuel"] > 0) & st["burnable"] & (st["burning"] == 0) & ~st["burned"]
    rgb[fuelv] = np.array([0.20, 0.62, 0.24]); pri[fuelv] = 2.0
    if "wet" in st:
        wetv = (st["wet"] > 0.5) & (st["burning"] == 0)
        rgb[wetv] = np.array([0.15, 0.45, 0.85]); pri[wetv] = 2.5
    rgb[st["burned"]] = np.array([0.10, 0.09, 0.09]); pri[st["burned"]] = 3.0
    fire = st["burning"] > 0
    hot = np.clip(st["timer"] / np.maximum(st["burn_time"], 1), 0, 1)
    fr = np.stack([np.ones_like(hot), 0.35 + 0.55 * hot, np.zeros_like(hot)], -1)
    rgb[fire] = fr[fire]; pri[fire] = 4.0
    return rgb, pri


def _project(rgb, pri, axis):
    idx = pri.argmax(axis=axis)
    return np.take_along_axis(rgb, np.expand_dims(idx, (axis, -1)), axis=axis).squeeze(axis)


def _act(st, W, rng):
    """One step under policy W (None = no-op baseline)."""
    a = suppress.N_ACT - 1 if W is None else suppress.policy_action(W, suppress.observe(st))
    suppress.apply_action(st, a)
    suppress.sim_step(st, rng)


def run_episode(scene, policy=None, seed=0, steps=90):
    st = suppress.init(scene)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        _act(st, policy, rng)
        if (st["burning"] > 0).sum() == 0 and st["t"] > 5:
            break
    return ie.metrics(st), st


def evaluate_pool(k, policy=None, seed0=200):
    """Baseline (no suppression) vs policy across k generated scenes → comparison rows."""
    W = None if policy is None else np.load(policy)
    rows = []
    for i in range(k):
        scene = generate.generate_scene(seed=seed0 + i)
        mb, _ = run_episode(scene, None, seed=i)
        mp, _ = run_episode(scene, W, seed=i)
        rows.append((mb, mp))
    return rows


def compare_gif(scene, policy, out=None, seed=0, steps=90, fps=10):
    """Side-by-side top views: no suppression (left) vs trained policy (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    W = np.load(policy)
    sB = suppress.init(scene); sP = suppress.init(scene)
    rngB = np.random.default_rng(seed); rngP = np.random.default_rng(seed)
    frames = []
    for _ in range(steps):
        def top(st):
            rgb, pri = _volume_rgb(st)
            return _project(rgb[:, :, 1:-1], pri[:, :, 1:-1], 2).transpose(1, 0, 2)
        fig, (aL, aR) = plt.subplots(1, 2, figsize=(9, 4.4))
        aL.imshow(top(sB), origin="lower"); aR.imshow(top(sP), origin="lower")
        mb, mp = ie.metrics(sB), ie.metrics(sP)
        aL.set_title(f"No suppression — burned {mb['burned_pct']:.0f}%", color="crimson", fontsize=10)
        aR.set_title(f"Trained sprinklers — burned {mp['burned_pct']:.0f}%  (water {sP['water']})",
                     color="seagreen", fontsize=10)
        for a in (aL, aR):
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle("Indoor fire suppression — RL zoned sprinklers", fontsize=12, weight="bold")
        fig.tight_layout(); fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
        frames.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
        plt.close(fig)
        _act(sB, None, rngB); _act(sP, W, rngP)
    out = out or os.path.join(ASSETS, "indoor_compare.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"saved {out}")


def _print_pool(rows):
    def avg(key, idx):
        return np.mean([r[idx][key] for r in rows])
    print("\n=== Indoor suppression eval (mean over scenes) ===")
    print(f"{'metric':18s}{'no-suppression':>18s}{'trained policy':>18s}")
    for key, label in [("burned_pct", "burned %"), ("rooms_lost", "rooms lost"),
                       ("damage_usd", "$ damage"), ("smoke_pct", "smoke %")]:
        print(f"{label:18s}{avg(key,0):>18.1f}{avg(key,1):>18.1f}")
    print(f"{'fuel saved %':18s}{100-avg('burned_pct',0):>18.1f}{100-avg('burned_pct',1):>18.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=0)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--scene", default=None, help="floorplan JSON (else a generated scene)")
    args = ap.parse_args()
    if args.pool:
        rows = evaluate_pool(args.pool, args.policy)
        _print_pool(rows)
        if args.policy:
            compare_gif(generate.generate_scene(seed=201), args.policy)
    else:
        scene = importer.load(args.scene) if args.scene else generate.generate_scene(seed=3)
        m, _ = run_episode(scene, None)
        print("no-suppression:", {k: round(v, 1) if isinstance(v, float) else v for k, v in m.items()})
