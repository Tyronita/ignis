"""
3D wildfire scene: the trained air-tanker over real terrain (slope-coupled spread).

  python -m ignis.scene3d          live rotating 3D window (TkAgg)
  python -m ignis.scene3d save      write assets/scene3d.gif

The 2D CA state is lifted onto an elevation heightmap; fire climbs uphill, the
surface is painted with the live fire state, flames/smoke rise as columns, and the
tanker flies above dropping water columns.
"""

import os
import sys
import numpy as np

SAVE = len(sys.argv) > 1 and sys.argv[1] == "save"
import matplotlib
matplotlib.use("Agg" if SAVE else "TkAgg")
import matplotlib.pyplot as plt
from . import fire_env as fe

ROOT = os.path.join(os.path.dirname(__file__), "..")
ZSCALE = 15.0

TERR = fe.make_terrain(3)
fe.set_terrain(TERR)
Z = TERR * ZSCALE
X, Yg = np.meshgrid(np.arange(fe.N), np.arange(fe.N))
FLY_Z = Z.max() + 7.0
W = np.load(os.path.join(ROOT, "policy3d.npy"))

# --- vegetation on the terrain: trees + bushes that ignite as the fire reaches them ---
_vrng = np.random.default_rng(1)
TREES = np.stack([_vrng.integers(2, fe.N - 2, 60), _vrng.integers(2, fe.N - 2, 60)], 1)
BUSHES = np.stack([_vrng.integers(2, fe.N - 2, 120), _vrng.integers(2, fe.N - 2, 120)], 1)
TREE_H = 6.0
_GREEN = np.array([0.13, 0.5, 0.13]); _ORANGE = np.array([1.0, 0.4, 0.05])
_DARK = np.array([0.10, 0.08, 0.06])


def _veg(pts, sc):
    x, y = pts[:, 0], pts[:, 1]
    b = sc.s["burning"][0][y, x] > 0
    burned = (sc.s["fuel"][0][y, x] <= 0) & ~b
    col = np.tile(_GREEN, (len(pts), 1)); col[burned] = _DARK; col[b] = _ORANGE
    return col, Z[y, x]


class Scene:
    def __init__(self, seed):
        self.reset(seed)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.s = fe.init_state(1, self.rng)
        self.t = 0
        self.prev_budget = int(self.s["budget"][0])
        self.dropped = False

    def step(self):
        fe.step(self.s, fe.policy_action(W[None], fe.features(self.s)), self.rng)
        self.dropped = int(self.s["budget"][0]) < self.prev_budget
        self.prev_budget = int(self.s["budget"][0])
        self.t += 1


def draw(ax, sc, azim):
    ax.cla(); ax.set_axis_off(); ax.view_init(elev=42, azim=azim)
    ax.set_zlim(0, FLY_Z + 6)
    rgb = fe.cell_rgb(sc.s)
    rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1), np.float32)], -1)
    ax.plot_surface(X, Yg, Z, facecolors=rgba, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    # vegetation: tree canopies (^) on trunks + low bushes (o), coloured by fire state
    tcol, tbase = _veg(TREES, sc)
    ax.scatter(TREES[:, 0], TREES[:, 1], tbase + TREE_H, c=tcol, marker="^", s=120, depthshade=False)
    ax.scatter(TREES[:, 0], TREES[:, 1], tbase + TREE_H * 0.45, c=[[0.4, 0.26, 0.13]], s=14, depthshade=False)
    bcol, bbase = _veg(BUSHES, sc)
    ax.scatter(BUSHES[:, 0], BUSHES[:, 1], bbase + 1.2, c=bcol, marker="o", s=40, depthshade=False)
    burning = sc.s["burning"][0] > 0
    if burning.any():
        by, bx = np.where(burning)
        if by.size > 450:
            idx = np.random.default_rng(0).choice(by.size, 450, replace=False)
            by, bx = by[idx], bx[idx]
        zt = Z[by, bx]
        inten = sc.s["timer"][0][by, bx] / fe.BURN_TIME
        ax.scatter(bx, by, zt + 0.6, c="orangered", s=22, depthshade=False)
        ax.scatter(bx, by, zt + 2.0 + 3 * inten, c="gold", s=9, depthshade=False)
        ax.scatter(bx, by, zt + 6.5, c="0.5", s=14, alpha=0.22, depthshade=False)
    ay, ax_ = int(sc.s["agent"][0, 0]), int(sc.s["agent"][0, 1])
    ax.scatter([ax_], [ay], [FLY_Z], marker="v", s=210, c="white",
               edgecolors="k", linewidths=1.5, depthshade=False)
    if sc.dropped:
        ax.plot([ax_, ax_], [ay, ay], [FLY_Z, Z[ay, ax_]], c="deepskyblue", lw=2, alpha=0.8)
    saved = fe.fuel_saved(sc.s)[0] * 100
    ax.set_title(f"Trained air-tanker over terrain  |  {saved:.0f}% fuel saved  "
                 f"|  water {int(sc.s['budget'][0])}",
                 color="seagreen", fontsize=12, weight="bold")


def main():
    sc = Scene(7)
    if SAVE:
        import imageio.v2 as imageio
        fig = plt.figure(figsize=(8, 6.6)); ax = fig.add_subplot(111, projection="3d")
        frames, azim = [], -60
        for _ in range(80):
            sc.step(); azim += 1.6; draw(ax, sc, azim); fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
            frames.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
        imageio.mimsave(os.path.join(ROOT, "assets", "scene3d.gif"), frames, fps=12, loop=0)
        print("saved assets/scene3d.gif")
        return

    fig = plt.figure(figsize=(9, 7.5)); ax = fig.add_subplot(111, projection="3d")
    st = {"azim": -60, "seed": 7}
    from matplotlib.animation import FuncAnimation

    def update(_):
        if sc.t >= fe.T:
            st["seed"] += 1; sc.reset(st["seed"])
        sc.step(); st["azim"] += 1.3; draw(ax, sc, st["azim"]); return []

    _ani = FuncAnimation(fig, update, interval=110, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
