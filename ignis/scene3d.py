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
DROP_FADE = 14   # frames a drop trajectory stays visible before fully fading
PATH_LEN = 40    # how many past flight positions to trail behind the tanker

TERR = fe.make_terrain(3)
fe.set_terrain(TERR)
Z = TERR * ZSCALE
X, Yg = np.meshgrid(np.arange(fe.N), np.arange(fe.N))
FLY_Z = Z.max() + 7.0
W = np.load(os.path.join(ROOT, "policy3d.npy"))


class Scene:
    def __init__(self, seed):
        self.reset(seed)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.s = fe.init_state(1, self.rng)
        self.t = 0
        self.prev_budget = int(self.s["budget"][0])
        self.dropped = False
        self.drops = []   # fading trail of past water-drop trajectories
        self.path = []    # breadcrumb trail of the tanker's flight path

    def step(self):
        fe.step(self.s, fe.policy_action(W[None], fe.features(self.s)), self.rng)
        self.dropped = int(self.s["budget"][0]) < self.prev_budget
        self.prev_budget = int(self.s["budget"][0])
        self.t += 1

        ay, ax_ = int(self.s["agent"][0, 0]), int(self.s["agent"][0, 1])
        self.path.append((ay, ax_))
        if len(self.path) > PATH_LEN:
            self.path.pop(0)

        for d in self.drops:
            d["age"] += 1
        self.drops = [d for d in self.drops if d["age"] < DROP_FADE]
        if self.dropped:
            self.drops.append({"ay": ay, "ax": ax_, "age": 0})


def draw(ax, sc, azim):
    ax.cla(); ax.set_axis_off(); ax.view_init(elev=42, azim=azim)
    ax.set_zlim(0, FLY_Z + 6)
    rgb = fe.cell_rgb(sc.s)
    rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1), np.float32)], -1)
    ax.plot_surface(X, Yg, Z, facecolors=rgba, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
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

    # flight path: where the tanker has flown, so its route reads as a trajectory
    if len(sc.path) > 1:
        py = [p[0] for p in sc.path]; px = [p[1] for p in sc.path]
        ax.plot(px, py, [FLY_Z] * len(sc.path), c="0.55", lw=1.3, ls="--", alpha=0.6)

    # water drops: fading trail so each drop stays visible for DROP_FADE frames
    # instead of flashing for a single frame
    for d in sc.drops:
        alpha = max(0.12, 1 - d["age"] / DROP_FADE)
        dy, dx, dz = d["ay"], d["ax"], Z[d["ay"], d["ax"]]
        ax.plot([dx, dx], [dy, dy], [FLY_Z, dz], c="deepskyblue", lw=2.5, alpha=alpha)
        ax.scatter([dx], [dy], [dz + 0.3], c="deepskyblue", s=70 * alpha,
                   alpha=alpha, depthshade=False)

    ax.scatter([ax_], [ay], [FLY_Z], marker="v", s=210, c="white",
               edgecolors="k", linewidths=1.5, depthshade=False)
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
