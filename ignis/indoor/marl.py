"""
Two-class multi-agent simulation: fire-engine RESPONDERS vs CIVILIANS.

- Class 1 · Responders (engine + operators): the engine parks at the entrance; after a
  response delay, operators advance toward the fire and hit it with a hose (extinguish +
  wet a radius), under a water budget. Reward: fire suppressed + civilians protected.
- Class 2 · Civilians: evacuate via the fire-aware shortest path (see evacuate.py), with a
  pre-movement delay so the fire genuinely threatens them.

The headline result is the coupling: responders contain the fire → civilians gain escape
time → fewer casualties + less damage ("lives saved"). Responder policy here is a
transparent heuristic (advance-and-attack); cooperative CEM/PPO training is the next step.

  python -m ignis.indoor.marl        # compare no-responders vs responders on the house + gif
"""

import os
import numpy as np
from . import importer, suppress, evacuate, indoor_env as ie, render3d as r3
from .playthrough import plan_rgb

HOUSE = os.path.join(os.path.dirname(__file__), "examples", "house.json")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")

RESP_N = 2
RESP_DELAY_S = 25.0        # fire-engine response/arrival time
RESP_SPEED = 4             # operator cells/step
HOSE_RADIUS = 3            # hose reach (cells)
RESP_WATER = 120           # total hose steps


class ResponderTeam:
    """Fire-engine operators: arrive, advance to the fire, hose it down."""
    def __init__(self, scene, n=RESP_N, delay_s=RESP_DELAY_S, seed=0):
        self.scene = scene
        self.delay = int(delay_s / evacuate.DT_S)
        ex = (scene.exits or [[scene.nx // 2, 0]])[0]
        self.pos = np.array([[int(ex[0]), min(scene.ny - 2, int(ex[1]) + 2)]] * n, np.int32)
        self.water = np.full(n, RESP_WATER)
        self.block = scene.solid[:, :, 1]
        self.trajectories = [[] for _ in range(n)]
        self.t = 0

    def _advance(self, x, y, tx, ty):
        for _ in range(RESP_SPEED):
            step = None
            for dx, dy in sorted(((1, 0), (-1, 0), (0, 1), (0, -1)),
                                 key=lambda d: abs(x + d[0] - tx) + abs(y + d[1] - ty)):
                a, b = x + dx, y + dy
                if 0 <= a < self.scene.nx and 0 <= b < self.scene.ny and not self.block[a, b]:
                    step = (a, b); break
            if step is None:
                break
            x, y = step
        return x, y

    def step(self, st):
        self.t += 1
        burning = (st["burning"] > 0).any(axis=2)
        fires = np.argwhere(burning)
        for i in range(len(self.pos)):
            x, y = self.pos[i]
            if self.t >= self.delay and len(fires) and self.water[i] > 0:
                d = np.abs(fires[:, 0] - x) + np.abs(fires[:, 1] - y)
                tx, ty = fires[d.argmin()]
                x, y = self._advance(x, y, tx, ty)
                # hose: extinguish + wet a radius (all z)
                x0, x1 = max(0, x - HOSE_RADIUS), x + HOSE_RADIUS + 1
                y0, y1 = max(0, y - HOSE_RADIUS), y + HOSE_RADIUS + 1
                st["wet"][x0:x1, y0:y1, :] = 1.0
                sub = st["burning"][x0:x1, y0:y1, :] > 0
                st["burning"][x0:x1, y0:y1, :][sub] = 0.0
                st["timer"][x0:x1, y0:y1, :][sub] = 0
                self.water[i] -= 1
                self.pos[i] = (x, y)
            self.trajectories[i].append((int(x), int(y)))


def run(scene, responders=True, n_civ=12, civ_delay_s=18, steps=140, seed=0, record=False):
    st = suppress.init(scene)
    ev = evacuate.Evacuation(scene, n_agents=n_civ, seed=seed,
                             start_delay=int(civ_delay_s / evacuate.DT_S))
    team = ResponderTeam(scene, seed=seed) if responders else None
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(steps):
        if team:
            team.step(st)
        suppress.sim_step(st, rng)
        em = ev.step(st)
        if record:
            frames.append((k, ie.metrics(st), em, ev, team))
        if em["still_inside"] == 0 and (st["burning"] > 0).sum() == 0 and k > 8:
            break
    m = ie.metrics(st)
    return dict(fuel_saved=100 * (1 - m["burned_pct"] / 100), burned_pct=m["burned_pct"],
                damage_usd=m["damage_usd"], rooms_lost=m["rooms_lost"],
                escaped=em["escaped"], casualties=em["casualties"],
                occupants=em["occupants"]), (st, ev, team, frames)


def compare(scene, seed=0):
    no, _ = run(scene, responders=False, seed=seed)
    yes, _ = run(scene, responders=True, seed=seed)
    print("=== Two-class MARL — civilians only vs civilians + fire-engine responders ===")
    print(f"{'metric':16s}{'no responders':>16s}{'+ responders':>16s}")
    for k, lab in [("burned_pct", "burned %"), ("damage_usd", "$ damage"),
                   ("rooms_lost", "rooms lost"), ("casualties", "casualties"),
                   ("escaped", "escaped")]:
        print(f"{lab:16s}{no[k]:>16.1f}{yes[k]:>16.1f}")
    print(f"\nresponders -> damage avoided ${no['damage_usd'] - yes['damage_usd']:.0f}, "
          f"burned {no['burned_pct'] - yes['burned_pct']:.0f} pts less, "
          f"{no['rooms_lost'] - yes['rooms_lost']:.0f} fewer rooms lost; "
          f"lives saved {yes['escaped'] - no['escaped']:+d} (egress-limited in this scene).")
    return no, yes


def render_gif(scene, out=None, steps=140, frames=20, down=3, seed=0, fps=10):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    st = suppress.init(scene)
    ev = evacuate.Evacuation(scene, n_agents=12, seed=seed, start_delay=int(18 / evacuate.DT_S))
    team = ResponderTeam(scene, seed=seed)
    rng = np.random.default_rng(seed)
    imgs, azim = [], -60
    civ_col = {0: "#1f6feb", 1: "#2ecc71", 2: "#e74c3c"}
    grab = set(np.linspace(0, steps - 1, frames).astype(int))
    for k in range(steps):
        team.step(st); suppress.sim_step(st, rng); em = ev.step(st)
        if k in grab:
            fig = plt.figure(figsize=(14, 6.6), dpi=100)
            axp = fig.add_subplot(1, 2, 1)
            axp.imshow(plan_rgb(st, scene), origin="lower", interpolation="nearest")
            for i in range(len(ev.pos)):                       # civilians
                axp.scatter(ev.pos[i, 0], ev.pos[i, 1], c=civ_col[int(ev.state[i])],
                            s=34, edgecolors="k", linewidths=0.5, zorder=5)
            for i in range(len(team.pos)):                     # responders (engine operators)
                tr = np.array(team.trajectories[i])
                if len(tr) > 1:
                    axp.plot(tr[:, 0], tr[:, 1], c="#8e44ad", lw=1.4, alpha=0.8)
                axp.scatter(team.pos[i, 0], team.pos[i, 1], c="#f1c40f", marker="s",
                            s=70, edgecolors="#8e44ad", linewidths=1.5, zorder=6)
            axp.set_xticks([]); axp.set_yticks([])
            axp.set_title(f"PLAN — escaped {em['escaped']}/{em['occupants']}  "
                          f"casualties {em['casualties']}   (yellow□=responders, blue=civilians)",
                          fontsize=10, color="#8e44ad", weight="bold")
            ax3 = fig.add_subplot(1, 2, 2, projection="3d")
            c = r3._downsample(r3._category(st), down); filled = c > 0
            fc = np.zeros(c.shape + (4,), np.float32)
            for code, col in r3._COLORS.items():
                fc[c == code] = col
            ax3.voxels(filled, facecolors=fc, edgecolor=(0, 0, 0, 0.05))
            ax3.set_axis_off(); ax3.view_init(elev=26, azim=azim)
            ax3.set_box_aspect((c.shape[0], c.shape[1], c.shape[2]))
            mm = ie.metrics(st)
            ax3.set_title(f"3D — step {mm['step']}  burned {mm['burned_pct']:.0f}%",
                          fontsize=10, color="#c0392b", weight="bold")
            fig.suptitle("Two-class MARL — fire-engine responders + civilians",
                         fontsize=12, weight="bold")
            fig.tight_layout(); fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
            imgs.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
            plt.close(fig); azim += 120 / frames
        if em["still_inside"] == 0 and (st["burning"] > 0).sum() == 0 and k > 8:
            break
    out = out or os.path.join(ASSETS, "marl.gif")
    imageio.mimsave(out, imgs, fps=fps, loop=0)
    print(f"saved {out}  ({len(imgs)} frames)")


def main():
    scene = importer.load(HOUSE)
    compare(scene)
    render_gif(scene)


if __name__ == "__main__":
    main()
