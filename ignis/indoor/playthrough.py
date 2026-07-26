"""
Synchronised playthrough — one run, two views (plan + 3D), both time-locked.

Runs the fire, the RL suppression policy (optional), and the evacuation together, and
renders a two-panel GIF: LEFT = architectural plan view (furniture, fire, smoke, exits,
occupant trajectories), RIGHT = 3D voxel orbit. Every run shows both views in sync.

  python -m ignis.indoor.playthrough                       # house, suppression + evacuation
  python -m ignis.indoor.playthrough --no-suppress
"""

import argparse
import os
import numpy as np
from . import importer, suppress, evacuate, render3d as r3, indoor_env as ie

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
HOUSE = os.path.join(os.path.dirname(__file__), "examples", "house.json")

# plan-view palette
_MATCOL = {"wood": (0.71, 0.51, 0.25), "foam": (0.29, 0.64, 0.87),
           "fabric": (0.29, 0.64, 0.87), "steel": (0.55, 0.55, 0.58),
           "glass": (0.6, 0.85, 0.9), "drywall": (0.8, 0.78, 0.75),
           "concrete": (0.5, 0.5, 0.52), "paper": (0.9, 0.9, 0.8)}


def plan_rgb(st, scene):
    nx, ny, nz = st["burning"].shape
    img = np.full((nx, ny, 3), 0.96, np.float32)                 # floor / air
    walls = st["solid"][:, :, 1:nz - 1].any(axis=2)
    img[walls] = (0.32, 0.32, 0.35)
    from . import materials as M
    matf = scene.material[:, :, 1]
    furn = (~scene.solid[:, :, 1]) & (matf != 0)
    for name, col in _MATCOL.items():
        img[furn & (matf == M.id_of(name))] = col
    img[st["burned"].any(axis=2)] = (0.12, 0.10, 0.10)
    wet = st.get("wet")
    if wet is not None:
        img[(wet > 0.5).any(axis=2) & ~st["burning"].any(axis=2)] = (0.55, 0.78, 0.95)
    img[st["burning"].any(axis=2)] = (1.0, 0.4, 0.05)
    for ex, ey in (scene.exits or []):
        img[max(0, ex-1):ex+2, max(0, ey-1):ey+2] = (0.15, 0.8, 0.3)
    return np.transpose(img, (1, 0, 2))                          # (y,x,3) for imshow


def run(scene, policy=None, steps=120, n_agents=8, seed=0, record_traj=True):
    st = suppress.init(scene)
    ev = evacuate.Evacuation(scene, n_agents=n_agents, seed=seed)
    rng = np.random.default_rng(seed)
    W = None if policy is None else np.load(policy)
    frames_state = []
    for k in range(steps):
        a = suppress.N_ACT - 1 if W is None else suppress.policy_action(W, suppress.observe(st))
        suppress.apply_action(st, a)
        suppress.sim_step(st, rng)
        em = ev.step(st)
        frames_state.append((k, ie.metrics(st), em,
                             ev.pos.copy(), ev.state.copy(),
                             [list(t) for t in ev.trajectories] if record_traj else None))
        if em["still_inside"] == 0 and (st["burning"] > 0).sum() == 0 and k > 5:
            break
    return st, ev, frames_state


def render_gif(scene, policy=None, out=None, steps=120, n_agents=8, frames=20,
               down=3, fps=10, figsize=(14, 6.6), dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    st = suppress.init(scene)
    ev = evacuate.Evacuation(scene, n_agents=n_agents, seed=0)
    rng = np.random.default_rng(0)
    W = None if policy is None else np.load(policy)
    grab = set(np.linspace(0, steps - 1, frames).astype(int))
    imgs, azim = [], -60
    for k in range(steps):
        a = suppress.N_ACT - 1 if W is None else suppress.policy_action(W, suppress.observe(st))
        suppress.apply_action(st, a); suppress.sim_step(st, rng); em = ev.step(st)
        if k in grab:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            # ---- plan view ----
            axp = fig.add_subplot(1, 2, 1)
            axp.imshow(plan_rgb(st, scene), origin="lower", interpolation="nearest")
            colors = {0: "#1f6feb", 1: "#2ecc71", 2: "#e74c3c"}
            for i in range(len(ev.pos)):
                tr = np.array(ev.trajectories[i])
                if len(tr) > 1:
                    axp.plot(tr[:, 0], tr[:, 1], color=colors[int(ev.state[i])], lw=1.2, alpha=0.7)
                axp.scatter(ev.pos[i, 0], ev.pos[i, 1], c=colors[int(ev.state[i])],
                            s=42, edgecolors="k", linewidths=0.6, zorder=5)
            axp.set_xticks([]); axp.set_yticks([])
            axp.set_title(f"PLAN — escaped {em['escaped']}/{em['occupants']}  "
                          f"casualties {em['casualties']}  inside {em['still_inside']}",
                          fontsize=11, color="#1f6feb", weight="bold")
            # ---- 3D view ----
            ax3 = fig.add_subplot(1, 2, 2, projection="3d")
            c = r3._downsample(r3._category(st), down); filled = c > 0
            fc = np.zeros(c.shape + (4,), np.float32)
            for code, col in r3._COLORS.items():
                fc[c == code] = col
            ax3.voxels(filled, facecolors=fc, edgecolor=(0, 0, 0, 0.05))
            ax3.set_axis_off(); ax3.view_init(elev=26, azim=azim)
            ax3.set_box_aspect((c.shape[0], c.shape[1], c.shape[2]))
            m = ie.metrics(st)
            ax3.set_title(f"3D — step {m['step']}  burned {m['burned_pct']:.0f}%  "
                          f"rooms {m['rooms_lost']}/{m['n_rooms']}",
                          fontsize=11, color="#c0392b", weight="bold")
            fig.suptitle("Ignis playthrough — synchronised plan + 3D  (blue=escaping, green=out, red=casualty)",
                         fontsize=12, weight="bold")
            fig.tight_layout(); fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
            imgs.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
            plt.close(fig); azim += 120 / frames
        if em["still_inside"] == 0 and (st["burning"] > 0).sum() == 0 and k > 5:
            break
    out = out or os.path.join(ASSETS, "playthrough.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, imgs, fps=fps, loop=0)
    print(f"saved {out}  ({len(imgs)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-suppress", action="store_true")
    ap.add_argument("--scene", default=HOUSE)
    args = ap.parse_args()
    scene = importer.load(args.scene)
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    pol = None if args.no_suppress else os.path.join(root, "policy_indoor.npy")
    if pol and not os.path.exists(pol):
        pol = None
    render_gif(scene, policy=pol)
