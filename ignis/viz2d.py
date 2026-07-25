"""
2D side-by-side: the SAME fire with no tanker (left) vs the trained tanker (right).

  python -m ignis.viz2d          live TkAgg window
  python -m ignis.viz2d save      write assets/compare2d.gif

Both panels are driven by identically-seeded fires, so the only difference is the
agent's water — an honest counterfactual.
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
W = np.load(os.path.join(ROOT, "policy.npy"))


class Pair:
    def __init__(self, seed):
        self.reset(seed)

    def reset(self, seed):
        self.rngA = np.random.default_rng(seed)
        self.rngB = np.random.default_rng(seed)
        self.sA = fe.init_state(1, self.rngA)
        self.sB = fe.init_state(1, self.rngB)
        self.t = 0

    def step(self):
        fe.step(self.sA, np.zeros(1, np.int32), self.rngA)
        fe.step(self.sB, fe.policy_action(W[None], fe.features(self.sB)), self.rngB)
        self.t += 1


def _saved(s):
    return float(fe.fuel_saved(s)[0]) * 100


def _titles(axL, axR, pair):
    axL.set_title(f"No tanker — {_saved(pair.sA):.0f}% fuel saved", color="crimson")
    axR.set_title(f"Trained tanker — {_saved(pair.sB):.0f}% fuel saved  "
                  f"(water {int(pair.sB['budget'][0])})", color="seagreen")


def main():
    pair = Pair(7)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.6))
    fig.suptitle("Wildfire suppression — trained air-tanker (RL)", fontsize=14, weight="bold")
    imL = axL.imshow(fe.render(pair.sA)); axL.axis("off")
    imR = axR.imshow(fe.render(pair.sB)); axR.axis("off")

    if SAVE:
        import imageio.v2 as imageio
        frames = []
        for _ in range(fe.T):
            imL.set_data(fe.render(pair.sA)); imR.set_data(fe.render(pair.sB))
            _titles(axL, axR, pair); fig.tight_layout(); fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
            frames.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
            pair.step()
        out = os.path.join(ROOT, "assets", "compare2d.gif")
        imageio.mimsave(out, frames, fps=12, loop=0)
        print("saved assets/compare2d.gif")
        return

    from matplotlib.animation import FuncAnimation
    seed = [7]

    def update(_):
        if pair.t >= fe.T:
            seed[0] += 1; pair.reset(seed[0])
        pair.step()
        imL.set_data(fe.render(pair.sA)); imR.set_data(fe.render(pair.sB))
        _titles(axL, axR, pair)
        return imL, imR

    _ani = FuncAnimation(fig, update, interval=90, blit=False, cache_frame_data=False)
    fig.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
