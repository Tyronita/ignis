"""
Indoor eval harness: importer -> sim -> metrics -> gif. Proves the pipeline end-to-end.

  python -m ignis.indoor.evals ignis/indoor/examples/apartment.json

Metrics (spread + materials): % fuel saved / burned, rooms lost, time-to-flashover,
$ material damage, smoke-filled %, containment. Trained suppression is a later step;
this run is the no-suppression baseline the future policy will be scored against.
"""

import os
import sys
import numpy as np
from . import importer, indoor_env as ie

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")


_LEGEND = [
    ("air",           (0.93, 0.94, 0.97)),
    ("smoke",         (0.62, 0.58, 0.66)),
    ("wall",          (0.55, 0.55, 0.58)),
    ("unburned fuel", (0.20, 0.62, 0.24)),
    ("burned out",    (0.10, 0.09, 0.09)),
    ("burning",       (1.00, 0.55, 0.00)),
]


def _volume_rgb(st):
    """Per-voxel RGB + a saliency priority for max-projection rendering."""
    nx, ny, nz = st["burning"].shape
    rgb = np.zeros((nx, ny, nz, 3), np.float32)
    pri = np.zeros((nx, ny, nz), np.float32)
    # air = faint
    rgb[st["is_air"]] = np.array([0.93, 0.94, 0.97]); pri[st["is_air"]] = 0.1
    # smoke-logged air (flamed_air proxy) = tinted, so the smoke_pct metric is visible
    smoke = st["is_air"] & st["flamed_air"]
    rgb[smoke] = np.array([0.62, 0.58, 0.66]); pri[smoke] = 0.5
    # walls = gray
    wall = st["solid"]; rgb[wall] = np.array([0.55, 0.55, 0.58]); pri[wall] = 1.0
    # unburned fuel = green
    fuelv = (st["fuel"] > 0) & st["burnable"] & (st["burning"] == 0) & ~st["burned"]
    rgb[fuelv] = np.array([0.20, 0.62, 0.24]); pri[fuelv] = 2.0
    # burned = dark
    rgb[st["burned"]] = np.array([0.10, 0.09, 0.09]); pri[st["burned"]] = 3.0
    # burning = hot
    fire = st["burning"] > 0
    hot = np.clip(st["timer"] / np.maximum(st["burn_time"], 1), 0, 1)
    fr = np.stack([np.ones_like(hot), 0.35 + 0.55 * hot, np.zeros_like(hot)], -1)
    rgb[fire] = fr[fire]; pri[fire] = 4.0
    return rgb, pri


def _project(rgb, pri, axis):
    idx = pri.argmax(axis=axis)
    return np.take_along_axis(rgb, np.expand_dims(idx, (axis, -1)), axis=axis).squeeze(axis)


def evaluate(path, seed=0, steps=90, record=True):
    scene = importer.load(path)
    st = ie.init_from_scene(scene)
    rng = np.random.default_rng(seed)
    frames, timeline = [], []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    legend_handles = [Patch(facecolor=c, edgecolor="0.4", label=name) for name, c in _LEGEND]

    for k in range(steps):
        if record and k % 1 == 0:
            rgb, pri = _volume_rgb(st)
            # project interior only (skip floor/ceiling and front/back walls) so the
            # room layout and the vertical flame plume are actually visible
            top = _project(rgb[:, :, 1:-1], pri[:, :, 1:-1], 2).transpose(1, 0, 2)   # (y,x,3)
            side = _project(rgb[:, 1:-1, :], pri[:, 1:-1, :], 1).transpose(1, 0, 2)  # (z,x,3)
            fig, (a0, a1) = plt.subplots(1, 2, figsize=(9, 4.2),
                                         gridspec_kw={"width_ratios": [1.4, 1]})
            a0.imshow(top, origin="lower"); a0.set_title("top view (rooms)", fontsize=10)
            a1.imshow(side, origin="lower", aspect="auto")
            a1.set_title("side view (fire climbs → ceiling)", fontsize=10)
            m = ie.metrics(st)
            fig.suptitle(f"step {m['step']:2d}  |  burned {m['burned_pct']:.0f}%  |  "
                         f"rooms lost {m['rooms_lost']}/{m['n_rooms']}  |  "
                         f"smoke {m['smoke_pct']:.0f}%  |  ${m['damage_usd']:.0f}",
                         fontsize=11, weight="bold")
            for a in (a0, a1):
                a.set_xticks([]); a.set_yticks([])
            fig.legend(handles=legend_handles, loc="lower center", ncol=len(_LEGEND),
                       frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
            frames.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
            plt.close(fig)
        ie.step(st, rng)
        timeline.append(ie.metrics(st))
        if timeline[-1]["active_flame"] == 0 and k > 6:
            break

    final = ie.metrics(st)
    if record:
        import imageio.v2 as imageio
        os.makedirs(ASSETS, exist_ok=True)
        imageio.mimsave(os.path.join(ASSETS, "indoor.gif"), frames, fps=10, loop=0)
        print("saved assets/indoor.gif")
    return final, timeline, scene


def _print_table(final, scene):
    print("\n=== Indoor fire eval (no suppression baseline) ===")
    print(f"scene           : {scene.room_names}  dims={scene.dims}")
    print(f"final step      : {final['step']}")
    print(f"fuel saved      : {final['fuel_saved_pct']:.1f}%   (burned {final['burned_pct']:.1f}%)")
    print(f"rooms lost      : {final['rooms_lost']}/{final['n_rooms']}")
    print(f"flashover step  : {final['flashover_step']}")
    print(f"material damage : ${final['damage_usd']:.0f}")
    print(f"smoke-filled    : {final['smoke_pct']:.1f}% of air volume")
    print(f"contained       : {final['contained']}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), "examples", "apartment.json")
    final, timeline, scene = evaluate(path)
    _print_table(final, scene)
