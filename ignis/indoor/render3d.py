"""
3D voxel renderer for indoor fire — matplotlib `ax.voxels`, orbiting camera → GIF.

Walls are drawn translucent so the fire inside is visible; furniture is green, active
flame orange, burned dark, wet blue. The volume is downsampled for speed. This is the
"see the 3D indoor sim" deliverable (assets/indoor3d.gif).

  python -m ignis.indoor.render3d            # render a generated building
  python -m ignis.indoor.render3d <scene.json>
"""

import os
import sys
import numpy as np
from . import indoor_env as ie

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")

# category codes ordered by saliency (higher wins a downsample block)
AIR, WALL, FURN, WET, BURNED, FIRE = 0, 1, 2, 3, 4, 5
_COLORS = {
    WALL:   (0.55, 0.55, 0.60, 0.09),
    FURN:   (0.20, 0.62, 0.24, 0.95),
    WET:    (0.15, 0.45, 0.85, 0.85),
    BURNED: (0.10, 0.09, 0.09, 0.95),
    FIRE:   (1.00, 0.45, 0.05, 0.98),
}


def _category(st):
    """Per-voxel category code (nx,ny,nz)."""
    c = np.zeros(st["burning"].shape, np.int8)
    c[st["solid"]] = WALL
    fuelv = st["solid_fuel"] & (st["burning"] == 0) & ~st["burned"]
    c[fuelv] = FURN
    if "wet" in st:
        c[st["wet"] & (st["burning"] == 0)] = WET
    c[st["burned"]] = BURNED
    c[st["burning"] > 0] = FIRE
    return c


def _downsample(c, d):
    """Block-max downsample (keeps the most salient voxel per block)."""
    if d <= 1:
        return c
    nx, ny, nz = c.shape
    px, py, pz = (-nx) % d, (-ny) % d, (-nz) % d
    c = np.pad(c, ((0, px), (0, py), (0, pz)))
    nx, ny, nz = c.shape
    return c.reshape(nx//d, d, ny//d, d, nz//d, d).max(axis=(1, 3, 5))


def render_gif(scene, out=None, steps=None, frames=20, down=2, seed=0, fps=11,
               figsize=(9.2, 8.2), dpi=108, total_deg=135, elev=24):
    """Higher-resolution, slower-orbit voxel render. `down` sets voxel-drawing detail
    (2 = more detail), figsize/dpi set output pixels, total_deg spans the whole clip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    st = ie.init_from_scene(scene)
    rng = np.random.default_rng(seed)
    steps = steps or 100
    grab = set(np.linspace(0, steps - 1, frames).astype(int))
    imgs, azim = [], -60
    for k in range(steps):
        if k in grab:
            c = _downsample(_category(st), down)
            filled = c > 0
            colors = np.zeros(c.shape + (4,), np.float32)
            for code, col in _COLORS.items():
                colors[c == code] = col
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
            ax.voxels(filled, facecolors=colors, edgecolor=(0, 0, 0, 0.06))
            ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((c.shape[0], c.shape[1], c.shape[2]))   # true proportions
            m = ie.metrics(st)
            ax.set_title(f"3D house fire — step {m['step']:2d}    burned {m['burned_pct']:.0f}%    "
                         f"rooms {m['rooms_lost']}/{m['n_rooms']}",
                         color="#c0392b", fontsize=14, weight="bold")
            fig.tight_layout(); fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
            imgs.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
            plt.close(fig); azim += total_deg / frames
        ie.step(st, rng)

    out = out or os.path.join(ASSETS, "indoor3d.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, imgs, fps=fps, loop=0)
    print(f"saved {out}  ({len(imgs)} frames)")
    return out


if __name__ == "__main__":
    from . import importer
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        scene = importer.load(sys.argv[1])
    else:  # default to the realistic furnished house
        scene = importer.load(os.path.join(os.path.dirname(__file__), "examples", "house.json"))
    render_gif(scene)
