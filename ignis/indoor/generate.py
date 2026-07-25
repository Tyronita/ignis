"""
Procedural indoor building generator → our VoxelScene (the RL-ready distribution).

A seeded guillotine partition splits the floor into rooms; every split inserts a
doorway (so the building is connected by construction), rooms are filled with
combustible furniture, and a random ignition point is chosen. Emits the existing
floorplan dict and reuses `importer.from_floorplan`, so there is a single
voxelization path shared with hand-authored scenes.

  python -m ignis.indoor.generate --seed 3 --rooms 4
"""

import argparse
import numpy as np
from . import importer, materials as M
from .schema import VoxelScene

DEFAULT_DIMS = (40, 28, 16)
DEFAULT_MIX = {"foam": 1.0, "fabric": 1.0, "wood": 2.0, "drywall": 0.6, "paper": 0.5}


def _split_largest(rects, rng, min_size):
    """Split the largest rectangle along its longer axis; return (wall, door)."""
    i = max(range(len(rects)), key=lambda k: (rects[k][2]-rects[k][0]) * (rects[k][3]-rects[k][1]))
    x0, y0, x1, y1 = rects[i]
    w, h = x1 - x0, y1 - y0
    vertical = w >= h if abs(w - h) > 2 else rng.random() < 0.5
    if (vertical and w < 2 * min_size) or (not vertical and h < 2 * min_size):
        return None
    if vertical:
        sx = rng.integers(x0 + min_size, x1 - min_size + 1)
        wall = {"a": [int(sx), int(y0)], "b": [int(sx), int(y1)]}
        door = {"at": [int(sx), int(rng.integers(y0 + 1, y1))]}
        rects[i] = (x0, y0, sx, y1); rects.append((sx, y0, x1, y1))
    else:
        sy = rng.integers(y0 + min_size, y1 - min_size + 1)
        wall = {"a": [int(x0), int(sy)], "b": [int(x1), int(sy)]}
        door = {"at": [int(rng.integers(x0 + 1, x1)), int(sy)]}
        rects[i] = (x0, y0, x1, sy); rects.append((x0, sy, x1, y1))
    return wall, door


def generate_floorplan(seed=0, dims=DEFAULT_DIMS, n_rooms=4,
                       furniture_density=0.5, material_mix=None, min_size=7):
    """Seeded parametric floorplan dict (schema-compatible)."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = dims
    mix = material_mix or DEFAULT_MIX
    names = list(mix); weights = np.array([mix[n] for n in names], float)
    weights /= weights.sum()

    interior = (1, 1, nx - 1, ny - 1)
    rects = [interior]
    walls = [
        {"a": [0, 0], "b": [nx-1, 0], "material": "concrete"},
        {"a": [nx-1, 0], "b": [nx-1, ny-1], "material": "concrete"},
        {"a": [nx-1, ny-1], "b": [0, ny-1], "material": "concrete"},
        {"a": [0, ny-1], "b": [0, 0], "material": "concrete"},
    ]
    doors = []
    tries = 0
    while len(rects) < n_rooms and tries < 4 * n_rooms:
        res = _split_largest(rects, rng, min_size)
        tries += 1
        if res is None:
            continue
        wall, door = res
        wall["material"] = "drywall"; wall["height"] = nz - 2
        walls.append(wall)
        door["width"] = 3; door["height"] = 5
        doors.append(door)

    rooms, furniture = [], []
    for i, (x0, y0, x1, y1) in enumerate(rects):
        rooms.append({"name": f"room{i}", "bbox": [int(x0), int(y0), int(x1), int(y1)],
                      "height": nz - 2})
        # furniture boxes
        n_items = 1 + int(furniture_density * rng.integers(1, 4))
        for _ in range(n_items):
            fw = rng.integers(3, max(4, (x1 - x0) // 2))
            fh = rng.integers(3, max(4, (y1 - y0) // 2))
            fx = rng.integers(x0 + 1, max(x0 + 2, x1 - fw))
            fy = rng.integers(y0 + 1, max(y0 + 2, y1 - fh))
            mat = names[rng.choice(len(names), p=weights)]
            top = int(rng.integers(2, 5))
            furniture.append({"bbox": [int(fx), int(fy), int(fx + fw), int(fy + fh)],
                              "z": [1, top], "material": mat})

    # ignite a random furniture box near its base
    f = furniture[rng.integers(len(furniture))]
    bx = (f["bbox"][0] + f["bbox"][2]) // 2
    by = (f["bbox"][1] + f["bbox"][3]) // 2
    ignition = [[int(bx), int(by), 1], [int(bx)+1, int(by), 1], [int(bx), int(by)+1, 1]]

    return {"dims": [nx, ny, nz], "cell_size_m": 0.25, "rooms": rooms, "walls": walls,
            "doors": doors, "furniture": furniture, "ignition": ignition,
            "budgets": {"water": 60, "time": 90}}


def is_connected(scene: VoxelScene):
    """BFS over walkable (non-solid) floor voxels — are all rooms reachable?"""
    solid = scene.solid
    walk = (~solid[:, :, 1])  # floor level slice (x,y)
    rid = scene.room_id[:, :, 1]
    nx, ny = walk.shape
    # seed from a walkable cell INSIDE a room (not the outer air-moat around the walls)
    start = np.argwhere((rid >= 0) & walk)
    if len(start) == 0:
        return False
    seen = np.zeros_like(walk)
    stack = [tuple(start[0])]
    seen[tuple(start[0])] = True
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = x + dx, y + dy
            if 0 <= a < nx and 0 <= b < ny and walk[a, b] and not seen[a, b]:
                seen[a, b] = True; stack.append((a, b))
    # each room is reachable if ANY of its walkable floor cells was visited
    rid = scene.room_id[:, :, 1]
    for r in range(rid.max() + 1):
        room_cells = (rid == r) & walk
        if room_cells.any() and not (seen & room_cells).any():
            return False
    return True


def generate_scene(seed=0, **kw):
    scene = importer.from_floorplan(generate_floorplan(seed=seed, **kw))
    return scene


def sample_pool(k, seed0=0, **kw):
    """A list of k distinct generated scenes for RL / evals."""
    return [generate_scene(seed=seed0 + i, **kw) for i in range(k)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rooms", type=int, default=4)
    ap.add_argument("--dims", type=int, nargs=3, default=list(DEFAULT_DIMS))
    args = ap.parse_args()
    sc = generate_scene(seed=args.seed, n_rooms=args.rooms, dims=tuple(args.dims))
    print(importer._summary(sc))
    print("connected:", is_connected(sc))
