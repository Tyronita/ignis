"""
Importers: turn a scene description into the voxel "our format" (schema.VoxelScene).

Default front-end: a JSON floorplan (rooms / walls / doors / furniture).
Experimental presets (NOT default): USD stage or a 3D mesh (glTF/OBJ) voxelized to
the same VoxelScene — gated behind optional deps so the core stays dependency-light.

  python -m ignis.indoor.importer ignis/indoor/examples/apartment.json
"""

import json
import sys
import numpy as np
from . import materials as M
from .schema import VoxelScene, validate_floorplan


def _rasterize_wall(mat, solid, a, b, height, mid):
    """Stamp a 1-voxel-thick vertical wall between a and b for z in [0,height)."""
    (x0, y0), (x1, y1) = a, b
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.round(np.linspace(x0, x1, steps)).astype(int)
    ys = np.round(np.linspace(y0, y1, steps)).astype(int)
    nx, ny, nz = mat.shape
    h = min(height, nz)
    for x, y in zip(xs, ys):
        if 0 <= x < nx and 0 <= y < ny:
            mat[x, y, :h] = mid
            solid[x, y, :h] = True


def from_floorplan(d):
    """floorplan dict -> VoxelScene."""
    validate_floorplan(d)
    nx, ny, nz = (int(v) for v in d["dims"])
    cell = float(d.get("cell_size_m", 0.25))
    mat = np.zeros((nx, ny, nz), np.int32)          # 0 = air
    solid = np.zeros((nx, ny, nz), bool)
    room_id = np.full((nx, ny, nz), -1, np.int32)
    room_names = []

    # structural floor + ceiling (concrete) so fire is contained and climbs to a ceiling
    cid = M.id_of("concrete")
    mat[:, :, 0] = cid;      solid[:, :, 0] = True
    mat[:, :, nz - 1] = cid; solid[:, :, nz - 1] = True

    # rooms = interior air volumes + room tagging
    for i, r in enumerate(d.get("rooms", [])):
        x0, y0, x1, y1 = r["bbox"]
        h = min(int(r.get("height", nz)), nz - 1)
        room_id[x0:x1, y0:y1, 1:h] = i
        room_names.append(r.get("name", f"room{i}"))

    # walls block spread
    for w in d.get("walls", []):
        _rasterize_wall(mat, solid, w["a"], w["b"],
                        int(w.get("height", nz - 1)), M.id_of(w.get("material", "concrete")))

    # doors carve gaps back into the walls (air, non-solid)
    for door in d.get("doors", []):
        x, y = door["at"]; wdt = int(door.get("width", 2)); dh = int(door.get("height", 3))
        x0, x1 = max(0, x - wdt // 2), min(nx, x + wdt // 2 + 1)
        y0, y1 = max(0, y - wdt // 2), min(ny, y + wdt // 2 + 1)
        mat[x0:x1, y0:y1, 1:1 + dh] = 0
        solid[x0:x1, y0:y1, 1:1 + dh] = False

    # furniture = combustible fill
    for f in d.get("furniture", []):
        x0, y0, x1, y1 = f["bbox"]; z0, z1 = f.get("z", [1, 3])
        mid = M.id_of(f.get("material", "wood"))
        mat[x0:x1, y0:y1, z0:z1] = mid
        solid[x0:x1, y0:y1, z0:z1] = False

    fuel = M._column("fuel_load", np.float32)[mat]
    fuel[solid] = 0.0
    ignition = [tuple(p) for p in d.get("ignition", [])]
    budgets = d.get("budgets", {})
    return VoxelScene((nx, ny, nz), cell, mat, solid, fuel, ignition,
                      budgets, room_id, room_names)


def load(path):
    with open(path) as fh:
        return from_floorplan(json.load(fh))


def voxelize_mesh(path, dims, cell_size_m=0.25):
    """EXPERIMENTAL preset (not default): voxelize a USD/glTF/OBJ scene.

    Requires `trimesh` (glTF/OBJ) or `pxr` (USD). Emits the same VoxelScene so the
    sim is agnostic to the front-end. Left as a stub for the next build.
    """
    raise NotImplementedError(
        "mesh/USD voxelization is an experimental preset — install trimesh (glTF/OBJ) "
        "or pxr/OpenUSD (USD), then map mesh submeshes to materials into a VoxelScene. "
        "Default path is the JSON floorplan importer (from_floorplan/load).")


def _summary(sc):
    solid_frac = sc.solid.mean() * 100
    fuel_vox = int((sc.fuel > 0).sum())
    return (f"VoxelScene dims={sc.dims} cell={sc.cell_size_m}m  "
            f"solid={solid_frac:.1f}%  fuel voxels={fuel_vox}  "
            f"rooms={sc.room_names}  ignition={sc.ignition}  budgets={sc.budgets}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "ignis/indoor/examples/apartment.json"
    print(_summary(load(path)))
