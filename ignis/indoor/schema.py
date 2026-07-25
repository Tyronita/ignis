"""
"Our format": the voxel scene that every front-end (JSON floorplan, USD, mesh
voxelizer) converts to. The indoor sim consumes ONLY this — importers are swappable.

VoxelScene
  dims        (nx, ny, nz)         grid resolution (configurable, not hardcoded)
  cell_size_m float                edge length of a voxel, metres
  material    int   (nx,ny,nz)     material id per voxel (0 = air)
  solid       bool  (nx,ny,nz)     obstacle mask (walls/glass/steel) — blocks spread
  fuel        float (nx,ny,nz)     fuel load per voxel (from material)
  ignition    list[(x,y,z)]        initially-burning voxels
  budgets     dict                 water / time / compute limits
  room_id     int   (nx,ny,nz)     which room a voxel belongs to (-1 = none), for evals

The JSON floorplan schema (input) is documented in floorplan_spec() below.
"""

from dataclasses import dataclass, field
import numpy as np
from . import materials as M


@dataclass
class VoxelScene:
    dims: tuple
    cell_size_m: float
    material: np.ndarray
    solid: np.ndarray
    fuel: np.ndarray
    ignition: list
    budgets: dict = field(default_factory=dict)
    room_id: np.ndarray = None
    room_names: list = field(default_factory=list)

    @property
    def nx(self): return self.dims[0]
    @property
    def ny(self): return self.dims[1]
    @property
    def nz(self): return self.dims[2]


def floorplan_spec():
    """Human-readable description of the accepted JSON floorplan schema."""
    return {
        "dims": "[nx, ny, nz] voxel grid (required)",
        "cell_size_m": "float, voxel edge length in metres (default 0.25)",
        "rooms": "[{name, bbox:[x0,y0,x1,y1], height:int}]  interior air volumes",
        "walls": "[{a:[x,y], b:[x,y], height:int, material:'concrete'}]  block spread",
        "doors": "[{at:[x,y], width:int, height:int}]  gaps carved back into walls",
        "furniture": "[{bbox:[x0,y0,x1,y1], z:[z0,z1], material:'foam'}]  combustible fill",
        "ignition": "[[x,y,z], ...]  initially-burning voxels",
        "budgets": "{water:int, time:int}  suppression limits",
    }


REQUIRED = ("dims",)


def validate_floorplan(d):
    """Raise ValueError on a malformed floorplan dict; return it if OK."""
    for k in REQUIRED:
        if k not in d:
            raise ValueError(f"floorplan missing required key '{k}'")
    dims = d["dims"]
    if not (isinstance(dims, (list, tuple)) and len(dims) == 3 and all(int(v) > 0 for v in dims)):
        raise ValueError("dims must be [nx, ny, nz], all > 0")
    for r in d.get("rooms", []):
        if "bbox" not in r or len(r["bbox"]) != 4:
            raise ValueError(f"room {r.get('name')} needs bbox [x0,y0,x1,y1]")
    for w in d.get("walls", []):
        M.id_of(w.get("material", "concrete"))   # validates material name
    for f in d.get("furniture", []):
        M.id_of(f.get("material", "wood"))
    return d
