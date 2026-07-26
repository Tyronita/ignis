"""
Export occupant trajectories in the VLGE telemetry shape (characterSnapshots).

Matches the fields of a VLGE session export so our simulated agent paths are
drop-in compatible with the platform's spatial/behavioural data schema:
per-frame position (pc), time, velocity/acceleration, displacement, movement_state,
heading; plus templateSize (world bounds) and sessionMetrics.

Voxel→world mapping: world.x = gx·cell, world.y = head height, world.z = gy·cell.

  python -m ignis.indoor.vlge_export        # run evacuation on the house, write VLGE-shaped JSON
"""

import json
import math
import os
import numpy as np
from . import importer, suppress, evacuate, indoor_env as ie

HEAD_M = 1.7
TICKS_PER_S = 10_000_000          # .NET-style 100 ns ticks
HOUSE = os.path.join(os.path.dirname(__file__), "examples", "house.json")


def _movement_state(v):
    return "running" if v > 1.5 else ("walk" if v > 0.1 else "stationary")


def _session_from_traj(traj, cell, dt):
    """One agent path (list of (gx,gy)) -> a VLGE-shaped session dict."""
    snaps, total_dist = [], 0.0
    prev = None
    speeds = []
    for i, (gx, gy) in enumerate(traj):
        x, z = gx * cell, gy * cell
        t = i * dt
        if prev is None:
            v = a = disp = 0.0; yaw = 0.0
        else:
            dx, dz = x - prev[0], z - prev[1]
            disp = math.hypot(dx, dz); v = disp / dt
            a = (v - speeds[-1]) / dt if speeds else 0.0
            yaw = math.degrees(math.atan2(dx, dz))
            total_dist += disp
        speeds.append(v)
        snaps.append({
            "pc": {"x": round(x, 3), "y": HEAD_M, "z": round(z, 3)},
            "rc": {"x": 0.0, "y": math.sin(math.radians(yaw) / 2), "z": 0.0,
                   "w": math.cos(math.radians(yaw) / 2)},
            "floorType": "",
            "timestamp": {"ticks": int(t * TICKS_PER_S)},
            "time_s": round(t, 3),
            "velocity_mps": round(v, 3),
            "acceleration_mps2": round(a, 3),
            "displacement_m": round(disp, 3),
            "height_above_ground": 0.0,
            "camera_yaw_deg": round(yaw, 1),
            "movement_state": _movement_state(v),
        })
        prev = (x, z)
    dur = max(1e-6, (len(traj) - 1) * dt)
    return {
        "mode": 0,
        "remoteUpdateFrequency": 1.0 / dt,
        "startDateTime": {"ticks": 0},
        "characterSnapshots": snaps,
        "sessionMetrics": {
            "session_duration_s": round(dur, 3),
            "total_frames": len(snaps),
            "total_distance_m": round(total_dist, 3),
            "avg_velocity_mps": round(total_dist / dur, 3),
            "max_velocity_mps": round(max(speeds) if speeds else 0.0, 3),
            "net_displacement_m": round(math.hypot(
                snaps[-1]["pc"]["x"] - snaps[0]["pc"]["x"],
                snaps[-1]["pc"]["z"] - snaps[0]["pc"]["z"]), 3) if snaps else 0.0,
        },
    }


def export(scene, out_dir="datasets/vlge", n_agents=8, seed=0, steps=120):
    st = suppress.init(scene)
    ev = evacuate.Evacuation(scene, n_agents=n_agents, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        suppress.sim_step(st, rng)
        m = ev.step(st)
        if m["still_inside"] == 0:
            break
    nx, ny, nz = scene.dims; cell = scene.cell_size_m
    template = {
        "description": "Ignis voxel house — VLGE-shaped bounds",
        "x_min": 0.0, "x_max": round(nx * cell, 3),
        "y_min": 0.0, "y_max": round(nz * cell, 3),
        "z_min": 0.0, "z_max": round(ny * cell, 3),
        "width_x": round(nx * cell, 3), "height_y": round(nz * cell, 3),
        "depth_z": round(ny * cell, 3), "unit": "world_units",
    }
    sessions = []
    for i, traj in enumerate(ev.trajectories):
        s = _session_from_traj(traj, cell, evacuate.DT_S)
        s["templateSize"] = template
        s["agent_id"] = i
        s["outcome"] = ["active", "escaped", "casualty"][int(ev.state[i])]
        sessions.append(s)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "vlge_trajectories.json")
    with open(path, "w") as fh:
        json.dump({"n_sessions": len(sessions), "source": "ignis.evacuate",
                   "schema": "VLGE characterSnapshots (subset)", "sessions": sessions}, fh, indent=1)
    print(f"saved {path}  ({len(sessions)} VLGE-shaped agent sessions, "
          f"{sum(len(s['characterSnapshots']) for s in sessions)} snapshots)")
    return path


if __name__ == "__main__":
    export(importer.load(HOUSE))
