"""
Evacuation task — occupants escape a burning building against the clock.

Occupants navigate the floor toward the nearest exit along a fire-aware shortest path
(a distance field re-solved each step as fire blocks cells). Walls and furniture block
movement; a burning cell is lethal. Trajectories are recorded (optional overlay).

Real constants (see PHYSICS.md): DT_S = 1 s/step, walking speed 1.2 m/s (reduced in
smoke). At 0.25 m voxels that is ~5 cells/step. This is a strong shortest-path baseline;
multi-agent RL (PettingZoo) is the documented upgrade.
"""

import numpy as np
from collections import deque

DT_S = 1.0          # seconds per simulation step
WALK_MPS = 1.2      # normal walking speed
SMOKE_SLOW = 0.5    # speed multiplier when moving through smoke-filled air


def _cells_per_step(cell_size_m, in_smoke=False):
    v = WALK_MPS * (SMOKE_SLOW if in_smoke else 1.0)
    return max(1, int(round(v * DT_S / cell_size_m)))


class Evacuation:
    def __init__(self, scene, n_agents=6, seed=0, start_delay=0):
        self.scene = scene
        self.cell = scene.cell_size_m
        self.start_delay = int(start_delay)   # pre-movement/recognition delay (steps)
        self.rng = np.random.default_rng(seed)
        self.exits = scene.exits or [[scene.nx // 2, 0]]
        # static navigation blockers: walls + furniture at floor level (z=1)
        z = 1
        from . import materials as M
        solid = scene.solid[:, :, z]
        furn = (~scene.solid[:, :, z]) & (scene.material[:, :, z] != 0)
        self.static_block = solid | furn
        # spawn agents on free floor cells inside rooms, away from ignition
        rid = scene.room_id[:, :, z]
        free = np.argwhere((rid >= 0) & ~self.static_block)
        pick = self.rng.choice(len(free), size=min(n_agents, len(free)), replace=False)
        self.pos = free[pick].astype(np.int32)                # (A,2)
        self.state = np.zeros(len(self.pos), np.int8)          # 0 active, 1 escaped, 2 casualty
        self.escaped_step = np.full(len(self.pos), -1, np.int32)
        self.trajectories = [[] for _ in self.pos]
        self.t = 0

    def _distance_field(self, hazard):
        """BFS distance to nearest exit over free (not blocked, not hazard) floor cells."""
        nx, ny = self.static_block.shape
        free = ~self.static_block & ~hazard
        dist = np.full((nx, ny), 1 << 30, np.int32)
        q = deque()
        for ex, ey in self.exits:
            for x in range(max(0, ex - 1), min(nx, ex + 2)):
                for y in range(max(0, ey - 1), min(ny, ey + 2)):
                    if 0 <= x < nx and 0 <= y < ny and (free[x, y] or (ex, ey) == (x, y)):
                        dist[x, y] = 0; q.append((x, y))
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = x + dx, y + dy
                if 0 <= a < nx and 0 <= b < ny and free[a, b] and dist[a, b] > dist[x, y] + 1:
                    dist[a, b] = dist[x, y] + 1; q.append((a, b))
        return dist

    def step(self, fire_state):
        hazard = (fire_state["burning"] > 0).any(axis=2)              # burning column = lethal
        if self.t < self.start_delay:                                 # not moving yet (recognition)
            for i, (x, y) in enumerate(self.pos):
                if self.state[i] == 0 and hazard[x, y]:
                    self.state[i] = 2
                self.trajectories[i].append((int(x), int(y)))
            self.t += 1
            return self.metrics()
        smoke = fire_state.get("flamed_air")
        smoke2d = smoke.any(axis=2) if smoke is not None else np.zeros_like(hazard)
        dist = self._distance_field(hazard)
        nx, ny = self.static_block.shape
        for i, (x, y) in enumerate(self.pos):
            if self.state[i]:
                self.trajectories[i].append((int(x), int(y)))
                continue
            steps = _cells_per_step(self.cell, in_smoke=bool(smoke2d[x, y]))
            for _ in range(steps):
                if hazard[x, y]:
                    self.state[i] = 2; break                          # caught by fire
                if dist[x, y] == 0:
                    self.state[i] = 1; self.escaped_step[i] = self.t; break
                # greedily descend the distance field
                best, bxy = dist[x, y], None
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = x + dx, y + dy
                    if 0 <= a < nx and 0 <= b < ny and dist[a, b] < best:
                        best, bxy = dist[a, b], (a, b)
                if bxy is None:
                    break                                             # blocked this step
                x, y = bxy
            self.pos[i] = (x, y)
            self.trajectories[i].append((int(x), int(y)))
        self.t += 1
        return self.metrics()

    def metrics(self):
        esc = int((self.state == 1).sum())
        dead = int((self.state == 2).sum())
        active = int((self.state == 0).sum())
        times = self.escaped_step[self.state == 1]
        return dict(occupants=len(self.pos), escaped=esc, casualties=dead, still_inside=active,
                    mean_escape_s=float(times.mean() * DT_S) if len(times) else -1.0,
                    all_out_s=float(times.max() * DT_S) if len(times) == len(self.pos) and len(times) else -1.0)
