import { Obstacle } from './Obstacle.js';
import { mulberry32 } from '../util/rng.js';

const ROBOT_RADIUS = 9;

// Produces a randomized room and guarantees, by construction, that it is
// navigable: obstacles are expanded by the robot's radius and checked with a
// grid BFS from spawn to exit before the room is accepted. A failed layout is
// simply re-rolled (with a shrinking obstacle budget after repeated failures)
// rather than patched, which keeps the generator honest — every accepted room
// really was verified, not just "probably fine."
export class EnvironmentGenerator {
  static generate(seed) {
    let obstacleBudget = null;
    for (let attempt = 1; attempt <= 250; attempt++) {
      const rng = mulberry32((seed ^ (attempt * 2654435761)) >>> 0);
      const room = EnvironmentGenerator._attempt(rng, obstacleBudget);
      if (room) {
        room.seed = seed;
        room.generationAttempts = attempt;
        return room;
      }
      if (attempt % 12 === 0) {
        obstacleBudget = Math.max(3, (obstacleBudget ?? 22) - 3);
      }
    }
    // Absolute fallback so the generator can never hang the simulation.
    const room = EnvironmentGenerator._emptyRoom(seed);
    room.generationAttempts = 250;
    return room;
  }

  static _attempt(rng, obstacleBudget) {
    const width = 520 + rng() * 340; // 520-860
    const height = 380 + rng() * 240; // 380-620
    const center = { x: width / 2, y: height / 2 };

    const side = Math.floor(rng() * 4); // 0 top, 1 right, 2 bottom, 3 left
    const gap = 46 + rng() * 30;
    let exit;
    if (side === 0) exit = { side, x: 20 + rng() * (width - 40 - gap), y: -6, w: gap, h: 12 };
    else if (side === 2) exit = { side, x: 20 + rng() * (width - 40 - gap), y: height - 6, w: gap, h: 12 };
    else if (side === 3) exit = { side, x: -6, y: 20 + rng() * (height - 40 - gap), w: 12, h: gap };
    else exit = { side, x: width - 6, y: 20 + rng() * (height - 40 - gap), w: 12, h: gap };
    exit.cx = exit.x + exit.w / 2;
    exit.cy = exit.y + exit.h / 2;

    const count = obstacleBudget ?? Math.floor(6 + rng() * 17); // 6-22
    const centerBias = rng(); // clustering strength toward the center
    const corridorMode = rng() < 0.35; // favor long thin wall segments this episode

    const obstacles = [];
    const spawnSafeR = 46;
    let placed = 0;
    let tries = 0;
    while (placed < count && tries < count * 25) {
      tries++;
      let w, h;
      if (corridorMode && rng() < 0.5) {
        if (rng() < 0.5) {
          w = 70 + rng() * 160;
          h = 14 + rng() * 10;
        } else {
          w = 14 + rng() * 10;
          h = 70 + rng() * 160;
        }
      } else {
        w = 24 + rng() * 70;
        h = 24 + rng() * 70;
      }

      let x, y;
      if (rng() < centerBias) {
        const spread = 60 + rng() * 140;
        x = center.x + (rng() * 2 - 1) * spread - w / 2;
        y = center.y + (rng() * 2 - 1) * spread - h / 2;
      } else {
        x = 10 + rng() * (width - 20 - w);
        y = 10 + rng() * (height - 20 - h);
      }
      x = Math.max(6, Math.min(x, width - 6 - w));
      y = Math.max(6, Math.min(y, height - 6 - h));

      const cand = new Obstacle(x, y, w, h);
      if (cand.intersectsCircle(center.x, center.y, spawnSafeR)) continue;
      if (cand.intersectsCircle(exit.cx, exit.cy, gap * 0.9)) continue;
      obstacles.push(cand);
      placed++;
    }

    const room = {
      width,
      height,
      center,
      exit,
      obstacles,
      obstacleCount: obstacles.length,
      obstacleDensity: (obstacles.reduce((s, o) => s + o.w * o.h, 0) / (width * height)),
      corridorMode,
      centerBias,
    };
    if (!EnvironmentGenerator._isNavigable(room)) return null;
    return room;
  }

  static _isNavigable(room) {
    const cell = 16;
    const pad = ROBOT_RADIUS + 3; // Minkowski-expand obstacles so the path fits the robot, not just a point
    const cols = Math.ceil(room.width / cell);
    const rows = Math.ceil(room.height / cell);
    const blocked = new Uint8Array(cols * rows);

    for (let gy = 0; gy < rows; gy++) {
      for (let gx = 0; gx < cols; gx++) {
        const cx = gx * cell + cell / 2;
        const cy = gy * cell + cell / 2;
        for (const o of room.obstacles) {
          if (cx > o.x - pad && cx < o.x + o.w + pad && cy > o.y - pad && cy < o.y + o.h + pad) {
            blocked[gy * cols + gx] = 1;
            break;
          }
        }
      }
    }

    const clampCol = (v) => Math.min(cols - 1, Math.max(0, Math.floor(v / cell)));
    const clampRow = (v) => Math.min(rows - 1, Math.max(0, Math.floor(v / cell)));
    const startGX = clampCol(room.center.x);
    const startGY = clampRow(room.center.y);
    const goalGX = clampCol(room.exit.cx);
    const goalGY = clampRow(room.exit.cy);
    if (blocked[startGY * cols + startGX]) return false;

    const seen = new Uint8Array(cols * rows);
    const queue = [[startGX, startGY]];
    seen[startGY * cols + startGX] = 1;
    let qi = 0;
    while (qi < queue.length) {
      const [gx, gy] = queue[qi++];
      if (gx === goalGX && gy === goalGY) return true;
      const neighbors = [
        [gx + 1, gy],
        [gx - 1, gy],
        [gx, gy + 1],
        [gx, gy - 1],
      ];
      for (const [nx, ny] of neighbors) {
        if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
        const idx = ny * cols + nx;
        if (seen[idx] || blocked[idx]) continue;
        seen[idx] = 1;
        queue.push([nx, ny]);
      }
    }
    return false;
  }

  static _emptyRoom(seed) {
    const width = 600;
    const height = 450;
    const center = { x: width / 2, y: height / 2 };
    const exit = { side: 1, x: width - 6, y: height / 2 - 25, w: 12, h: 50, cx: width, cy: height / 2 };
    return {
      width,
      height,
      center,
      exit,
      obstacles: [],
      obstacleCount: 0,
      obstacleDensity: 0,
      corridorMode: false,
      centerBias: 0,
      seed,
    };
  }
}
