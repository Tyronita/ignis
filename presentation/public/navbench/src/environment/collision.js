// Shared collision queries used by both the robot's naive sensing and the
// simulation's authoritative collision/success checks, so the two can never
// disagree about what "blocked" means.

export function isInExitZone(room, x, y, r = 0) {
  const e = room.exit;
  return x > e.x - r && x < e.x + e.w + r && y > e.y - r && y < e.y + e.h + r;
}

// True if a circle at (x,y,r) overlaps a wall or obstacle. The exit gap is a
// deliberate hole in the wall check, not an obstacle, so the robot can pass
// through it freely.
export function isBlocked(room, x, y, r) {
  const outOfBounds = x - r < 0 || x + r > room.width || y - r < 0 || y + r > room.height;
  if (outOfBounds) {
    return !isInExitZone(room, x, y, r);
  }
  for (const o of room.obstacles) {
    if (o.intersectsCircle(x, y, r)) return true;
  }
  return false;
}

// True once the robot has actually walked out through the exit gap (not just
// standing near it) — position is outside the room rect AND within the gap span.
export function hasReachedExit(room, x, y) {
  const outside = x < 0 || x > room.width || y < 0 || y > room.height;
  if (!outside) return false;
  return isInExitZone(room, x, y, 4);
}
