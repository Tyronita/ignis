export class Obstacle {
  constructor(x, y, w, h) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
  }

  intersectsCircle(cx, cy, r) {
    const nx = Math.max(this.x, Math.min(cx, this.x + this.w));
    const ny = Math.max(this.y, Math.min(cy, this.y + this.h));
    const dx = cx - nx;
    const dy = cy - ny;
    return dx * dx + dy * dy < r * r;
  }
}
