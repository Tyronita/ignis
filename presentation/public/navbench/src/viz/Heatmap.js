// Rooms vary in size every episode, so raw pixel coordinates aren't
// comparable across episodes. Collision locations are normalized into
// room-relative [0,1] space before binning, so the accumulated heatmap shows
// *where within a room* (e.g. "near the exit-side wall", "center") collisions
// cluster, independent of any single room's actual dimensions.
export class Heatmap {
  constructor(resolution = 40) {
    this.res = resolution;
    this.grid = new Float32Array(resolution * resolution);
    this.maxVal = 0;
    this.totalCollisions = 0;
  }

  record(episodeData) {
    if (episodeData.result !== 'collision' || !episodeData.collision_location) return;
    const { x, y } = episodeData.collision_location;
    const nx = Math.min(this.res - 1, Math.max(0, Math.floor((x / episodeData.room_width) * this.res)));
    const ny = Math.min(this.res - 1, Math.max(0, Math.floor((y / episodeData.room_height) * this.res)));
    const idx = ny * this.res + nx;
    this.grid[idx] += 1;
    this.maxVal = Math.max(this.maxVal, this.grid[idx]);
    this.totalCollisions++;
  }

  reset() {
    this.grid.fill(0);
    this.maxVal = 0;
    this.totalCollisions = 0;
  }

  // Single hue (status-critical red), alpha-encoded by magnitude — a
  // sequential ramp expressed as opacity rather than discrete color steps.
  render(ctx, width, height) {
    ctx.fillStyle = '#0d0d0d';
    ctx.fillRect(0, 0, width, height);
    if (this.maxVal <= 0) return;
    const cw = width / this.res;
    const ch = height / this.res;
    for (let gy = 0; gy < this.res; gy++) {
      for (let gx = 0; gx < this.res; gx++) {
        const v = this.grid[gy * this.res + gx];
        if (v <= 0) continue;
        const t = v / this.maxVal;
        ctx.fillStyle = `rgba(208,59,59,${0.12 + 0.78 * t})`;
        ctx.fillRect(gx * cw, gy * ch, cw + 0.5, ch + 0.5);
      }
    }
  }
}
