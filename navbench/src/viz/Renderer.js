// Draws the current room, obstacles, exit gap, trajectory trail, and robot.
// Rooms vary in size, so the room is centered inside a fixed-size logical
// canvas rather than the canvas being resized per episode — keeps the
// viewport stable to look at while the room itself visibly changes shape.

// Long, thin obstacles are corridor-forming "walls" in the physics; blocky
// ones are "furniture". Both now render with real furniture art (elongated
// ones get the table image stretched along their length, like a long table
// or counter) — no plain colored rectangles standing in for objects.
function isWallLike(o) {
  return Math.max(o.w, o.h) / Math.min(o.w, o.h) > 2.4;
}

function loadImg(src, onReady) {
  const img = new Image();
  img.onload = onReady;
  img.src = src;
  return img;
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.LOGICAL_W = canvas.width;
    this.LOGICAL_H = canvas.height;
    this.scale = 1;
    this._lastCssW = 0;

    this.robotImgReady = false;
    this.chairImgReady = false;
    this.tableImgReady = false;
    this.robotImg = loadImg('assets/robot.png', () => { this.robotImgReady = true; });
    this.chairImg = loadImg('assets/chair.png', () => { this.chairImgReady = true; });
    this.tableImg = loadImg('assets/table.png', () => { this.tableImgReady = true; });

    window.addEventListener('resize', () => this._resize());
    this._resize();
  }

  // Renders at the canvas's actual on-screen resolution × devicePixelRatio,
  // rather than a fixed 860×620 buffer stretched by CSS — the stretch is what
  // was making everything (the robot especially) look soft/low-res.
  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    const cssW = rect.width || this.LOGICAL_W;
    if (Math.abs(cssW - this._lastCssW) < 0.5) return;
    this._lastCssW = cssW;
    const dpr = window.devicePixelRatio || 1;
    const cssH = cssW * (this.LOGICAL_H / this.LOGICAL_W);
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.scale = this.canvas.width / this.LOGICAL_W;
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
  }

  draw(room, robot, trail, flash) {
    this._resize();
    const ctx = this.ctx;
    ctx.save();
    ctx.scale(this.scale, this.scale);

    const W = this.LOGICAL_W;
    const H = this.LOGICAL_H;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0d0d0d';
    ctx.fillRect(0, 0, W, H);

    const ox = (W - room.width) / 2;
    const oy = (H - room.height) / 2;
    ctx.save();
    ctx.translate(ox, oy);

    this._drawFloor(ctx, room);
    this._drawWallsAndExit(ctx, room);
    this._drawObstacles(ctx, room);
    this._drawTrail(ctx, trail);
    this._drawSpawnMarker(ctx, room);
    this._drawRobot(ctx, robot);

    if (flash) {
      ctx.strokeStyle = flash === 'success' ? '#17b869' : '#e5484d';
      ctx.lineWidth = 6;
      ctx.strokeRect(3, 3, room.width - 6, room.height - 6);
    }

    ctx.restore();
    ctx.restore();
  }

  _drawFloor(ctx, room) {
    const grad = ctx.createRadialGradient(
      room.width / 2, room.height / 2, 0,
      room.width / 2, room.height / 2, Math.max(room.width, room.height) * 0.7
    );
    grad.addColorStop(0, '#1c1c1b');
    grad.addColorStop(1, '#131312');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, room.width, room.height);

    ctx.strokeStyle = 'rgba(255,255,255,0.035)';
    ctx.lineWidth = 1;
    for (let gx = 0; gx <= room.width; gx += 40) {
      ctx.beginPath();
      ctx.moveTo(gx + 0.5, 0);
      ctx.lineTo(gx + 0.5, room.height);
      ctx.stroke();
    }
    for (let gy = 0; gy <= room.height; gy += 40) {
      ctx.beginPath();
      ctx.moveTo(0, gy + 0.5);
      ctx.lineTo(room.width, gy + 0.5);
      ctx.stroke();
    }
  }

  _drawWallsAndExit(ctx, room) {
    ctx.strokeStyle = '#4a5568';
    ctx.lineWidth = 6;
    ctx.strokeRect(3, 3, room.width - 6, room.height - 6);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.strokeRect(3, 3, room.width - 6, room.height - 6);

    const e = room.exit;
    // Punch the gap through the wall.
    ctx.fillStyle = '#131312';
    ctx.fillRect(e.x - 4, e.y - 4, e.w + 8, e.h + 8);

    // Door-swing arc — the standard architectural symbol, doubles as an
    // unmistakable "this is a floor plan" cue.
    const r = e.side === 0 || e.side === 2 ? e.w : e.h;
    ctx.strokeStyle = 'rgba(210,218,232,0.32)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    if (e.side === 0) {
      ctx.moveTo(e.x, 0);
      ctx.lineTo(e.x, r);
      ctx.arc(e.x, 0, r, Math.PI / 2, Math.PI, false);
    } else if (e.side === 2) {
      ctx.moveTo(e.x, room.height);
      ctx.lineTo(e.x, room.height - r);
      ctx.arc(e.x, room.height, r, -Math.PI / 2, 0, true);
    } else if (e.side === 3) {
      ctx.moveTo(0, e.y);
      ctx.lineTo(r, e.y);
      ctx.arc(0, e.y, r, 0, Math.PI / 2, false);
    } else {
      ctx.moveTo(room.width, e.y);
      ctx.lineTo(room.width - r, e.y);
      ctx.arc(room.width, e.y, r, Math.PI, Math.PI * 1.5, false);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = '#1fbf6b';
    ctx.lineWidth = 2.5;
    ctx.strokeRect(e.x, e.y, e.w, e.h);
  }

  _drawObstacles(ctx, room) {
    for (const o of room.obstacles) {
      // Every obstacle renders as real furniture art now — no flat colored
      // boxes. Elongated ones (former "wall" segments) get the table image
      // stretched along their length; blocky ones get a table or chair
      // depending on footprint size. The image is always stretched to fill
      // that obstacle's exact box, so furniture scales with the obstacle.
      const useTable = isWallLike(o) || o.w * o.h > 4500;
      const img = useTable ? this.tableImg : this.chairImg;
      const ready = useTable ? this.tableImgReady : this.chairImgReady;

      ctx.save();
      ctx.shadowColor = 'rgba(0,0,0,0.5)';
      ctx.shadowBlur = 10;
      ctx.shadowOffsetY = 4;
      if (ready) {
        ctx.drawImage(img, o.x, o.y, o.w, o.h);
      }
      ctx.restore();
    }
  }

  _drawTrail(ctx, trail) {
    if (trail.length < 2) return;
    for (let i = 1; i < trail.length; i++) {
      const t = i / trail.length;
      ctx.strokeStyle = `rgba(76,141,255,${0.15 + 0.55 * t})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(trail[i - 1].x, trail[i - 1].y);
      ctx.lineTo(trail[i].x, trail[i].y);
      ctx.stroke();
    }
  }

  _drawSpawnMarker(ctx, room) {
    const c = room.center;
    ctx.strokeStyle = '#6b6960';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(c.x - 6, c.y);
    ctx.lineTo(c.x + 6, c.y);
    ctx.moveTo(c.x, c.y - 6);
    ctx.lineTo(c.x, c.y + 6);
    ctx.stroke();
  }

  _drawRobot(ctx, robot) {
    // The full robot art, anchored at its base (the wheeled platform) rather
    // than centered — same convention as a top-down character sprite, where
    // the figure stands "on" its logical (x,y) and extends upward, instead
    // of being sliced in half by centering a tall image on a point.
    const w = robot.radius * 2.7;
    const aspect = this.robotImg.naturalWidth
      ? this.robotImg.naturalHeight / this.robotImg.naturalWidth
      : 1380 / 546;
    const h = w * aspect;
    const baseX = robot.x;
    const baseY = robot.y;

    ctx.beginPath();
    ctx.ellipse(baseX, baseY, w * 0.42, w * 0.16, 0, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.fill();

    if (this.robotImgReady) {
      ctx.drawImage(this.robotImg, baseX - w / 2, baseY - h, w, h);
    } else {
      ctx.fillStyle = '#4c8dff';
      ctx.beginPath();
      ctx.arc(baseX, baseY - h / 2, w / 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Heading chevron at the feet — direction without spinning the body art.
    const cx = baseX + Math.cos(robot.heading) * (w * 0.62);
    const cy = baseY + Math.sin(robot.heading) * (w * 0.62);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(robot.heading);
    ctx.beginPath();
    ctx.moveTo(5, 0);
    ctx.lineTo(-3, -4);
    ctx.lineTo(-3, 4);
    ctx.closePath();
    ctx.fillStyle = '#4c8dff';
    ctx.fill();
    ctx.strokeStyle = '#0d0d0d';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
  }
}
