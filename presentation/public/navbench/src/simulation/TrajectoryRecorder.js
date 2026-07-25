// Records one episode at the per-timestep granularity of a real robotics
// trajectory dataset. Static per-episode facts (seed, room, obstacle layout)
// live once on the episode object rather than being repeated on every
// timestep — the same information the spec's per-timestep schema asks for,
// organized the way an actual trajectory dataset file would be.
export class TrajectoryRecorder {
  constructor(room, episodeIndex) {
    this.room = room;
    this.episodeIndex = episodeIndex;
    this.timesteps = [];
    this.result = null;
  }

  record(t, robot) {
    this.timesteps.push({
      timestamp: +t.toFixed(3),
      position: { x: +robot.x.toFixed(2), y: +robot.y.toFixed(2) },
      heading: +robot.heading.toFixed(4),
      velocity: +robot.linearVelocity.toFixed(2),
      angular_velocity: +robot.angularVelocity.toFixed(4),
      collided: false,
    });
  }

  finish(result) {
    this.result = result;
    if (this.timesteps.length) {
      this.timesteps[this.timesteps.length - 1].collided = result === 'collision';
    }
  }

  toEpisodeSummary() {
    return {
      episodeIndex: this.episodeIndex,
      room_seed: this.room.seed,
      room_width: +this.room.width.toFixed(1),
      room_height: +this.room.height.toFixed(1),
      obstacle_count: this.room.obstacleCount,
      obstacle_density: +this.room.obstacleDensity.toFixed(4),
      obstacle_layout: this.room.obstacles.map((o) => ({ x: o.x, y: o.y, w: o.w, h: o.h })),
      exit: { side: this.room.exit.side, x: this.room.exit.x, y: this.room.exit.y, w: this.room.exit.w, h: this.room.exit.h },
      result: this.result,
      success: this.result === 'success',
      runtime: this.timesteps.length ? this.timesteps[this.timesteps.length - 1].timestamp : 0,
      total_distance: this._pathLength(),
      collision_location: this.result === 'collision' ? this.timesteps[this.timesteps.length - 1].position : null,
      timesteps: this.timesteps,
    };
  }

  _pathLength() {
    let d = 0;
    for (let i = 1; i < this.timesteps.length; i++) {
      const a = this.timesteps[i - 1].position;
      const b = this.timesteps[i].position;
      d += Math.hypot(b.x - a.x, b.y - a.y);
    }
    return +d.toFixed(1);
  }
}
