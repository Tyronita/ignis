// Keeps a lightweight summary (no per-timestep data) of every episode ever
// run, so running statistics stay cheap across hundreds of episodes even
// though full trajectories are only kept for a bounded recent window
// elsewhere (see main.js).
export class MetricsCollector {
  constructor() {
    this.episodes = [];
  }

  record(episodeData) {
    const { timesteps, obstacle_layout, ...summary } = episodeData;
    this.episodes.push(summary);
  }

  get count() {
    return this.episodes.length;
  }

  get successRate() {
    return this.count ? this.episodes.filter((e) => e.success).length / this.count : 0;
  }

  get avgSurvivalTime() {
    return this._avg((e) => e.runtime);
  }

  get avgPathLength() {
    return this._avg((e) => e.total_distance);
  }

  get resultCounts() {
    const counts = { success: 0, collision: 0, timeout: 0 };
    for (const e of this.episodes) counts[e.result] = (counts[e.result] || 0) + 1;
    return counts;
  }

  recent(n = 10) {
    return this.episodes.slice(-n);
  }

  _avg(fn) {
    if (!this.count) return 0;
    return this.episodes.reduce((s, e) => s + fn(e), 0) / this.count;
  }
}
