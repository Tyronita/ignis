import { EnvironmentGenerator } from '../environment/EnvironmentGenerator.js';
import { isBlocked, hasReachedExit } from '../environment/collision.js';
import { Robot } from '../robot/Robot.js';
import { TrajectoryRecorder } from './TrajectoryRecorder.js';
import { makeSeed } from '../util/rng.js';

const DT = 1 / 60;
const MAX_EPISODE_TIME = 25; // seconds of simulated time before a run is called a timeout
const MAX_TICKS_PER_FRAME = 4000; // bounds worst-case main-thread time per rendered frame

// Orchestrates one episode at a time and the transition between them. Speed
// is a tick multiplier rather than a separate "batch mode" code path — a
// speed of 1 is real-time and a speed of 60 fast-forwards through many
// episodes per second while still rendering, so "hundreds of episodes,
// displayed quickly" is the same loop, just turned up.
export class SimulationRunner {
  constructor({ controller, metrics, heatmap, onEpisodeEnd, onTick }) {
    this.controller = controller;
    this.metrics = metrics;
    this.heatmap = heatmap;
    this.onEpisodeEnd = onEpisodeEnd;
    this.onTick = onTick;
    this.episodeIndex = 0;
    this.speed = 1;
    this.running = false;
    this.episodeLimit = null;
    this._startEpisode();
  }

  setController(controller) {
    this.controller = controller;
    if (this.controller.reset) this.controller.reset();
  }

  start(episodeLimit = null) {
    this.episodeLimit = episodeLimit;
    this.running = true;
  }

  pause() {
    this.running = false;
  }

  setSpeed(speed) {
    this.speed = speed;
  }

  _startEpisode(seed) {
    this.episodeIndex++;
    const useSeed = seed ?? makeSeed();
    this.room = EnvironmentGenerator.generate(useSeed);
    this.robot = new Robot(this.room.center.x, this.room.center.y, Math.random() * Math.PI * 2);
    if (this.controller.reset) this.controller.reset();
    this.recorder = new TrajectoryRecorder(this.room, this.episodeIndex);
    this.elapsed = 0;
    this.ended = false;
  }

  // Called once per rendered frame. Advances `speed` physics ticks (capped),
  // stepping straight into the next episode if one finishes mid-batch.
  update() {
    if (!this.running) return;
    let ticks = Math.min(MAX_TICKS_PER_FRAME, Math.round(this.speed));
    while (ticks-- > 0) {
      this._tick();
      if (this.episodeLimit && this.episodeIndex > this.episodeLimit) {
        this.running = false;
        break;
      }
    }
  }

  _tick() {
    this.elapsed += DT;
    const env = { isBlocked: (x, y, r) => isBlocked(this.room, x, y, r) };
    const command = this.controller.decide(this.robot, env);
    this.robot.step(DT, command);
    this.recorder.record(this.elapsed, this.robot);

    let result = null;
    if (hasReachedExit(this.room, this.robot.x, this.robot.y)) result = 'success';
    else if (isBlocked(this.room, this.robot.x, this.robot.y, this.robot.radius)) result = 'collision';
    else if (this.elapsed >= MAX_EPISODE_TIME) result = 'timeout';

    if (result) {
      this.ended = true;
      this.recorder.finish(result);
      const episodeData = this.recorder.toEpisodeSummary();
      this.metrics.record(episodeData);
      this.heatmap.record(episodeData);
      if (this.onEpisodeEnd) this.onEpisodeEnd(episodeData, this.room);
      this._startEpisode();
    } else if (this.onTick) {
      this.onTick(this.robot, this.room, this.elapsed);
    }
  }
}
