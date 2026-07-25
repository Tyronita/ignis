import { RobotPolicyController } from './robot/RobotPolicyController.js';
import { HumanController } from './robot/HumanController.js';
import { MetricsCollector } from './simulation/MetricsCollector.js';
import { Heatmap } from './viz/Heatmap.js';
import { SimulationRunner } from './simulation/SimulationRunner.js';
import { Renderer } from './viz/Renderer.js';
import { UI } from './viz/UI.js';
import { DatasetExporter } from './export/DatasetExporter.js';

const RECENT_FULL_EPISODES_CAP = 40; // bounds memory: full per-timestep data kept only for the last N episodes

const dom = {
  mainCanvas: document.getElementById('mainCanvas'),
  heatmapCanvas: document.getElementById('heatmapCanvas'),
  liveDot: document.getElementById('liveDot'),
  statusText: document.getElementById('statusText'),
  btnStartPause: document.getElementById('btnStartPause'),
  btnRunBatch: document.getElementById('btnRunBatch'),
  btnReset: document.getElementById('btnReset'),
  speedButtons: Array.from(document.querySelectorAll('button.speed')),
  controllerSelect: document.getElementById('controllerSelect'),
  btnExportEpisode: document.getElementById('btnExportEpisode'),
  btnExportSession: document.getElementById('btnExportSession'),
  statEpisode: document.getElementById('statEpisode'),
  statRuntime: document.getElementById('statRuntime'),
  statRoomDims: document.getElementById('statRoomDims'),
  statObstacles: document.getElementById('statObstacles'),
  statLastResult: document.getElementById('statLastResult'),
  statCount: document.getElementById('statCount'),
  statSuccessRate: document.getElementById('statSuccessRate'),
  statAvgSurvival: document.getElementById('statAvgSurvival'),
  statAvgPath: document.getElementById('statAvgPath'),
  barSuccess: document.getElementById('barSuccess'),
  barCollision: document.getElementById('barCollision'),
  barTimeout: document.getElementById('barTimeout'),
  legendCounts: document.getElementById('legendCounts'),
};

const metrics = new MetricsCollector();
const heatmap = new Heatmap();
const renderer = new Renderer(dom.mainCanvas);

let trail = [];
let flash = null;
let flashUntil = 0;
const recentFullEpisodes = [];

let policyController = new RobotPolicyController();
let humanController = null; // created lazily so we don't attach listeners unless used
let activeControllerKind = 'policy';

const runner = new SimulationRunner({
  controller: policyController,
  metrics,
  heatmap,
  onTick: () => {
    trail.push({ x: runner.robot.x, y: runner.robot.y });
    if (trail.length > 2000) trail.shift();
  },
  onEpisodeEnd: (episodeData) => {
    ui.onEpisodeEnd(episodeData);
    flash = episodeData.result;
    flashUntil = performance.now() + 350;
    recentFullEpisodes.push(episodeData);
    if (recentFullEpisodes.length > RECENT_FULL_EPISODES_CAP) recentFullEpisodes.shift();
    trail = [];
  },
});

const ui = new UI(dom, {
  onToggleRun: () => {
    if (runner.running) {
      runner.pause();
      ui.setRunning(false);
    } else {
      runner.start();
      ui.setRunning(true);
    }
  },
  onRunBatch: (n) => {
    runner.start(runner.episodeIndex + n);
    ui.setRunning(true);
  },
  onReset: () => {
    metrics.episodes.length = 0;
    heatmap.reset();
  },
  onSpeedChange: (speed) => runner.setSpeed(speed),
  onControllerChange: (kind) => {
    activeControllerKind = kind;
    if (kind === 'human') {
      if (!humanController) humanController = new HumanController();
      runner.setController(humanController);
    } else {
      runner.setController(policyController);
    }
  },
  onExportEpisode: () => {
    if (!recentFullEpisodes.length) return;
    DatasetExporter.downloadEpisode(recentFullEpisodes[recentFullEpisodes.length - 1]);
  },
  onExportSession: () => {
    if (!recentFullEpisodes.length) return;
    DatasetExporter.downloadSession(recentFullEpisodes);
  },
});
ui.setRunning(false);

// Debug hook for automated testing / console inspection — harmless in production.
window.__navbench = { runner, metrics, heatmap, recentFullEpisodes, renderer, ui, getTrail: () => trail };

function loop() {
  requestAnimationFrame(loop);
  runner.update();

  const activeFlash = flash && performance.now() < flashUntil ? flash : null;
  if (flash && performance.now() >= flashUntil) flash = null;

  renderer.draw(runner.room, runner.robot, trail, activeFlash);
  ui.refresh({ episodeIndex: runner.episodeIndex, elapsed: runner.elapsed, room: runner.room }, metrics);
}
requestAnimationFrame(loop);
