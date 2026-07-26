const RESULT_LABEL = { success: 'SUCCESS', collision: 'COLLISION', timeout: 'TIMEOUT' };

export class UI {
  constructor(dom, callbacks) {
    this.dom = dom;
    this.cb = callbacks;
    this.lastResult = null;
    this._wire();
  }

  _wire() {
    this.dom.btnStartPause.addEventListener('click', () => this.cb.onToggleRun());
    this.dom.btnRunBatch.addEventListener('click', () => this.cb.onRunBatch(100));
    this.dom.btnReset.addEventListener('click', () => this.cb.onReset());
    this.dom.speedButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        this.dom.speedButtons.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        this.cb.onSpeedChange(Number(btn.dataset.speed));
      });
    });
    this.dom.controllerSelect.addEventListener('change', (e) => this.cb.onControllerChange(e.target.value));
    this.dom.btnExportEpisode.addEventListener('click', () => this.cb.onExportEpisode());
    this.dom.btnExportSession.addEventListener('click', () => this.cb.onExportSession());
  }

  setRunning(running) {
    this.dom.btnStartPause.textContent = running ? '⏸ Pause' : '▶ Run';
    this.dom.btnStartPause.classList.toggle('primary', !running);
    if (this.dom.liveDot) this.dom.liveDot.classList.toggle('on', running);
    if (this.dom.statusText) this.dom.statusText.textContent = running ? 'Running' : 'Paused';
  }

  onEpisodeEnd(episodeData) {
    this.lastResult = episodeData.result;
    this.dom.statLastResult.textContent = RESULT_LABEL[episodeData.result];
    this.dom.statLastResult.className = 'badge ' + episodeData.result;
  }

  refresh({ episodeIndex, elapsed, room }, metrics) {
    this.dom.statEpisode.textContent = episodeIndex;
    this.dom.statRuntime.textContent = elapsed.toFixed(1) + 's';
    this.dom.statRoomDims.textContent = `${room.width.toFixed(0)}×${room.height.toFixed(0)}`;
    this.dom.statObstacles.textContent = room.obstacleCount;

    this.dom.statCount.textContent = metrics.count;
    this.dom.statSuccessRate.textContent = (metrics.successRate * 100).toFixed(1) + '%';
    this.dom.statAvgSurvival.textContent = metrics.avgSurvivalTime.toFixed(1) + 's';
    this.dom.statAvgPath.textContent = metrics.avgPathLength.toFixed(0) + 'px';

    const counts = metrics.resultCounts;
    const total = Math.max(1, metrics.count);
    this.dom.barSuccess.style.width = (counts.success / total) * 100 + '%';
    this.dom.barCollision.style.width = (counts.collision / total) * 100 + '%';
    this.dom.barTimeout.style.width = (counts.timeout / total) * 100 + '%';
    this.dom.legendCounts.innerHTML =
      `<span class="item"><span class="dot" style="background:var(--good)"></span><b>${counts.success}</b>&nbsp;success</span>` +
      `<span class="item"><span class="dot" style="background:var(--critical)"></span><b>${counts.collision}</b>&nbsp;collision</span>` +
      `<span class="item"><span class="dot" style="background:var(--baseline)"></span><b>${counts.timeout}</b>&nbsp;timeout</span>`;
  }
}
