// Same decide()/reset() interface as RobotPolicyController, so
// SimulationRunner can swap between them with zero changes elsewhere. This is
// the seam the "future human data mode" hangs off of: point this controller's
// output at the TrajectoryRecorder instead of the policy's, and a human's
// keystrokes become demonstration data in the exact same recorded format.
export class HumanController {
  constructor() {
    this.keys = {};
    this._onDown = (e) => { this.keys[e.key.toLowerCase()] = true; };
    this._onUp = (e) => { this.keys[e.key.toLowerCase()] = false; };
    window.addEventListener('keydown', this._onDown);
    window.addEventListener('keyup', this._onUp);
  }

  reset() {}

  decide() {
    let linear = 0;
    let angular = 0;
    if (this.keys['arrowup'] || this.keys['w']) linear = 110;
    if (this.keys['arrowdown'] || this.keys['s']) linear = -60;
    if (this.keys['arrowleft'] || this.keys['a']) angular -= 3.2;
    if (this.keys['arrowright'] || this.keys['d']) angular += 3.2;
    return { linear, angular };
  }

  dispose() {
    window.removeEventListener('keydown', this._onDown);
    window.removeEventListener('keyup', this._onUp);
  }
}
