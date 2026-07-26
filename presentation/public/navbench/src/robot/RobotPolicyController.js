// Deliberately naive "bug" navigation policy: move forward; if the forward
// probe is blocked, rotate to a random new heading and try again. No mapping,
// no planning, no wall-following. This is the point — it's what a policy
// trained on predictable factory-floor layouts looks like when it meets a
// room it has never seen: it does not fail gracefully, it just gets stuck.
const FORWARD_SPEED = 95; // px/s
const TURN_RATE = Math.PI * 1.6; // rad/s while turning
const PROBE_EXTRA = 16;
const TURN_CREEP = FORWARD_SPEED * 0.15;

export class RobotPolicyController {
  constructor() {
    this.reset();
  }

  reset() {
    this.state = 'forward';
    this.targetHeading = 0;
    this.turnDir = 1;
  }

  decide(robot, environment) {
    const probeDist = robot.radius + PROBE_EXTRA;
    const px = robot.x + Math.cos(robot.heading) * probeDist;
    const py = robot.y + Math.sin(robot.heading) * probeDist;
    const blocked = environment.isBlocked(px, py, robot.radius * 1.1);

    if (this.state === 'forward') {
      if (!blocked) return { linear: FORWARD_SPEED, angular: 0 };
      this.state = 'turning';
      const delta = (Math.random() < 0.5 ? -1 : 1) * (Math.PI / 3 + Math.random() * ((Math.PI * 2) / 3)); // 60-180deg
      this.targetHeading = robot.heading + delta;
      this.turnDir = Math.sign(delta) || 1;
    }

    const diff = this._angleDiff(this.targetHeading, robot.heading);
    if (Math.abs(diff) < 0.05) {
      this.state = 'forward';
      return { linear: FORWARD_SPEED, angular: 0 };
    }
    return { linear: TURN_CREEP, angular: this.turnDir * TURN_RATE };
  }

  _angleDiff(a, b) {
    let d = (a - b) % (Math.PI * 2);
    if (d > Math.PI) d -= Math.PI * 2;
    if (d < -Math.PI) d += Math.PI * 2;
    return d;
  }
}
