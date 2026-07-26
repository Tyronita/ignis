export class Robot {
  constructor(x, y, heading, radius = 9) {
    this.x = x;
    this.y = y;
    this.heading = heading;
    this.radius = radius;
    this.linearVelocity = 0;
    this.angularVelocity = 0;
  }

  step(dt, command) {
    this.linearVelocity = command.linear;
    this.angularVelocity = command.angular;
    this.heading += this.angularVelocity * dt;
    this.x += Math.cos(this.heading) * this.linearVelocity * dt;
    this.y += Math.sin(this.heading) * this.linearVelocity * dt;
  }
}
