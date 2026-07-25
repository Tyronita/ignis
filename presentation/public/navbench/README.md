# NAVBENCH — Navigation Policy Robustness Benchmark

A browser-based research visualization, not a game. It exists to make one
question visible: **how do navigation policies trained on predictable
layouts fail when they leave them, and how would richer, real-environment
behavioral data change that?**

A deliberately naive "bug" policy — move forward, randomly rotate when
blocked, no mapping, no planning — is run against hundreds of procedurally
generated rooms. Every room is verified navigable before the episode starts,
so failures are attributable to the policy, not to an unsolvable room.

## Running it

ES modules require a real origin, not `file://`. From this directory:

```sh
python3 -m http.server 8080
# then open http://localhost:8080
```

## Architecture

```
src/
  util/rng.js                    seeded PRNG (mulberry32) — every room is reproducible from its seed
  environment/
    Obstacle.js                  axis-aligned rect + circle-intersection test
    collision.js                 shared isBlocked/hasReachedExit — the robot's sensing and the
                                  simulation's authoritative collision/success checks both call this,
                                  so they can never disagree about what "blocked" means
    EnvironmentGenerator.js       randomizes room dims, obstacle count/size/placement/density/corridors,
                                  exit position — then verifies navigability with a grid BFS (obstacles
                                  expanded by the robot's radius) before accepting the room. A failed
                                  layout is re-rolled, not patched.
  robot/
    Robot.js                     position/heading/velocity state + integration step
    RobotPolicyController.js     the naive bug policy (forward-probe sensing, random rotate when blocked)
    HumanController.js           same decide()/reset() interface, driven by WASD/arrow keys instead —
                                  the seam the future human-data mode hangs off of
  simulation/
    TrajectoryRecorder.js        per-timestep recording in the shape of a real trajectory dataset
    MetricsCollector.js          running stats across all episodes (lightweight, no per-timestep data)
    SimulationRunner.js          episode loop: generate room, run controller vs. physics, detect
                                  collision/success/timeout, hand off to metrics/heatmap, start next episode
  viz/
    Heatmap.js                   collision locations normalized to each room's own [0,1]x[0,1] space
                                  before binning, so rooms of different sizes are comparable
    Renderer.js                  draws the room/obstacles/exit/trajectory/robot each frame
    UI.js                        side panel + controls, wired to callbacks from main.js
  export/
    DatasetExporter.js           downloads episode(s) as JSON
  main.js                        wires all of the above together; owns the recent-episode buffer
                                  and the render loop
```

## Design decisions worth knowing about

- **Navigability is guaranteed by verification, not by construction.** The
  generator places obstacles randomly, then runs a BFS on a grid where
  obstacles are expanded by the robot's radius (so a "path" actually fits the
  robot, not just a mathematical point). If the exit isn't reachable from
  spawn, the room is discarded and re-rolled — with a shrinking obstacle
  budget after repeated failures, so generation can never hang.
- **The heatmap is room-relative, not pixel-absolute.** Room dimensions
  change every episode, so a raw-pixel heatmap would be meaningless. Collision
  coordinates are normalized to `[0,1]×[0,1]` of that episode's own room
  before being binned into a fixed grid.
- **The controller interface is the seam for human data collection.**
  `RobotPolicyController` and `HumanController` expose the same
  `decide(robot, environment) -> {linear, angular}` / `reset()` shape.
  Swapping the dropdown in the UI swaps the controller with zero changes to
  `SimulationRunner`. Recording a human's `decide()` outputs through the exact
  same `TrajectoryRecorder` — in the exact same procedurally generated rooms —
  is what would turn this into imitation-learning demonstration data. Not
  implemented here, deliberately.
- **Full per-timestep data is only kept for a bounded recent window**
  (`RECENT_FULL_EPISODES_CAP` in `main.js`, default 40). `MetricsCollector`
  keeps lightweight summaries for every episode ever run, so running stats
  stay cheap across hundreds of episodes even though full trajectories
  aren't retained forever.
- **Speed is a tick multiplier, not a separate batch-mode code path.** 1× is
  real-time; 40× / 150× fast-forward through many physics ticks per rendered
  frame, so "run hundreds of episodes, displayed quickly" is the same loop
  turned up, not a second simulation implementation to keep in sync.

## Exported dataset shape

Per timestep: `{ timestamp, position: {x,y}, heading, velocity, angular_velocity, collided }`.
Per episode (attached once, not repeated per timestep): `room_seed`,
`room_width`/`room_height`, `obstacle_count`, `obstacle_density`,
`obstacle_layout`, `exit`, `result`, `success`, `runtime`, `total_distance`,
`collision_location`, `timesteps`.
