"""
Gymnasium environments so anyone can `gym.make(...)` Ignis and train their own agents.

    import gymnasium as gym
    import ignis.gym_env            # registers the envs
    env = gym.make("Ignis-Indoor-v0")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Two envs:
  Ignis-Wildfire-v0  — 2D wildfire; discrete air-tanker moves (stay/up/down/left/right).
                       (Prior art: JaxWildfire, arXiv:2512.06102 — cited; this is a CPU/NumPy env.)
  Ignis-Indoor-v0    — 3D structural fire; discrete zoned-sprinkler action (novel: JaxWildfire
                       is 2D and excludes structural fire).

Gymnasium is an optional dependency; importing this module without it raises a clear error.
Multi-agent (PettingZoo) and a JAX/Gymnax backend are documented scale-up paths.
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError("Ignis Gym envs need `gymnasium` — pip install gymnasium") from e

from . import fire_env as fe


class IgnisWildfireEnv(gym.Env):
    """2D wildfire suppression. Reward = fuel saved this step (− newly burned fraction)."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, n=48, terrain=False, max_steps=None):
        super().__init__()
        from .config import SimConfig
        SimConfig(n=n).apply()
        if terrain:
            fe.set_terrain(fe.make_terrain(3))
        self.max_steps = max_steps or fe.T
        self.observation_space = spaces.Box(-np.inf, np.inf, (fe.N_FEAT,), np.float32)
        self.action_space = spaces.Discrete(fe.N_ACTIONS)
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.s = fe.init_state(1, self._rng)
        self.t = 0
        self._prev_saved = float(fe.fuel_saved(self.s)[0])
        return fe.features(self.s)[0], {}

    def step(self, action):
        fe.step(self.s, np.array([int(action)], np.int32), self._rng)
        self.t += 1
        saved = float(fe.fuel_saved(self.s)[0])
        reward = saved - self._prev_saved           # positive only if we prevent burning
        self._prev_saved = saved
        terminated = bool((self.s["burning"] > 0).sum() == 0 and self.t > 3)
        truncated = self.t >= self.max_steps
        info = {"fuel_saved": saved}
        return fe.features(self.s)[0], float(reward), terminated, truncated, info

    def render(self):
        return fe.render(self.s)


class IgnisIndoorEnv(gym.Env):
    """3D structural-fire suppression via zoned ceiling sprinklers.

    Built on the same VoxelScene sim; the suppression mechanic + observation come from
    ignis.indoor.suppress so training and the Gym env share one code path.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, scene=None, seed_pool=8, max_steps=90):
        super().__init__()
        from .indoor import suppress, generate
        self._sup = suppress
        self._scene = scene
        self._pool = generate.sample_pool(seed_pool) if scene is None else [scene]
        self.max_steps = max_steps
        gx, gy = suppress.ZONES
        self.action_space = spaces.Discrete(gx * gy + 1)   # +1 = no-op
        self.observation_space = spaces.Box(0.0, 1.0, (gx * gy + 1,), np.float32)
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        scene = self._scene or self._pool[int(self._rng.integers(len(self._pool)))]
        self.st = self._sup.init(scene)
        self.t = 0
        self._prev = self._sup.fuel_saved(self.st)
        return self._sup.observe(self.st), {}

    def step(self, action):
        self._sup.apply_action(self.st, int(action))
        self._sup.sim_step(self.st, self._rng)
        self.t += 1
        saved = self._sup.fuel_saved(self.st)
        reward = saved - self._prev
        self._prev = saved
        terminated = bool((self.st["burning"] > 0).sum() == 0 and self.t > 5)
        truncated = self.t >= self.max_steps
        return self._sup.observe(self.st), float(reward), terminated, truncated, {"fuel_saved": saved}


def _register():
    from gymnasium.envs.registration import register, registry
    for env_id, entry in [("Ignis-Wildfire-v0", "ignis.gym_env:IgnisWildfireEnv"),
                          ("Ignis-Indoor-v0", "ignis.gym_env:IgnisIndoorEnv")]:
        if env_id not in registry:
            register(id=env_id, entry_point=entry)


_register()
