"""
PPO (deep RL) on Ignis-Indoor-v0, and a CEM-vs-PPO-vs-baseline comparison.

PPO trains an MLP policy with clipped policy-gradient + GAE — a genuine deep-RL
algorithm — on the same Gym environment CEM optimises, so the comparison is
apples-to-apples (same env/reward, different learner + policy class).

  python -m ignis.indoor.train_ppo --steps 120000

Note: CPU-only here (torch was downgraded to +cpu by the SB3 install). Small env, fine.
"""

import argparse
import os
import numpy as np


def train(total_steps=120_000, n_envs=6, seed=0):
    import gymnasium as gym
    import ignis.gym_env  # noqa: registers envs
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    vec = make_vec_env("Ignis-Indoor-v0", n_envs=n_envs, seed=seed,
                       env_kwargs={"seed_pool": 6})
    model = PPO("MlpPolicy", vec, verbose=0, seed=seed,
                n_steps=256, batch_size=256, gae_lambda=0.95, gamma=0.99,
                ent_coef=0.01, learning_rate=3e-4)
    print(f"training PPO (MlpPolicy) for {total_steps} steps on {n_envs} envs ...")
    model.learn(total_timesteps=total_steps)
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    model.save(os.path.join(root, "ppo_indoor"))
    print("saved ppo_indoor.zip")
    return model


def _eval(model, k=10, seed0=700):
    """Baseline vs CEM vs PPO on the SAME held-out generated scenes → mean fuel saved."""
    from . import suppress, generate
    from ..gym_env import IgnisIndoorEnv
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    Wcem = np.load(os.path.join(root, "policy_indoor.npy")) \
        if os.path.exists(os.path.join(root, "policy_indoor.npy")) else None
    base, cem, ppo = [], [], []
    for i in range(k):
        sc = generate.generate_scene(seed=seed0 + i)
        base.append(suppress.rollout(None, sc, np.random.default_rng(i))[0])
        if Wcem is not None:
            cem.append(suppress.rollout(Wcem, sc, np.random.default_rng(i))[0])
        env = IgnisIndoorEnv(scene=sc)
        obs, _ = env.reset(seed=i)
        done = False; info = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(a))
            done = term or trunc
        ppo.append(info.get("fuel_saved", 0.0))
    return (np.mean(base), np.mean(cem) if cem else float("nan"), np.mean(ppo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--n_envs", type=int, default=6)
    args = ap.parse_args()
    model = train(args.steps, args.n_envs)
    b, c, p = _eval(model)
    print(f"\n=== held-out fuel saved ===\nbaseline {b*100:.1f}%   CEM {c*100:.1f}%   PPO {p*100:.1f}%")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = ["no-agent", "CEM", "PPO (deep RL)"]; vals = [b*100, c*100, p*100]
        colors = ["#7f8c8d", "#e67e22", "#2980b9"]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar(names, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10)
        ax.set_ylabel("% fuel saved (held-out)")
        ax.set_title("Suppression: CEM vs PPO vs baseline (same Gym env)")
        ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "cem_vs_ppo.png")
        fig.savefig(out, dpi=120); print("saved assets/cem_vs_ppo.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
