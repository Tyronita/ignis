"""
Spatial fire-dataset export with provenance — Ignis as a data generator.

Every rollout is recorded as per-step voxel fire-state + agent action + outcome metrics,
written as a compressed NPZ plus a JSON provenance manifest (seed, config, scene source,
git SHA, timestamp, license). This backs the "future data-collection plan" and the event's
spatial-data-with-transparent-provenance theme.

  python -m ignis.dataset --episodes 5 --out datasets/indoor
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return "unknown"


def _downsample_occ(burning, burned, d=2):
    """Compact per-step channels (fire, burned) at reduced resolution (uint8)."""
    def red(a):
        nx, ny, nz = a.shape
        px, py, pz = (-nx) % d, (-ny) % d, (-nz) % d
        a = np.pad(a.astype(np.float32), ((0, px), (0, py), (0, pz)))
        nx, ny, nz = a.shape
        return (a.reshape(nx//d, d, ny//d, d, nz//d, d).max(axis=(1, 3, 5)) > 0).astype(np.uint8)
    return red(burning > 0), red(burned)


def export_indoor_episode(seed, out_dir, down=2, steps=90, policy=None):
    """Record one indoor rollout to NPZ + manifest. Returns the manifest dict."""
    from .indoor import suppress, generate
    scene = generate.generate_scene(seed=seed)
    st = suppress.init(scene)
    rng = np.random.default_rng(seed)
    fire_frames, burned_frames, actions, metrics = [], [], [], []
    W = None if policy is None else np.load(policy)
    for _ in range(steps):
        a = suppress.N_ACT - 1 if W is None else suppress.policy_action(W, suppress.observe(st))
        suppress.apply_action(st, a)
        suppress.sim_step(st, rng)
        f, b = _downsample_occ(st["burning"], st["burned"], down)
        fire_frames.append(f); burned_frames.append(b); actions.append(a)
        from .indoor import indoor_env as ie
        metrics.append(ie.metrics(st))
        if (st["burning"] > 0).sum() == 0 and st["t"] > 5:
            break

    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"episode_{seed:04d}.npz")
    np.savez_compressed(npz, fire=np.array(fire_frames, np.uint8),
                        burned=np.array(burned_frames, np.uint8),
                        actions=np.array(actions, np.int16),
                        dims=np.array(scene.dims), down=down)
    manifest = {
        "episode": os.path.basename(npz),
        "generator": "ignis.indoor.generate.generate_scene",
        "seed": int(seed),
        "scene_source": "procedural",
        "dims": list(scene.dims),
        "cell_size_m": scene.cell_size_m,
        "downsample": down,
        "steps_recorded": len(actions),
        "policy": os.path.basename(policy) if policy else "none (no-suppression baseline)",
        "final_metrics": metrics[-1] if metrics else {},
        "channels": ["fire", "burned"],
        "action_space": f"discrete zoned sprinklers ({suppress.ZONES[0]}x{suppress.ZONES[1]} + no-op)",
        "provenance": {
            "git_sha": _git_sha(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tool": "Ignis",
            "license": "MIT",
        },
    }
    with open(os.path.join(out_dir, f"episode_{seed:04d}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="datasets/indoor")
    ap.add_argument("--policy", default=None)
    args = ap.parse_args()
    index = []
    for i in range(args.episodes):
        m = export_indoor_episode(args.seed0 + i, args.out, policy=args.policy)
        index.append(m)
        print(f"  wrote {m['episode']}  burned={m['final_metrics'].get('burned_pct',0):.0f}%")
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump({"n_episodes": len(index), "episodes": index}, fh, indent=2)
    print(f"\nwrote {len(index)} episodes + index.json + per-episode provenance to {args.out}/")


if __name__ == "__main__":
    main()
