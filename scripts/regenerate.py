"""
Regenerate every artifact from scratch, then sync them into the presentation.

This is the maintained entry point — one command reproduces the whole submission:
  python scripts/regenerate.py

Steps: train (flat + terrain) -> save all GIFs/curves -> run indoor eval ->
copy assets/ into presentation/public/ so the React deck always shows fresh media.
"""

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = {**os.environ, "PYTHONPATH": ""}

STEPS = [
    ("train air-tanker (flat)",       [sys.executable, "-m", "ignis.train"]),
    ("train air-tanker (terrain)",    [sys.executable, "-m", "ignis.train3d"]),
    ("render 2D side-by-side gif",    [sys.executable, "-m", "ignis.viz2d", "save"]),
    ("render 3D terrain scene gif",   [sys.executable, "-m", "ignis.scene3d", "save"]),
    ("render 3D house voxel gif",     [sys.executable, "-m", "ignis.indoor.render3d"]),
    ("train indoor suppression (CEM)",[sys.executable, "-m", "ignis.indoor.suppress"]),
    ("synchronised plan+3D playthrough (fire+suppression+evacuation)",
        [sys.executable, "-m", "ignis.indoor.playthrough"]),
    ("two-class MARL (fire-engine responders vs civilians)",
        [sys.executable, "-m", "ignis.indoor.marl"]),
    ("safe building design (ASET-RSET) comparison",
        [sys.executable, "-m", "ignis.indoor.safety"]),
    ("hyperparameter comparison",
        [sys.executable, "-m", "ignis.indoor.hpo_compare"]),
    ("VLGE-shaped trajectory export",
        [sys.executable, "-m", "ignis.indoor.vlge_export"]),
    # PPO (deep RL) needs the `rl` extra (stable-baselines3); run separately:
    #   python -m ignis.indoor.train_ppo --steps 150000
    ("indoor baseline vs trained + compare gif",
        [sys.executable, "-m", "ignis.indoor.evals", "--pool", "8", "--policy", "policy_indoor.npy"]),
    ("export sample spatial dataset",
        [sys.executable, "-m", "ignis.dataset", "--episodes", "3", "--out", "datasets/indoor",
         "--policy", "policy_indoor.npy"]),
]


def run():
    for name, cmd in STEPS:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)

    # sync media into the React deck
    src = os.path.join(ROOT, "assets")
    dst = os.path.join(ROOT, "presentation", "public")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.lower().endswith((".gif", ".png")):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print("\nsynced assets -> presentation/public/")
    print("done. all artifacts under assets/.")


if __name__ == "__main__":
    run()
