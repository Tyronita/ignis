"""
Fire-safety scoring + safe-building design (see SAFETY.md for the standards).

Estimates the performance-based criterion ASET − RSET from the simulation, reports the
computed safety variables, rolls them into a 0–100 score, and demonstrates a
code-minimum vs. optimised building comparison.

  python -m ignis.indoor.safety            # score the house + an optimised variant
"""

import copy
import json
import os
import numpy as np
from . import importer, suppress, evacuate, indoor_env as ie
from . import materials as M

HEAD_M = 1.7            # tenability assessed at head height
DETECTION_S = 20.0      # smoke-alarm detection delay (NFPA 72 typical)
PREMOVE_S = 15.0        # pre-movement / recognition time
UNTENABLE = 0.30        # fraction of floor hazard/smoke = untenable
HOUSE = os.path.join(os.path.dirname(__file__), "examples", "house.json")


def compute_aset(scene, seed=0, steps=160):
    """Available Safe Egress Time (s): free-burn until the floor becomes untenable."""
    st = suppress.init(scene)
    rng = np.random.default_rng(seed)
    interior = (scene.room_id[:, :, 1] >= 0)
    ni = max(1, int(interior.sum()))
    for k in range(steps):
        suppress.sim_step(st, rng)
        hazard = (st["burning"] > 0).any(axis=2)
        smoke = st["flamed_air"].any(axis=2) if "flamed_air" in st else np.zeros_like(hazard)
        cov = ((hazard | smoke) & interior).sum() / ni
        if cov > UNTENABLE:
            return (k + 1) * evacuate.DT_S
    return steps * evacuate.DT_S


def compute_rset(scene, n_agents=8, seed=0):
    """Required Safe Egress Time (s) = detection + pre-movement + travel to exit."""
    st = suppress.init(scene)
    ev = evacuate.Evacuation(scene, n_agents=n_agents, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        suppress.sim_step(st, rng)
        m = ev.step(st)
        if m["still_inside"] == 0:
            break
    travel = m["all_out_s"] if m["all_out_s"] >= 0 else 200.0
    return DETECTION_S + PREMOVE_S + travel, m


def _max_travel_m(scene):
    """Longest exit-access travel distance from any spawnable floor cell (metres)."""
    ev = evacuate.Evacuation(scene, n_agents=1, seed=0)
    dist = ev._distance_field(np.zeros(scene.solid[:, :, 1].shape, bool))
    reachable = dist[dist < (1 << 29)]
    return float(reachable.max() * scene.cell_size_m) if len(reachable) else -1.0


def safety_report(scene, seed=0):
    aset = compute_aset(scene, seed)
    rset, evm = compute_rset(scene, seed=seed)
    margin = aset - rset
    upholstered = int(((scene.material == M.id_of("foam")) | (scene.material == M.id_of("fabric"))).sum())
    rated_walls = float(((scene.solid) & np.isin(scene.material,
                        [M.id_of("concrete"), M.id_of("drywall")])).sum() / max(1, scene.solid.sum()))
    vars_ = dict(
        n_exits=len(scene.exits),
        max_travel_m=round(_max_travel_m(scene), 1),
        sprinklered=scene.budgets.get("water", 0) > 0,
        rated_boundary_frac=round(rated_walls, 2),
        upholstered_voxels=upholstered,
        ASET_s=round(aset, 1), RSET_s=round(rset, 1), margin_s=round(margin, 1),
        escaped=evm["escaped"], occupants=evm["occupants"], casualties=evm["casualties"],
    )
    # 0–100 score: margin (safe if >0), everyone out, no casualties, short travel, sprinklered
    score = (35 * (1 / (1 + np.exp(-margin / 20)))            # ASET-RSET margin (sigmoid)
             + 30 * (evm["escaped"] / max(1, evm["occupants"]))
             + 15 * (1 - evm["casualties"] / max(1, evm["occupants"]))
             + 10 * (scene.budgets.get("water", 0) > 0)
             + 10 * max(0, 1 - vars_["max_travel_m"] / 30))
    vars_["safety_score"] = round(float(score), 1)
    vars_["code_ok"] = bool(vars_["n_exits"] >= 2 and vars_["sprinklered"] and margin > 0)
    return vars_


def _optimised(base_dict):
    """A safer design: add a 2nd exit on the far side + upgrade upholstery to low-FSI."""
    d = copy.deepcopy(base_dict)
    nx, ny, nz = d["dims"]
    d.setdefault("exits", []).append([nx - 12, ny - 1])          # 2nd exit, opposite corner
    d["doors"].append({"at": [nx - 12, ny - 1], "width": 3, "height": 7})
    for f in d["furniture"]:                                     # Class-A finish swap for soft goods
        if f["material"] in ("foam", "fabric"):
            f["material"] = "drywall"
    return d


def main():
    base_dict = json.load(open(HOUSE))
    base = importer.from_floorplan(base_dict)
    opt = importer.from_floorplan(_optimised(base_dict))
    rb, ro = safety_report(base), safety_report(opt)

    print("=== Safe building design — code-minimum vs optimised ===")
    keys = ["n_exits", "max_travel_m", "sprinklered", "upholstered_voxels",
            "ASET_s", "RSET_s", "margin_s", "casualties", "code_ok", "safety_score"]
    print(f"{'variable':20s}{'code-min':>14s}{'optimised':>14s}")
    for k in keys:
        print(f"{k:20s}{str(rb[k]):>14s}{str(ro[k]):>14s}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        labels = ["ASET s", "RSET s", "margin s", "casualties", "safety score"]
        b = [rb["ASET_s"], rb["RSET_s"], rb["margin_s"], rb["casualties"], rb["safety_score"]]
        o = [ro["ASET_s"], ro["RSET_s"], ro["margin_s"], ro["casualties"], ro["safety_score"]]
        x = np.arange(len(labels)); w = 0.38
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(x - w/2, b, w, label="code-minimum", color="#c0392b")
        ax.bar(x + w/2, o, w, label="optimised (2nd exit + Class-A)", color="#2ecc71")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend()
        ax.set_title("Safe building design — ASET−RSET & outcomes"); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "safety.png")
        fig.savefig(out, dpi=120); print("saved assets/safety.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
