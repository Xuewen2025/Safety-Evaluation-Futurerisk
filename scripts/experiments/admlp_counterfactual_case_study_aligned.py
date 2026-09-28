#!/usr/bin/env python3
"""AD-MLP open-loop counterfactual case study, re-run with the SAME
reference-frame-selection methodology as tcp_counterfactual_case_study.py /
uniad_counterfactual_case_study.py / vad_counterfactual_case_study.py: scan
several candidate reference frames per clip and keep the one with the
largest own-prediction L2 error, instead of admlp_counterfactual_case_study.py's
earliest-valid-frame rule.

Purpose: admlp_counterfactual_case_studies.json (earliest valid frame) and
the TCP/UniAD/VAD counterfactual files (worst-divergence frame) are NOT
methodologically comparable to each other -- AD-MLP's real trajectories
land on an arbitrary, typically uneventful moment while the other three
land on each planner's own hardest moment, which is why AD-MLP's real-R
baseline (0.611) looks so much higher than the other three's (0.20-0.24).
This script produces an AD-MLP dataset selected the same way as the other
three, so a planner-level comparison (mean R, substituted collision rate)
is no longer confounded by that difference.

Does NOT overwrite admlp_counterfactual_case_studies.json (used by the
existing instance-level ranking-accuracy table) or case_study_failure.png
(a fixed single diagnosed instance, independent of this file) -- writes to
a distinctly named output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "Bench2DriveZoo"))

from futurerisk.adapters.bench2drive import load_clip_frames
from futurerisk.config import DEFAULT_CONFIG
from scripts.experiments.admlp_case_study import build_admlp_input, load_admlp_model, predict_future_global
from scripts.experiments.admlp_counterfactual_case_study import build_real_and_counterfactual, score_instance
from scripts.evaluate_crossdomain_v1_to_bench2drive import load_checkpoint

MIN_I = 20
FUTURE_FRAMES = 30  # 3.0s at 10Hz, matching AD-MLP's 6-waypoint/0.5s-spacing output
MIN_SPEED = 1.0
STRIDE = 20


def candidate_reference_frames(frames):
    """Every STRIDE-th frame with a full [-20, +30]-frame window (AD-MLP
    needs 2s of history AND its own 3s future window, unlike UniAD/VAD/TCP
    which only need the forward window) and non-degenerate ego speed."""
    idxs = sorted(frames.keys())
    out = []
    for i in idxs[MIN_I::STRIDE]:
        if frames[i].get("speed", 0.0) < MIN_SPEED:
            continue
        if i - 20 < idxs[0]:
            continue
        if i + FUTURE_FRAMES <= idxs[-1] and all((i + d) in frames for d in range(-20, FUTURE_FRAMES + 1)):
            out.append(i)
    return out


def main():
    admlp_net = load_admlp_model()
    carm_model, carm_cfg, ckpt = load_checkpoint()

    clip_files = sorted(Path("dataset/Bench2Drive-Base").glob("*.tar.gz"))
    candidates = [f for f in clip_files if any(k in f.name for k in ("Accident", "Crossing", "Junction", "Cut"))]
    print(f"{len(candidates)} candidate clips")

    results = []
    for f in candidates[:15]:
        clip_name, frames = load_clip_frames(str(f))
        ref_frames = candidate_reference_frames(frames)
        if not ref_frames:
            print(f"  {clip_name}: skip (no candidate reference frames)")
            continue
        best = None
        for i in ref_frames:
            xy_pred, _ = predict_future_global(admlp_net, frames, i)
            real_future = np.array([[frames[i + d]["x"], frames[i + d]["y"]] for d in [5, 10, 15, 20, 25, 30]])
            l2 = np.linalg.norm(xy_pred - real_future, axis=1).mean()
            if best is None or l2 > best[2]:
                best = (i, xy_pred, l2)
        i, xy_pred, l2_mean = best
        try:
            real_inst, cf_inst = build_real_and_counterfactual(str(f), i, xy_pred, DEFAULT_CONFIG)
        except ValueError as e:
            print(f"  {clip_name}: skip ({e})")
            continue
        real_score = score_instance(carm_model, carm_cfg, real_inst)
        cf_score = score_instance(carm_model, carm_cfg, cf_inst)
        row = {
            "clip_name": clip_name, "ref_frame": i, "admlp_l2_mean": float(l2_mean),
            "n_agents": real_score["n_agents"],
            "R_real": real_score["R"], "collision_real": real_score["geometric_collision"],
            "R_admlp_counterfactual": cf_score["R"], "collision_admlp_counterfactual": cf_score["geometric_collision"],
        }
        results.append(row)
        print(f"  {clip_name}  ref={i}  n_agents={row['n_agents']}  admlp_L2={row['admlp_l2_mean']:.2f}m"
              f"  R_real={row['R_real']:.3f}(coll={row['collision_real']})"
              f"  R_admlp_cf={row['R_admlp_counterfactual']:.3f}(coll={row['collision_admlp_counterfactual']})")

    out_path = Path("dataset/results_v1/admlp_counterfactual_case_studies_aligned.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out_path} ({len(results)} scenarios)")


if __name__ == "__main__":
    main()
