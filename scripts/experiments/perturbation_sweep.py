#!/usr/bin/env python3
"""Experiment 1 (root.tex Sec. IV-E, "parametric perturbation sweep"):
a single, real DeepAccident test-split instance with a genuine native
collision, subjected to a controlled rigid lateral shift of ego's entire
path, sweeping the shift magnitude delta from 0 (the original, colliding
trajectory) up through a close call to a comfortably-safe trajectory --
comparing CARM's R against raw 3D-TTC thresholding and PET at every delta.

Instance selection (fixed, not re-run per invocation): among the v1
checkpoint's own held-out test split (`ckpt["split"]["test"]`), we scanned
every instance with a genuine DeepAccident native collision label and kept
those whose *worst* agent (min tau_3D over time) both (a) actually collides
geometrically and (b) has a "car-following" or "crossing" dominant
interaction type -- i.e. a single clear near-miss/collision partner, not a
diffuse multi-agent pileup, even though DeepAccident's own scenes often
carry a couple dozen background agent tracks. Picked the smallest-agent-
count car-following example so the "one scenario, one interaction" premise
holds as cleanly as possible:
    Town04_type001_subtype0002_000001__ego_vehicle_behind__r0060
(21 tracked agents total; the actual collision is a rear-end/close
sideswipe against agent id 18999, tau_3D counting straight down to 0.0 at
the final frame, i.e. a genuine end-of-window collision, not a mid-window
one -- see the selection scan in this experiment's design notes).

Perturbation applied (the "single scalar delta = ego trajectory DOF" from
the design): a *rigid parallel shift* of ego's entire (x, y) path by delta
meters, applied along ego's own per-frame heading normal
(nx, ny) = (-sin(heading), cos(heading)) -- i.e. purely lateral in ego's
own frame at every instant, computed from the *original* (unperturbed)
heading trace, which is itself left untouched, along with z and
length/width/height. Every surrounding agent's track, and the shared
`times` array, are left completely unchanged. This is a "what if ego had
been standing/driving delta meters further left/right the whole time"
counterfactual -- not a realistic control law, but a clean, single-DOF,
monotonic geometric knob for stress-testing r(t)-style metrics' smoothness,
exactly per the design brief.

At each delta we rebuild a `RawTrajectoryInstance` (only `ego_box`
replaced) and re-run `preprocess_instance` with the SAME DeepAccident
threshold table this instance's whole cache was built from (loaded from
`dataset/DeepAccident-preprocessed/DeepAccident_thresholds.json`, Algorithm
2's fixed preprocessing artifact) and the v1 checkpoint's own
`FutureRiskConfig` (horizon_s=4.0, num_bins=20, everything else default).
Three scores are recomputed from scratch at each delta:
  - CARM's R, via the v1 checkpoint (`predict_R`, identical convention to
    `scripts/evaluate_crossdomain_v1_to_bench2drive.py`);
  - the raw-3D-TTC baseline score (`evaluation.raw_ttc_baseline_score`) on
    the min tau_3D over agents and time;
  - PET (`ssm_baselines.instance_ssm_scores` -> `pet_score`).

`agent_collision_flags` is passed through unchanged from the cached
instance (needed only to satisfy `RawTrajectoryInstance`'s dataclass
contract and `preprocess_instance`'s internal T_evt computation, which none
of the three reported scores depend on -- verified by inspection: T_evt
never reaches `CARM.forward`, `raw_ttc_baseline_score`, or
`ssm_baselines.post_encroachment_time`, all three of which recompute
directly from box geometry). It is therefore never re-derived for the
shifted ego path, and is irrelevant to every number reported here.

Smoothness is reported both qualitatively (the figure) and quantitatively:
for each curve, the max absolute first difference between adjacent
delta-steps, and the curve's total variation (sum of absolute first
differences) over the full sweep.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futurerisk.data_pipeline import load_preprocessed_instance, preprocess_instance, raw_instance_from_preprocessed
from futurerisk.event_proxy import load_threshold_tables
from futurerisk.evaluation import raw_ttc_baseline_score
from futurerisk.geometry import BoxTrajectory
from futurerisk.safety_prior import context_dependent_margin
from futurerisk.ssm_baselines import instance_ssm_scores, pet_score
from scripts.evaluate_crossdomain_v1_to_bench2drive import load_checkpoint, predict_R

INSTANCE_ID = "Town04_type001_subtype0002_000001__ego_vehicle_behind__r0060"
DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
THRESHOLD_TABLE_PATH = "dataset/DeepAccident-preprocessed/DeepAccident_thresholds.json"
RESULTS_DIR = Path("dataset/results_v1")
FIG_DIR = Path("icra/figures")
LOGS_DIR = Path("dataset/logs")

DELTA_MIN, DELTA_MAX, N_DELTA = 0.0, 6.0, 55


def setup_logging(log_path: Path):
    log = logging.getLogger("perturbation_sweep")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def shifted_ego(ego: BoxTrajectory, delta: float) -> BoxTrajectory:
    """Rigid lateral shift of ego's path by `delta` meters, perpendicular
    to ego's own (unchanged) per-frame heading. Positive delta = left."""
    nx, ny = -np.sin(ego.heading), np.cos(ego.heading)
    return BoxTrajectory(
        x=ego.x + delta * nx, y=ego.y + delta * ny, z=ego.z.copy(), heading=ego.heading.copy(),
        length=ego.length.copy(), width=ego.width.copy(), height=ego.height.copy(),
    )


def smoothness_metrics(deltas: np.ndarray, values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=float)
    diffs = np.diff(finite)
    return {
        "max_abs_first_diff": float(np.max(np.abs(diffs))) if len(diffs) else 0.0,
        "total_variation": float(np.sum(np.abs(diffs))) if len(diffs) else 0.0,
        "is_monotonic_nonincreasing": bool(np.all(diffs <= 1e-9)),
        "is_monotonic_nondecreasing": bool(np.all(diffs >= -1e-9)),
    }


def normalize01(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    lo, hi = np.min(v), np.max(v)
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(LOGS_DIR / f"perturbation_sweep_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    log.info(f"[1/4] Loading v1 checkpoint + DeepAccident threshold table + instance {INSTANCE_ID} ...")
    model, cfg, ckpt = load_checkpoint()
    assert INSTANCE_ID in ckpt["split"]["test"], "selected instance must be in the checkpoint's own held-out test split"
    tables = load_threshold_tables(THRESHOLD_TABLE_PATH)
    table = tables["DeepAccident"]

    p = Path(DEEPACCIDENT_PREPROCESSED) / f"{INSTANCE_ID}.npz"
    pre0 = load_preprocessed_instance(str(p))
    raw0 = raw_instance_from_preprocessed(pre0)
    worst_agent0 = min(pre0.tau_3d, key=lambda a: np.min(pre0.tau_3d[a]))
    log.info(f"      n_agents={len(raw0.agent_boxes)}  T={len(raw0.times)}  worst_agent={worst_agent0}"
             f"  native_collision at delta=0 (min tau_3D)={np.min(pre0.tau_3d[worst_agent0]):.3f}")

    deltas = np.linspace(DELTA_MIN, DELTA_MAX, N_DELTA)
    log.info(f"[2/4] Sweeping delta in [{DELTA_MIN}, {DELTA_MAX}] ({N_DELTA} points) ...")

    R_vals, ttc_vals, pet_vals, min_tau3d_vals, worst_margin_vals, pet_raw_vals = [], [], [], [], [], []
    for i, delta in enumerate(deltas):
        new_ego = shifted_ego(raw0.ego_box, float(delta))
        raw = dataclasses.replace(raw0, ego_box=new_ego)
        pre = preprocess_instance(raw, table, cfg=cfg)

        min_tau3d = min(float(np.min(pre.tau_3d[a])) for a in pre.tau_3d)
        margins = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
        worst_margin = float(np.min(np.concatenate(margins)))

        R = predict_R(model, cfg, raw, pre)
        ttc_score = float(raw_ttc_baseline_score(np.array([min_tau3d]), ttc_cap=cfg.ttc_cap)[0])
        ssm = instance_ssm_scores(raw.ego_box, raw.agent_boxes, pre.times, cfg.dt)
        pet_sc = float(pet_score(np.array([ssm["pet_min"]]))[0])

        R_vals.append(R)
        ttc_vals.append(ttc_score)
        pet_vals.append(pet_sc)
        min_tau3d_vals.append(min_tau3d)
        worst_margin_vals.append(worst_margin)
        pet_raw_vals.append(ssm["pet_min"])
        if (i + 1) % 10 == 0 or i == 0:
            log.info(f"      delta={delta:5.2f}  min_tau3d={min_tau3d if np.isfinite(min_tau3d) else float('inf'):.3f}"
                     f"  R={R:.4f}  ttc_score={ttc_score:.3f}  pet_score={pet_sc:.3f}")

    R_vals = np.array(R_vals)
    ttc_vals = np.array(ttc_vals)
    pet_vals = np.array(pet_vals)

    log.info("[3/4] Smoothness metrics ...")
    smooth = {
        "R_predicted_CARM": smoothness_metrics(deltas, R_vals),
        "raw_ttc_score": smoothness_metrics(deltas, ttc_vals),
        "pet_score": smoothness_metrics(deltas, pet_vals),
    }
    for name, m in smooth.items():
        log.info(f"      {name:20s} max|d1|={m['max_abs_first_diff']:.4f}"
                 f"  total_variation={m['total_variation']:.4f}"
                 f"  monotonic(non-incr)={m['is_monotonic_nonincreasing']}")

    results = {
        "instance_id": INSTANCE_ID,
        "worst_agent": worst_agent0,
        "delta_min": DELTA_MIN, "delta_max": DELTA_MAX, "n_delta": N_DELTA,
        "deltas": deltas.tolist(),
        "raw": {
            "R_predicted_CARM": R_vals.tolist(),
            "raw_ttc_score": ttc_vals.tolist(),
            "pet_score": pet_vals.tolist(),
            "min_tau3d_over_agents": [None if not np.isfinite(v) else v for v in min_tau3d_vals],
            "worst_margin": [None if not np.isfinite(v) else v for v in worst_margin_vals],
            "pet_min_raw_seconds": [None if not np.isfinite(v) else v for v in pet_raw_vals],
        },
        "normalized_0_1_for_plot": {
            "R_predicted_CARM": normalize01(R_vals).tolist(),
            "raw_ttc_score": normalize01(ttc_vals).tolist(),
            "pet_score": normalize01(pet_vals).tolist(),
        },
        "smoothness_metrics": smooth,
    }
    out_path = RESULTS_DIR / "perturbation_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"      wrote {out_path}")

    log.info("[4/4] Rendering figure ...")
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 5.4), sharex=True)
    ax0, ax1 = axes

    ax0.plot(deltas, normalize01(R_vals), label="CARM (R)", color="#4C72B0", linewidth=1.8)
    ax0.plot(deltas, normalize01(ttc_vals), label="raw 3D-TTC", color="#C44E52", linewidth=1.4, linestyle="--")
    ax0.plot(deltas, normalize01(pet_vals), label="PET", color="#55A868", linewidth=1.4, linestyle=":")
    ax0.set_ylabel("score (min-max normalized)")
    ax0.set_title(f"Lateral-shift perturbation sweep\n({INSTANCE_ID})", fontsize=9)
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(alpha=0.3)

    ax1.plot(deltas, R_vals, label="CARM (R), raw scale", color="#4C72B0", linewidth=1.8)
    ax1.plot(deltas, ttc_vals, label="raw 3D-TTC, raw scale", color="#C44E52", linewidth=1.4, linestyle="--")
    ax1.plot(deltas, pet_vals, label="PET, raw scale", color="#55A868", linewidth=1.4, linestyle=":")
    ax1.set_xlabel("lateral shift delta (m)")
    ax1.set_ylabel("score (raw scale)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = FIG_DIR / "perturbation_sweep.png"
    fig.savefig(fig_path, dpi=200)
    log.info(f"      wrote {fig_path}")

    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
