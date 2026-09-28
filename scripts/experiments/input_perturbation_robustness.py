#!/usr/bin/env python3
"""Experiment 2 (root.tex Sec. IV-E, "input-perturbation robustness"):
does CARM's R swing sharply under small, realistic input perturbations that
should NOT change the risk assessment much -- additive coordinate noise
(sensor/logging jitter), a changed resampling rate, and simulated
multi-object-tracking timestamp jitter?

Instance sample: a FIXED, deterministic subset of the v1 checkpoint's own
held-out test split (`ckpt["split"]["test"]`, 1656 instances) -- every
20th instance in the split's own (fixed, seeded) order, i.e.
`split["test"][::20]`, giving 83 instances. No cherry-picking: this is a
plain stride subsample of an already-random, already-fixed split.

Perturbation types (each compared against the ORIGINAL, already-cached
R for the same instance, computed once from `load_preprocessed_instance` +
`predict_R` with no modification):

  (a) Additive coordinate noise: i.i.d. N(0, sigma^2) added independently
      to every (x, y) at every timestep, for ego AND every surrounding
      agent, at sigma in {0.1, 0.3, 0.5} meters (typical logged-perception
      position-noise magnitudes). M=15 repeats per (instance, sigma) with
      independent, deterministically-seeded RNGs (seeded from a hash of
      instance_id/sigma/repeat_index, so the whole sweep is reproducible
      without a global RNG state threaded through the loop).

  (b) Changed resampling rate: 2x (a denser grid, linear interpolation of
      x/y/z and a heading-unwrap-then-interpolate-then-rewrap of heading,
      onto twice as many frames spanning the identical time window) and
      0.5x (exact subsampling -- every other original frame, no
      interpolation at all) of the original dt=0.1s. Both require `cfg.dt`
      (and `cfg.ttc_delta_step`, which `preprocess_instance` also passes as
      the per-frame spacing into `compute_3d_ttc`) to be updated to match
      the new inter-frame spacing -- `cfg.horizon_s`/`cfg.num_bins` (hence
      the model's bin grid) are left untouched, so the checkpoint's
      architecture still applies unmodified. Chosen over "resample back
      onto the original dt grid" because that would just reproduce the
      original instance for the 2x case (interpolate denser, then
      immediately re-sample back to the original points) and would throw
      away exactly the effect being tested (dt-dependent finite-difference
      derivatives feeding evasive severity / closing speed / tau_phy)
      for the 0.5x case's real information loss.

  (c) Simulated sensor jitter: each SURROUNDING AGENT's track (never
      ego's own -- ego's pose is the observing platform's own odometry,
      not a tracked object subject to multi-object-tracking
      desynchronization) is independently time-shifted by
      tau_shift ~ Uniform(-0.3, 0.3) seconds and re-interpolated back onto
      the common time grid (x/y/z linear, heading via
      unwrap-interpolate-rewrap; timestamps outside the original span hold
      the boundary value, `np.interp`'s default edge behavior). M=15
      repeats per instance with independent per-repeat, per-agent draws.

All raw per-instance/per-condition/per-repeat R values (both R_original and
R_perturbed) are saved to the results JSON; the summary numbers (mean/std/
max |R_perturbed - R_original|, aggregated over all instances and repeats
within each condition) are reported plainly, including if some condition
turns out not to be robust -- this experiment is a stress test, not a
confirmation exercise.
"""
from __future__ import annotations

import dataclasses
import hashlib
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

from futurerisk.config import FutureRiskConfig
from futurerisk.data_pipeline import (
    RawTrajectoryInstance, load_preprocessed_instance, preprocess_instance, raw_instance_from_preprocessed,
)
from futurerisk.event_proxy import load_threshold_tables
from futurerisk.geometry import BoxTrajectory
from scripts.evaluate_crossdomain_v1_to_bench2drive import load_checkpoint, predict_R

DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
THRESHOLD_TABLE_PATH = "dataset/DeepAccident-preprocessed/DeepAccident_thresholds.json"
RESULTS_DIR = Path("dataset/results_v1")
FIG_DIR = Path("icra/figures")
LOGS_DIR = Path("dataset/logs")

TEST_STRIDE = 20            # split["test"][::20] -> 83 instances
NOISE_SIGMAS = (0.1, 0.3, 0.5)
NOISE_REPEATS = 15
JITTER_MAX_SHIFT_S = 0.3
JITTER_REPEATS = 15


def setup_logging(log_path: Path):
    log = logging.getLogger("input_perturbation_robustness")
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


def seeded_rng(*parts) -> np.random.Generator:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def _copy_box(box: BoxTrajectory, x=None, y=None) -> BoxTrajectory:
    return BoxTrajectory(
        x=box.x.copy() if x is None else x, y=box.y.copy() if y is None else y,
        z=box.z.copy(), heading=box.heading.copy(),
        length=box.length.copy(), width=box.width.copy(), height=box.height.copy(),
    )


def add_position_noise(raw: RawTrajectoryInstance, sigma: float, rng: np.random.Generator) -> RawTrajectoryInstance:
    ego = raw.ego_box
    new_ego = _copy_box(ego, x=ego.x + rng.normal(0.0, sigma, size=ego.x.shape),
                         y=ego.y + rng.normal(0.0, sigma, size=ego.y.shape))
    new_agents = {}
    for aid, box in raw.agent_boxes.items():
        new_agents[aid] = _copy_box(box, x=box.x + rng.normal(0.0, sigma, size=box.x.shape),
                                     y=box.y + rng.normal(0.0, sigma, size=box.y.shape))
    return dataclasses.replace(raw, ego_box=new_ego, agent_boxes=new_agents)


def _interp_box_onto(box: BoxTrajectory, src_times: np.ndarray, dst_times: np.ndarray) -> BoxTrajectory:
    x = np.interp(dst_times, src_times, box.x)
    y = np.interp(dst_times, src_times, box.y)
    z = np.interp(dst_times, src_times, box.z)
    heading_unwrapped = np.unwrap(box.heading)
    heading = np.interp(dst_times, src_times, heading_unwrapped)
    heading = (heading + np.pi) % (2 * np.pi) - np.pi
    length = np.interp(dst_times, src_times, box.length)
    width = np.interp(dst_times, src_times, box.width)
    height = np.interp(dst_times, src_times, box.height)
    return BoxTrajectory(x=x, y=y, z=z, heading=heading, length=length, width=width, height=height)


def upsample_double(raw: RawTrajectoryInstance, cfg: FutureRiskConfig) -> tuple:
    times = raw.times
    new_times = np.linspace(times[0], times[-1], 2 * (len(times) - 1) + 1)
    new_ego = _interp_box_onto(raw.ego_box, times, new_times)
    new_agents = {aid: _interp_box_onto(box, times, new_times) for aid, box in raw.agent_boxes.items()}
    # nearest-original-frame lookup for the (unused-downstream, but
    # dataclass-required) collision flags
    nearest = np.clip(np.round(np.linspace(0, len(times) - 1, len(new_times))).astype(int), 0, len(times) - 1)
    new_flags = {aid: flags[nearest] for aid, flags in raw.agent_collision_flags.items()}
    new_raw = dataclasses.replace(raw, times=new_times, ego_box=new_ego, agent_boxes=new_agents,
                                   agent_collision_flags=new_flags)
    new_dt = float(new_times[1] - new_times[0])
    new_cfg = dataclasses.replace(cfg, dt=new_dt, ttc_delta_step=new_dt)
    return new_raw, new_cfg


def downsample_half(raw: RawTrajectoryInstance, cfg: FutureRiskConfig) -> tuple:
    idx = np.arange(0, len(raw.times), 2)
    new_times = raw.times[idx]
    new_ego = BoxTrajectory(x=raw.ego_box.x[idx], y=raw.ego_box.y[idx], z=raw.ego_box.z[idx],
                             heading=raw.ego_box.heading[idx], length=raw.ego_box.length[idx],
                             width=raw.ego_box.width[idx], height=raw.ego_box.height[idx])
    new_agents = {}
    for aid, box in raw.agent_boxes.items():
        new_agents[aid] = BoxTrajectory(x=box.x[idx], y=box.y[idx], z=box.z[idx], heading=box.heading[idx],
                                         length=box.length[idx], width=box.width[idx], height=box.height[idx])
    new_flags = {aid: flags[idx] for aid, flags in raw.agent_collision_flags.items()}
    new_raw = dataclasses.replace(raw, times=new_times, ego_box=new_ego, agent_boxes=new_agents,
                                   agent_collision_flags=new_flags)
    new_dt = float(new_times[1] - new_times[0]) if len(new_times) > 1 else cfg.dt
    new_cfg = dataclasses.replace(cfg, dt=new_dt, ttc_delta_step=new_dt)
    return new_raw, new_cfg


def jitter_agents(raw: RawTrajectoryInstance, max_shift_s: float, rng: np.random.Generator) -> RawTrajectoryInstance:
    times = raw.times
    new_agents = {}
    for aid, box in raw.agent_boxes.items():
        shift = rng.uniform(-max_shift_s, max_shift_s)
        # agent's true position at nominal time t is approximated by its
        # observed trace evaluated at (t - shift) -- a positive shift means
        # "recorded late" (position looks like it lags the true track).
        src_times = times - shift
        x = np.interp(times, src_times, box.x)
        y = np.interp(times, src_times, box.y)
        z = np.interp(times, src_times, box.z)
        heading_unwrapped = np.unwrap(box.heading)
        heading = np.interp(times, src_times, heading_unwrapped)
        heading = (heading + np.pi) % (2 * np.pi) - np.pi
        new_agents[aid] = BoxTrajectory(x=x, y=y, z=z, heading=heading, length=box.length.copy(),
                                         width=box.width.copy(), height=box.height.copy())
    return dataclasses.replace(raw, agent_boxes=new_agents)


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(LOGS_DIR / f"input_perturbation_robustness_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    log.info("[1/4] Loading v1 checkpoint + DeepAccident threshold table ...")
    model, cfg, ckpt = load_checkpoint()
    tables = load_threshold_tables(THRESHOLD_TABLE_PATH)
    table = tables["DeepAccident"]
    test_ids = ckpt["split"]["test"][::TEST_STRIDE]
    log.info(f"      {len(test_ids)} instances (every {TEST_STRIDE}th of {len(ckpt['split']['test'])} test instances)")

    per_instance = []
    for i, iid in enumerate(test_ids):
        p = Path(DEEPACCIDENT_PREPROCESSED) / f"{iid}.npz"
        if not p.exists():
            log.info(f"      SKIP {iid}: no cached preprocessing file")
            continue
        pre0 = load_preprocessed_instance(str(p))
        if pre0.ego_box is None:
            continue
        raw0 = raw_instance_from_preprocessed(pre0)
        R_orig = predict_R(model, cfg, raw0, pre0)

        entry = {"instance_id": iid, "n_agents": len(raw0.agent_boxes), "R_original": R_orig,
                 "noise": {str(s): [] for s in NOISE_SIGMAS}, "jitter": [], "resample_2x": None,
                 "resample_0p5x": None}

        # (a) additive coordinate noise
        for sigma in NOISE_SIGMAS:
            for rep in range(NOISE_REPEATS):
                rng = seeded_rng(iid, "noise", sigma, rep)
                raw_p = add_position_noise(raw0, sigma, rng)
                pre_p = preprocess_instance(raw_p, table, cfg=cfg)
                R_p = predict_R(model, cfg, raw_p, pre_p)
                entry["noise"][str(sigma)].append(R_p)

        # (b) resampling rate
        raw_2x, cfg_2x = upsample_double(raw0, cfg)
        pre_2x = preprocess_instance(raw_2x, table, cfg=cfg_2x)
        entry["resample_2x"] = predict_R(model, cfg_2x, raw_2x, pre_2x)

        raw_half, cfg_half = downsample_half(raw0, cfg)
        pre_half = preprocess_instance(raw_half, table, cfg=cfg_half)
        entry["resample_0p5x"] = predict_R(model, cfg_half, raw_half, pre_half)

        # (c) sensor jitter (per-agent time-shift)
        for rep in range(JITTER_REPEATS):
            rng = seeded_rng(iid, "jitter", rep)
            raw_j = jitter_agents(raw0, JITTER_MAX_SHIFT_S, rng)
            pre_j = preprocess_instance(raw_j, table, cfg=cfg)
            R_j = predict_R(model, cfg, raw_j, pre_j)
            entry["jitter"].append(R_j)

        per_instance.append(entry)
        if (i + 1) % 10 == 0 or i + 1 == len(test_ids):
            log.info(f"      ... {i+1}/{len(test_ids)} instances done ({time.time()-t0:.1f}s elapsed)")

    log.info("[2/4] Aggregating |R_perturbed - R_original| per condition ...")
    conditions = {}
    for sigma in NOISE_SIGMAS:
        diffs = []
        for e in per_instance:
            diffs.extend(abs(r - e["R_original"]) for r in e["noise"][str(sigma)])
        diffs = np.array(diffs)
        conditions[f"noise_sigma_{sigma}"] = {
            "n": int(len(diffs)), "mean_abs_dR": float(diffs.mean()), "std_abs_dR": float(diffs.std()),
            "max_abs_dR": float(diffs.max()), "p95_abs_dR": float(np.percentile(diffs, 95)),
        }
    jitter_diffs = np.array([abs(r - e["R_original"]) for e in per_instance for r in e["jitter"]])
    conditions["sensor_jitter"] = {
        "n": int(len(jitter_diffs)), "mean_abs_dR": float(jitter_diffs.mean()), "std_abs_dR": float(jitter_diffs.std()),
        "max_abs_dR": float(jitter_diffs.max()), "p95_abs_dR": float(np.percentile(jitter_diffs, 95)),
    }
    for tag, key in (("resample_2x", "resample_2x"), ("resample_0p5x", "resample_0p5x")):
        diffs = np.array([abs(e[key] - e["R_original"]) for e in per_instance])
        conditions[tag] = {
            "n": int(len(diffs)), "mean_abs_dR": float(diffs.mean()), "std_abs_dR": float(diffs.std()),
            "max_abs_dR": float(diffs.max()), "p95_abs_dR": float(np.percentile(diffs, 95)),
        }

    for name, m in conditions.items():
        log.info(f"      {name:20s} n={m['n']:5d}  mean|dR|={m['mean_abs_dR']:.4f}"
                 f"  std={m['std_abs_dR']:.4f}  max={m['max_abs_dR']:.4f}  p95={m['p95_abs_dR']:.4f}")

    results = {
        "test_stride": TEST_STRIDE, "n_instances": len(per_instance),
        "noise_sigmas": list(NOISE_SIGMAS), "noise_repeats": NOISE_REPEATS,
        "jitter_max_shift_s": JITTER_MAX_SHIFT_S, "jitter_repeats": JITTER_REPEATS,
        "per_instance": per_instance,
        "condition_summary": conditions,
    }
    out_path = RESULTS_DIR / "input_perturbation_robustness.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"[3/4] wrote {out_path}")

    log.info("[4/4] Rendering figure ...")
    labels = [f"noise\nsigma={s}m" for s in NOISE_SIGMAS] + ["sensor\njitter", "resample\n2x", "resample\n0.5x"]
    box_data = [
        [abs(r - e["R_original"]) for e in per_instance for r in e["noise"][str(s)]] for s in NOISE_SIGMAS
    ] + [
        [abs(r - e["R_original"]) for e in per_instance for r in e["jitter"]],
        [abs(e["resample_2x"] - e["R_original"]) for e in per_instance],
        [abs(e["resample_0p5x"] - e["R_original"]) for e in per_instance],
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bp = ax.boxplot(box_data, labels=labels, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)
    ax.set_ylabel("|R_perturbed - R_original|")
    ax.set_title(f"CARM R robustness to input perturbations (n={len(per_instance)} DeepAccident test instances)",
                 fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = FIG_DIR / "input_perturbation.png"
    fig.savefig(fig_path, dpi=200)
    log.info(f"      wrote {fig_path}")

    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
