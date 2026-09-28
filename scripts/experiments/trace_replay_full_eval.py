#!/usr/bin/env python3
"""Cross-domain generalization: the CARM checkpoint trained ONLY on
DeepAccident (`carm_deepaccident_v1.pt`, native-anchored target) evaluated
ZERO-SHOT on TRACE-replay -- real CARLA 0.9.15 physics replays of all 52
TRACE (github.com/NahianSalsabil/carla-benchmark) scenarios, produced by
`Reconstruction-Pipeline/scripts/ns_launch_scene_bm_logging.py` (the TRACE
authors' own benchmark-replay script, instrumented to log per-tick state)
run headless via `-nullrhi` (no GPU rendering at all -- this host's NVIDIA
driver has no GL/Vulkan userspace libraries, so ordinary `-RenderOffScreen`
still fails to initialize a render device; `-nullrhi` bypasses rendering
entirely and only needed the physics/game-logic path, which is all this
evaluation requires).

This is the SAME table-production script as `trace_full_eval.py` (the
kinematic-reconstruction version), pointed at
`futurerisk/adapters/trace_replay.py`'s preprocessed output instead. The
key methodological difference this fixes (see trace_replay.py's module
docstring): the collision label here is CARLA's own
`sensor.other.collision` event, independent of any TTC/geometry formula --
not the geometric-SAT-overlap label `trace.py` had to fall back to, which
was close to tautological with several rule-based baselines. Real
closed-loop vehicle dynamics (actual acceleration/braking, not a
constant-speed assumption) also mean the `-m` (margin) proxy in Table 2'
should no longer be uniformly +inf/undefined the way it was for every
method in the kinematic-reconstruction version.

Every baseline is scored zero-shot, identically to trace_full_eval.py:
rule-based methods are pure functions of the (now real-physics)
BoxTrajectory; Lu et al. is fit ONCE on DeepAccident TRAIN (never on any
TRACE data); GSSM uses its released SHRP2 checkpoint zero-shot; CARM uses
the DeepAccident-only v1 checkpoint zero-shot.

Caveats that still apply (see trace_replay.py docstring):
  1. Both vehicles are always spawned as `vehicle.tesla.model3` regardless
     of the real crash report's actual vehicle models/sizes (the TRACE
     authors' own simplification, not introduced here).
  2. n=52 independent underlying scenarios, not one per windowed instance
     -- every reported n is a window count.
  3. A `BasicAgent` closed-loop controller replayed each scenario's given
     waypoints/speed at real physics -- not every one of the 52 reproduced
     an actual collision once real control/dynamics were in the loop (see
     the per-scenario `status` field); non-colliding replays are kept as
     genuine negatives, not discarded.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futurerisk.baselines_extra import LuEtAlReproduction, build_ssm_sequence_features, instance_rss_score
from futurerisk.baselines_gssm import (
    GSSMUnavailableError, build_gssm_instance_rows, instance_gssm_score_from_intensity,
    score_rows_via_subprocess,
)
from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import discrimination_metrics, raw_ttc_baseline_score
from futurerisk.forward_ttc_baseline import forward_ttc_all_agents
from futurerisk.safety_prior import analytic_trajectory_risk, context_dependent_margin
from futurerisk.ssm_baselines import deceleration_rate_to_avoid_crash, drac_score, instance_ssm_scores, pet_score
from scripts.evaluate_crossdomain_v1_to_bench2drive import load_checkpoint, predict_R
from scripts.experiments.v1_full_eval import severity_proxies

DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
TRACE_PREPROCESSED = "dataset/TRACE-replay-preprocessed"
TRACE_REPLAY_DIR = "external/carla-benchmark/Benchmark/replay"

# Judgment call (added after manually inspecting the outlier rows in a first
# run of this script): 4/49 replays -- 510026, 510131, 510230, 510332 -- hit
# CARLA's 3000-tick (150s) cap without completing their route or colliding
# (`status == "no_collision_tick_cap_reached"`). Direct inspection of
# replay_510332.json's per-tick vx/vy (e.g. tick 995-999: vx swings
# -38 -> -40 -> -17 -> +10 -> +15 m/s within 5 ticks/0.25s) shows the vehicle
# stuck oscillating/spinning in place, not driving -- consistent with these
# being the ONLY 4 scenarios that ran the full 3000 ticks (every completed
# or collided scenario stopped well under 200 ticks). Differentiating this
# noise twice for evasive-severity/margin blows up to physically-impossible
# values (observed max_evasive_severity=573333, vs. real hard braking's
# single-digit-to-~15 m/s^2 range), and all 4 contribute exactly 116 windows
# each -- 464 of the 491 collision-free windows (94.5%) -- so they dominate
# Table 2' (both the a_max sign and the suspiciously-exact rho_vs_neg_m=+-1.0
# from a single one of these 4 scenarios, 510332, being the sole source of
# every finite worst_margin point in the un-filtered run). Excluded here from
# BOTH tables: they are simulation artifacts, not genuine safe-driving
# negatives, and their risk *scores* (not just the severity proxies) are
# built from the same corrupted velocity signal, so Table 1' isn't safe to
# trust for these windows either. The 1 scenario that legitimately completed
# its route without colliding (`no_collision_route_completed`, 1 window) is
# kept as a genuine negative.
EXCLUDED_STATUSES = {"no_collision_tick_cap_reached"}
CHECKPOINT_PATH = "dataset/checkpoints/carm_deepaccident_v1.pt"
RESULTS_DIR = Path("dataset/results_trace_replay")
LOGS_DIR = Path("dataset/logs")
DT = 1.0 / 20.0  # CARLA's own synchronous-mode tick rate used for the replay, NOT the model's 0.1s
TTC_CAP = 5.0

METHOD_LABELS = {
    "R_predicted": "FutureRisk (CARM)",
    "y_analytic": "y_analytic",
    "raw_ttc_score": "raw 3D-TTC",
    "fwd_ttc_2d_score": "Forward TTC (2-D)",
    "fwd_ttc_1d_score": "Forward TTC (1-D)",
    "pet_score": "PET",
    "rss_score": "RSS",
    "lu_score": "Lu et al.",
    "gssm_score": "GSSM (Jiao et al., zero-shot)",
}
TABLE_METHOD_ORDER = [
    "y_analytic", "raw_ttc_score", "fwd_ttc_2d_score", "fwd_ttc_1d_score",
    "pet_score", "rss_score", "lu_score", "gssm_score", "R_predicted",
]


def load_scenario_status(replay_dir: str) -> dict:
    """crash_id (str) -> replay `status`, read directly from
    `Benchmark/replay/replay_<crash_id>.json` (not carried through the
    preprocessing pipeline, since `RawTrajectoryInstance` has no field for
    it -- see `EXCLUDED_STATUSES` for why this is needed here)."""
    out = {}
    for p in Path(replay_dir).glob("replay_*.json"):
        data = json.loads(p.read_text())
        out[str(data["crash_id"])] = data["status"]
    return out


def setup_logging(log_path: Path):
    log = logging.getLogger("trace_replay_full_eval")
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


def fit_lu_on_deepaccident_train(cfg, log, seed: int = 0) -> LuEtAlReproduction:
    import torch
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    split = ckpt["split"]
    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)

    feats, ttc_summary = [], []
    for iid in split["train"]:
        p = Path(DEEPACCIDENT_PREPROCESSED) / f"{iid}.npz"
        if not p.exists():
            continue
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None or not pre.tau_3d:
            continue
        entry = native_labels.get(iid)
        if entry is None or entry.get("collision_label") is None:
            continue
        worst_agent = min(pre.tau_3d, key=lambda a: np.min(pre.tau_3d[a]))
        raw = raw_instance_from_preprocessed(pre)
        drac_t = deceleration_rate_to_avoid_crash(raw.ego_box, raw.agent_boxes[worst_agent], cfg.dt)
        feats.append(build_ssm_sequence_features(pre.tau_3d[worst_agent], drac_t, pre.evasive_severity[worst_agent]))
        ttc_summary.append(min(float(np.min(pre.tau_3d[worst_agent])), cfg.ttc_cap))
    X_train = np.stack(feats)
    lu = LuEtAlReproduction(input_dim=X_train.shape[1], latent_dim=8, n_clusters=4, seed=seed)
    lu.fit(X_train, np.array(ttc_summary), epochs=80)
    log.info(f"      Lu et al. fit on {len(feats)} DeepAccident-train instances")
    return lu


# Bug fix (found by direct inspection after CARM scored suspiciously low on
# a first run of this script): `cfg` here is the checkpoint's TRAINING
# config, `cfg.dt=0.1` (10Hz, matching DeepAccident/Bench2Drive) -- but
# TRACE-replay's cached raw trajectories are natively 20Hz (REPLAY_DT=0.05,
# see trace_replay.py/preprocess_trace_replay.py). Every OTHER baseline in
# this file is already called with the correct `DT=1/20` (see
# forward_ttc_all_agents/instance_ssm_scores/instance_rss_score/
# deceleration_rate_to_avoid_crash below); only CARM's own `predict_R` path
# was missed. `build_agent_feature_v1` (features_v1.py) uses `cfg.dt` for
# TWO things: `np.gradient(x, cfg.dt)` (velocity/accel/yaw-rate -- wrong dt
# scales these by dt_used/dt_actual, verified as exactly 0.5x for velocity
# on a real instance) and `_bin_sample_indices` (`idx = round(bin_edge_s /
# cfg.dt)`, converting the model's fixed-in-seconds bin edges into an array
# index -- with the wrong dt this only reaches half the intended horizon_s,
# verified as 1.9s of real elapsed time instead of the intended 3.8s on the
# same instance). Neither `CARM.__init__` (only reads cfg.num_bins/
# delta_max/sigma_min/sigma_max/beta_residual/composition_eps, checked
# directly in carm_model.py) nor `aggregate_instance_risk` depend on dt, so
# it's safe to keep the model/checkpoint's `cfg` as-is everywhere else (e.g.
# `cfg.T_m`, `cfg.ttc_cap` just below) and only swap `dt` for the feature
# tensors CARM actually consumes -- `cfg_feat` below, a `dt=DT` copy of
# `cfg`, built once in `main()` and passed here for that sole purpose.
def score_trace_instance(model, cfg, cfg_feat, pre) -> dict:
    raw = raw_instance_from_preprocessed(pre)
    R_pred = predict_R(model, cfg_feat, raw, pre)

    margins = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
    y_analytic = analytic_trajectory_risk(margins, T_m=cfg.T_m) if margins else 0.0

    tau3d_min = min((np.min(pre.tau_3d[a]) for a in pre.tau_3d), default=np.inf)
    ttc_score = float(raw_ttc_baseline_score(np.array([tau3d_min]), ttc_cap=cfg.ttc_cap)[0])

    tau_2d = forward_ttc_all_agents(raw.ego_box, raw.agent_boxes, dt=DT, cap=TTC_CAP, ignore_turning=False)
    tau_1d = forward_ttc_all_agents(raw.ego_box, raw.agent_boxes, dt=DT, cap=TTC_CAP, ignore_turning=True)
    min_tau_2d = min((float(np.min(t)) for t in tau_2d.values()), default=np.inf)
    min_tau_1d = min((float(np.min(t)) for t in tau_1d.values()), default=np.inf)
    fwd_2d_score = float(raw_ttc_baseline_score(np.array([min_tau_2d]), ttc_cap=TTC_CAP)[0])
    fwd_1d_score = float(raw_ttc_baseline_score(np.array([min_tau_1d]), ttc_cap=TTC_CAP)[0])

    ssm = instance_ssm_scores(raw.ego_box, raw.agent_boxes, pre.times, DT)
    pet_sc = float(pet_score(np.array([ssm["pet_min"]]))[0])
    rss_sc = instance_rss_score(raw.ego_box, raw.agent_boxes, pre.interaction_type, DT)

    geometric_collision_label = int(any(f.any() for f in raw.agent_collision_flags.values()))

    if pre.tau_3d:
        worst_agent = min(pre.tau_3d, key=lambda a: np.min(pre.tau_3d[a]))
        drac_t = deceleration_rate_to_avoid_crash(raw.ego_box, raw.agent_boxes[worst_agent], DT)
        lu_feat = build_ssm_sequence_features(pre.tau_3d[worst_agent], drac_t, pre.evasive_severity[worst_agent])
    else:
        lu_feat = np.zeros(3 * 16, dtype=np.float32)

    gssm_rows = build_gssm_instance_rows(raw.ego_box, raw.agent_boxes, DT)

    row = {
        "instance_id": pre.instance_id,
        "scenario_id": pre.instance_id.split("__w")[0],
        "R_predicted": R_pred, "y_analytic": float(y_analytic), "raw_ttc_score": ttc_score,
        "fwd_ttc_2d_score": fwd_2d_score, "fwd_ttc_1d_score": fwd_1d_score,
        "pet_score": pet_sc, "rss_score": rss_sc,
        "geometric_collision_label": geometric_collision_label,
    }
    row.update(severity_proxies(pre))
    return row, lu_feat, gssm_rows


def spearman_block(rows: list, score_keys: list, proxy_keys: list, collision_key: str, log) -> dict:
    free = [r for r in rows if r[collision_key] == 0]
    n_scenarios = len(set(r["scenario_id"] for r in free))
    out = {"n_windows": len(free), "n_underlying_scenarios": n_scenarios}
    for proxy in proxy_keys:
        vals = np.array([r[proxy] for r in free])
        finite = np.isfinite(vals)
        for score in score_keys:
            scores = np.array([r[score] for r in free])[finite]
            v = vals[finite]
            key = f"{score}_vs_{proxy}"
            if len(v) < 10 or np.all(v == v[0]) or np.all(scores == scores[0]):
                out[key] = None
                continue
            rho, p = spearmanr(scores, v)
            sign = -1.0 if proxy == "worst_margin" else 1.0
            out[key] = {"n": int(len(v)), "spearman_rho": float(sign * rho), "p_value": float(p)}
    return out


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(LOGS_DIR / f"trace_replay_full_eval_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    log.info("[1/5] Loading CARM v1 checkpoint (DeepAccident-only, zero-shot on TRACE-replay) ...")
    model, cfg, ckpt = load_checkpoint()
    # cfg.dt=0.1 (10Hz, the training rate) -- TRACE-replay's cached raw
    # trajectories are natively 20Hz. cfg_feat is a dt-corrected copy used
    # ONLY for building CARM's input tensors (see score_trace_instance's
    # docstring comment for the bug this fixes); every other use of `cfg`
    # below (Lu et al. fitting, T_m, ttc_cap) is dt-independent and
    # deliberately still uses the checkpoint's real `cfg`.
    cfg_feat = dataclasses.replace(cfg, dt=DT)

    log.info("[2/5] Fitting Lu et al. reproduction on DeepAccident TRAIN (never on TRACE) ...")
    lu = fit_lu_on_deepaccident_train(cfg, log)

    status_by_crash_id = load_scenario_status(TRACE_REPLAY_DIR)
    excluded_ids = {cid for cid, s in status_by_crash_id.items() if s in EXCLUDED_STATUSES}
    log.info(f"      Excluding {len(excluded_ids)} scenario(s) with status in {EXCLUDED_STATUSES}: {sorted(excluded_ids)}")

    log.info(f"[3/5] Scoring every TRACE-replay instance (zero-shot) from {TRACE_PREPROCESSED} ...")
    rows, lu_feats, gssm_boundaries, gssm_X, gssm_s = [], [], [], [], []
    n_skipped = 0
    cursor = 0
    for p in sorted(Path(TRACE_PREPROCESSED).glob("*.npz")):
        crash_id = p.stem.split("__w")[0].replace("tracereplay_", "")
        if crash_id in excluded_ids:
            n_skipped += 1
            continue
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        row, lu_feat, gssm_rows = score_trace_instance(model, cfg, cfg_feat, pre)
        rows.append(row)
        lu_feats.append(lu_feat)
        if gssm_rows is None:
            gssm_boundaries.append((cursor, cursor))
        else:
            X, s = gssm_rows
            gssm_X.append(X)
            gssm_s.append(s)
            gssm_boundaries.append((cursor, cursor + len(s)))
            cursor += len(s)
    n_scenarios = len(set(r["scenario_id"] for r in rows))
    n_collisions = sum(r["geometric_collision_label"] for r in rows)
    log.info(f"      -> {len(rows)} TRACE-replay windowed instances from {n_scenarios} underlying scenarios scored"
             f" ({n_collisions} positive / {len(rows)-n_collisions} negative), {n_skipped} windows skipped"
             f" from {len(excluded_ids)} excluded scenario(s)")

    X_lu = np.stack(lu_feats)
    for r, s in zip(rows, lu.score(X_lu)):
        r["lu_score"] = float(s)

    log.info("[4/5] Scoring GSSM (released SHRP2 checkpoint, zero-shot, one batched subprocess call) ...")
    try:
        X_all = np.concatenate(gssm_X, axis=0)
        s_all = np.concatenate(gssm_s, axis=0)
        intensity_all = score_rows_via_subprocess(X_all, s_all, batch_tag="trace_replay_full_eval")
        for r, (lo, hi) in zip(rows, gssm_boundaries):
            r["gssm_score"] = instance_gssm_score_from_intensity(intensity_all[lo:hi]) if hi > lo else 0.0
    except GSSMUnavailableError as exc:
        log.error(f"GSSM scoring failed: {exc}")
        for r in rows:
            r["gssm_score"] = None

    import csv
    with open(RESULTS_DIR / "predictions_trace_replay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    log.info("[5/5] Table 1' (AUROC/AUPRC vs CARLA collision-sensor label) and Table 2' (rho vs a_max / -m) ...")
    results = {
        "n_windows": len(rows), "n_underlying_scenarios": n_scenarios,
        "caveats": [
            "collision label is CARLA's own sensor.other.collision event -- independent of any TTC/geometry formula",
            "both vehicles always spawned as vehicle.tesla.model3 regardless of the real crash report's vehicle models (TRACE authors' own simplification)",
            "instances from the same scenario are correlated -- n_windows != n_independent_samples",
            "not every scenario's closed-loop BasicAgent replay reproduced an actual collision -- non-colliding replays kept as real negatives",
            f"excluded {len(excluded_ids)} scenario(s) with status in {sorted(EXCLUDED_STATUSES)} ({sorted(excluded_ids)}) "
            "-- these hit the 3000-tick cap with the vehicle stuck oscillating in place (inspected directly), "
            "not driving; see EXCLUDED_STATUSES docstring in this script",
        ],
    }

    lbl = np.array([r["geometric_collision_label"] for r in rows])
    table1 = {}
    for key in TABLE_METHOD_ORDER:
        scores = np.array([r[key] if r.get(key) is not None else np.nan for r in rows])
        if np.any(np.isnan(scores)):
            table1[METHOD_LABELS[key]] = None
            continue
        m = discrimination_metrics(scores, lbl)
        table1[METHOD_LABELS[key]] = {"auroc": m["auroc"], "auprc": m["auprc"]}
        log.info(f"      [Table 1'] {METHOD_LABELS[key]:32s} AUROC={m['auroc']:.3f}  AUPRC={m['auprc']:.3f}")
    results["table1_instance_consistency"] = {
        "n": len(rows), "n_collisions": int(lbl.sum()), "n_underlying_scenarios": n_scenarios,
        "methods": table1,
    }

    valid_keys = [k for k in TABLE_METHOD_ORDER if all(r.get(k) is not None for r in rows)]
    fine = spearman_block(rows, valid_keys, ["max_evasive_severity", "worst_margin"], "geometric_collision_label", log)
    table2 = {}
    for key in TABLE_METHOD_ORDER:
        a_max_entry = fine.get(f"{key}_vs_max_evasive_severity")
        m_entry = fine.get(f"{key}_vs_worst_margin")
        table2[METHOD_LABELS[key]] = {"rho_vs_a_max": a_max_entry, "rho_vs_neg_m": m_entry}
        rho_a = a_max_entry["spearman_rho"] if a_max_entry else float("nan")
        rho_m = m_entry["spearman_rho"] if m_entry else float("nan")
        log.info(f"      [Table 2'] {METHOD_LABELS[key]:32s} rho_vs_a_max={rho_a:+.3f}  rho_vs_-m={rho_m:+.3f}")
    results["table2_finegrained"] = {
        "n_windows": fine["n_windows"], "n_underlying_scenarios": fine["n_underlying_scenarios"],
        "methods": table2,
    }

    with open(RESULTS_DIR / "trace_replay_full_eval_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"wrote {RESULTS_DIR / 'trace_replay_full_eval_summary.json'}")
    log.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
