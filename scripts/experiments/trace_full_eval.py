#!/usr/bin/env python3
"""Cross-domain generalization: the CARM checkpoint trained ONLY on
DeepAccident (`carm_deepaccident_v1.pt`, native-anchored target) evaluated
ZERO-SHOT on TRACE (github.com/NahianSalsabil/carla-benchmark, 52 real
NHTSA-crash reconstructions, kinematically rebuilt into trajectories by
`futurerisk/adapters/trace.py` -- see that module's docstring for every
reconstruction judgment call and caveat before trusting these numbers).

Produces the TRACE-domain analogues of the paper's two tables, covering ALL
8 baselines + CARM (not a subset):

  Table 1' (instance_consistency, TRACE)   -- AUROC/AUPRC vs. the
      geometric collision label (see trace.py docstring, judgment call 5:
      this is NOT an independent native label the way DeepAccident's is).
  Table 2' (finegrained_baselines, TRACE)  -- Spearman rho(method, a_max
      proxy) and rho(method, -m proxy), collision-free TRACE subset, same
      `spearman_block` convention/sign-flip as `v1_full_eval.py`.

Every baseline is scored zero-shot: rule-based methods (y_analytic, raw
3D-TTC, Forward TTC 2D/1D, PET, RSS) are pure functions of the reconstructed
BoxTrajectory, no fitting. Lu et al. (2022) is fit ONCE on DeepAccident
TRAIN (same seed/recipe as v1_full_eval.py) and scored zero-shot on TRACE
-- never fit on any TRACE data. GSSM uses the released SHRP2 checkpoint,
zero-shot, same as everywhere else in this project. CARM uses the
DeepAccident-only v1 checkpoint, zero-shot.

IMPORTANT caveats to report alongside any number this script prints (see
`futurerisk/adapters/trace.py` module docstring for the full justification
of each):
  1. Trajectories are a kinematic RECONSTRUCTION (constant speed along an
     extended polyline, solved for simultaneous impact timing), not a
     CARLA physics replay -- evasive-severity/a_max signal is dominated by
     path curvature, not braking, and is weaker evidence than
     DeepAccident's own simulated physics.
  2. The collision label is geometric (SAT overlap on paths explicitly
     routed through the annotated crash_point), not an independent
     native label -- close to tautological with the ground truth for the
     window containing the impact, same caveat as this project's own
     Bench2Drive adapter.
  3. n=52 independent underlying scenarios, not one per windowed instance
     (273 total windows) -- every reported n is a window count, not an
     independent-sample count; treat significance/CI claims accordingly.
"""
from __future__ import annotations

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
TRACE_PREPROCESSED = "dataset/TRACE-preprocessed"
CHECKPOINT_PATH = "dataset/checkpoints/carm_deepaccident_v1.pt"
RESULTS_DIR = Path("dataset/results_trace")
LOGS_DIR = Path("dataset/logs")
DT = 0.1
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
# Table 1/2 in the paper report these 8 + CARM (DRAC is computed/available
# but not one of the 8 requested rows, kept internal for the Lu et al.
# feature builder only, same role it plays in v1_full_eval.py).
TABLE_METHOD_ORDER = [
    "y_analytic", "raw_ttc_score", "fwd_ttc_2d_score", "fwd_ttc_1d_score",
    "pet_score", "rss_score", "lu_score", "gssm_score", "R_predicted",
]


def setup_logging(log_path: Path):
    log = logging.getLogger("trace_full_eval")
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
    """Fit Lu et al.'s reproduction on DeepAccident TRAIN ONLY -- exactly
    v1_full_eval.py's recipe/seed -- so it is evaluated zero-shot on TRACE,
    never fit on any TRACE data (same protocol as CARM and GSSM)."""
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


def score_trace_instance(model, cfg, pre) -> dict:
    raw = raw_instance_from_preprocessed(pre)
    R_pred = predict_R(model, cfg, raw, pre)

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

    ssm = instance_ssm_scores(raw.ego_box, raw.agent_boxes, pre.times, cfg.dt)
    pet_sc = float(pet_score(np.array([ssm["pet_min"]]))[0])
    rss_sc = instance_rss_score(raw.ego_box, raw.agent_boxes, pre.interaction_type, cfg.dt)

    geometric_collision_label = int(any(f.any() for f in raw.agent_collision_flags.values()))

    if pre.tau_3d:
        worst_agent = min(pre.tau_3d, key=lambda a: np.min(pre.tau_3d[a]))
        drac_t = deceleration_rate_to_avoid_crash(raw.ego_box, raw.agent_boxes[worst_agent], cfg.dt)
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
    log = setup_logging(LOGS_DIR / f"trace_full_eval_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    log.info("[1/5] Loading CARM v1 checkpoint (DeepAccident-only, zero-shot on TRACE) ...")
    model, cfg, ckpt = load_checkpoint()

    log.info("[2/5] Fitting Lu et al. reproduction on DeepAccident TRAIN (never on TRACE) ...")
    lu = fit_lu_on_deepaccident_train(cfg, log)

    log.info(f"[3/5] Scoring every TRACE instance (zero-shot) from {TRACE_PREPROCESSED} ...")
    rows, lu_feats, gssm_boundaries, gssm_X, gssm_s = [], [], [], [], []
    cursor = 0
    for p in sorted(Path(TRACE_PREPROCESSED).glob("*.npz")):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        row, lu_feat, gssm_rows = score_trace_instance(model, cfg, pre)
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
    log.info(f"      -> {len(rows)} TRACE windowed instances from {n_scenarios} underlying scenarios scored")

    X_lu = np.stack(lu_feats)
    for r, s in zip(rows, lu.score(X_lu)):
        r["lu_score"] = float(s)

    log.info("[4/5] Scoring GSSM (released SHRP2 checkpoint, zero-shot, one batched subprocess call) ...")
    try:
        X_all = np.concatenate(gssm_X, axis=0)
        s_all = np.concatenate(gssm_s, axis=0)
        intensity_all = score_rows_via_subprocess(X_all, s_all, batch_tag="trace_full_eval")
        for r, (lo, hi) in zip(rows, gssm_boundaries):
            r["gssm_score"] = instance_gssm_score_from_intensity(intensity_all[lo:hi]) if hi > lo else 0.0
    except GSSMUnavailableError as exc:
        log.error(f"GSSM scoring failed: {exc}")
        for r in rows:
            r["gssm_score"] = None

    import csv
    with open(RESULTS_DIR / "predictions_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    log.info("[5/5] Table 1' (AUROC/AUPRC vs geometric collision label) and Table 2' (rho vs a_max / -m) ...")
    results = {
        "n_windows": len(rows), "n_underlying_scenarios": n_scenarios,
        "caveats": [
            "trajectories are a kinematic reconstruction, not a CARLA physics replay (futurerisk/adapters/trace.py)",
            "collision label is geometric (SAT overlap on paths routed through the annotated crash_point), not an independent native label",
            "instances from the same scenario are correlated -- n_windows != n_independent_samples",
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

    with open(RESULTS_DIR / "trace_full_eval_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"wrote {RESULTS_DIR / 'trace_full_eval_summary.json'}")
    log.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
