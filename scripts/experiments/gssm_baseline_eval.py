#!/usr/bin/env python3
"""GSSM (Jiao et al. 2026, Nature Machine Intelligence; arXiv:2505.13556)
baseline evaluation -- adds one new row, "GSSM (Jiao et al., zero-shot)",
to the two tables `v1_full_eval.py` already builds:

  Table I  (instance_consistency)   -- AUROC/AUPRC vs. binary collision
                                        label, DeepAccident in-domain
                                        (native ground truth, test split)
                                        + Bench2Drive cross-domain
                                        (geometric ground truth).
  Table II (finegrained_baselines)  -- Spearman's rho(GSSM, a(t)) on the
                                        collision-free DeepAccident subset
                                        (train+val+test pooled, same
                                        pooling `v1_full_eval.py` uses for
                                        this table), a(t) = the
                                        evasive-severity proxy
                                        (`max_evasive_severity`), same as
                                        the other 7 methods' rho(., a(t)).

This script does NOT recompute CARM/y_analytic/raw-TTC/PET/DRAC/RSS/Lu et
al. -- it scores GSSM on the EXACT SAME instances (same split, same native
labels file, same severity-proxy computation, same discrimination_metrics/
spearmanr calls) that `v1_full_eval.py` already used for those methods, so
the resulting numbers are a like-for-like new row, not a separately-defined
evaluation.

Path taken: released checkpoint, zero-shot (see `futurerisk/baselines_gssm.py`
module docstring for the full justification and the exact feature/score
definitions). No retraining, no fine-tuning, no collision label used in
computing the GSSM score at any point -- only in the discrimination/rho
metrics computed against it afterward, same as every other baseline here.

Honest scope notes (see also `dataset/results_v1/gssm_baseline_notes.md`):
- The released checkpoint's `current+acc` encoder needs 13 purely
  kinematic features per (ego, agent, frame); DeepAccident/Bench2Drive's
  `BoxTrajectory` fields supply all 13 with no invented data (see
  `futurerisk/baselines_gssm.py:gssm_current_acc_features` for the two
  documented approximations: ego longitudinal acceleration from
  finite-differenced speed, and velocity direction assumed equal to
  heading -- the latter matching the official repo's OWN training-time
  convention, not a simplification introduced here).
- GSSM's own `environment`/`profiles` encoder variants are NOT used
  (SHRP2-specific weather/lighting/road-surface categories and 2.5s
  kinematic-history windows have no faithful DeepAccident/Bench2Drive
  equivalent) -- this evaluates the SafeBaseline/current+acc checkpoint
  specifically, the most capable released variant usable with zero
  invented input.
- GSSM's own instance/frame-level score ("intensity") is aggregated to
  one instance-level scalar via max over (agent, frame) -- the same
  "worst agent, worst moment" convention `instance_rss_score`/
  `instance_ssm_scores` already use for RSS/PET/DRAC in this project.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futurerisk.baselines_gssm import (
    GSSMUnavailableError, build_gssm_instance_rows, instance_gssm_score_from_intensity,
    score_rows_via_subprocess,
)
from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import discrimination_metrics
from scripts.experiments.v1_full_eval import severity_proxies

DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
CHECKPOINT_PATH = "dataset/checkpoints/carm_deepaccident_v1.pt"
RESULTS_DIR = Path("dataset/results_v1")
LOGS_DIR = Path("dataset/logs")
DT = 0.1  # matches FutureRiskConfig.dt / GSSM's own 10Hz training rate

METHOD_LABEL = "GSSM (Jiao et al., zero-shot)"


def setup_logging(log_path: Path):
    log = logging.getLogger("gssm_baseline_eval")
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


def load_split() -> dict:
    """Read only the {'train','val','test': [instance_id, ...]} split out
    of the CARM v1 checkpoint -- the same split `v1_full_eval.py` uses --
    without constructing/running the CARM model itself (GSSM scoring does
    not need R_predicted)."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    return ckpt["split"]


def collect_instance(pre, geometric_only: bool = False) -> dict:
    raw = raw_instance_from_preprocessed(pre)
    rows = build_gssm_instance_rows(raw.ego_box, raw.agent_boxes, DT)
    geometric_collision_label = int(any(f.any() for f in raw.agent_collision_flags.values()))
    entry = {
        "instance_id": pre.instance_id,
        "geometric_collision_label": geometric_collision_label,
        "rows": rows,  # (X, s) tuple or None
    }
    if not geometric_only:
        entry.update(severity_proxies(pre))
    return entry


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(LOGS_DIR / f"gssm_baseline_eval_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    log.info("[1/5] Loading split from CARM v1 checkpoint (same split as v1_full_eval.py) ...")
    split = load_split()
    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)
    for k, v in split.items():
        log.info(f"      {k}: {len(v)} instance ids")

    log.info("[2/5] Building GSSM current+acc features for DeepAccident train/val/test ...")
    da_entries = {"train": [], "val": [], "test": []}
    for part in ("train", "val", "test"):
        for iid in split[part]:
            p = Path(DEEPACCIDENT_PREPROCESSED) / f"{iid}.npz"
            if not p.exists():
                continue
            pre = load_preprocessed_instance(str(p))
            if pre.ego_box is None:
                continue
            lbl_entry = native_labels.get(iid)
            if lbl_entry is None or lbl_entry.get("collision_label") is None:
                continue
            entry = collect_instance(pre)
            entry["native_collision_label"] = int(lbl_entry["collision_label"])
            da_entries[part].append(entry)
        log.info(f"      {part}: {len(da_entries[part])} instances")

    log.info("[3/5] Building GSSM current+acc features for Bench2Drive (cross-domain) ...")
    bd_entries = []
    for i, p in enumerate(sorted(Path(BENCH2DRIVE_PREPROCESSED).glob("*.npz"))):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        bd_entries.append(collect_instance(pre, geometric_only=False))
        if (i + 1) % 1000 == 0:
            log.info(f"      ... {i+1} instances built")
    log.info(f"      -> {len(bd_entries)} Bench2Drive instances")

    log.info("[4/5] Scoring ALL rows with the released GSSM checkpoint (one batched subprocess call) ...")
    all_entries = da_entries["train"] + da_entries["val"] + da_entries["test"] + bd_entries
    X_list, s_list, boundaries = [], [], []
    cursor = 0
    for e in all_entries:
        if e["rows"] is None:
            boundaries.append((cursor, cursor))
            continue
        X, s = e["rows"]
        X_list.append(X)
        s_list.append(s)
        boundaries.append((cursor, cursor + len(s)))
        cursor += len(s)
    if not X_list:
        log.error("No instances with agents found -- cannot score GSSM. Aborting.")
        sys.exit(1)
    X_all = np.concatenate(X_list, axis=0)
    s_all = np.concatenate(s_list, axis=0)
    log.info(f"      total rows to score: {len(s_all)} (across {len(all_entries)} instances)")

    try:
        intensity_all = score_rows_via_subprocess(X_all, s_all, batch_tag="gssm_full_eval")
    except GSSMUnavailableError as exc:
        log.error(f"GSSM scoring failed: {exc}")
        notes = RESULTS_DIR / "gssm_baseline_notes.md"
        notes.write_text(
            "# GSSM baseline -- FAILED\n\n"
            f"GSSM inference subprocess failed:\n\n```\n{exc}\n```\n\n"
            "No numbers were fabricated. See the traceback above for what to fix.\n"
        )
        sys.exit(1)

    for e, (lo, hi) in zip(all_entries, boundaries):
        e["gssm_score"] = instance_gssm_score_from_intensity(intensity_all[lo:hi]) if hi > lo else 0.0
    log.info(f"      scoring done in {time.time()-t0:.1f}s total")

    log.info("[5/5] Computing Table I (AUROC/AUPRC) and Table II (rho vs a(t)) ...")
    results = {"method_label": METHOD_LABEL, "checkpoint": "SafeBaseline/current+acc/model_final_71epoch.pth (zero-shot)"}

    # --- Table I: DeepAccident in-domain (native label, test split only, matching v1_full_eval.py) ---
    da_test_lbl = np.array([e["native_collision_label"] for e in da_entries["test"]])
    da_test_scores = np.array([e["gssm_score"] for e in da_entries["test"]])
    da_metrics = discrimination_metrics(da_test_scores, da_test_lbl)
    results["deepaccident_indomain"] = {
        "n": len(da_entries["test"]), "n_collisions": int(da_test_lbl.sum()),
        "auroc": da_metrics["auroc"], "auprc": da_metrics["auprc"],
    }
    log.info(f"      DeepAccident in-domain   n={len(da_entries['test'])} coll={int(da_test_lbl.sum())} "
             f"AUROC={da_metrics['auroc']:.3f} AUPRC={da_metrics['auprc']:.3f}")

    # --- Table I: Bench2Drive cross-domain (geometric label) ---
    bd_lbl = np.array([e["geometric_collision_label"] for e in bd_entries])
    bd_scores = np.array([e["gssm_score"] for e in bd_entries])
    bd_metrics = discrimination_metrics(bd_scores, bd_lbl)
    results["bench2drive_crossdomain"] = {
        "n": len(bd_entries), "n_collisions": int(bd_lbl.sum()),
        "auroc": bd_metrics["auroc"], "auprc": bd_metrics["auprc"],
    }
    log.info(f"      Bench2Drive cross-domain n={len(bd_entries)} coll={int(bd_lbl.sum())} "
             f"AUROC={bd_metrics['auroc']:.3f} AUPRC={bd_metrics['auprc']:.3f}")

    # --- Table II: DeepAccident collision-free (native label==0), all splits pooled ---
    da_all = da_entries["train"] + da_entries["val"] + da_entries["test"]
    free = [e for e in da_all if e["native_collision_label"] == 0]
    finegrained = {}
    for proxy, sign in (("max_evasive_severity", 1.0), ("worst_margin", -1.0), ("min_tau3d", -1.0)):
        vals = np.array([e[proxy] for e in free])
        finite = np.isfinite(vals)
        scores = np.array([e["gssm_score"] for e in free])[finite]
        v = vals[finite]
        if len(v) < 10 or np.all(v == v[0]) or np.all(scores == scores[0]):
            finegrained[f"gssm_score_vs_{proxy}"] = None
            continue
        rho, p = spearmanr(scores, v)
        finegrained[f"gssm_score_vs_{proxy}"] = {"n": int(len(v)), "spearman_rho": float(sign * rho), "p_value": float(p)}
    results["finegrained_baselines"] = finegrained
    primary = finegrained.get("gssm_score_vs_max_evasive_severity")
    if primary:
        log.info(f"      rho(GSSM, a(t)) = {primary['spearman_rho']:+.3f} (n={primary['n']})")
    else:
        log.info("      rho(GSSM, a(t)) = NA")

    # --- Table II, Bench2Drive cross-domain (geometric label==0), same proxies ---
    bd_free = [e for e in bd_entries if e["geometric_collision_label"] == 0]
    log.info(f"      Bench2Drive collision-free: {len(bd_free)}/{len(bd_entries)}")
    finegrained_bd = {}
    for proxy, sign in (("max_evasive_severity", 1.0), ("worst_margin", -1.0), ("min_tau3d", -1.0)):
        vals = np.array([e[proxy] for e in bd_free])
        finite = np.isfinite(vals)
        scores = np.array([e["gssm_score"] for e in bd_free])[finite]
        v = vals[finite]
        if len(v) < 10 or np.all(v == v[0]) or np.all(scores == scores[0]):
            finegrained_bd[f"gssm_score_vs_{proxy}"] = None
            continue
        rho, p = spearmanr(scores, v)
        finegrained_bd[f"gssm_score_vs_{proxy}"] = {"n": int(len(v)), "spearman_rho": float(sign * rho), "p_value": float(p)}
    results["finegrained_baselines_bench2drive"] = finegrained_bd
    primary_bd = finegrained_bd.get("gssm_score_vs_max_evasive_severity")
    if primary_bd:
        log.info(f"      [Bench2Drive] rho(GSSM, a(t)) = {primary_bd['spearman_rho']:+.3f} (n={primary_bd['n']})")
    else:
        log.info("      [Bench2Drive] rho(GSSM, a(t)) = NA")

    with open(RESULTS_DIR / "gssm_baseline_eval.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"wrote {RESULTS_DIR / 'gssm_baseline_eval.json'}")
    log.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
