#!/usr/bin/env python3
"""Forward-projected, deployable analytic-TTC baseline evaluation -- adds
TWO new rows to the tables `v1_full_eval.py` already builds, following the
exact same pattern as `gssm_baseline_eval.py` (same split, same native
labels, same severity-proxy computation, same discrimination_metrics /
spearmanr calls, so the resulting numbers are like-for-like new rows, not
a separately-defined evaluation):

  "Forward TTC 2-D (Li et al. 2023, ours)" -- `forward_ttc_baseline
      .forward_ttc_ctrv(..., ignore_turning=False)`: the CTRV
      (constant-turn-rate-and-velocity) instantiation of Li et al.
      (arXiv:2312.07019) Section 4.1's 2-D kinematic-bicycle-model SSM
      framework, held at each frame's own current (v, omega).
  "Forward TTC 1-D (Hayward 1972, naive)" -- the SAME function with
      `ignore_turning=True`: forces omega=0 for every frame, i.e. the
      classic longitudinal-only constant-velocity TTC that is the
      omega=0 sub-case of the identical framework.

This script does NOT recompute CARM/y_analytic/raw-TTC(oracle)/PET/DRAC/
RSS/Lu et al./GSSM -- it scores these two new methods on the EXACT SAME
instances (same split, same native labels file, same severity-proxy
computation) that `v1_full_eval.py` (and `gssm_baseline_eval.py`) already
used, so the resulting numbers slot directly into the same table.

Bonus analysis (Section [3/4] below): a direct, like-for-like replication
of Li et al.'s own headline claim -- the 1D-vs-2D forward-TTC discrepancy
-- computed on THIS project's own datasets rather than theirs, split into
"non-critical" (raw 2D forward-TTC >= 1.5s) and "critical" (< 1.5s) buckets
to match how they report it (up to ~300% / ~20% respectively in their
paper).

No external checkpoint, no subprocess, no GPU -- pure numpy on already-
cached `PreprocessedInstance` fields, so this is much cheaper than
`gssm_baseline_eval.py` and needs no external dependency to be available.
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

from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import discrimination_metrics, raw_ttc_baseline_score
from futurerisk.forward_ttc_baseline import forward_ttc_all_agents
from scripts.experiments.v1_full_eval import severity_proxies

DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
CHECKPOINT_PATH = "dataset/checkpoints/carm_deepaccident_v1.pt"
RESULTS_DIR = Path("dataset/results_v1")
LOGS_DIR = Path("dataset/logs")
DT = 0.1
TTC_CAP = 5.0
CRITICAL_TTC_S = 1.5  # matches Li et al.'s own critical/non-critical split

METHOD_LABEL_2D = "Forward TTC 2-D (Li et al. 2023, ours)"
METHOD_LABEL_1D = "Forward TTC 1-D (Hayward 1972, naive)"


def setup_logging(log_path: Path):
    log = logging.getLogger("forward_ttc_baseline_eval")
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
    """Same split `v1_full_eval.py`/`gssm_baseline_eval.py` use, read
    straight out of the CARM v1 checkpoint without constructing the model."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    return ckpt["split"]


def collect_instance(pre, geometric_only: bool = False) -> dict:
    raw = raw_instance_from_preprocessed(pre)
    geometric_collision_label = int(any(f.any() for f in raw.agent_collision_flags.values()))
    entry = {
        "instance_id": pre.instance_id,
        "geometric_collision_label": geometric_collision_label,
    }
    if raw.agent_boxes:
        tau_2d = forward_ttc_all_agents(raw.ego_box, raw.agent_boxes, dt=DT, cap=TTC_CAP, ignore_turning=False)
        tau_1d = forward_ttc_all_agents(raw.ego_box, raw.agent_boxes, dt=DT, cap=TTC_CAP, ignore_turning=True)
        min_tau_2d = min(float(np.min(t)) for t in tau_2d.values())
        min_tau_1d = min(float(np.min(t)) for t in tau_1d.values())
    else:
        min_tau_2d = float("inf")
        min_tau_1d = float("inf")
    entry["min_tau_2d"] = min_tau_2d
    entry["min_tau_1d"] = min_tau_1d
    entry["score_2d"] = float(raw_ttc_baseline_score(np.array([min_tau_2d]), ttc_cap=TTC_CAP)[0])
    entry["score_1d"] = float(raw_ttc_baseline_score(np.array([min_tau_1d]), ttc_cap=TTC_CAP)[0])
    if not geometric_only:
        entry.update(severity_proxies(pre))
    return entry


def discrepancy_report(entries: list, log) -> dict:
    """Direct replication of Li et al.'s own 1D-vs-2D divergence statistic
    on this project's data: relative |tau_2d - tau_1d| / tau_1d, split into
    non-critical (tau_2d >= CRITICAL_TTC_S) and critical (< CRITICAL_TTC_S)
    buckets, restricted to instances where both are finite (a real
    conflict was predicted by both methods -- otherwise "relative
    divergence" is undefined, not just large)."""
    both_finite = [e for e in entries if np.isfinite(e["min_tau_2d"]) and np.isfinite(e["min_tau_1d"])]
    if not both_finite:
        return {"n": 0, "note": "no instances where both methods predicted a finite forward TTC"}
    tau2d = np.array([e["min_tau_2d"] for e in both_finite])
    tau1d = np.array([e["min_tau_1d"] for e in both_finite])
    nonzero_1d = tau1d > 1e-6
    rel_div = np.full_like(tau1d, np.nan)
    rel_div[nonzero_1d] = np.abs(tau2d[nonzero_1d] - tau1d[nonzero_1d]) / tau1d[nonzero_1d]

    critical = tau2d < CRITICAL_TTC_S
    out = {"n": int(len(both_finite))}
    for name, mask in (("non_critical (tau_2d>=1.5s)", ~critical), ("critical (tau_2d<1.5s)", critical)):
        vals = rel_div[mask & nonzero_1d]
        if len(vals) == 0:
            out[name] = None
            continue
        out[name] = {
            "n": int(len(vals)),
            "mean_relative_divergence_pct": float(100 * np.mean(vals)),
            "median_relative_divergence_pct": float(100 * np.median(vals)),
            "max_relative_divergence_pct": float(100 * np.max(vals)),
        }
        log.info(f"      {name}: n={len(vals)} mean_rel_div={100*np.mean(vals):.1f}% "
                  f"median={100*np.median(vals):.1f}% max={100*np.max(vals):.1f}%")
    return out


def table_i_row(entries_test, bd_entries, label, score_key, log):
    lbl = np.array([e["native_collision_label"] for e in entries_test])
    scores = np.array([e[score_key] for e in entries_test])
    da_metrics = discrimination_metrics(scores, lbl)
    bd_lbl = np.array([e["geometric_collision_label"] for e in bd_entries])
    bd_scores = np.array([e[score_key] for e in bd_entries])
    bd_metrics = discrimination_metrics(bd_scores, bd_lbl)
    log.info(f"      [{label}] DeepAccident in-domain  n={len(entries_test)} coll={int(lbl.sum())} "
              f"AUROC={da_metrics['auroc']:.3f} AUPRC={da_metrics['auprc']:.3f}")
    log.info(f"      [{label}] Bench2Drive cross-domain n={len(bd_entries)} coll={int(bd_lbl.sum())} "
              f"AUROC={bd_metrics['auroc']:.3f} AUPRC={bd_metrics['auprc']:.3f}")
    return {
        "deepaccident_indomain": {"n": len(entries_test), "n_collisions": int(lbl.sum()),
                                   "auroc": da_metrics["auroc"], "auprc": da_metrics["auprc"]},
        "bench2drive_crossdomain": {"n": len(bd_entries), "n_collisions": int(bd_lbl.sum()),
                                     "auroc": bd_metrics["auroc"], "auprc": bd_metrics["auprc"]},
    }


def table_ii_row(da_all_free, score_key, log, label):
    finegrained = {}
    for proxy, sign in (("max_evasive_severity", 1.0), ("worst_margin", -1.0), ("min_tau3d", -1.0)):
        vals = np.array([e[proxy] for e in da_all_free])
        finite = np.isfinite(vals)
        scores = np.array([e[score_key] for e in da_all_free])[finite]
        v = vals[finite]
        if len(v) < 10 or np.all(v == v[0]) or np.all(scores == scores[0]):
            finegrained[f"{score_key}_vs_{proxy}"] = None
            continue
        rho, p = spearmanr(scores, v)
        finegrained[f"{score_key}_vs_{proxy}"] = {"n": int(len(v)), "spearman_rho": float(sign * rho), "p_value": float(p)}
    primary = finegrained.get(f"{score_key}_vs_max_evasive_severity")
    if primary:
        log.info(f"      [{label}] rho(score, a(t)) = {primary['spearman_rho']:+.3f} (n={primary['n']})")
    else:
        log.info(f"      [{label}] rho(score, a(t)) = NA")
    return finegrained


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(LOGS_DIR / f"forward_ttc_baseline_eval_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    log.info("[1/4] Loading split from CARM v1 checkpoint (same split as v1_full_eval.py) ...")
    split = load_split()
    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)
    for k, v in split.items():
        log.info(f"      {k}: {len(v)} instance ids")

    log.info("[2/4] Computing forward-TTC (2D CTRV + 1D naive) for DeepAccident train/val/test ...")
    da_entries = {"train": [], "val": [], "test": []}
    for part in ("train", "val", "test"):
        for i, iid in enumerate(split[part]):
            if (i + 1) % 500 == 0:
                log.info(f"      ... {part}: {i+1}/{len(split[part])} instance ids processed")
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

    log.info("[3/4] Computing forward-TTC (2D CTRV + 1D naive) for Bench2Drive (cross-domain) ...")
    bd_entries = []
    for i, p in enumerate(sorted(Path(BENCH2DRIVE_PREPROCESSED).glob("*.npz"))):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        bd_entries.append(collect_instance(pre, geometric_only=False))
        if (i + 1) % 1000 == 0:
            log.info(f"      ... {i+1} instances built")
    log.info(f"      -> {len(bd_entries)} Bench2Drive instances")
    log.info(f"      scoring done in {time.time()-t0:.1f}s total")

    log.info("[4/4] Computing Table I (AUROC/AUPRC), Table II (rho vs a(t)), and the 1D-vs-2D discrepancy report ...")
    results = {"method_label_2d": METHOD_LABEL_2D, "method_label_1d": METHOD_LABEL_1D}

    log.info("   -- Table I --")
    results["table_i_2d"] = table_i_row(da_entries["test"], bd_entries, "2D", "score_2d", log)
    results["table_i_1d"] = table_i_row(da_entries["test"], bd_entries, "1D", "score_1d", log)

    log.info("   -- Table II (DeepAccident collision-free, all splits pooled) --")
    da_all = da_entries["train"] + da_entries["val"] + da_entries["test"]
    free = [e for e in da_all if e["native_collision_label"] == 0]
    results["table_ii_2d"] = table_ii_row(free, "score_2d", log, "2D")
    results["table_ii_1d"] = table_ii_row(free, "score_1d", log, "1D")

    log.info("   -- Table II, Bench2Drive cross-domain (geometric-collision-free) --")
    bd_free = [e for e in bd_entries if e["geometric_collision_label"] == 0]
    log.info(f"      Bench2Drive collision-free: {len(bd_free)}/{len(bd_entries)}")
    results["table_ii_2d_bench2drive"] = table_ii_row(bd_free, "score_2d", log, "2D, Bench2Drive")
    results["table_ii_1d_bench2drive"] = table_ii_row(bd_free, "score_1d", log, "1D, Bench2Drive")

    log.info("   -- 1D-vs-2D forward-TTC discrepancy (Li et al.'s own headline comparison, replicated on our data) --")
    log.info("      DeepAccident (train+val+test pooled):")
    results["discrepancy_deepaccident"] = discrepancy_report(da_all, log)
    log.info("      Bench2Drive:")
    results["discrepancy_bench2drive"] = discrepancy_report(bd_entries, log)

    with open(RESULTS_DIR / "forward_ttc_baseline_eval.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"wrote {RESULTS_DIR / 'forward_ttc_baseline_eval.json'}")
    log.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
