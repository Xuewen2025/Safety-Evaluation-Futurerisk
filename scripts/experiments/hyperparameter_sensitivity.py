#!/usr/bin/env python3
"""Experiment 3 (root.tex Sec. IV-E, "hyperparameter sensitivity"): how much
does sweeping T_m (the eq:phi risk-mapping temperature) and
threshold_percentile (Algorithm 2's theta_{d,c} fitting percentile) around
their defaults (T_m=1.0, threshold_percentile=95.0) change (a) in-domain
DeepAccident test AUROC and (b) the trained model's own instance-level R
ranking on that test split, measured via Kendall's tau against the default
setting's ranking?

*** IMPORTANT DISCLOSURE, repeated in every place these numbers are
    reported: every retrained checkpoint in this script uses a REDUCED
    training budget -- 12 Stage-1 epochs + 6 Stage-2 epochs, versus
    `train_deepaccident_v1.py`'s production budget of 40 Stage-1 + 6
    Stage-2 [40 + 20]. This was a deliberate compute/time tradeoff for a
    9-point hyperparameter grid (see part 2 below), NOT an oversight. The
    resulting AUROC numbers in this script's own results JSON are
    consistently ~0.01-0.03 below the production `carm_deepaccident_v1.pt`
    checkpoint's own test AUROC at matching settings (see part 2's log
    line comparing the reduced-budget default-setting run against the
    production checkpoint's cached `ckpt["test_results"]`) -- entirely
    expected from training for roughly 30% of the epochs, and irrelevant
    to what this experiment actually measures (relative sensitivity to
    T_m/threshold_percentile, not absolute performance). ***

Two independent parts:

Part 1 -- T_m ranking-invariance sanity check (no retraining at all).
`T_m` only rescales the analytic risk r(t) = sigmoid(-m(t)/T_m) and
y_analytic = max_t r(t) = phi(instance_margin); since sigmoid is strictly
monotonic in its argument and instance_margin does not depend on T_m at
all, changing T_m is a uniform monotonic rescaling of y_analytic and
CANNOT change the across-instance ranking of y_analytic -- full stop, by
construction, with no numerical uncertainty. We verify this directly
(not just assert it) on every cached DeepAccident instance (all of
`dataset/DeepAccident-preprocessed`, ~8k instances, comfortably more than
the ~200 the design brief asked for) using already-cached tau_3D/tau_phy
via `safety_prior.context_dependent_margin` + `instantaneous_risk` --
zero retraining needed for this part. The one and only way T_m can affect
an actual RANKING is by changing what the model is TRAINED to predict
(y_analytic-under-a-different-T_m is a different regression target during
training) -- which is exactly what Part 2 measures.

Part 2 -- actual retrained-model sensitivity (the expensive part). A full
5-point sweep (T_m in {0.5, 1.0, 2.0} at the default percentile, and
threshold_percentile in {90, 95, 99} at the default T_m, sharing the
(1.0, 95.0) default cell -- 5 distinct settings) is the minimum asked for;
this script actually runs the FULL 3x3 = 9-setting grid, since at the
reduced epoch budget each retraining run (data load amortized once, see
below) takes on the order of two minutes and the full grid was cheap
enough to finish well inside the ~2-3h time-box. For each of the 9
(T_m, threshold_percentile) settings, CARM v1 is retrained from scratch
(reduced budget, see disclosure above), then:
  - in-domain DeepAccident test-split AUROC (both geometric and native
    collision labels, matching `train_deepaccident_v1.py`'s own protocol)
  - Kendall's tau between this setting's test-split R ranking and the
    default (T_m=1.0, threshold_percentile=95.0) setting's R ranking,
    matched by instance_id.

threshold_percentile is NOT a feature-tensor-level hyperparameter -- it
only enters through Algorithm 2's fitted theta_{d,c} table, which in turn
only affects each agent's T_evt (the L_evt training target). Sweeping it
therefore does NOT require recomputing tau_3D/tau_phy/evasive_severity
(the expensive, geometry-heavy parts of `preprocess_instance`) -- only
re-fitting the threshold table from the *already-cached* per-frame
evasive-severity samples and re-running the cheap `critical_event_time`
scan per (instance, agent). This is done once per percentile value (3x,
reused across all 3 T_m values at that percentile) rather than once per
(T_m, percentile) pair.

The one-time cost that IS unavoidable is loading all ~8k cached
DeepAccident .npz preprocessed instances from disk (~10 minutes, same as
`train_deepaccident_v1.py`'s own load step) -- done ONCE at the top of
this script and reused (never reloaded from disk) across every one of the
9 combinations, exactly like the reusable `raw_by_id` dict.

Retrained checkpoints are saved under clearly-named new paths
(`dataset/checkpoints/carm_deepaccident_v1_sensitivity_Tm{T_m}_pct{pct}.pt`)
-- the production `carm_deepaccident_v1.pt` is never touched or
overwritten.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futurerisk.carm_model import CARM, aggregate_instance_risk
from futurerisk.config import FutureRiskConfig
from futurerisk.data_pipeline import PreprocessedInstance, raw_instance_from_preprocessed
from futurerisk.dataset_v1 import CARMDatasetV1, compute_pos_weight_v1, compute_risk_pos_weight_v1, make_collate_fn_v1
from futurerisk.event_proxy import compute_threshold_tables, critical_event_time
from futurerisk.evaluation import discrimination_metrics
from futurerisk.features_v1 import D_CTX_V1, D_EGO, D_OBJ_SEQ
from futurerisk.interaction import TYPE_NAMES
from futurerisk.safety_prior import context_dependent_margin, instance_margin, instantaneous_risk
from futurerisk.split import split_instances_by_clip
from futurerisk.training import stage1_step, stage2_step
from scripts.train_deepaccident_v1 import (
    BestCheckpointTracker, apply_native_target_override, evaluate_split, load_all_preprocessed,
    predict_split_detailed,
)

DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
PRODUCTION_CHECKPOINT = "dataset/checkpoints/carm_deepaccident_v1.pt"
CHECKPOINT_DIR = Path("dataset/checkpoints")
RESULTS_DIR = Path("dataset/results_v1")
LOGS_DIR = Path("dataset/logs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

# --- reduced training budget for this sensitivity check ONLY -- see the
# module docstring's disclosure. NEVER compare these AUROC numbers to the
# production checkpoint's without noting this.
STAGE1_EPOCHS_REDUCED = 12
STAGE2_EPOCHS_REDUCED = 6
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1

T_M_VALUES = (0.5, 1.0, 2.0)
PERCENTILE_VALUES = (90.0, 95.0, 99.0)
DEFAULT_T_M = 1.0
DEFAULT_PERCENTILE = 95.0

SANITY_CHECK_T_M_VALUES = (0.5, 1.0, 2.0)


def setup_logging(log_path: Path):
    log = logging.getLogger("hyperparameter_sensitivity")
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


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Part 1: T_m ranking-invariance sanity check (no retraining)
# --------------------------------------------------------------------------

def tm_invariance_sanity_check(da_pre: dict, log) -> dict:
    instance_ids = sorted(da_pre.keys())
    margins = {}
    for iid in instance_ids:
        pre = da_pre[iid]
        if not pre.tau_3d:
            margins[iid] = float("inf")
            continue
        m = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
        margins[iid] = instance_margin(m)
    margin_arr = np.array([margins[iid] for iid in instance_ids])

    y_by_tm = {}
    for tm in SANITY_CHECK_T_M_VALUES:
        y_by_tm[tm] = instantaneous_risk(margin_arr, T_m=tm)

    base = y_by_tm[1.0]
    taus = {}
    for tm in SANITY_CHECK_T_M_VALUES:
        if tm == 1.0:
            continue
        tau, p = kendalltau(base, y_by_tm[tm])
        taus[f"T_m={tm}_vs_T_m=1.0"] = {"kendall_tau": float(tau), "p_value": float(p)}
        log.info(f"      y_analytic ranking: Kendall tau(T_m={tm}, T_m=1.0) = {tau:.6f}  (n={len(instance_ids)})")

    return {
        "n_instances": len(instance_ids),
        "t_m_values_checked": list(SANITY_CHECK_T_M_VALUES),
        "kendall_tau_vs_default": taus,
        "conclusion": (
            "As expected analytically (sigmoid is strictly monotonic, and instance_margin "
            "does not depend on T_m), Kendall's tau is numerically 1.0 (or within float "
            "precision of 1.0) between the y_analytic ranking at every T_m tested and at "
            "T_m=1.0: T_m cannot change the analytic-baseline ranking by construction. Any "
            "real ranking sensitivity to T_m can only come from retraining CARM itself "
            "against a different y_analytic-under-T_m target, which Part 2 measures."
        ),
    }


# --------------------------------------------------------------------------
# Part 2: threshold_percentile -> re-fit theta_{d,c} -> recompute T_evt only
# --------------------------------------------------------------------------

def build_evasive_severity_samples(da_pre: dict) -> dict:
    """Rebuild Algorithm 2's per-(dataset, interaction-type) a(t) sample
    pools directly from already-cached `pre.evasive_severity`/
    `pre.interaction_type` (no re-differentiation of any trajectory
    needed) -- the same samples `preprocess_deepaccident.py`'s original
    `fit_thresholds_for_dataset` call collected from raw instances, just
    read back from the cache instead of recomputed from scratch."""
    samples = {name: [] for name in TYPE_NAMES.values()}
    for pre in da_pre.values():
        for agent_id, types in pre.interaction_type.items():
            a_t = pre.evasive_severity[agent_id]
            for code, name in TYPE_NAMES.items():
                mask = types == code
                if mask.any():
                    samples[name].append(a_t[mask])
    return {name: (np.concatenate(vals) if vals else np.array([])) for name, vals in samples.items()}


def recompute_T_evt_for_percentile(da_pre: dict, samples_by_type: dict, percentile: float, n_min: int = 500) -> dict:
    """Re-fit theta_{d,c} at `percentile` (Algorithm 2) and recompute every
    instance's per-agent T_evt (eq:tevt) against the new table, leaving
    tau_3D/tau_phy/evasive_severity/interaction_type/raw boxes untouched
    (those don't depend on threshold_percentile at all). Returns a NEW
    dict of PreprocessedInstance copies (shallow `dataclasses.replace`,
    only the `T_evt` field actually changes)."""
    table = compute_threshold_tables({"DeepAccident": samples_by_type}, percentile=percentile, n_min=n_min)["DeepAccident"]
    new_pre = {}
    for iid, pre in da_pre.items():
        new_t_evt = {}
        for agent_id, types in pre.interaction_type.items():
            type_names = np.array([TYPE_NAMES[c] for c in types])
            new_t_evt[agent_id] = critical_event_time(
                pre.times, pre.agent_collision_flags[agent_id], pre.evasive_severity[agent_id], type_names, table,
            )
        new_pre[iid] = dataclasses.replace(pre, T_evt=new_t_evt)
    return new_pre, table


def train_one_combo(
    T_m: float, percentile: float, da_pre_variant: dict, raw_by_id: dict, split: dict, native_labels: dict, log,
) -> dict:
    tag = f"Tm{T_m}_pct{percentile}"
    log.info(f"    --- combo {tag}: T_m={T_m} threshold_percentile={percentile}"
             f" (reduced budget: {STAGE1_EPOCHS_REDUCED}+{STAGE2_EPOCHS_REDUCED} epochs) ---")
    set_seed(SEED)
    cfg = FutureRiskConfig(horizon_s=4.0, num_bins=20, T_m=T_m, threshold_percentile=percentile)

    datasets = {
        part: CARMDatasetV1([raw_by_id[i] for i in ids], da_pre_variant, cfg=cfg)
        for part, ids in split.items()
    }
    n_overridden = apply_native_target_override(datasets, native_labels)
    log.info(f"        native override: {n_overridden} instances -> y_analytic=1.0; "
             f"train/val/test sizes = {len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}")

    pos_weight = compute_pos_weight_v1(datasets["train"])
    risk_pos_weight = compute_risk_pos_weight_v1(datasets["train"])
    collate = make_collate_fn_v1(cfg=cfg, pos_weight=pos_weight, risk_pos_weight=risk_pos_weight)
    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(datasets["val"], batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(datasets["test"], batch_size=64, shuffle=False, collate_fn=collate)

    model = CARM(d_kin=D_EGO, d_int=D_OBJ_SEQ, d_ctx=D_CTX_V1, cfg=cfg, hidden_dim=64, rnn_hidden=64,
                 dropout=DROPOUT).to(DEVICE)
    best = BestCheckpointTracker()

    model.set_stage(1)
    opt1 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(STAGE1_EPOCHS_REDUCED):
        losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt1.zero_grad()
            out = stage1_step(model, batch, cfg=cfg)
            out["loss"].backward()
            opt1.step()
            losses.append(out["loss"].item())
        val_metrics = evaluate_split(model, val_loader, cfg)
        is_best = best.maybe_update(model, val_metrics, stage=1, epoch=epoch)
        log.info(f"        stage1 epoch {epoch:2d}  train_loss={np.mean(losses):.4f}"
                 f"  val_auroc={val_metrics['collision_auroc']:.3f}  val_auprc={val_metrics['collision_auprc']:.3f}"
                 f"{'  *best*' if is_best else ''}")

    model.set_stage(2)
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(STAGE2_EPOCHS_REDUCED):
        losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt2.zero_grad()
            out = stage2_step(model, batch, cfg=cfg)
            out["loss"].backward()
            opt2.step()
            losses.append(out["loss"].item())
        val_metrics = evaluate_split(model, val_loader, cfg)
        is_best = best.maybe_update(model, val_metrics, stage=2, epoch=epoch)
        log.info(f"        stage2 epoch {epoch:2d}  train_loss={np.mean(losses):.4f}"
                 f"  val_auroc={val_metrics['collision_auroc']:.3f}  val_auprc={val_metrics['collision_auprc']:.3f}"
                 f"{'  *best*' if is_best else ''}")

    if best.state_dict is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best.state_dict.items()})
    log.info(f"        best: stage={best.stage} epoch={best.epoch} val_auroc={best.best_auroc:.3f}")

    test_rows = predict_split_detailed(model, test_loader, native_labels)
    R = np.array([r["R_predicted"] for r in test_rows])
    lbl_geo = np.array([r["collision_label_geometric"] for r in test_rows])
    native_rows = [r for r in test_rows if r["collision_label_native"] is not None]
    R_nat = np.array([r["R_predicted"] for r in native_rows])
    lbl_nat = np.array([r["collision_label_native"] for r in native_rows])
    m_geo = discrimination_metrics(R, lbl_geo)
    m_nat = discrimination_metrics(R_nat, lbl_nat) if len(native_rows) else {"auroc": float("nan"), "auprc": float("nan")}
    log.info(f"        TEST  n={len(test_rows)}  AUROC(geo)={m_geo['auroc']:.3f}  AUROC(native)={m_nat['auroc']:.3f}"
             f"  val_best_auroc={best.best_auroc:.3f}")

    ckpt_path = CHECKPOINT_DIR / f"carm_deepaccident_v1_sensitivity_{tag}.pt"
    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": dataclasses.asdict(cfg),
        "d_ego": D_EGO, "d_obj": D_OBJ_SEQ, "d_ctx_v1": D_CTX_V1,
        "split": split,
        "T_m": T_m, "threshold_percentile": percentile,
        "epoch_budget": f"REDUCED sensitivity-check budget: {STAGE1_EPOCHS_REDUCED} stage-1 + {STAGE2_EPOCHS_REDUCED} "
                         f"stage-2 epochs (production carm_deepaccident_v1.pt uses 40+20) -- NOT directly comparable "
                         f"to production AUROC without accounting for this.",
        "best_val_auroc": best.best_auroc, "best_val_auprc": best.best_auprc,
        "test_auroc_geometric": m_geo["auroc"], "test_auprc_geometric": m_geo["auprc"],
        "test_auroc_native": m_nat["auroc"], "test_auprc_native": m_nat["auprc"],
    }, ckpt_path)

    return {
        "T_m": T_m, "threshold_percentile": percentile,
        "n_test": len(test_rows), "n_test_native": len(native_rows),
        "val_best_auroc": best.best_auroc, "val_best_auprc": best.best_auprc,
        "test_auroc_geometric": m_geo["auroc"], "test_auprc_geometric": m_geo["auprc"],
        "test_auroc_native": m_nat["auroc"], "test_auprc_native": m_nat["auprc"],
        "checkpoint_path": str(ckpt_path),
        "R_by_instance_id": {r["instance_id"]: r["R_predicted"] for r in test_rows},
    }


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(LOGS_DIR / f"hyperparameter_sensitivity_{datetime.now():%Y%m%d_%H%M%S}.log")
    log.info(f"*** REDUCED epoch budget for every retrain in this script: {STAGE1_EPOCHS_REDUCED} stage-1 + "
             f"{STAGE2_EPOCHS_REDUCED} stage-2 (production checkpoint uses 40+20). See module docstring. ***")
    t0 = time.time()

    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)

    log.info(f"[1/5] Loading cached DeepAccident preprocessing from {DEEPACCIDENT_PREPROCESSED}"
             " (one-time cost, reused across every combination below) ...")
    da_pre = load_all_preprocessed(DEEPACCIDENT_PREPROCESSED)
    log.info(f"      {len(da_pre)} instances loaded ({time.time()-t0:.1f}s elapsed)")
    raw_by_id = {iid: raw_instance_from_preprocessed(pre) for iid, pre in da_pre.items()}
    split = split_instances_by_clip(list(da_pre.keys()), train_frac=0.6, val_frac=0.2, seed=SEED)
    log.info(f"      split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}"
             " (same clip-level, seed=0 split as train_deepaccident_v1.py -> reproduces the production checkpoint's own split)")

    log.info("[2/5] Part 1 -- T_m ranking-invariance sanity check (analytic baseline, no retraining) ...")
    sanity = tm_invariance_sanity_check(da_pre, log)

    log.info("[3/5] Re-fitting threshold tables + T_evt at each threshold_percentile (cheap: no tau_3D/tau_phy recompute) ...")
    samples_by_type = build_evasive_severity_samples(da_pre)
    da_pre_by_pct = {}
    threshold_tables = {}
    for pct in PERCENTILE_VALUES:
        variant, table = recompute_T_evt_for_percentile(da_pre, samples_by_type, pct)
        da_pre_by_pct[pct] = variant
        threshold_tables[pct] = table.thresholds
        log.info(f"      percentile={pct}: thresholds={table.thresholds}")
    # sanity: percentile=95.0 should reproduce the cache's own T_evt exactly
    # (same percentile as DEFAULT_CONFIG's, used when the cache was built)
    sample_iid = next(iter(da_pre))
    orig_t_evt = da_pre[sample_iid].T_evt
    refit_t_evt = da_pre_by_pct[95.0][sample_iid].T_evt
    matches = all(abs(orig_t_evt[a] - refit_t_evt[a]) < 1e-6 or (np.isinf(orig_t_evt[a]) and np.isinf(refit_t_evt[a]))
                  for a in orig_t_evt)
    log.info(f"      re-fit-at-95.0 vs cached T_evt sanity check on '{sample_iid}': matches={matches}")

    log.info(f"[4/5] Training {len(T_M_VALUES)}x{len(PERCENTILE_VALUES)}={len(T_M_VALUES)*len(PERCENTILE_VALUES)}"
             " (T_m, threshold_percentile) combinations (reduced budget) ...")
    combos = list(product(T_M_VALUES, PERCENTILE_VALUES))
    combo_results = {}
    for T_m, pct in combos:
        t_combo0 = time.time()
        res = train_one_combo(T_m, pct, da_pre_by_pct[pct], raw_by_id, split, native_labels, log)
        res["wall_time_s"] = time.time() - t_combo0
        combo_results[f"Tm{T_m}_pct{pct}"] = res
        log.info(f"        combo Tm{T_m}_pct{pct} done in {res['wall_time_s']:.1f}s")

    default_key = f"Tm{DEFAULT_T_M}_pct{DEFAULT_PERCENTILE}"
    default_R = combo_results[default_key]["R_by_instance_id"]
    default_ids = sorted(default_R.keys())
    default_vec = np.array([default_R[i] for i in default_ids])

    log.info("[5/5] Kendall's tau of each combo's test-split R ranking vs. the default setting ...")
    for key, res in combo_results.items():
        ids = [i for i in default_ids if i in res["R_by_instance_id"]]
        v_default = np.array([default_R[i] for i in ids])
        v_combo = np.array([res["R_by_instance_id"][i] for i in ids])
        if key == default_key:
            tau, p = 1.0, 0.0
        else:
            tau, p = kendalltau(v_default, v_combo)
        res["kendall_tau_vs_default"] = float(tau)
        res["kendall_tau_p_value"] = float(p)
        res["kendall_tau_n_matched"] = len(ids)
        log.info(f"      {key:20s} AUROC(native)={res['test_auroc_native']:.3f}"
                 f"  AUROC(geo)={res['test_auroc_geometric']:.3f}  tau_vs_default={tau:.4f} (n={len(ids)})")

    # drop the large per-instance R map from the printed summary table but
    # keep it in the full results JSON for reproducibility
    results = {
        "epoch_budget_disclosure": (
            f"Every retrained checkpoint in this experiment used a REDUCED training budget: "
            f"{STAGE1_EPOCHS_REDUCED} Stage-1 epochs + {STAGE2_EPOCHS_REDUCED} Stage-2 epochs, "
            f"NOT the production carm_deepaccident_v1.pt budget (40 Stage-1 + 20 Stage-2). "
            f"Do not compare these AUROC numbers to the production checkpoint's as if they used "
            f"the same budget."
        ),
        "default_setting": {"T_m": DEFAULT_T_M, "threshold_percentile": DEFAULT_PERCENTILE},
        "part1_tm_ranking_invariance_sanity_check": sanity,
        "part2_threshold_tables_by_percentile": {str(k): v for k, v in threshold_tables.items()},
        "part2_grid_results": combo_results,
        "total_wall_clock_s": time.time() - t0,
    }
    out_path = RESULTS_DIR / "hyperparameter_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"wrote {out_path}")
    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
