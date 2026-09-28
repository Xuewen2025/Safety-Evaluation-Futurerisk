#!/usr/bin/env python3
"""v1 interpretability evidence (root.tex Sec. exp_interpretability):
  - formula-anchored fraction for FutureRisk (Table interpretability_tradeoff)
  - anchored-fraction DISTRIBUTION, not just its mean (root.tex Sec.
    exp_tradeoff review item P0-1): every unmasked agent's per-agent
    anchored fraction is kept (not flattened away into one scalar) and
    grouped by (a) native/geometric collision label and (b) whether that
    agent's ground-truth critical-event label T_evt (eq:tevt, the same
    label the hazard head is supervised on) is finite ("triggered") or
    infinite ("no event") within the horizon -- i.e. whether the
    interaction was ever actually safety-critical, not a threshold on the
    model's own q_k belief.
  - residual ablation: Stage-1-only (g_r forced to 0, no retrain needed --
    Stage 2 only trains g_r, encoder/hazard/safe_ttc are untouched by it)
    vs. full CARM (Table residual_ablation)
  - g_r distribution by interaction type (Figure residual_distribution)
  - learned tau_safe correction by (interaction type, closing-speed bin)
    (Figure learned_correction)

No retraining: everything here is inference-only on the existing v1
checkpoint (single seed -- see note in `main()` below on why this is not
yet a multi-seed stability result), computed once over DeepAccident's
in-domain test split (native label) and Bench2Drive's cross-domain test
set (geometric label).
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import discrimination_metrics
from futurerisk.features_v1 import build_instance_batch_item_v1
from futurerisk.interaction import TYPE_NAMES
from scripts.evaluate_crossdomain_v1_to_bench2drive import DEVICE, load_checkpoint

DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
RESULTS_DIR = Path("dataset/results_v1")
LOGS_DIR = Path("dataset/logs")


def setup_logging(log_path: Path):
    log = logging.getLogger("v1_interpretability")
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


def _logit_np(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@torch.no_grad()
def per_instance_diagnostics(model, cfg, raw, pre):
    item = build_instance_batch_item_v1(raw, pre, cfg)
    if item["x_ego"].shape[0] == 0:
        return None
    x_ego = torch.from_numpy(item["x_ego"]).to(DEVICE)
    X_obj = torch.from_numpy(item["X_obj"]).to(DEVICE)
    x_ctx = torch.from_numpy(item["x_ctx_v1"]).to(DEVICE)
    tau_phy_seq = torch.from_numpy(item["tau_phy_seq"]).to(DEVICE)
    bin_edges = torch.tensor(cfg.bin_edges, dtype=torch.float32, device=DEVICE)

    h = model.encoder(x_ego, X_obj, x_ctx)
    out = model(x_kin=x_ego, X_int=X_obj, x_ctx=x_ctx, tau_phy_seq=tau_phy_seq, bin_edges=bin_edges)
    residual_raw = model.residual_head.mlp(h).squeeze(-1)          # (J,)
    beta_tanh_gr = model.beta * torch.tanh(residual_raw)            # (J,)
    logit_R_tilde = _logit_torch(out["R_tilde"])                    # (J,)

    R_full = out["R"].cpu().numpy()                                  # per-agent, full CARM
    R_tilde = out["R_tilde"].cpu().numpy()                           # per-agent, Stage-1-only (no residual)
    g_r = residual_raw.cpu().numpy()
    formula_frac_terms = (beta_tanh_gr.abs() / logit_R_tilde.abs().clamp_min(1e-6)).cpu().numpy()

    # instance-level aggregation, noisy-OR, matching aggregate_instance_risk
    R_instance_full = 1.0 - np.prod(1.0 - R_full)
    R_instance_stage1 = 1.0 - np.prod(1.0 - R_tilde)

    v_close = []
    from futurerisk.data_pipeline import _relative_closing_speed
    for aid in item["agent_ids"]:
        vc = _relative_closing_speed(raw.ego_box, raw.agent_boxes[aid], cfg.dt)
        v_close.append(float(np.mean(vc)))

    itypes = [TYPE_NAMES.get(pre.dominant_interaction_type.get(aid, 4), "other") for aid in item["agent_ids"]]

    # Ground-truth critical-event trigger per agent: T_evt (eq:tevt) is
    # already computed by build_instance_batch_item_v1 (same label used to
    # supervise the hazard head, S3.6.2/S3.7) -- finite iff a collision or
    # harsh-evasive-severity threshold crossing ever occurred for that agent
    # within the horizon. Using this (not a threshold on the model's own
    # q_k) keeps the "triggered" grouping independent of the very quantity
    # (g_r / anchored fraction) whose distribution we are trying to explain.
    critical_event_triggered = np.isfinite(item["T_evt"]).tolist()

    return {
        "agent_ids": list(item["agent_ids"]),
        "R_instance_full": float(R_instance_full), "R_instance_stage1": float(R_instance_stage1),
        "g_r": g_r.tolist(), "formula_frac_terms": formula_frac_terms.tolist(),
        "anchored_frac": (1.0 - formula_frac_terms).tolist(),
        "critical_event_triggered": critical_event_triggered,
        "interaction_types": itypes, "v_close": v_close,
    }


def _logit_torch(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


def _bootstrap_ci(values, stat_fn=np.median, n_boot=10000, seed=0):
    """Percentile bootstrap 95% CI of `stat_fn` over `values` (resampling
    test agents with replacement). This quantifies sampling uncertainty of
    the single-seed estimate, not training-seed variance -- see the
    `note_single_seed` field written alongside every use of this helper."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = stat_fn(values[idx], axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _group_summary(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None,
                "median_bootstrap_ci95": [None, None]}
    lo, hi = _bootstrap_ci(values, np.median)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)), "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)), "q75": float(np.percentile(values, 75)),
        "median_bootstrap_ci95": [lo, hi],
    }


def _two_group_comparison(a, b):
    """Two-sided Mann-Whitney U test between two independent groups of
    per-agent anchored fractions."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"u_statistic": None, "p_value": None}
    u, pval = sstats.mannwhitneyu(a, b, alternative="two-sided")
    return {"u_statistic": float(u), "p_value": float(pval)}


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(LOGS_DIR / f"v1_interpretability_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    model, cfg, ckpt = load_checkpoint()
    split = ckpt["split"]
    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)

    log.info("[1/4] DeepAccident in-domain test (native label) ...")
    da_full_g_r, da_full_frac, da_by_type_g_r = [], [], defaultdict(list)
    da_R_full, da_R_stage1, da_lbl = [], [], []
    per_agent_records = []  # one row per (dataset, instance, agent) -- P0-1 distribution evidence
    for iid in split["test"]:
        p = Path(DEEPACCIDENT_PREPROCESSED) / f"{iid}.npz"
        if not p.exists():
            continue
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        entry = native_labels.get(iid)
        if entry is None or entry.get("collision_label") is None:
            continue
        raw = raw_instance_from_preprocessed(pre)
        d = per_instance_diagnostics(model, cfg, raw, pre)
        if d is None:
            continue
        da_R_full.append(d["R_instance_full"]); da_R_stage1.append(d["R_instance_stage1"])
        native_lbl = int(entry["collision_label"])
        da_lbl.append(native_lbl)
        da_full_g_r.extend(d["g_r"]); da_full_frac.extend(d["formula_frac_terms"])
        for t, g in zip(d["interaction_types"], d["g_r"]):
            da_by_type_g_r[t].append(g)
        for aid, af, itype, trig in zip(
            d["agent_ids"], d["anchored_frac"], d["interaction_types"], d["critical_event_triggered"]
        ):
            per_agent_records.append({
                "dataset": "deepaccident", "instance_id": iid, "agent_id": aid,
                "anchored_frac": af, "interaction_type": itype,
                # native label is instance-level (DeepAccident's ground-truth
                # accident annotation does not identify which agent), so it
                # is broadcast to every agent in a colliding instance.
                "native_collision_label": native_lbl,
                "critical_event_triggered": bool(trig),
            })

    da_R_full, da_R_stage1, da_lbl = np.array(da_R_full), np.array(da_R_stage1), np.array(da_lbl)
    m_full_in = discrimination_metrics(da_R_full, da_lbl)
    m_stage1_in = discrimination_metrics(da_R_stage1, da_lbl)
    log.info(f"      in-domain: full CARM AUROC={m_full_in['auroc']:.3f} AUPRC={m_full_in['auprc']:.3f}"
             f"   Stage-1-only AUROC={m_stage1_in['auroc']:.3f} AUPRC={m_stage1_in['auprc']:.3f}")

    formula_anchored_fraction = float(1.0 - np.mean(da_full_frac))
    log.info(f"      formula-anchored fraction (FutureRisk) = {formula_anchored_fraction:.4f}")
    log.info(f"      g_r stats: mean={np.mean(da_full_g_r):.4f} std={np.std(da_full_g_r):.4f}")
    for t, vals in da_by_type_g_r.items():
        log.info(f"        g_r by type  {t:24s} n={len(vals):5d} mean={np.mean(vals):+.4f} std={np.std(vals):.4f}")

    log.info("[2/4] Bench2Drive cross-domain test (geometric label) ...")
    bd_R_full, bd_R_stage1, bd_lbl = [], [], []
    bd_by_type_g_r = defaultdict(list)
    for i, p in enumerate(sorted(Path(BENCH2DRIVE_PREPROCESSED).glob("*.npz"))):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        raw = raw_instance_from_preprocessed(pre)
        d = per_instance_diagnostics(model, cfg, raw, pre)
        if d is None:
            continue
        bd_R_full.append(d["R_instance_full"]); bd_R_stage1.append(d["R_instance_stage1"])
        bd_lbl.append(int(any(f.any() for f in raw.agent_collision_flags.values())))
        for t, g in zip(d["interaction_types"], d["g_r"]):
            bd_by_type_g_r[t].append(g)
        for aid, af, itype, trig in zip(
            d["agent_ids"], d["anchored_frac"], d["interaction_types"], d["critical_event_triggered"]
        ):
            flags = raw.agent_collision_flags.get(aid)
            per_agent_records.append({
                "dataset": "bench2drive", "instance_id": p.stem, "agent_id": aid,
                "anchored_frac": af, "interaction_type": itype,
                # Bench2Drive's geometric collision flag IS per-agent (unlike
                # DeepAccident's native label), so this grouping is finer here.
                "native_collision_label": int(bool(flags is not None and flags.any())),
                "critical_event_triggered": bool(trig),
            })
        if (i + 1) % 1000 == 0:
            log.info(f"      ... {i+1} instances scored")

    bd_R_full, bd_R_stage1, bd_lbl = np.array(bd_R_full), np.array(bd_R_stage1), np.array(bd_lbl)
    m_full_cross = discrimination_metrics(bd_R_full, bd_lbl)
    m_stage1_cross = discrimination_metrics(bd_R_stage1, bd_lbl)
    log.info(f"      cross-domain: full CARM AUROC={m_full_cross['auroc']:.3f} AUPRC={m_full_cross['auprc']:.3f}"
             f"   Stage-1-only AUROC={m_stage1_cross['auroc']:.3f} AUPRC={m_stage1_cross['auprc']:.3f}")

    log.info("[3/4] Anchored-fraction distribution (P0-1: mean -> distribution) ...")
    da_records = [r for r in per_agent_records if r["dataset"] == "deepaccident"]
    bd_records = [r for r in per_agent_records if r["dataset"] == "bench2drive"]

    def _split(records, key):
        pos = [r["anchored_frac"] for r in records if r[key]]
        neg = [r["anchored_frac"] for r in records if not r[key]]
        return pos, neg

    def _panel(records):
        coll_pos, coll_neg = _split(records, "native_collision_label")
        evt_pos, evt_neg = _split(records, "critical_event_triggered")
        return {
            "overall": _group_summary([r["anchored_frac"] for r in records]),
            "native_collision": _group_summary(coll_pos),
            "no_collision": _group_summary(coll_neg),
            "collision_vs_no_collision_mannwhitney": _two_group_comparison(coll_pos, coll_neg),
            "critical_event_triggered": _group_summary(evt_pos),
            "no_critical_event": _group_summary(evt_neg),
            "critical_event_mannwhitney": _two_group_comparison(evt_pos, evt_neg),
        }

    anchored_fraction_distribution = {
        # DeepAccident: native_collision_label is instance-level (broadcast
        # to every agent in a colliding instance -- DeepAccident's ground
        # truth does not identify which agent collided).
        "deepaccident_indomain": _panel(da_records),
        # Bench2Drive: agent_collision_flags is genuinely per-agent, so this
        # panel is the finer-grained confirmatory one.
        "bench2drive_crossdomain": _panel(bd_records),
        "note_single_seed": (
            "All numbers above come from ONE training seed "
            "(dataset/checkpoints/carm_deepaccident_v1.pt); no multi-seed "
            "checkpoint exists in this project as of this run. The 95% CIs "
            "are a percentile bootstrap (10000 resamples) over test agents "
            "WITHIN this one seed's predictions -- they quantify sampling "
            "uncertainty of the test-set estimate, NOT training-seed "
            "variance, and must not be reported as multi-seed stability."
        ),
    }
    da_p = anchored_fraction_distribution["deepaccident_indomain"]
    log.info(f"      DeepAccident overall: median={da_p['overall']['median']:.3f} "
             f"IQR=[{da_p['overall']['q25']:.3f},{da_p['overall']['q75']:.3f}] (n={da_p['overall']['n']})")
    log.info(f"      collision vs no-collision median: {da_p['native_collision']['median']:.3f} vs "
             f"{da_p['no_collision']['median']:.3f}  (Mann-Whitney p={da_p['collision_vs_no_collision_mannwhitney']['p_value']:.2e})")
    log.info(f"      critical-event triggered vs not, median: {da_p['critical_event_triggered']['median']:.3f} vs "
             f"{da_p['no_critical_event']['median']:.3f}  (Mann-Whitney p={da_p['critical_event_mannwhitney']['p_value']:.2e})")

    per_agent_csv = RESULTS_DIR / "v1_interpretability_per_agent.csv"
    with open(per_agent_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "instance_id", "agent_id", "anchored_frac", "interaction_type",
            "native_collision_label", "critical_event_triggered",
        ])
        writer.writeheader()
        writer.writerows(per_agent_records)
    log.info(f"wrote {per_agent_csv} ({len(per_agent_records)} agent-level rows)")

    log.info("[4/4] Writing results ...")
    results = {
        "formula_anchored_fraction": formula_anchored_fraction,
        "anchored_fraction_distribution": anchored_fraction_distribution,
        "residual_ablation": {
            "stage1_only": {
                "auroc_indomain": m_stage1_in["auroc"], "auprc_indomain": m_stage1_in["auprc"],
                "auroc_crossdomain": m_stage1_cross["auroc"], "auprc_crossdomain": m_stage1_cross["auprc"],
            },
            "full_carm": {
                "auroc_indomain": m_full_in["auroc"], "auprc_indomain": m_full_in["auprc"],
                "auroc_crossdomain": m_full_cross["auroc"], "auprc_crossdomain": m_full_cross["auprc"],
            },
        },
        "g_r_by_type_deepaccident": {t: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))}
                                      for t, v in da_by_type_g_r.items()},
        "g_r_by_type_bench2drive": {t: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))}
                                     for t, v in bd_by_type_g_r.items()},
    }
    with open(RESULTS_DIR / "v1_interpretability_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"wrote {RESULTS_DIR / 'v1_interpretability_summary.json'}")
    log.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
