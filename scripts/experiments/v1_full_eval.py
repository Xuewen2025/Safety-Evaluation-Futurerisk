#!/usr/bin/env python3
"""Full evaluation for the v1 protocol (DeepAccident-only training,
native-anchored target, 4s window, ego/object/road features), covering the
root.tex tables that need real numbers:

  Table instance_consistency  -- DeepAccident in-domain (native) +
                                  Bench2Drive cross-domain (geometric),
                                  5 methods (CARM, y_analytic, raw_ttc, RSS,
                                  Lu et al.)
  Table ssm_consistency       -- rho(R, TTC/PET/DRAC/RSS) by scenario
                                  category, DeepAccident in-domain
  Table finegrained_baselines -- rho(method, severity proxy) for all 7
                                  methods, DeepAccident in-domain
  Table per_interaction_type  -- DeepAccident->Bench2Drive zero-shot,
                                  AUROC/AUPRC/ECE/lead-time per type
  Figure severity_bins data   -- box-plot-ready binned R by a(t)

All 7 scored methods (CARM, y_analytic, raw_ttc, PET, DRAC, RSS, Lu et al.)
are built from the same universal BoxTrajectory-level fields for both
datasets -- no dataset-specific input (ssm_baselines.py / baselines_extra.py
convention).
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
from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import (
    discrimination_metrics, early_warning_lead_time, group_discrimination_metrics,
    operating_point_threshold, raw_ttc_baseline_score, reliability_diagram,
)
from futurerisk.interaction import TYPE_NAMES
from futurerisk.safety_prior import analytic_trajectory_risk, context_dependent_margin, instantaneous_risk
from futurerisk.ssm_baselines import deceleration_rate_to_avoid_crash, drac_score, instance_ssm_scores, pet_score
from scripts.evaluate_crossdomain_v1_to_bench2drive import load_checkpoint, predict_R

DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
RESULTS_DIR = Path("dataset/results_v1")
LOGS_DIR = Path("dataset/logs")


def setup_logging(log_path: Path):
    log = logging.getLogger("v1_full_eval")
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


def instance_dominant_type(pre) -> str:
    if not pre.tau_3d:
        return TYPE_NAMES[4]
    worst_agent = min(pre.tau_3d, key=lambda a: np.min(pre.tau_3d[a]))
    code = pre.dominant_interaction_type.get(worst_agent, 4)
    return TYPE_NAMES.get(code, "other")


def severity_proxies(pre) -> dict:
    if not pre.tau_3d:
        return {"min_tau3d": float("inf"), "worst_margin": float("inf"), "max_evasive_severity": 0.0}
    min_tau3d = min(float(np.min(pre.tau_3d[a])) for a in pre.tau_3d)
    margins = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
    worst_margin = float(np.min(np.concatenate(margins))) if margins else float("inf")
    max_sev = max(float(np.max(pre.evasive_severity[a])) for a in pre.evasive_severity) if pre.evasive_severity else 0.0
    return {"min_tau3d": min_tau3d, "worst_margin": worst_margin, "max_evasive_severity": max_sev}


def analytic_margin_and_risk(pre, T_m: float):
    margins = [context_dependent_margin(pre.tau_3d[aid], pre.tau_phy[aid]) for aid in pre.tau_3d]
    if not margins:
        return np.full(len(pre.times), np.inf), np.zeros(len(pre.times))
    margin_t = np.min(np.stack(margins), axis=0)
    risk_t = instantaneous_risk(margin_t, T_m=T_m)
    return margin_t, risk_t


def score_instance(model, cfg, pre, ssm_seq_len=16) -> dict:
    raw = raw_instance_from_preprocessed(pre)
    R_pred = predict_R(model, cfg, raw, pre)

    margins = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
    y_analytic = analytic_trajectory_risk(margins, T_m=cfg.T_m) if margins else 0.0

    tau3d_min = min((np.min(pre.tau_3d[a]) for a in pre.tau_3d), default=np.inf)
    ttc_score = float(raw_ttc_baseline_score(np.array([tau3d_min]), ttc_cap=cfg.ttc_cap)[0])

    ssm = instance_ssm_scores(raw.ego_box, raw.agent_boxes, pre.times, cfg.dt)
    pet_sc = float(pet_score(np.array([ssm["pet_min"]]))[0])
    drac_sc = float(drac_score(np.array([ssm["drac_max"]]))[0])
    rss_sc = instance_rss_score(raw.ego_box, raw.agent_boxes, pre.interaction_type, cfg.dt)

    geometric_collision_label = int(any(f.any() for f in raw.agent_collision_flags.values()))

    # SSM sequence features for the Lu et al. reproduction: use the worst
    # (min-tau3d) agent's own tau_3d/DRAC(t)/evasive_severity sequences.
    if pre.tau_3d:
        worst_agent = min(pre.tau_3d, key=lambda a: np.min(pre.tau_3d[a]))
        drac_t = deceleration_rate_to_avoid_crash(raw.ego_box, raw.agent_boxes[worst_agent], cfg.dt)
        ssm_feat = build_ssm_sequence_features(
            pre.tau_3d[worst_agent], drac_t, pre.evasive_severity[worst_agent], seq_len=ssm_seq_len,
        )
    else:
        ssm_feat = np.zeros(3 * ssm_seq_len, dtype=np.float32)

    row = {
        "instance_id": pre.instance_id,
        "R_predicted": R_pred, "y_analytic": float(y_analytic), "raw_ttc_score": ttc_score,
        "pet_score": pet_sc, "drac_score": drac_sc, "rss_score": rss_sc,
        "geometric_collision_label": geometric_collision_label,
        "dominant_type": instance_dominant_type(pre),
    }
    row.update(severity_proxies(pre))
    return row, ssm_feat


def spearman_block(rows: list, score_keys: list, proxy_keys: list, collision_key: str) -> dict:
    free = [r for r in rows if r[collision_key] == 0]
    out = {}
    for proxy in proxy_keys:
        vals = np.array([r[proxy] for r in free])
        finite = np.isfinite(vals)
        for score in score_keys:
            scores = np.array([r[score] for r in free])[finite]
            v = vals[finite]
            if len(v) < 10 or np.all(v == v[0]) or np.all(scores == scores[0]):
                out[f"{score}_vs_{proxy}"] = None
                continue
            rho, p = spearmanr(scores, v)
            sign = -1.0 if proxy in ("min_tau3d", "worst_margin") else 1.0
            out[f"{score}_vs_{proxy}"] = {"n": int(len(v)), "spearman_rho": float(sign * rho), "p_value": float(p)}
    return out


def binned_means(rows: list, proxy: str, score_keys: list, collision_key: str, n_bins: int = 5) -> list:
    free = [r for r in rows if r[collision_key] == 0 and np.isfinite(r[proxy])]
    if len(free) < n_bins * 5:
        return []
    vals = np.array([r[proxy] for r in free])
    order = np.argsort(vals)
    bins = np.array_split(order, n_bins)
    out = []
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        entry = {"bin": i, "n": len(idx), f"{proxy}_range": [float(vals[idx].min()), float(vals[idx].max())]}
        for score in score_keys:
            arr = np.array([free[j][score] for j in idx])
            entry[f"{score}_mean"] = float(arr.mean())
            entry[f"{score}_median"] = float(np.median(arr))
        out.append(entry)
    return out


METHOD_LABELS = {
    "R_predicted": "FutureRisk (CARM)", "y_analytic": "y_analytic", "raw_ttc_score": "raw 3D-TTC",
    "pet_score": "PET", "drac_score": "DRAC", "rss_score": "RSS", "lu_score": "Lu et al. (reproduced)",
}


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(LOGS_DIR / f"v1_full_eval_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    log.info("[1/6] Loading v1 checkpoint ...")
    model, cfg, ckpt = load_checkpoint()
    split = ckpt["split"]
    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)

    log.info("[2/6] Scoring DeepAccident train/val/test (in-domain, native label) ...")
    da_rows = {"train": [], "val": [], "test": []}
    da_ssm_feats = {"train": [], "val": [], "test": []}
    for part in ("train", "val", "test"):
        for iid in split[part]:
            p = Path(DEEPACCIDENT_PREPROCESSED) / f"{iid}.npz"
            if not p.exists():
                continue
            pre = load_preprocessed_instance(str(p))
            if pre.ego_box is None:
                continue
            entry = native_labels.get(iid)
            if entry is None or entry.get("collision_label") is None:
                continue
            row, feat = score_instance(model, cfg, pre)
            row["native_collision_label"] = int(entry["collision_label"])
            da_rows[part].append(row)
            da_ssm_feats[part].append(feat)
        log.info(f"      {part}: {len(da_rows[part])} instances")

    log.info("[3/6] Fitting Lu et al. reproduction on DeepAccident TRAIN, scoring all splits ...")
    X_train = np.stack(da_ssm_feats["train"])
    # ranking proxy for "how safe" a cluster is (higher = safer), used only to
    # label clusters after k-means, never to fit the autoencoder itself
    ttc_summary_train = np.array([min(r["min_tau3d"], cfg.ttc_cap) if np.isfinite(r["min_tau3d"]) else cfg.ttc_cap
                                   for r in da_rows["train"]])
    lu = LuEtAlReproduction(input_dim=X_train.shape[1], latent_dim=8, n_clusters=4, seed=0)
    lu.fit(X_train, ttc_summary_train, epochs=80)
    for part in ("train", "val", "test"):
        if not da_rows[part]:
            continue
        X_part = np.stack(da_ssm_feats[part])
        scores = lu.score(X_part)
        for r, s in zip(da_rows[part], scores):
            r["lu_score"] = float(s)

    log.info("[4/6] Scoring Bench2Drive (cross-domain, zero-shot, geometric label) ...")
    bd_rows = []
    bd_ssm_feats = []
    for i, p in enumerate(sorted(Path(BENCH2DRIVE_PREPROCESSED).glob("*.npz"))):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        row, feat = score_instance(model, cfg, pre)
        bd_rows.append(row)
        bd_ssm_feats.append(feat)
        if (i + 1) % 1000 == 0:
            log.info(f"      ... {i+1} instances scored")
    X_bd = np.stack(bd_ssm_feats)
    for r, s in zip(bd_rows, lu.score(X_bd)):
        r["lu_score"] = float(s)
    log.info(f"      -> {len(bd_rows)} Bench2Drive instances scored")

    all_rows_csv = {"deepaccident_test": da_rows["test"], "bench2drive_crossdomain": bd_rows}
    for name, rows in all_rows_csv.items():
        import csv
        with open(RESULTS_DIR / f"predictions_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    method_keys = ["R_predicted", "y_analytic", "raw_ttc_score", "pet_score", "drac_score", "rss_score", "lu_score"]

    log.info("[5/6] Table instance_consistency (5-method subset: CARM/y_analytic/raw_ttc/RSS/Lu) ...")
    subset5 = ["R_predicted", "y_analytic", "raw_ttc_score", "rss_score", "lu_score"]
    results = {}

    da_test_lbl = np.array([r["native_collision_label"] for r in da_rows["test"]])
    da_test_scores = {k: np.array([r[k] for r in da_rows["test"]]) for k in method_keys}
    results["deepaccident_indomain"] = {
        "n": len(da_rows["test"]), "n_collisions": int(da_test_lbl.sum()),
        "methods": {METHOD_LABELS[k]: discrimination_metrics(da_test_scores[k], da_test_lbl) for k in subset5},
    }
    for k in subset5:
        m = results["deepaccident_indomain"]["methods"][METHOD_LABELS[k]]
        log.info(f"      DeepAccident in-domain  {METHOD_LABELS[k]:28s} AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f}")

    bd_lbl = np.array([r["geometric_collision_label"] for r in bd_rows])
    bd_scores = {k: np.array([r[k] for r in bd_rows]) for k in method_keys}
    results["bench2drive_crossdomain"] = {
        "n": len(bd_rows), "n_collisions": int(bd_lbl.sum()),
        "methods": {METHOD_LABELS[k]: discrimination_metrics(bd_scores[k], bd_lbl) for k in subset5},
    }
    for k in subset5:
        m = results["bench2drive_crossdomain"]["methods"][METHOD_LABELS[k]]
        log.info(f"      Bench2Drive cross-domain  {METHOD_LABELS[k]:28s} AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f}")

    log.info("[6/6] SSM consistency, fine-grained baselines, per-type breakdown ...")
    # SSM consistency: rho(R, TTC/PET/DRAC/RSS) by scenario category, DeepAccident in-domain (all splits pooled for sample size)
    da_all = da_rows["train"] + da_rows["val"] + da_rows["test"]
    groups = np.array([r["dominant_type"] for r in da_all])
    ssm_consistency = {}
    category_map = {"car-following": "Car-following (stable)", "oncoming": "Oncoming", "crossing": "Crossing"}
    for cat_key, cat_label in category_map.items():
        mask = groups == cat_key
        if mask.sum() < 10:
            continue
        R = np.array([r["R_predicted"] for r in da_all])[mask]
        entry = {}
        for ssm_key, ssm_label in (("raw_ttc_score", "TTC"), ("pet_score", "PET"), ("drac_score", "DRAC"), ("rss_score", "RSS")):
            s = np.array([r[ssm_key] for r in da_all])[mask]
            if np.all(s == s[0]) or np.all(R == R[0]):
                entry[ssm_label] = None
                continue
            rho, p = spearmanr(R, s)
            entry[ssm_label] = {"n": int(mask.sum()), "rho": float(rho), "p": float(p)}
        ssm_consistency[cat_label] = entry
        log.info(f"      {cat_label:24s} " + "  ".join(
            f"{k}={v['rho']:+.3f}" if v else f"{k}=NA" for k, v in entry.items()))
    # "Abrupt speed/heading" category: top-quartile of ego net heading change magnitude proxy via max_evasive_severity as stand-in severity axis
    sev = np.array([r["max_evasive_severity"] for r in da_all])
    thresh = np.quantile(sev, 0.75)
    mask = sev >= thresh
    if mask.sum() >= 10:
        R = np.array([r["R_predicted"] for r in da_all])[mask]
        entry = {}
        for ssm_key, ssm_label in (("raw_ttc_score", "TTC"), ("pet_score", "PET"), ("drac_score", "DRAC"), ("rss_score", "RSS")):
            s = np.array([r[ssm_key] for r in da_all])[mask]
            rho, p = spearmanr(R, s) if not (np.all(s == s[0]) or np.all(R == R[0])) else (float("nan"), float("nan"))
            entry[ssm_label] = {"n": int(mask.sum()), "rho": float(rho), "p": float(p)}
        ssm_consistency["Abrupt speed/heading (top-quartile evasive severity)"] = entry
        log.info("      Abrupt speed/heading    " + "  ".join(f"{k}={v['rho']:+.3f}" for k, v in entry.items()))
    results["ssm_consistency"] = ssm_consistency

    # Fine-grained baselines: rho(method, proxy), DeepAccident in-domain, collision-free (native)
    proxy_keys = ["max_evasive_severity", "worst_margin", "min_tau3d"]
    for r in da_all:
        r["_native_lbl_for_spearman"] = r.get("native_collision_label", 0)
    finegrained = spearman_block(da_all, method_keys, proxy_keys, "native_collision_label")
    results["finegrained_baselines"] = finegrained
    for k in method_keys:
        v = finegrained.get(f"{k}_vs_max_evasive_severity")
        if v:
            log.info(f"      rho({METHOD_LABELS[k]}, a(t)) = {v['spearman_rho']:+.3f} (n={v['n']})")

    results["severity_bins"] = {
        proxy: binned_means(da_all, proxy, method_keys, "native_collision_label") for proxy in proxy_keys
    }

    # Per-interaction-type cross-domain (Bench2Drive), with ECE + lead time
    bd_groups = np.array([r["dominant_type"] for r in bd_rows])
    per_type_disc = group_discrimination_metrics(
        np.array([r["R_predicted"] for r in bd_rows]), bd_lbl, bd_groups,
    )
    # operating threshold from DeepAccident VAL non-colliding (since that's the trained domain's own val set)
    val_traces = []
    for r_row, iid in zip(da_rows["val"], split["val"]):
        if r_row["native_collision_label"] == 1:
            continue
        p = Path(DEEPACCIDENT_PREPROCESSED) / f"{iid}.npz"
        if not p.exists():
            continue
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        _, risk_t = analytic_margin_and_risk(pre, cfg.T_m)
        val_traces.append(risk_t)
    threshold = operating_point_threshold(val_traces, target_fpr=0.05) if val_traces else 0.5
    log.info(f"      operating threshold (DeepAccident val, 5% FPR): {threshold:.4f}")

    per_type_full = {}
    for t in sorted(set(bd_groups.tolist())) + ["overall"]:
        mask = np.ones(len(bd_rows), dtype=bool) if t == "overall" else (bd_groups == t)
        n = int(mask.sum())
        if n == 0:
            continue
        sub_lbl = bd_lbl[mask]
        sub_R = np.array([r["R_predicted"] for r in bd_rows])[mask]
        m = discrimination_metrics(sub_R, sub_lbl)
        rd = reliability_diagram(sub_R, sub_lbl, n_bins=10) if len(set(sub_lbl.tolist())) > 1 else None
        # lead time: use geometric collision time as timing (Bench2Drive has no native timing either)
        lead_times = []
        sub_rows = [r for r, keep in zip(bd_rows, mask) if keep]
        for r in sub_rows:
            if r["geometric_collision_label"] != 1:
                continue
            p = Path(BENCH2DRIVE_PREPROCESSED) / f"{r['instance_id']}.npz"
            pre = load_preprocessed_instance(str(p))
            raw = raw_instance_from_preprocessed(pre)
            collision_time = None
            for aid, flags in raw.agent_collision_flags.items():
                if flags.any():
                    idx = int(np.argmax(flags))
                    tt = float(pre.times[idx])
                    collision_time = tt if collision_time is None else min(collision_time, tt)
            if collision_time is None:
                continue
            _, risk_t = analytic_margin_and_risk(pre, cfg.T_m)
            lt = early_warning_lead_time(risk_t, pre.times, threshold, collision_time)
            if lt is not None:
                lead_times.append(lt)
        per_type_full[t] = {
            "n": n, "n_collisions": int(sub_lbl.sum()), "auroc": m["auroc"], "auprc": m["auprc"],
            "ece": rd.ece if rd is not None else None,
            "mean_lead_time_s": float(np.mean(lead_times)) if lead_times else None,
            "n_lead_time_samples": len(lead_times),
        }
        log.info(f"      {t:24s} n={n:5d} coll={int(sub_lbl.sum()):4d} AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f}"
                 f" ECE={per_type_full[t]['ece']}"
                 f" lead_time={per_type_full[t]['mean_lead_time_s']}")
    results["per_interaction_type_crossdomain"] = per_type_full

    with open(RESULTS_DIR / "v1_full_eval_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"wrote {RESULTS_DIR / 'v1_full_eval_summary.json'}")
    log.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
