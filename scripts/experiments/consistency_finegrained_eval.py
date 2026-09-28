#!/usr/bin/env python3
"""Consolidated low-cost experiments from experiment_design.txt, computed in
one pass over the (corrected-tau_phy) Bench2Drive-Base-only checkpoint,
evaluated both in-domain (Bench2Drive test split) and cross-domain
(DeepAccident, zero-shot, native collision label as ground truth). Covers:

  一.1  instance-level consistency (AUROC/AUPRC) -- in-domain AND
        cross-domain, not just cross-domain as before.
  一.3  consistency with classic SSMs (PET/DRAC) -- reported overall AND
        broken out per interaction type, so "SSMs are reliable in
        car-following but fragile in oncoming/crossing" is a checkable
        claim rather than an assertion.
  二.1  fine-grained discrimination: Spearman correlation of R against
        continuous severity proxies (worst-case margin, min tau_3D, max
        evasive severity) on collision-free trajectories only (collision
        rate can't see this axis at all).
  二.3  same proxies, correlated against y_analytic/raw_ttc/PET/DRAC too,
        for a head-to-head "how much more does the learned part resolve
        vs. a pure analytic formula" comparison.
  三.1  AUROC/AUPRC broken out per interaction type (not just pooled).

All five scored methods (CARM R, y_analytic, raw_ttc, PET, DRAC) are built
from the SAME universal BoxTrajectory-level fields (position/heading/extent
over a shared time axis) for every dataset -- no dataset-specific input,
matching ssm_baselines.py's module docstring and this project's existing
raw_ttc_baseline_score/y_analytic convention.
"""
from __future__ import annotations

import csv
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

from futurerisk.carm_model import CARM, aggregate_instance_risk
from futurerisk.config import FutureRiskConfig
from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import compare_methods, group_discrimination_metrics
from futurerisk.features import build_instance_batch_item
from futurerisk.interaction import TYPE_NAMES
from futurerisk.safety_prior import analytic_trajectory_risk, context_dependent_margin
from futurerisk.ssm_baselines import drac_score, instance_ssm_scores, pet_score
from futurerisk.evaluation import raw_ttc_baseline_score

CHECKPOINT_PATH = "dataset/checkpoints/carm_bench2drive_base.pt"
BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
RESULTS_DIR = Path("dataset/results_experiments")
LOGS_DIR = Path("dataset/logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("consistency_finegrained_eval")


def setup_logging(log_path: Path) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def load_checkpoint():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    cfg = FutureRiskConfig(**{k: v for k, v in ckpt["cfg"].items() if k != "physical_prior_by_class"})
    model = CARM(
        d_kin=ckpt["d_kin"], d_int=ckpt["d_int"], d_ctx=ckpt["d_ctx"], cfg=cfg,
        hidden_dim=64, rnn_hidden=64, dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def predict_R(model: CARM, cfg: FutureRiskConfig, raw, pre) -> float:
    item = build_instance_batch_item(raw, pre, cfg)
    if item["x_kin"].shape[0] == 0:
        return 0.0
    x_kin = torch.from_numpy(item["x_kin"]).to(DEVICE)
    X_int = torch.from_numpy(item["X_int"]).to(DEVICE)
    x_ctx = torch.from_numpy(item["x_ctx"]).to(DEVICE)
    tau_phy_seq = torch.from_numpy(item["tau_phy_seq"]).to(DEVICE)
    bin_edges = torch.tensor(cfg.bin_edges, dtype=torch.float32, device=DEVICE)
    out = model(x_kin=x_kin, X_int=X_int, x_ctx=x_ctx, tau_phy_seq=tau_phy_seq, bin_edges=bin_edges)
    R_instance = aggregate_instance_risk(out["R"].unsqueeze(0))
    return float(R_instance.item())


def instance_dominant_type(pre) -> str:
    """One group label per instance: the dominant interaction type of
    whichever agent has the smallest min-tau_3D (the most critical
    interaction in the instance -- a reasonable proxy for "what kind of
    conflict is driving this instance's risk", used purely for grouping
    metrics, not as a model input)."""
    if not pre.tau_3d:
        return TYPE_NAMES[4]  # "other"
    worst_agent = min(pre.tau_3d, key=lambda a: np.min(pre.tau_3d[a]))
    code = pre.dominant_interaction_type.get(worst_agent, 4)
    return TYPE_NAMES.get(code, "other")


def severity_proxies(pre) -> dict:
    """Continuous, collision-rate-blind severity proxies for one instance
    (S3.5's own harsh-evasive severity, worst-case margin, worst-case
    3D-TTC) -- used only for the fine-grained binning/Spearman experiment,
    never as a model input."""
    if not pre.tau_3d:
        return {"min_tau3d": float("inf"), "worst_margin": float("inf"), "max_evasive_severity": 0.0}
    min_tau3d = min(float(np.min(pre.tau_3d[a])) for a in pre.tau_3d)
    margins = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
    worst_margin = float(np.min(np.concatenate(margins))) if margins else float("inf")
    max_sev = max(float(np.max(pre.evasive_severity[a])) for a in pre.evasive_severity) if pre.evasive_severity else 0.0
    return {"min_tau3d": min_tau3d, "worst_margin": worst_margin, "max_evasive_severity": max_sev}


def score_instance(model, cfg, pre) -> dict:
    raw = raw_instance_from_preprocessed(pre)
    R_pred = predict_R(model, cfg, raw, pre)

    y_analytic = 0.0
    margins = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
    if margins:
        y_analytic = analytic_trajectory_risk(margins, T_m=cfg.T_m)

    tau3d_min = min((np.min(pre.tau_3d[a]) for a in pre.tau_3d), default=np.inf)
    ttc_score = float(raw_ttc_baseline_score(np.array([tau3d_min]), ttc_cap=cfg.ttc_cap)[0])

    ssm = instance_ssm_scores(raw.ego_box, raw.agent_boxes, pre.times, cfg.dt)
    pet_sc = float(pet_score(np.array([ssm["pet_min"]]))[0])
    drac_sc = float(drac_score(np.array([ssm["drac_max"]]))[0])

    geometric_collision_label = int(any(f.any() for f in raw.agent_collision_flags.values()))

    row = {
        "instance_id": pre.instance_id,
        "R_predicted": R_pred, "y_analytic": y_analytic, "raw_ttc_score": ttc_score,
        "pet_score": pet_sc, "drac_score": drac_sc,
        "geometric_collision_label": geometric_collision_label,
        "dominant_type": instance_dominant_type(pre),
    }
    row.update(severity_proxies(pre))
    return row


def spearman_block(rows: list, score_keys: list, proxy_keys: list, collision_key: str) -> dict:
    """S3-two.1/two.3: Spearman rho between each score and each severity
    proxy, restricted to collision-free instances (proxy_key values of
    +inf, i.e. no near-miss at all for that agent set, are dropped)."""
    free = [r for r in rows if r[collision_key] == 0]
    out = {}
    for proxy in proxy_keys:
        vals = np.array([r[proxy] for r in free])
        finite = np.isfinite(vals)
        for score in score_keys:
            scores = np.array([r[score] for r in free])[finite]
            v = vals[finite]
            if len(v) < 10 or np.all(v == v[0]):
                out[f"{score}_vs_{proxy}"] = None
                continue
            rho, p = spearmanr(scores, v)
            # min_tau3d/worst_margin: LOWER proxy = MORE dangerous, so a
            # good risk score should be NEGATIVELY correlated with them;
            # max_evasive_severity: HIGHER = more dangerous, positive
            # correlation expected. Report the "alignment" sign-corrected
            # so all methods read on the same "higher = better aligned"
            # scale regardless of which proxy's natural direction.
            sign = -1.0 if proxy in ("min_tau3d", "worst_margin") else 1.0
            out[f"{score}_vs_{proxy}"] = {"n": int(len(v)), "spearman_rho": float(sign * rho), "p_value": float(p)}
    return out


def binned_means(rows: list, proxy: str, score_keys: list, collision_key: str, n_bins: int = 5) -> list:
    """Quantile-bin collision-free instances by `proxy`, report each
    score's mean per bin -- the boxplot-equivalent numeric summary
    experiment_design.txt asks for (S-two.1)."""
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


def run_one_domain(name: str, rows: list, collision_key: str) -> dict:
    lbl = np.array([r[collision_key] for r in rows])
    method_scores = {
        "CARM (R)": np.array([r["R_predicted"] for r in rows]),
        "analytic-only (y_analytic)": np.array([r["y_analytic"] for r in rows]),
        "raw 3D-TTC": np.array([r["raw_ttc_score"] for r in rows]),
        "PET": np.array([r["pet_score"] for r in rows]),
        "DRAC": np.array([r["drac_score"] for r in rows]),
    }
    overall = compare_methods(method_scores, lbl)
    log.info(f"[{name}] overall (n={len(rows)}, collisions={int(lbl.sum())}):")
    for m, v in overall.items():
        log.info(f"    {m:28s} AUROC={v['auroc']:.3f}  AUPRC={v['auprc']:.3f}")

    groups = np.array([r["dominant_type"] for r in rows])
    per_type = {m: group_discrimination_metrics(s, lbl, groups) for m, s in method_scores.items()}
    log.info(f"[{name}] per-interaction-type AUROC (三.1):")
    for t in sorted(set(groups.tolist())):
        line = f"    {t:24s}"
        for m in method_scores:
            entry = per_type[m].get(t, {})
            line += f"  {m.split(' ')[0]:>10s}={entry.get('auroc', float('nan')):.3f}(n={int(entry.get('n',0))})"
        log.info(line)

    proxy_keys = ["min_tau3d", "worst_margin", "max_evasive_severity"]
    score_keys = list(row_key for row_key in ("R_predicted", "y_analytic", "raw_ttc_score", "pet_score", "drac_score"))
    spearman = spearman_block(rows, score_keys, proxy_keys, collision_key)
    log.info(f"[{name}] Spearman rho (sign-corrected, collision-free only, 二.1/二.3):")
    for k, v in spearman.items():
        if v is not None:
            log.info(f"    {k:40s} rho={v['spearman_rho']:+.3f} (n={v['n']}, p={v['p_value']:.2e})")

    binned = {proxy: binned_means(rows, proxy, score_keys, collision_key) for proxy in proxy_keys}

    return {
        "overall": overall, "per_interaction_type": per_type, "spearman": spearman, "binned": binned,
        "n_instances": len(rows), "n_collisions": int(lbl.sum()),
    }


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"consistency_finegrained_eval_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")
    t0 = time.time()

    log.info(f"[1/4] Loading checkpoint {CHECKPOINT_PATH} ...")
    model, cfg, ckpt = load_checkpoint()
    bd_split = ckpt["split"]
    log.info(f"      best_val_auroc={ckpt.get('best_val_auroc')} best_val_auprc={ckpt.get('best_val_auprc')}")

    log.info("[2/4] Scoring Bench2Drive test split (in-domain) ...")
    bd_rows = []
    for iid in bd_split["test"]:
        p = Path(BENCH2DRIVE_PREPROCESSED) / f"{iid}.npz"
        if not p.exists():
            continue
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        bd_rows.append(score_instance(model, cfg, pre))
    log.info(f"      -> {len(bd_rows)} Bench2Drive test instances scored")

    log.info("[3/4] Scoring DeepAccident (cross-domain, zero-shot, native label) ...")
    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)
    da_rows = []
    for i, p in enumerate(sorted(Path(DEEPACCIDENT_PREPROCESSED).glob("*.npz"))):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        entry = native_labels.get(pre.instance_id)
        if entry is None or entry["collision_label"] is None:
            continue
        row = score_instance(model, cfg, pre)
        row["native_collision_label"] = int(entry["collision_label"])
        da_rows.append(row)
        if (i + 1) % 2000 == 0:
            log.info(f"      ... {i+1} files scanned, {len(da_rows)} usable so far")
    log.info(f"      -> {len(da_rows)} DeepAccident instances scored (native label available)")

    for name, rows, key in (("bench2drive_test_indomain", bd_rows, "geometric_collision_label"),
                             ("deepaccident_crossdomain_native", da_rows, "native_collision_label")):
        with open(RESULTS_DIR / f"predictions_{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    log.info("[4/4] Computing consistency + fine-grained + per-type results ...")
    results = {
        "bench2drive_test_indomain": run_one_domain("Bench2Drive-test(in-domain)", bd_rows, "geometric_collision_label"),
        "deepaccident_crossdomain_native": run_one_domain("DeepAccident(cross-domain,native)", da_rows, "native_collision_label"),
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"      wrote {RESULTS_DIR / 'summary.json'}")
    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
