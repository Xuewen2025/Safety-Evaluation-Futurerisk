#!/usr/bin/env python3
"""S3.10 cross-domain generalization evaluation: the CARM checkpoint
trained ONLY on Bench2Drive-Base (`carm_bench2drive_base.pt`) is evaluated
zero-shot on DeepAccident, which it has never seen during training.

Metrics (S3.10):
    1. Discrimination: AUROC/AUPRC of trajectory-level R against the
       binary true-collision label -- for the model, the analytic-only
       y_analytic (no learned component), and raw 3D-TTC thresholding.
    2. Early-warning lead time: operating-point threshold chosen from the
       Bench2Drive VAL split (non-colliding instances, target 5% FPR),
       applied to DeepAccident's colliding instances' analytic r(t) trace.
    3. Calibration: reliability diagram / ECE of the model's R against
       empirical collision frequency on DeepAccident.

Honest scope note: PET/DRAC baselines mentioned in S3.10 are not
implemented anywhere in this package (no generator for them exists), so
they're omitted here rather than faked.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from futurerisk.carm_model import CARM, aggregate_instance_risk
from futurerisk.config import FutureRiskConfig
from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import (
    discrimination_metrics, early_warning_lead_time, operating_point_threshold,
    raw_ttc_baseline_score, reliability_diagram,
)
from futurerisk.features import D_CTX, D_INT, D_KIN, build_instance_batch_item
from futurerisk.safety_prior import context_dependent_margin, instantaneous_risk

CHECKPOINT_PATH = "dataset/checkpoints/carm_bench2drive_base.pt"
BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
RESULTS_DIR = Path("dataset/results_crossdomain")
LOGS_DIR = Path("dataset/logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("evaluate_crossdomain")


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
    # physical_prior_by_class holds nested PhysicalPriorParams dicts asdict()'d;
    # FutureRiskConfig's default_factory already builds a correct table, and
    # since it's fixed/deterministic (not learned), reuse the default rather
    # than reconstruct nested dataclasses from plain dicts.
    model = CARM(
        # dropout must match training (0.1) -- it changes _mlp's *architecture*
        # (whether a Dropout layer/index exists in the Sequential at all), not
        # just its behavior, so a mismatched value breaks state_dict loading
        # even though .eval() would make dropout a no-op either way.
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
    R_per_agent = out["R"].unsqueeze(0)  # (1, J)
    R_instance = aggregate_instance_risk(R_per_agent)
    return float(R_instance.item())


def analytic_margin_and_risk(pre, cfg: FutureRiskConfig):
    margins = [context_dependent_margin(pre.tau_3d[aid], pre.tau_phy[aid]) for aid in pre.tau_3d]
    if not margins:
        return np.full(len(pre.times), np.inf), np.zeros(len(pre.times))
    margin_t = np.min(np.stack(margins), axis=0)
    risk_t = instantaneous_risk(margin_t, T_m=cfg.T_m)
    return margin_t, risk_t


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"evaluate_crossdomain_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")
    t0 = time.time()

    log.info(f"[1/5] Loading checkpoint {CHECKPOINT_PATH} (Bench2Drive-Base-only training) ...")
    model, cfg, ckpt = load_checkpoint()
    log.info(f"      d_kin={ckpt['d_kin']} d_int={ckpt['d_int']} d_ctx={ckpt['d_ctx']}"
             f" best_val_auroc={ckpt.get('best_val_auroc')} best_val_auprc={ckpt.get('best_val_auprc')}")
    bench2drive_split = ckpt["split"]

    log.info(f"[2/5] Loading DeepAccident preprocessed instances from {DEEPACCIDENT_PREPROCESSED} ...")
    da_pre_by_id = {}
    for p in Path(DEEPACCIDENT_PREPROCESSED).glob("*.npz"):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is not None:
            da_pre_by_id[pre.instance_id] = pre
    log.info(f"      -> {len(da_pre_by_id)} usable DeepAccident instances")

    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)
    log.info(f"      -> {len(native_labels)} native (ground-truth, tau_3D-independent) collision labels loaded"
             f" from {NATIVE_LABELS_PATH}")

    log.info("[3/5] Running the Bench2Drive-only model on every DeepAccident instance (zero-shot) ...")
    rows = []
    da_case_traces = {}  # for lead-time metric: instance_id -> (times, risk_t, collision_time or None)
    for i, (iid, pre) in enumerate(da_pre_by_id.items()):
        raw = raw_instance_from_preprocessed(pre)
        R_pred = predict_R(model, cfg, raw, pre)
        y_analytic = 0.0
        margins = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
        if margins:
            from futurerisk.safety_prior import analytic_trajectory_risk
            y_analytic = analytic_trajectory_risk(margins, T_m=cfg.T_m)
        geometric_collision_label = int(any(f.any() for f in raw.agent_collision_flags.values()))
        native_entry = native_labels.get(iid)
        native_collision_label = int(native_entry["collision_label"]) if native_entry is not None else None
        tau3d_min = min((np.min(pre.tau_3d[a]) for a in pre.tau_3d), default=np.inf)
        ttc_score = float(raw_ttc_baseline_score(np.array([tau3d_min]), ttc_cap=cfg.ttc_cap)[0])

        margin_t, risk_t = analytic_margin_and_risk(pre, cfg)
        collision_time = None
        for aid, flags in raw.agent_collision_flags.items():
            if flags.any():
                idx = int(np.argmax(flags))
                t = float(pre.times[idx])
                collision_time = t if collision_time is None else min(collision_time, t)
        da_case_traces[iid] = (pre.times, risk_t, collision_time)

        rows.append({
            "instance_id": iid, "num_agents": len(pre.tau_3d),
            "R_predicted": R_pred, "y_analytic": y_analytic, "raw_ttc_score": ttc_score,
            "geometric_collision_label": geometric_collision_label,
            "native_collision_label": native_collision_label,
        })
        if (i + 1) % 1000 == 0:
            log.info(f"      ... {i+1}/{len(da_pre_by_id)} instances scored")

    csv_path = RESULTS_DIR / "predictions_deepaccident_zeroshot.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"      wrote {len(rows)} rows -> {csv_path}")

    log.info("[4/5] Discrimination metrics (S3.10 metric 1): model vs. analytic-only vs. raw-TTC baseline ...")
    log.info("      Ground truth = DeepAccident's OWN collision_intensity/collision_*_vehicle_id annotation,"
             " independent of tau_3D -- NOT the geometric flag (using the geometric flag here would make"
             " y_analytic/raw_ttc's scores tautological with the label, since both are tau_3D-derived).")
    valid_rows = [r for r in rows if r["native_collision_label"] is not None]
    log.info(f"      ({len(rows) - len(valid_rows)} instances missing a native label, excluded from this section)")
    R = np.array([r["R_predicted"] for r in valid_rows])
    y = np.array([r["y_analytic"] for r in valid_rows])
    ttc = np.array([r["raw_ttc_score"] for r in valid_rows])
    lbl = np.array([r["native_collision_label"] for r in valid_rows])
    geo_lbl = np.array([r["geometric_collision_label"] for r in valid_rows])
    n_pos = int(lbl.sum())
    agreement = float((lbl == geo_lbl).mean())
    log.info(f"      n={len(valid_rows)}  collisions(native)={n_pos} ({100*n_pos/len(valid_rows):.1f}%)"
             f"  collisions(geometric)={int(geo_lbl.sum())} ({100*geo_lbl.mean():.1f}%)"
             f"  label agreement={agreement:.3f}")
    for name, scores in (("CARM (Bench2Drive-only, zero-shot)", R), ("analytic-only (y_analytic)", y),
                         ("raw 3D-TTC baseline", ttc)):
        m = discrimination_metrics(scores, lbl)
        log.info(f"      {name:40s}  AUROC={m['auroc']:.3f}  AUPRC={m['auprc']:.3f}")

    log.info("[5/5] Early-warning lead time + calibration ...")
    # Operating threshold from Bench2Drive VAL (non-colliding instances, target 5% FPR)
    val_traces = []
    for iid in bench2drive_split["val"]:
        p = Path(BENCH2DRIVE_PREPROCESSED) / f"{iid}.npz"
        if not p.exists():
            continue
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        raw = raw_instance_from_preprocessed(pre)
        is_collision = any(f.any() for f in raw.agent_collision_flags.values())
        if is_collision:
            continue
        _, risk_t = analytic_margin_and_risk(pre, cfg)
        val_traces.append(risk_t)
    threshold = operating_point_threshold(val_traces, target_fpr=0.05) if val_traces else 0.5
    log.info(f"      operating threshold (from {len(val_traces)} non-colliding Bench2Drive val instances,"
             f" target 5% FPR): {threshold:.4f}")

    # Cohort = instances DeepAccident's own annotation says truly collided;
    # collision_time itself still comes from our geometric detector (the
    # only source with frame-level timing) -- an instance natively flagged
    # as a collision but where the geometric detector never found a matching
    # overlap (window edge effects, near-collision-then-recover physics
    # difference, etc.) has no usable impact time and is skipped, reported
    # explicitly below rather than silently dropped.
    lead_times = []
    n_colliding = 0
    n_no_geometric_time = 0
    for r in valid_rows:
        if r["native_collision_label"] != 1:
            continue
        n_colliding += 1
        times, risk_t, collision_time = da_case_traces[r["instance_id"]]
        if collision_time is None:
            n_no_geometric_time += 1
            continue
        lt = early_warning_lead_time(risk_t, times, threshold, collision_time)
        if lt is not None:
            lead_times.append(lt)
    lead_times = np.array(lead_times)
    log.info(f"      DeepAccident colliding instances (native label): {n_colliding}"
             f" ({n_no_geometric_time} of these have no geometric impact-time match, excluded below)")
    denom = n_colliding - n_no_geometric_time
    log.info(f"      warned before impact: {len(lead_times)}/{denom}"
             f" ({100*len(lead_times)/max(denom,1):.1f}%)")
    if len(lead_times):
        log.info(f"      lead time (s): mean={lead_times.mean():.2f} median={np.median(lead_times):.2f}"
                 f" min={lead_times.min():.2f} max={lead_times.max():.2f}")

    rd = reliability_diagram(R, lbl, n_bins=10)
    log.info(f"      calibration ECE (model R vs. true collision frequency): {rd.ece:.4f}")
    for i in range(len(rd.bin_counts)):
        if rd.bin_counts[i] > 0:
            log.info(f"        bin [{rd.bin_edges[i]:.1f},{rd.bin_edges[i+1]:.1f}) n={rd.bin_counts[i]:5d}"
                     f"  confidence={rd.bin_confidence[i]:.3f}  empirical_freq={rd.bin_accuracy[i]:.3f}")

    summary = {
        "n_instances": len(valid_rows), "n_instances_missing_native_label": len(rows) - len(valid_rows),
        "n_collisions_native": n_pos, "n_collisions_geometric": int(geo_lbl.sum()),
        "native_geometric_label_agreement": agreement,
        "auroc_model": discrimination_metrics(R, lbl)["auroc"],
        "auprc_model": discrimination_metrics(R, lbl)["auprc"],
        "auroc_analytic": discrimination_metrics(y, lbl)["auroc"],
        "auprc_analytic": discrimination_metrics(y, lbl)["auprc"],
        "auroc_raw_ttc": discrimination_metrics(ttc, lbl)["auroc"],
        "auprc_raw_ttc": discrimination_metrics(ttc, lbl)["auprc"],
        "operating_threshold": threshold,
        "n_colliding_instances_native": n_colliding,
        "n_colliding_no_geometric_time_match": n_no_geometric_time,
        "n_warned_before_impact": len(lead_times),
        "lead_time_mean_s": float(lead_times.mean()) if len(lead_times) else None,
        "lead_time_median_s": float(np.median(lead_times)) if len(lead_times) else None,
        "calibration_ece": rd.ece,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"      wrote {RESULTS_DIR / 'summary.json'}")

    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
