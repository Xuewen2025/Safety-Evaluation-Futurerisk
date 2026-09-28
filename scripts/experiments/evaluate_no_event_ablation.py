#!/usr/bin/env python3
"""Consolidated, side-by-side evaluation of Full CARM vs. the w/o-L_evt
ablation (carm_deepaccident_v1_no_event.pt), using IDENTICAL evaluation
code for both checkpoints so the comparison is apples-to-apples (and, as a
side effect, freshly re-verifies Full CARM's own in-domain/cross-domain
numbers rather than trusting whatever is currently written in root.tex).

For each checkpoint, computes:
  - AUROC_in / AUPRC_in   : DeepAccident test split, NATIVE collision label
  - AUROC_cross / AUPRC_cross : Bench2Drive (all 3679 instances, zero-shot),
                                 geometric collision label
  - rho(R, a_max)         : DeepAccident train+val+test pooled,
                             native-collision-free subset (same pooling
                             Table finegrained_baselines/v1_full_eval.py use)
  - rho(R, -m)            : same subset, restricted to instances where the
                             geometric margin is finite

a_max / m follow the ablation spec's own definitions (max evasive-severity
proxy / worst-case geometric margin), matching `severity_proxies()` in
v1_full_eval.py exactly (same function, imported not reimplemented).
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

from futurerisk.carm_model import CARM, aggregate_instance_risk
from futurerisk.config import FutureRiskConfig
from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import discrimination_metrics
from futurerisk.features_v1 import build_instance_batch_item_v1
from scripts.experiments.v1_full_eval import severity_proxies

DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
RESULTS_DIR = Path("dataset/results_v1")
LOGS_DIR = Path("dataset/logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "full_carm": "dataset/checkpoints/carm_deepaccident_v1.pt",
    "no_event": "dataset/checkpoints/carm_deepaccident_v1_no_event.pt",
}

log = logging.getLogger("evaluate_no_event_ablation")


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


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = FutureRiskConfig(**{k: v for k, v in ckpt["cfg"].items() if k != "physical_prior_by_class"})
    model = CARM(
        d_kin=ckpt["d_ego"], d_int=ckpt["d_obj"], d_ctx=ckpt["d_ctx_v1"], cfg=cfg,
        hidden_dim=64, rnn_hidden=64, dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def predict_R(model: CARM, cfg: FutureRiskConfig, raw, pre) -> float:
    item = build_instance_batch_item_v1(raw, pre, cfg)
    if item["x_ego"].shape[0] == 0:
        return 0.0
    x_ego = torch.from_numpy(item["x_ego"]).to(DEVICE)
    X_obj = torch.from_numpy(item["X_obj"]).to(DEVICE)
    x_ctx_v1 = torch.from_numpy(item["x_ctx_v1"]).to(DEVICE)
    tau_phy_seq = torch.from_numpy(item["tau_phy_seq"]).to(DEVICE)
    bin_edges = torch.tensor(cfg.bin_edges, dtype=torch.float32, device=DEVICE)
    out = model(x_kin=x_ego, X_int=X_obj, x_ctx=x_ctx_v1, tau_phy_seq=tau_phy_seq, bin_edges=bin_edges)
    R_instance = aggregate_instance_risk(out["R"].unsqueeze(0))
    return float(R_instance.item())


def evaluate_one(label: str, ckpt_path: str, native_labels: dict, bd_paths: list, log) -> dict:
    log.info(f"--- {label} ({ckpt_path}) ---")
    model, cfg, ckpt = load_checkpoint(ckpt_path)
    split = ckpt["split"]
    test_ids = set(split["test"])

    log.info(f"      scoring all DeepAccident instances (train+val+test pooled) ...")
    da_rows = []
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
            raw = raw_instance_from_preprocessed(pre)
            R_pred = predict_R(model, cfg, raw, pre)
            row = {"instance_id": iid, "part": part, "R_predicted": R_pred,
                   "native_collision_label": int(lbl_entry["collision_label"])}
            row.update(severity_proxies(pre))
            da_rows.append(row)
    log.info(f"      -> {len(da_rows)} DeepAccident instances scored")

    test_rows = [r for r in da_rows if r["part"] == "test"]
    R_test = np.array([r["R_predicted"] for r in test_rows])
    lbl_test = np.array([r["native_collision_label"] for r in test_rows])
    m_in = discrimination_metrics(R_test, lbl_test)
    log.info(f"      in-domain (native, test)  n={len(test_rows)} coll={int(lbl_test.sum())}"
             f"  AUROC={m_in['auroc']:.4f}  AUPRC={m_in['auprc']:.4f}")

    log.info(f"      scoring all Bench2Drive instances (zero-shot) ...")
    bd_rows = []
    for i, p in enumerate(bd_paths):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        raw = raw_instance_from_preprocessed(pre)
        R_pred = predict_R(model, cfg, raw, pre)
        geometric_collision_label = int(any(f.any() for f in raw.agent_collision_flags.values()))
        bd_rows.append({"instance_id": pre.instance_id, "R_predicted": R_pred,
                         "geometric_collision_label": geometric_collision_label})
        if (i + 1) % 1000 == 0:
            log.info(f"      ... {i+1}/{len(bd_paths)} Bench2Drive instances scored")
    R_bd = np.array([r["R_predicted"] for r in bd_rows])
    lbl_bd = np.array([r["geometric_collision_label"] for r in bd_rows])
    m_cross = discrimination_metrics(R_bd, lbl_bd)
    log.info(f"      cross-domain (geometric, Bench2Drive)  n={len(bd_rows)} coll={int(lbl_bd.sum())}"
             f"  AUROC={m_cross['auroc']:.4f}  AUPRC={m_cross['auprc']:.4f}")

    free = [r for r in da_rows if r["native_collision_label"] == 0]
    sev = np.array([r["max_evasive_severity"] for r in free])
    R_free = np.array([r["R_predicted"] for r in free])
    rho_sev, p_sev = spearmanr(R_free, sev)
    log.info(f"      rho(R, a_max)  n={len(free)}  rho={rho_sev:.4f}  p={p_sev:.2e}")

    margins = np.array([r["worst_margin"] for r in free])
    finite = np.isfinite(margins)
    n_m = int(finite.sum())
    if n_m >= 10 and len(set(margins[finite].tolist())) > 1:
        rho_m, p_m = spearmanr(R_free[finite], margins[finite])
        rho_m = -1.0 * rho_m  # sign convention matches Table finegrained_baselines: rho(R, -m)
    else:
        rho_m, p_m = float("nan"), float("nan")
    log.info(f"      rho(R, -m)     n={n_m}  rho={rho_m:.4f}  p={p_m:.2e}" if n_m >= 10 else
             f"      rho(R, -m)     n={n_m} (too few finite margins)")

    return {
        "checkpoint": ckpt_path,
        "in_domain": {"n": len(test_rows), "n_collisions": int(lbl_test.sum()),
                      "auroc": m_in["auroc"], "auprc": m_in["auprc"]},
        "cross_domain": {"n": len(bd_rows), "n_collisions": int(lbl_bd.sum()),
                          "auroc": m_cross["auroc"], "auprc": m_cross["auprc"]},
        "finegrained": {
            "rho_vs_a_max": {"n": len(free), "spearman_rho": float(rho_sev), "p_value": float(p_sev)},
            "rho_vs_neg_m": {"n": n_m, "spearman_rho": float(rho_m), "p_value": float(p_m)},
        },
    }


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"evaluate_no_event_ablation_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")
    t0 = time.time()

    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)
    bd_paths = sorted(Path(BENCH2DRIVE_PREPROCESSED).glob("*.npz"))
    log.info(f"Bench2Drive: {len(bd_paths)} preprocessed instances found")

    results = {}
    for label, ckpt_path in CHECKPOINTS.items():
        results[label] = evaluate_one(label, ckpt_path, native_labels, bd_paths, log)

    with open(RESULTS_DIR / "no_event_ablation_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"wrote {RESULTS_DIR / 'no_event_ablation_comparison.json'}")
    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
