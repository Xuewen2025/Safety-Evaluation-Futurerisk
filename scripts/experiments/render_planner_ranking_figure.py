#!/usr/bin/env python3
"""Render the planner-ranking success-case figure (root.tex
fig:case_study_planner_ranking): r(t), m(t), q_k for the real recorded ego
trajectory vs. UniAD's substituted counterfactual on the same clip, the
best-separated concordant pair in uniad_counterfactual_case_studies.json
(AccidentTwoWays_Town12_Route1102_Weather10, ref_frame=60): real R=0.037
(no collision) vs. UniAD counterfactual R=0.914 (genuine collision) --
the metric correctly ranks the substituted trajectory as far more
dangerous, the mirror image of the AD-MLP failure case in
case_study_failure.png.

Self-contained (does not import render_case_study_figures.py / the ADMLP
case-study chain, whose ADMLP.model dependency is unavailable in this
environment) -- compute_traces/render_three_panel are duplicated here from
render_case_study_figures.py rather than imported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "Bench2DriveZoo"))

from futurerisk.config import DEFAULT_CONFIG
from futurerisk.data_pipeline import preprocess_instance
from futurerisk.event_proxy import load_threshold_tables
from futurerisk.features_v1 import build_instance_batch_item_v1
from futurerisk.safety_prior import context_dependent_margin, instantaneous_risk
from scripts.evaluate_crossdomain_v1_to_bench2drive import DEVICE, load_checkpoint
from scripts.experiments.uniad_counterfactual_case_study import build_real_and_counterfactual

FIG_DIR = Path("icra/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
_THRESHOLD_TABLE = load_threshold_tables("dataset/Bench2Drive-Base-preprocessed/Bench2Drive_thresholds.json")["Bench2Drive"]
plt.rcParams.update({"font.size": 8})

CLIP_NAME = "AccidentTwoWays_Town12_Route1102_Weather10"
REF_FRAME = 60


@torch.no_grad()
def compute_traces(model, cfg, inst):
    pre = preprocess_instance(inst, _THRESHOLD_TABLE, cfg=cfg)
    margins = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
    m_t = np.min(np.stack(margins), axis=0) if margins else np.full(len(inst.times), np.inf)
    r_t = instantaneous_risk(m_t, T_m=cfg.T_m)

    item = build_instance_batch_item_v1(inst, pre, cfg)
    if item["x_ego"].shape[0] == 0:
        q_k = np.zeros(cfg.num_bins)
    else:
        x_ego = torch.from_numpy(item["x_ego"]).to(DEVICE)
        X_obj = torch.from_numpy(item["X_obj"]).to(DEVICE)
        x_ctx = torch.from_numpy(item["x_ctx_v1"]).to(DEVICE)
        tau_phy_seq = torch.from_numpy(item["tau_phy_seq"]).to(DEVICE)
        bin_edges = torch.tensor(cfg.bin_edges, dtype=torch.float32, device=DEVICE)
        out = model(x_kin=x_ego, X_int=X_obj, x_ctx=x_ctx, tau_phy_seq=tau_phy_seq, bin_edges=bin_edges)
        worst_idx = int(np.argmin([np.min(pre.tau_3d[a]) for a in pre.tau_3d])) if pre.tau_3d else 0
        q_k = out["q"][worst_idx].cpu().numpy()
    return inst.times, m_t, r_t, q_k


def render_three_panel(traces_by_label: dict, out_path: Path, title_suffix: str):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    colors = {"real": "#4C72B0", "UniAD counterfactual": "#C44E52"}
    for label, (times, m_t, r_t, q_k) in traces_by_label.items():
        c = colors.get(label, None)
        axes[0].plot(times, r_t, label=label, color=c)
        m_plot = np.clip(m_t, -10, 10)
        axes[1].plot(times, m_plot, label=label, color=c)
        bin_times = np.linspace(0, times[-1], len(q_k))
        axes[2].plot(bin_times, q_k, label=label, color=c, marker="o", markersize=2)
    axes[0].set_title("(a) $r(t)$"); axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("$r(t)$")
    axes[1].set_title("(b) $m(t)$"); axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("$m(t)$ (clipped $\\pm10$)")
    axes[1].axhline(0, color="gray", linewidth=0.6)
    axes[2].set_title("(c) $q_k$"); axes[2].set_xlabel("t (s)"); axes[2].set_ylabel("$q_k$")
    axes[0].legend(fontsize=6, loc="upper left")
    fig.suptitle(title_suffix, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    carm_model, carm_cfg, ckpt = load_checkpoint()

    with open("dataset/results_v1/uniad_predictions.json") as fh:
        predictions = json.load(fh)
    row = next(r for r in predictions if r["clip_name"] == CLIP_NAME and r["ref_frame"] == REF_FRAME)
    xy_pred = np.array(row["pred_xy"])

    real_inst, cf_inst = build_real_and_counterfactual(row["tar_path"], REF_FRAME, xy_pred, DEFAULT_CONFIG)

    traces = {
        "real": compute_traces(carm_model, carm_cfg, real_inst),
        "UniAD counterfactual": compute_traces(carm_model, carm_cfg, cf_inst),
    }
    render_three_panel(
        traces, FIG_DIR / "case_study_planner_ranking.png",
        "Ranking success: AccidentTwoWays -- real (R=0.037, no collision) vs.\n"
        "UniAD counterfactual (R=0.914, geometric collision)",
    )


if __name__ == "__main__":
    main()
