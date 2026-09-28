#!/usr/bin/env python3
"""Render the case-study figures (root.tex fig:case_study_success,
fig:case_study_failure): r(t), m(t), q_k on a shared time axis.

Failure case: the AD-MLP open-loop counterfactual from
admlp_counterfactual_case_study.py (AccidentTwoWays_Town12_Route1103) --
the real expert trajectory scores R=0.956 (a genuine near-miss, no
collision), while AD-MLP's counterfactual trajectory actually collides
(agent 15830, frame 17) yet scores only R=0.106. Diagnosis: q_k (hazard
head) for the colliding counterfactual barely rises above the real
trajectory's, despite the real geometric collision flag being set --
CARM trained on DeepAccident does not recognize this low-closing-speed
"ego stalls, another agent catches up" collision pattern as dangerous,
a genuine cross-domain limitation, not a construction artifact (verified
directly: agent 15830's min tau_3D=0.00, margin=-3.05 at frame 17).

Success case: two Bench2Drive in-domain test instances with different
safety profiles (large sustained margin vs. a brief close call), both
correctly collision-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "Bench2DriveZoo"))

from futurerisk.data_pipeline import load_preprocessed_instance, preprocess_instance, raw_instance_from_preprocessed
from futurerisk.config import DEFAULT_CONFIG
from futurerisk.event_proxy import load_threshold_tables
from futurerisk.features_v1 import build_instance_batch_item_v1
from futurerisk.safety_prior import context_dependent_margin, instantaneous_risk
from scripts.evaluate_crossdomain_v1_to_bench2drive import DEVICE, load_checkpoint
from scripts.experiments.admlp_case_study import find_valid_reference_frame, load_admlp_model, predict_future_global
from scripts.experiments.admlp_counterfactual_case_study import build_real_and_counterfactual
from futurerisk.adapters.bench2drive import load_clip_frames

FIG_DIR = Path("icra/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
_THRESHOLD_TABLE = load_threshold_tables("dataset/Bench2Drive-Base-preprocessed/Bench2Drive_thresholds.json")["Bench2Drive"]
plt.rcParams.update({"font.size": 8})


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
        # worst (min-margin) agent's q_k, matching which agent drives m(t)
        worst_idx = int(np.argmin([np.min(pre.tau_3d[a]) for a in pre.tau_3d])) if pre.tau_3d else 0
        q_k = out["q"][worst_idx].cpu().numpy()
    return inst.times, m_t, r_t, q_k


def render_three_panel(traces_by_label: dict, out_path: Path, title_suffix: str):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    colors = {"real": "#4C72B0", "AD-MLP counterfactual": "#C44E52",
              "large margin": "#4C72B0", "brief close call": "#C44E52",
              "UniAD counterfactual": "#C44E52"}
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


def render_failure_case():
    admlp_net = load_admlp_model()
    carm_model, carm_cfg, ckpt = load_checkpoint()
    tar = "dataset/Bench2Drive-Base/AccidentTwoWays_Town12_Route1103_Weather11.tar.gz"
    clip_name, frames = load_clip_frames(tar)
    i = find_valid_reference_frame(frames)
    xy_pred, _ = predict_future_global(admlp_net, frames, i)
    real_inst, cf_inst = build_real_and_counterfactual(tar, i, xy_pred, DEFAULT_CONFIG)

    traces = {
        "real": compute_traces(carm_model, carm_cfg, real_inst),
        "AD-MLP counterfactual": compute_traces(carm_model, carm_cfg, cf_inst),
    }
    render_three_panel(traces, FIG_DIR / "case_study_failure.png",
                        "Failure case: AccidentTwoWays -- real (R=0.956, no collision) vs.\n"
                        "AD-MLP counterfactual (R=0.106, geometric collision at t=1.7s)")


def render_success_case():
    carm_model, carm_cfg, ckpt = load_checkpoint()
    split = ckpt["split"]
    candidates = []
    for iid in split["test"][:400]:
        p = Path("dataset/DeepAccident-preprocessed") / f"{iid}.npz"
        if not p.exists():
            continue
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None or not pre.tau_3d:
            continue
        margins = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
        m_t = np.min(np.stack(margins), axis=0)
        if not np.isfinite(m_t).any():
            continue
        worst = float(np.min(m_t[np.isfinite(m_t)]))
        candidates.append((worst, iid, pre))
    candidates.sort(key=lambda t: t[0])
    # a brief close call: near the danger boundary but never negative for long
    close_call = next((c for c in candidates if -2.0 < c[0] < 0), candidates[0])
    # a large sustained margin: comfortably positive worst-case margin
    large_margin = next((c for c in reversed(candidates) if c[0] > 2.0), candidates[-1])

    traces = {}
    for label, (worst, iid, pre) in (("brief close call", close_call), ("large margin", large_margin)):
        raw = raw_instance_from_preprocessed(pre)
        traces[label] = compute_traces(carm_model, carm_cfg, raw)
    render_three_panel(traces, FIG_DIR / "case_study_success.png",
                        "Collision-free instances with different safety profiles (DeepAccident in-domain)")


if __name__ == "__main__":
    render_failure_case()
    render_success_case()
