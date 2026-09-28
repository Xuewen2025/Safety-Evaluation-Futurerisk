#!/usr/bin/env python3
"""Render root.tex fig:case_study_uniad_vs_vad: the single-scenario case
behind Sec. IV-C's explanation of Table IV's UniAD/VAD rank swap.

On AccidentTwoWays_Town12_Route1105_Weather13, UniAD-Tiny's substituted
trajectory (ref frame 160) has a brief, early close encounter with agent
41934 (center-to-center distance down to 2.77m at t=0.6s) that registers
as an actual box overlap (geometric collision=1), then the two separate
for the rest of the window; CARM scores it R=0.962. VAD-Base's substituted
trajectory on the same clip (ref frame 140, a few frames earlier -- each
model's own worst-prediction-divergence scan can land on a different
reference frame, per Sec. IV-C's protocol) never overlaps any agent's box
within the observed 4s window (geometric collision=0) -- but its distance
to agent 41977 shrinks *monotonically* the entire window, from 57.8m at
t=0 to just 3.4m at t=4s (the window's edge), still closing at roughly
4 m/s when the horizon cuts off. Because this convergence never actually
crosses into a projected box overlap within the fixed 4s horizon, the
closed-form analytic margin m(t) stays at +infinity throughout (r(t)==0
the whole trace) -- collision rate and the closed-form geometric proxy
both see nothing here. Yet CARM's hazard head partially picks up the
closing trend (q_k rises to 0.45, panel c) and the full model scores it
R=0.975 -- almost identical to UniAD's actual collision, and higher than
UniAD's. This is not "FutureRisk imagining danger nothing else sees": a
car closing from 57.8m to 3.4m in 4 seconds and still approaching when the
window ends is a trajectory any human would flag as heading toward
trouble; a binary collision-rate metric (and even our own geometric proxy)
simply cannot see that at all until the window either extends further or
an actual overlap occurs, while FutureRisk's learned hazard head can pick
up on the closing dynamics directly.

This single instance also explains most of Table IV's rank swap:
dropping it from VAD's 15-scenario mean R (0.423) brings VAD's mean down
to 0.383, matching UniAD's mean R (0.382) almost exactly -- computed
inline below from the same cached counterfactual JSONs Table IV itself
was built from, not re-derived by hand.
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

from futurerisk.data_pipeline import preprocess_instance
from futurerisk.event_proxy import load_threshold_tables
from futurerisk.features_v1 import build_instance_batch_item_v1
from futurerisk.geometry import rect_corners
from futurerisk.safety_prior import context_dependent_margin, instantaneous_risk
from scripts.evaluate_crossdomain_v1_to_bench2drive import DEVICE, load_checkpoint
from scripts.experiments.uniad_counterfactual_case_study import (
    build_real_and_counterfactual as build_uniad_rc,
)
from scripts.experiments.vad_counterfactual_case_study import (
    build_real_and_counterfactual as build_vad_rc,
)

FIG_DIR = Path("icra/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
_THRESHOLD_TABLE = load_threshold_tables("dataset/Bench2Drive-Base-preprocessed/Bench2Drive_thresholds.json")["Bench2Drive"]
plt.rcParams.update({"font.size": 8})

CLIP_NAME = "AccidentTwoWays_Town12_Route1105_Weather13"


@torch.no_grad()
def compute_traces(model, cfg, inst, highlight_agent_id=None):
    """Trajectory-level m(t)/r(t) (min/derived over ALL agents, unaffected
    by which single agent panel (c) highlights) plus one agent's q_k.

    `highlight_agent_id`, when given, picks panel (c)'s agent explicitly --
    used here to show the specific agent each model's story is actually
    about (UniAD: the one it collides with; VAD: the one it is
    monotonically closing on), rather than an automatic tie-breaking rule.
    Neither `argmin(tau_3D)` (ties arbitrarily when every agent's analytic
    tau_3D is +infinity, as happens for VAD below -- no agent's box-overlap
    projection ever fires within the window) nor `argmax(R)` (picks
    whichever agent CARM itself rates worst, which for UniAD here turns
    out to be a different, unrelated agent to the one it actually collides
    with) reliably points at the agent the surrounding text is discussing,
    so the caller must say which one explicitly.
    """
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
        agent_ids = list(inst.agent_boxes.keys())
        if highlight_agent_id is not None and highlight_agent_id in agent_ids:
            idx = agent_ids.index(highlight_agent_id)
        else:
            idx = int(torch.argmax(out["R"]).item())
        q_k = out["q"][idx].cpu().numpy()
    return inst.times, m_t, r_t, q_k


def _to_local_frame(ego_box, other_box, dt: float):
    """Transform ego's and one agent's (x, y, heading) into a frame with
    origin at ego's own position at t=0 and ego's own heading at t=0
    pointing along +y ("forward is up"), so the two scenes (UniAD's and
    VAD's, at completely different real-world coordinates) can be drawn at
    comparable scale/orientation side by side."""
    x0, y0 = ego_box.x[0], ego_box.y[0]
    theta0 = ego_box.heading[0] - np.pi / 2  # rotate so heading0 maps to +y

    def rot(px, py):
        dx, dy = px - x0, py - y0
        c, s = np.cos(-theta0), np.sin(-theta0)
        return c * dx - s * dy, s * dx + c * dy

    ego_lx, ego_ly = rot(ego_box.x, ego_box.y)
    other_lx, other_ly = rot(other_box.x, other_box.y)
    return ego_lx, ego_ly, ego_box.heading - theta0, other_lx, other_ly, other_box.heading - theta0


def _draw_box(ax, cx, cy, heading, length, width, color, alpha=1.0, zorder=3):
    corners = rect_corners(np.array([cx]), np.array([cy]), np.array([heading]),
                            np.array([length]), np.array([width]))[0]
    poly = plt.Polygon(corners, closed=True, facecolor=color, edgecolor="black",
                        linewidth=0.5, alpha=alpha, zorder=zorder)
    ax.add_patch(poly)


def render_birdseye(ax, ego_box, agent_box, times, key_idx, key_label, ego_color, agent_color):
    """Top-down path plot: ego's and one agent's center-trajectory lines
    (faded near t=0, solid near the key moment) plus each box drawn as an
    actual rectangle at t=0 (light) and at the key moment (solid,
    outlined), so the reader sees not just a number but where the two
    vehicles' bodies actually are relative to each other."""
    ego_lx, ego_ly, ego_lh, agent_lx, agent_ly, agent_lh = _to_local_frame(ego_box, agent_box, times[1] - times[0])

    ax.plot(ego_lx, ego_ly, color=ego_color, linewidth=1.2, alpha=0.6, zorder=2)
    ax.plot(agent_lx, agent_ly, color=agent_color, linewidth=1.2, alpha=0.6, zorder=2)

    _draw_box(ax, ego_lx[0], ego_ly[0], ego_lh[0], ego_box.length[0], ego_box.width[0],
              ego_color, alpha=0.25, zorder=2)
    _draw_box(ax, agent_lx[0], agent_ly[0], agent_lh[0], agent_box.length[0], agent_box.width[0],
              agent_color, alpha=0.25, zorder=2)
    _draw_box(ax, ego_lx[key_idx], ego_ly[key_idx], ego_lh[key_idx],
              ego_box.length[key_idx], ego_box.width[key_idx], ego_color, alpha=0.95, zorder=4)
    _draw_box(ax, agent_lx[key_idx], agent_ly[key_idx], agent_lh[key_idx],
              agent_box.length[key_idx], agent_box.width[key_idx], agent_color, alpha=0.95, zorder=4)

    d_key = np.hypot(ego_lx[key_idx] - agent_lx[key_idx], ego_ly[key_idx] - agent_ly[key_idx])
    ax.annotate(f"t={times[key_idx]:.1f}s\n{key_label}\nd={d_key:.1f}m",
                xy=((ego_lx[key_idx] + agent_lx[key_idx]) / 2, (ego_ly[key_idx] + agent_ly[key_idx]) / 2),
                xytext=(6, 6), textcoords="offset points", fontsize=5.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("local x (m)"); ax.set_ylabel("local y (m, ego heading at t=0)")


def render_four_panel(traces_by_label: dict, birdseye_specs: list, out_path: Path, title_suffix: str):
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.3))
    colors = {"UniAD counterfactual (collides, R=0.962)": "#C44E52",
              "VAD counterfactual (no collision, R=0.975)": "#4C72B0"}
    for label, (times, m_t, r_t, q_k) in traces_by_label.items():
        c = colors.get(label, None)
        axes[0].plot(times, r_t, label=label, color=c)
        bin_times = np.linspace(0, times[-1], len(q_k))
        axes[3].plot(bin_times, q_k, label=label, color=c, marker="o", markersize=2)
    axes[0].set_title("(a) $r(t)$"); axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("$r(t)$")
    axes[0].legend(fontsize=5, loc="upper left")
    axes[3].set_title("(c) $q_k$"); axes[3].set_xlabel("t (s)"); axes[3].set_ylabel("$q_k$")

    for ax_idx, spec in zip((1, 2), birdseye_specs):
        spec = dict(spec)
        panel_title = spec.pop("panel_title")
        render_birdseye(axes[ax_idx], **spec)
        axes[ax_idx].set_title(f"(b{ax_idx}) {panel_title}", fontsize=7)

    fig.suptitle(title_suffix, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    carm_model, carm_cfg, ckpt = load_checkpoint()

    with open("dataset/results_v1/uniad_predictions.json") as fh:
        uniad_preds = {r["clip_name"]: r for r in json.load(fh)}
    with open("dataset/results_v1/vad_predictions.json") as fh:
        vad_preds = {r["clip_name"]: r for r in json.load(fh)}

    u = uniad_preds[CLIP_NAME]
    v = vad_preds[CLIP_NAME]
    tar_path = u["tar_path"]

    _, uniad_cf_inst = build_uniad_rc(tar_path, u["ref_frame"], np.array(u["pred_xy"]), carm_cfg)
    _, vad_cf_inst = build_vad_rc(tar_path, v["ref_frame"], np.array(v["pred_xy"]), carm_cfg)

    traces = {
        # panel (c) highlights the specific agent each finding is about:
        # the one UniAD's path actually collides with, and the one VAD's
        # path is monotonically closing on (Sec. IV-C; both confirmed by
        # direct distance/collision-flag inspection, not by an automatic
        # tie-breaking rule -- see compute_traces' docstring).
        "UniAD counterfactual (collides, R=0.962)":
            compute_traces(carm_model, carm_cfg, uniad_cf_inst, highlight_agent_id="41934"),
        "VAD counterfactual (no collision, R=0.975)":
            compute_traces(carm_model, carm_cfg, vad_cf_inst, highlight_agent_id="41977"),
    }

    # Bird's-eye panels: key_idx=6 is UniAD's closest approach within its
    # collision window (t=0.6s, d=2.77m, still inside the t=0.0-0.9s
    # overlap span); key_idx=-1 is VAD's window edge (t=4.0s, d=3.36m,
    # still closing) -- both picked from the direct distance/collision-flag
    # inspection in this module's docstring, not an automatic rule.
    birdseye_specs = [
        dict(ego_box=uniad_cf_inst.ego_box, agent_box=uniad_cf_inst.agent_boxes["41934"],
             times=uniad_cf_inst.times, key_idx=6, key_label="closest (collision)",
             ego_color="#C44E52", agent_color="#F0A0A8",
             panel_title="UniAD vs. agent 41934 (collides)"),
        dict(ego_box=vad_cf_inst.ego_box, agent_box=vad_cf_inst.agent_boxes["41977"],
             times=vad_cf_inst.times, key_idx=-1, key_label="window ends\n(still closing)",
             ego_color="#4C72B0", agent_color="#9DB8DC",
             panel_title="VAD vs. agent 41977 (no collision)"),
    ]

    render_four_panel(
        traces, birdseye_specs, FIG_DIR / "case_study_uniad_vs_vad.png",
        f"{CLIP_NAME}: VAD closes 57.8m$\\to$3.4m (still converging at t=4s,\n"
        "no box overlap yet) and scores almost as high as UniAD's actual collision",
    )

    # Cross-check the "drop the one instance" arithmetic against the cached
    # counterfactual JSONs Table IV itself was built from.
    with open("dataset/results_v1/vad_counterfactual_case_studies.json") as fh:
        vad_cf = json.load(fh)
    r_values = [row["R_vad_counterfactual"] for row in vad_cf]
    full_mean = sum(r_values) / len(r_values)
    without_1105 = [row["R_vad_counterfactual"] for row in vad_cf if "Route1105" not in row["clip_name"]]
    dropped_mean = sum(without_1105) / len(without_1105)
    print(f"VAD mean R over all {len(r_values)}: {full_mean:.3f}; "
          f"dropping Route1105: {dropped_mean:.3f} (n={len(without_1105)})")


if __name__ == "__main__":
    main()
