"""Feature-tensor assembly (S3.6.1): turns a `RawTrajectoryInstance` plus
its cached `PreprocessedInstance` labels into the actual per-agent numpy
tensors CARM consumes (x_kin, X_int, x_ctx, tau_phy_seq) and the two
training targets (T_evt, y_analytic).

This is the piece that was explicitly deferred in earlier iterations of
this package (see README's "Dataset adapters are not included" /
`adapters/bench2drive.py`'s own scope) -- building it is what turns cached
preprocessing output into something `carm_model.CARM.forward` can actually
run on.

Feature choices below are a reasonable, minimal instantiation of S3.6.1's
feature list (relative kinematics / full-horizon relative motion+clearance
/ semantic context) using only what's available from `RawTrajectoryInstance`
+ `PreprocessedInstance` -- not the only valid instantiation, but internally
consistent and documented so it's easy to extend (e.g. road curvature and
speed limit are currently coarse placeholders since Bench2Drive-Mini's
`anno` frames don't carry a ready-made curvature/speed-limit field without
pulling in the separate `Bench2Drive-Map` HD-map dataset).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import FutureRiskConfig, DEFAULT_CONFIG
from .data_pipeline import PreprocessedInstance, RawTrajectoryInstance
from .interaction import NAME_TO_TYPE, TYPE_NAMES, wrap_to_pi
from .safety_prior import analytic_trajectory_risk, context_dependent_margin

AGENT_CLASS_VOCAB = ["vehicle", "cyclist", "pedestrian"]

D_KIN = 8   # [dx, dy, rel_vx, rel_vy, rel_heading, closing_speed, ego_speed, agent_speed]
D_INT = 6   # [dx, dy, dz, rel_vx, rel_vy, signed_clearance] per bin
# x_ctx = interaction-type *fractional* distribution (5) + relative-heading
# summary stats (3) + ego net heading change (1) + agent-class one-hot (3).
# Revised from an earlier version that used a single dominant-type one-hot +
# one crude curvature scalar: a real failure case (a left-turn-across-
# oncoming-traffic collision scored ~0 by the model) traced back to that
# dominant-type *mode* washing out the signal -- one specific agent's
# interaction-type breakdown for that instance was 26/58 frames
# "merging/lane-change", 20/58 "other", 0/58 "oncoming", even though the
# conflict was visibly an oncoming/crossing one. Handing the model the full
# fractional distribution plus the raw relative-heading range (which
# Algorithm 1's heuristic derives its categories from in the first place)
# lets it access the underlying geometric signal directly instead of
# depending entirely on one hard, mode-collapsed label. All of this comes
# from already-cached raw trajectories + interaction_type arrays -- no
# re-parsing of source archives needed.
D_CTX = len(TYPE_NAMES) + 3 + 1 + len(AGENT_CLASS_VOCAB)


def _bin_sample_indices(cfg: FutureRiskConfig, n_frames: int) -> np.ndarray:
    """Map the K hazard-bin left edges (0..H) onto the nearest raw-frame
    index at the instance's own dt, clipped to the instance's actual
    length (a window may be exactly `cfg.horizon_s` long, but this stays
    robust if it's ever shorter)."""
    edges = cfg.bin_edges[: cfg.num_bins]  # K left edges, excludes b_K
    idx = np.round(edges / cfg.dt).astype(int)
    return np.clip(idx, 0, max(n_frames - 1, 0))


def build_agent_feature(
    raw: RawTrajectoryInstance, pre: PreprocessedInstance, agent_id: str,
    cfg: FutureRiskConfig = DEFAULT_CONFIG,
) -> Dict[str, np.ndarray]:
    ego = raw.ego_box
    agent = raw.agent_boxes[agent_id]
    T = len(raw.times)
    dt = cfg.dt

    ego_vx, ego_vy = np.gradient(ego.x, dt), np.gradient(ego.y, dt)
    agent_vx, agent_vy = np.gradient(agent.x, dt), np.gradient(agent.y, dt)

    # --- x_kin: current-time (t=0) relative kinematics ---------------------
    dx0, dy0 = agent.x[0] - ego.x[0], agent.y[0] - ego.y[0]
    dist0 = max(float(np.hypot(dx0, dy0)), 1e-3)
    rel_vx0, rel_vy0 = agent_vx[0] - ego_vx[0], agent_vy[0] - ego_vy[0]
    closing0 = -(rel_vx0 * dx0 / dist0 + rel_vy0 * dy0 / dist0)
    rel_heading0 = float(wrap_to_pi(np.array(agent.heading[0] - ego.heading[0])))
    ego_speed0 = float(np.hypot(ego_vx[0], ego_vy[0]))
    agent_speed0 = float(np.hypot(agent_vx[0], agent_vy[0]))
    x_kin = np.array(
        [dx0, dy0, rel_vx0, rel_vy0, rel_heading0, closing0, ego_speed0, agent_speed0], dtype=np.float32,
    )

    # --- X_int: full-horizon (bidirectional) relative motion + signed clearance, K bins ---
    idx = _bin_sample_indices(cfg, T)
    dx, dy, dz = agent.x[idx] - ego.x[idx], agent.y[idx] - ego.y[idx], agent.z[idx] - ego.z[idx]
    rel_vx, rel_vy = agent_vx[idx] - ego_vx[idx], agent_vy[idx] - ego_vy[idx]
    center_dist = np.hypot(dx, dy)
    # crude signed clearance proxy: center distance minus each box's
    # half-diagonal in BEV (not the exact SAT distance eq:margin's tau_3D
    # is built from, but a cheap, smooth, sign-consistent stand-in: negative
    # when boxes are close enough to plausibly overlap, positive otherwise)
    half_diag_e = 0.5 * np.hypot(ego.length[idx], ego.width[idx])
    half_diag_j = 0.5 * np.hypot(agent.length[idx], agent.width[idx])
    signed_clearance = center_dist - (half_diag_e + half_diag_j)
    X_int = np.stack([dx, dy, dz, rel_vx, rel_vy, signed_clearance], axis=-1).astype(np.float32)
    if X_int.shape[0] < cfg.num_bins:  # pad if a window is ever shorter than one full horizon
        pad = np.zeros((cfg.num_bins - X_int.shape[0], D_INT), dtype=np.float32)
        X_int = np.concatenate([X_int, pad], axis=0)

    # --- x_ctx: semantic scene context --------------------------------------
    # (1) fractional interaction-type distribution over the full window,
    # not just its mode -- see the D_CTX comment above for why.
    full_types = pre.interaction_type.get(agent_id)
    type_frac = np.zeros(len(TYPE_NAMES), dtype=np.float32)
    if full_types is not None and len(full_types) > 0:
        for code in TYPE_NAMES:
            type_frac[code] = float(np.mean(full_types == code))
    else:
        dom_type = pre.dominant_interaction_type.get(agent_id, NAME_TO_TYPE["other"])
        type_frac[dom_type] = 1.0

    # (2) raw relative-heading range over the window (mean/min/max of
    # |wrap_to_pi(agent.heading - ego.heading)|) -- the underlying signal
    # Algorithm 1's categories are derived from, given to the model directly
    # rather than only through the (possibly mode-collapsed) categories.
    rel_heading_full = wrap_to_pi(agent.heading[idx] - ego.heading[idx])
    abs_rel_heading = np.abs(rel_heading_full)
    rel_heading_stats = np.array(
        [abs_rel_heading.mean(), abs_rel_heading.min(), abs_rel_heading.max()], dtype=np.float32,
    ) if len(idx) > 0 else np.zeros(3, dtype=np.float32)

    # (3) ego's own net heading change over the window (signed) -- how much
    # and which way ego is turning, directly relevant to turn-across-traffic
    # conflicts specifically (the failure case this was added for).
    ego_net_heading_change = float(wrap_to_pi(np.array(ego.heading[idx[-1]] - ego.heading[idx[0]]))) if len(idx) > 1 else 0.0

    agent_class = raw.agent_classes.get(agent_id, "vehicle") if raw.agent_classes else "vehicle"
    class_onehot = np.zeros(len(AGENT_CLASS_VOCAB), dtype=np.float32)
    if agent_class in AGENT_CLASS_VOCAB:
        class_onehot[AGENT_CLASS_VOCAB.index(agent_class)] = 1.0

    x_ctx = np.concatenate([
        type_frac, rel_heading_stats, [ego_net_heading_change], class_onehot,
    ]).astype(np.float32)

    # --- tau_phy^j(b_k): per-bin physical prior, resampled onto the same bins ---
    tau_phy_full = pre.tau_phy.get(agent_id)
    if tau_phy_full is None:
        tau_phy_seq = np.full(cfg.num_bins, cfg.tau_phy_clip[1], dtype=np.float32)
    else:
        tau_phy_seq = tau_phy_full[idx].astype(np.float32)
        if len(tau_phy_seq) < cfg.num_bins:
            pad = np.full(cfg.num_bins - len(tau_phy_seq), tau_phy_seq[-1] if len(tau_phy_seq) else cfg.tau_phy_clip[1])
            tau_phy_seq = np.concatenate([tau_phy_seq, pad]).astype(np.float32)

    T_evt = float(pre.T_evt.get(agent_id, np.inf))

    return dict(x_kin=x_kin, X_int=X_int, x_ctx=x_ctx, tau_phy_seq=tau_phy_seq, T_evt=T_evt)


def build_instance_label(
    raw: RawTrajectoryInstance, pre: PreprocessedInstance, cfg: FutureRiskConfig = DEFAULT_CONFIG,
) -> float:
    """y_analytic (S3.7): the dense, self-supervised analytic risk target
    from eq:phi, computed once per instance from every agent's margin."""
    margins = [context_dependent_margin(pre.tau_3d[aid], pre.tau_phy[aid]) for aid in pre.tau_3d]
    if not margins:
        return 0.0  # eq:agg's empty-product base case: no agents -> R=0
    return analytic_trajectory_risk(margins, T_m=cfg.T_m)


def instance_collision_label(raw: RawTrajectoryInstance) -> int:
    """Binary true-collision label for one instance (any agent, any frame)
    -- used only as a sanity-check/evaluation signal (S3.10-style), never
    as a training target (FutureRisk is explicitly outcome-agnostic, S3.1)."""
    return int(any(flags.any() for flags in raw.agent_collision_flags.values()))


def build_instance_batch_item(
    raw: RawTrajectoryInstance, pre: PreprocessedInstance, cfg: FutureRiskConfig = DEFAULT_CONFIG,
) -> Dict[str, np.ndarray]:
    """Every agent's feature tensors + this instance's two labels, still at
    the single-instance level (padding across instances into a batch is
    `dataset.collate_batch`'s job)."""
    agent_ids: List[str] = list(raw.agent_boxes.keys())
    feats = [build_agent_feature(raw, pre, aid, cfg) for aid in agent_ids]

    def stack(key, shape):
        if not feats:
            return np.zeros((0, *shape), dtype=np.float32)
        return np.stack([f[key] for f in feats]).astype(np.float32)

    return dict(
        instance_id=raw.instance_id,
        agent_ids=agent_ids,
        x_kin=stack("x_kin", (D_KIN,)),
        X_int=stack("X_int", (cfg.num_bins, D_INT)),
        x_ctx=stack("x_ctx", (D_CTX,)),
        tau_phy_seq=stack("tau_phy_seq", (cfg.num_bins,)),
        T_evt=np.array([f["T_evt"] for f in feats], dtype=np.float32) if feats else np.zeros((0,), dtype=np.float32),
        y_analytic=np.float32(build_instance_label(raw, pre, cfg)),
        collision_label=instance_collision_label(raw),
    )
