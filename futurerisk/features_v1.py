"""v1 feature schema -- a from-scratch redesign of `features.py`'s tensor
assembly, organized into three explicit, dataset-agnostic conceptual
groups instead of the original x_kin/X_int/x_ctx split (which mixed
ego-only and relative-to-object quantities together):

  1. **Ego info** (`x_ego`, D_EGO) -- facts about the ego vehicle alone,
     independent of any specific surrounding agent: speed, longitudinal
     acceleration, yaw rate, net heading change over the window. Any
     dataset that logs an ego trajectory can supply this.
  2. **Object info**: a per-bin sequence (`X_obj`, D_OBJ_SEQ, replaces the
     old X_int) of the surrounding agent's relative position/velocity/
     clearance over time, plus a static current-frame snapshot + class
     (folded into `x_ctx_v1`, see below) -- any dataset with tracked
     surrounding-agent boxes can supply this.
  3. **Road info** -- folded into `x_ctx_v1` alongside the object-static
     snapshot: the interaction-type fractional distribution (already a
     road/lane-geometry-derived quantity, Algorithm 1) and an ego
     path-curvature proxy computed purely from ego's own trajectory (mean
     |yaw rate| over the window) -- deliberately *not* dependent on any
     dataset-specific map/lane-graph field (unlike the raw
     `ego_lane_change_times`, which `raw_instance_from_preprocessed`
     doesn't even reconstruct from cache), so this degrades gracefully to
     "whatever the interaction-type classifier already produced" on any
     dataset, map or no map.

Model architecture is unchanged (`CARM.forward` still takes exactly three
per-agent tensors + a time-varying one); this module only changes what
goes into each of those three tensors, keeping the ego/object/road framing
purely a feature-design and documentation change, not a new model input
slot.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import FutureRiskConfig, DEFAULT_CONFIG
from .data_pipeline import PreprocessedInstance, RawTrajectoryInstance
from .features import AGENT_CLASS_VOCAB, build_instance_label, instance_collision_label
from .interaction import TYPE_NAMES, wrap_to_pi

D_EGO = 4          # [ego_speed, ego_accel_lon, ego_yaw_rate, ego_net_heading_change]
D_OBJ_SEQ = 5      # [dx, dy, rel_vx, rel_vy, signed_clearance] per bin
# x_ctx_v1 = object static snapshot (7: dx0,dy0,rel_vx0,rel_vy0,rel_heading0,
# closing0,agent_speed0) + agent class one-hot (3) + interaction-type
# fractional distribution (5, road/lane-geometry-derived) + ego path
# curvature proxy (1).
D_CTX_V1 = 7 + len(AGENT_CLASS_VOCAB) + len(TYPE_NAMES) + 1


def _bin_sample_indices(cfg: FutureRiskConfig, n_frames: int) -> np.ndarray:
    edges = cfg.bin_edges[: cfg.num_bins]
    idx = np.round(edges / cfg.dt).astype(int)
    return np.clip(idx, 0, max(n_frames - 1, 0))


def build_agent_feature_v1(
    raw: RawTrajectoryInstance, pre: PreprocessedInstance, agent_id: str,
    cfg: FutureRiskConfig = DEFAULT_CONFIG,
) -> Dict[str, np.ndarray]:
    ego = raw.ego_box
    agent = raw.agent_boxes[agent_id]
    T = len(raw.times)
    dt = cfg.dt

    ego_vx, ego_vy = np.gradient(ego.x, dt), np.gradient(ego.y, dt)
    agent_vx, agent_vy = np.gradient(agent.x, dt), np.gradient(agent.y, dt)

    # --- 1. Ego info: ego-only, no reference to any specific object -------
    ego_speed_series = np.hypot(ego_vx, ego_vy)
    ego_accel_series = np.gradient(ego_speed_series, dt)
    ego_heading_unwrapped = np.unwrap(ego.heading)
    ego_yaw_rate_series = np.gradient(ego_heading_unwrapped, dt)
    ego_net_heading_change = float(wrap_to_pi(np.array(ego.heading[-1] - ego.heading[0])))
    x_ego = np.array([
        ego_speed_series[0], ego_accel_series[0], ego_yaw_rate_series[0], ego_net_heading_change,
    ], dtype=np.float32)

    # --- 2. Object info: per-bin relative motion + clearance sequence ------
    idx = _bin_sample_indices(cfg, T)
    dx, dy = agent.x[idx] - ego.x[idx], agent.y[idx] - ego.y[idx]
    rel_vx, rel_vy = agent_vx[idx] - ego_vx[idx], agent_vy[idx] - ego_vy[idx]
    center_dist = np.hypot(dx, dy)
    half_diag_e = 0.5 * np.hypot(ego.length[idx], ego.width[idx])
    half_diag_j = 0.5 * np.hypot(agent.length[idx], agent.width[idx])
    signed_clearance = center_dist - (half_diag_e + half_diag_j)
    X_obj = np.stack([dx, dy, rel_vx, rel_vy, signed_clearance], axis=-1).astype(np.float32)
    if X_obj.shape[0] < cfg.num_bins:
        pad = np.zeros((cfg.num_bins - X_obj.shape[0], D_OBJ_SEQ), dtype=np.float32)
        X_obj = np.concatenate([X_obj, pad], axis=0)

    # --- Object static snapshot (t=0) + class, folded into x_ctx_v1 --------
    dx0, dy0 = agent.x[0] - ego.x[0], agent.y[0] - ego.y[0]
    dist0 = max(float(np.hypot(dx0, dy0)), 1e-3)
    rel_vx0, rel_vy0 = agent_vx[0] - ego_vx[0], agent_vy[0] - ego_vy[0]
    closing0 = -(rel_vx0 * dx0 / dist0 + rel_vy0 * dy0 / dist0)
    rel_heading0 = float(wrap_to_pi(np.array(agent.heading[0] - ego.heading[0])))
    agent_speed0 = float(np.hypot(agent_vx[0], agent_vy[0]))
    obj_static = np.array([dx0, dy0, rel_vx0, rel_vy0, rel_heading0, closing0, agent_speed0], dtype=np.float32)

    agent_class = raw.agent_classes.get(agent_id, "vehicle") if raw.agent_classes else "vehicle"
    class_onehot = np.zeros(len(AGENT_CLASS_VOCAB), dtype=np.float32)
    if agent_class in AGENT_CLASS_VOCAB:
        class_onehot[AGENT_CLASS_VOCAB.index(agent_class)] = 1.0

    # --- 3. Road info: interaction-type distribution + ego path curvature --
    full_types = pre.interaction_type.get(agent_id)
    type_frac = np.zeros(len(TYPE_NAMES), dtype=np.float32)
    if full_types is not None and len(full_types) > 0:
        for code in TYPE_NAMES:
            type_frac[code] = float(np.mean(full_types == code))
    ego_path_curvature = float(np.mean(np.abs(ego_yaw_rate_series))) if len(ego_yaw_rate_series) else 0.0

    x_ctx_v1 = np.concatenate([
        obj_static, class_onehot, type_frac, [ego_path_curvature],
    ]).astype(np.float32)

    # --- tau_phy^j(b_k): unchanged from features.py -------------------------
    tau_phy_full = pre.tau_phy.get(agent_id)
    if tau_phy_full is None:
        tau_phy_seq = np.full(cfg.num_bins, cfg.tau_phy_clip[1], dtype=np.float32)
    else:
        tau_phy_seq = tau_phy_full[idx].astype(np.float32)
        if len(tau_phy_seq) < cfg.num_bins:
            pad = np.full(cfg.num_bins - len(tau_phy_seq), tau_phy_seq[-1] if len(tau_phy_seq) else cfg.tau_phy_clip[1])
            tau_phy_seq = np.concatenate([tau_phy_seq, pad]).astype(np.float32)

    T_evt = float(pre.T_evt.get(agent_id, np.inf))

    return dict(x_ego=x_ego, X_obj=X_obj, x_ctx_v1=x_ctx_v1, tau_phy_seq=tau_phy_seq, T_evt=T_evt)


def build_instance_batch_item_v1(
    raw: RawTrajectoryInstance, pre: PreprocessedInstance, cfg: FutureRiskConfig = DEFAULT_CONFIG,
) -> Dict[str, np.ndarray]:
    agent_ids: List[str] = list(raw.agent_boxes.keys())
    feats = [build_agent_feature_v1(raw, pre, aid, cfg) for aid in agent_ids]

    def stack(key, shape):
        if not feats:
            return np.zeros((0, *shape), dtype=np.float32)
        return np.stack([f[key] for f in feats]).astype(np.float32)

    return dict(
        instance_id=raw.instance_id,
        agent_ids=agent_ids,
        x_ego=stack("x_ego", (D_EGO,)),
        X_obj=stack("X_obj", (cfg.num_bins, D_OBJ_SEQ)),
        x_ctx_v1=stack("x_ctx_v1", (D_CTX_V1,)),
        tau_phy_seq=stack("tau_phy_seq", (cfg.num_bins,)),
        T_evt=np.array([f["T_evt"] for f in feats], dtype=np.float32) if feats else np.zeros((0,), dtype=np.float32),
        y_analytic=np.float32(build_instance_label(raw, pre, cfg)),
        collision_label=instance_collision_label(raw),
    )
