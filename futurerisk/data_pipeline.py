"""Data pipeline across datasets (S3.9).

Per-dataset preprocessing, run independently for DeepAccident / nuPlan /
Bench2Drive: resample to a common resolution, compute 3D-TTC, interaction
type, evasive severity; fit per-(dataset, interaction-type) thresholds
(a full pass, done once, *before* label generation); generate T_evt and
tau_phy per trajectory instance; cache to disk (one file per instance, or
sharded for large-scale nuPlan). Dataset identity is kept in metadata only,
never in the model-input feature tensor (S3.9 item 2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from .config import FutureRiskConfig, DEFAULT_CONFIG
from .event_proxy import (
    ThresholdTable, compute_threshold_tables, critical_event_time,
    evasive_severity, lateral_acceleration, longitudinal_deceleration,
    save_threshold_tables, load_threshold_tables,
)
from .geometry import BoxTrajectory, compute_3d_ttc
from .interaction import TYPE_NAMES, classify_interaction_type, dominant_interaction_type
from .safety_prior import analytic_safe_ttc_prior_for_class


@dataclass
class RawTrajectoryInstance:
    """Standardized, dataset-agnostic representation of one executed
    trajectory instance (either logged human driving or a closed-loop
    rollout), after per-dataset extraction/resampling to a common rate.

    Concretely: ego pose/box trace, a dict of surrounding-agent pose/box
    traces keyed by agent id, per-(t, agent) collision flags, and lane-graph
    derived merge/lane-change flags if available. `dataset` and
    `instance_id` are metadata only (S3.9 item 2: kept out of the model
    input feature tensor).
    """
    dataset: str
    instance_id: str
    times: np.ndarray                      # (T,), seconds, common dt
    ego_box: BoxTrajectory
    agent_boxes: Dict[str, BoxTrajectory]
    agent_collision_flags: Dict[str, np.ndarray]     # id -> (T,) bool
    lane_graph_available: bool = False
    ego_lane_change_times: Optional[np.ndarray] = None
    agent_lane_change_times: Optional[Dict[str, np.ndarray]] = field(default=None)
    # Object class per agent (e.g. "vehicle"/"truck"/"bus"/"cyclist"/
    # "pedestrian"), used both as an x_ctx model feature (S3.6.1) and, here,
    # to select object-class-specific physical-prior constants (S3.4) via
    # FutureRiskConfig.get_physical_prior_params -- unclassified agents (key
    # missing, or not present in this dict at all) fall back to the "other"
    # (most conservative) profile.
    agent_classes: Optional[Dict[str, str]] = field(default=None)


@dataclass
class PreprocessedInstance:
    """Cached preprocessing output for one trajectory instance (S3.9: "cache
    as preprocessed tensors, e.g. one .npz/.pt file per instance")."""
    dataset: str
    instance_id: str
    times: np.ndarray
    tau_3d: Dict[str, np.ndarray]                 # agent id -> (T,)
    interaction_type: Dict[str, np.ndarray]        # agent id -> (T,) int codes
    evasive_severity: Dict[str, np.ndarray]        # agent id -> (T,)
    tau_phy: Dict[str, np.ndarray]                 # agent id -> (T,)
    T_evt: Dict[str, float]                        # agent id -> scalar (may be inf)
    dominant_interaction_type: Dict[str, int] = field(default_factory=dict)  # agent id -> mode(c(t,j))
    # Raw trajectories, cached alongside the derived labels above so that
    # feature-tensor assembly (features.py) never has to re-parse the
    # original dataset archives to rebuild x_kin/X_int -- without this, every
    # retrain would re-pay the full per-clip decompression cost (e.g. ~85
    # minutes for Bench2Drive-Base's 1000 clips) just to get positions back.
    # Optional/None for instances cached before this field existed.
    ego_box: Optional[BoxTrajectory] = None
    agent_boxes: Optional[Dict[str, BoxTrajectory]] = None
    agent_classes: Optional[Dict[str, str]] = None
    agent_collision_flags: Optional[Dict[str, np.ndarray]] = None


def _relative_closing_speed(ego_box: BoxTrajectory, agent_box: BoxTrajectory, dt: float) -> np.ndarray:
    """v_rel_close(t): closing-speed component along the current relative
    bearing (S3.4). Positive = closing (gap shrinking).

    BUG FIX (found via `ssm_baselines.py`'s PET/DRAC unit tests, which
    asserted the correct physical sign and failed against this function):
    an earlier version negated this dot product, which is backwards --
    verified directly with a synthetic unambiguous-closing pair (ego
    stationary, agent approaching head-on at 10 m/s) returning -10, and an
    unambiguous-receding pair returning +10, exactly the opposite of the
    "positive = closing" contract this function and eq:phy-prior both rely
    on. `bearing` points from ego to agent; `dist` shrinks over time
    (closing) iff d(dist)/dt = (agent_v - ego_v)*bearing < 0, i.e. iff
    (ego_v - agent_v)*bearing > 0 -- so the *unnegated* dot product below
    is already "positive when closing", confirmed by the same synthetic
    pair now returning +10/-10 respectively as expected.

    This fed `analytic_safe_ttc_prior_for_class` (eq:phy-prior:
    tau_phy = t_react + v_rel_close/a_avail + eps_unc) with a sign-flipped
    v_rel_close for every instance in every dataset preprocessed before
    this fix -- genuinely-closing (dangerous) pairs got an artificially
    *small* tau_phy (often floored at the 0.2s clip-range minimum, since
    a_avail>0 makes the flipped-negative term dominate), and genuinely
    safely-receding pairs got an artificially *large* one. Every
    preprocessed cache, trained checkpoint, and y_analytic/margin number
    reported before this fix used the corrupted tau_phy and needs
    regenerating."""
    dx = agent_box.x - ego_box.x
    dy = agent_box.y - ego_box.y
    dist = np.sqrt(dx ** 2 + dy ** 2)
    dist_safe = np.clip(dist, 1e-3, None)
    bearing_x, bearing_y = dx / dist_safe, dy / dist_safe

    ego_vx = np.gradient(ego_box.x, dt)
    ego_vy = np.gradient(ego_box.y, dt)
    agent_vx = np.gradient(agent_box.x, dt)
    agent_vy = np.gradient(agent_box.y, dt)
    rel_vx, rel_vy = ego_vx - agent_vx, ego_vy - agent_vy

    # component of (ego_v - agent_v) along the ego->agent bearing, positive
    # exactly when the gap is shrinking (see docstring derivation above).
    closing = rel_vx * bearing_x + rel_vy * bearing_y
    return closing


def _per_agent_evasive_severity(
    ego: BoxTrajectory, agent_box: BoxTrajectory, dt: float,
    ego_a_lon: np.ndarray, ego_a_lat: np.ndarray,
) -> np.ndarray:
    """a(t) for one (ego, agent j) pair, combining ego's and (where
    available -- here always, since agent poses are observed) agent j's own
    harsh-braking/hard-steering signal (S3.5.1: "a_lon(t) (ego and, where
    available, agent j)"). We take the elementwise max of the two parties'
    evasive-severity scalars: a harsh maneuver by *either* party marks the
    interaction as safety-critical at that time."""
    ego_sev = evasive_severity(ego_a_lon, ego_a_lat)

    agent_speed = np.sqrt(np.gradient(agent_box.x, dt) ** 2 + np.gradient(agent_box.y, dt) ** 2)
    agent_a_lon = longitudinal_deceleration(agent_speed, dt)
    agent_a_lat = lateral_acceleration(agent_speed, dt, heading=agent_box.heading)
    agent_sev = evasive_severity(agent_a_lon, agent_a_lat)

    return np.maximum(ego_sev, agent_sev)


def preprocess_instance(
    instance: RawTrajectoryInstance,
    threshold_table: ThresholdTable,
    cfg: FutureRiskConfig = DEFAULT_CONFIG,
) -> PreprocessedInstance:
    """One full pass of S3.9 item 1's per-instance computations, assuming
    thresholds have *already* been fit for this instance's dataset (a
    separate full-dataset pass, see `fit_thresholds_for_dataset`)."""
    dt = cfg.dt
    tau_3d, itype, sev, tau_phy, t_evt, dom_type = {}, {}, {}, {}, {}, {}

    ego = instance.ego_box
    ego_speed = np.sqrt(np.gradient(ego.x, dt) ** 2 + np.gradient(ego.y, dt) ** 2)
    ego_a_lon = longitudinal_deceleration(ego_speed, dt)
    ego_a_lat = lateral_acceleration(ego_speed, dt, heading=ego.heading)

    for agent_id, agent_box in instance.agent_boxes.items():
        tau3d = compute_3d_ttc(ego, agent_box, dt=cfg.ttc_delta_step, cap=cfg.ttc_cap)
        tau_3d[agent_id] = tau3d

        merge_flags = None
        if instance.lane_graph_available:
            from .interaction import merge_flag_from_lane_graph
            merge_flags = merge_flag_from_lane_graph(
                instance.times,
                instance.ego_lane_change_times if instance.ego_lane_change_times is not None else np.array([]),
                (instance.agent_lane_change_times or {}).get(agent_id, np.array([])),
                window_s=cfg.merge_window_s,
            )
        types = classify_interaction_type(
            ego.heading, agent_box.heading, ego.x, ego.y, agent_box.x, agent_box.y,
            lane_graph_available=instance.lane_graph_available,
            is_merging_or_lane_changing=merge_flags,
            cfg=cfg,
        )
        itype[agent_id] = types
        dom_type[agent_id] = dominant_interaction_type(types)

        a_t = _per_agent_evasive_severity(ego, agent_box, dt, ego_a_lon, ego_a_lat)
        sev[agent_id] = a_t

        v_close = _relative_closing_speed(ego, agent_box, dt)
        agent_class = (instance.agent_classes or {}).get(agent_id, "other")
        tau_phy_j = analytic_safe_ttc_prior_for_class(v_close, agent_class, cfg=cfg)
        tau_phy[agent_id] = tau_phy_j

        type_names = np.array([TYPE_NAMES[c] for c in types])
        t_evt[agent_id] = critical_event_time(
            instance.times, instance.agent_collision_flags[agent_id], a_t, type_names, threshold_table,
        )

    return PreprocessedInstance(
        dataset=instance.dataset, instance_id=instance.instance_id, times=instance.times,
        tau_3d=tau_3d, interaction_type=itype, evasive_severity=sev, tau_phy=tau_phy, T_evt=t_evt,
        dominant_interaction_type=dom_type,
        # cache the raw trajectories too (see PreprocessedInstance's
        # docstring note) so feature-tensor assembly never needs the
        # original archives again
        ego_box=instance.ego_box, agent_boxes=instance.agent_boxes,
        agent_classes=instance.agent_classes, agent_collision_flags=instance.agent_collision_flags,
    )


def collect_evasive_severity_samples(
    instances: Iterable[RawTrajectoryInstance], cfg: FutureRiskConfig = DEFAULT_CONFIG,
) -> Dict[str, np.ndarray]:
    """First pass over one dataset: collect a(t) samples per interaction
    type (needed before thresholds can be fit, S3.9 item 1 / S3.5.2)."""
    dt = cfg.dt
    samples: Dict[str, List[float]] = {name: [] for name in TYPE_NAMES.values()}

    for instance in instances:
        ego = instance.ego_box
        ego_speed = np.sqrt(np.gradient(ego.x, dt) ** 2 + np.gradient(ego.y, dt) ** 2)
        ego_a_lon = longitudinal_deceleration(ego_speed, dt)
        ego_a_lat = lateral_acceleration(ego_speed, dt, heading=ego.heading)

        for agent_id, agent_box in instance.agent_boxes.items():
            merge_flags = None
            if instance.lane_graph_available:
                from .interaction import merge_flag_from_lane_graph
                merge_flags = merge_flag_from_lane_graph(
                    instance.times,
                    instance.ego_lane_change_times if instance.ego_lane_change_times is not None else np.array([]),
                    (instance.agent_lane_change_times or {}).get(agent_id, np.array([])),
                    window_s=cfg.merge_window_s,
                )
            types = classify_interaction_type(
                ego.heading, agent_box.heading, ego.x, ego.y, agent_box.x, agent_box.y,
                lane_graph_available=instance.lane_graph_available,
                is_merging_or_lane_changing=merge_flags, cfg=cfg,
            )
            a_t = _per_agent_evasive_severity(ego, agent_box, dt, ego_a_lon, ego_a_lat)
            for code, name in TYPE_NAMES.items():
                mask = types == code
                if mask.any():
                    samples[name].extend(a_t[mask].tolist())

    return {name: np.array(vals) for name, vals in samples.items()}


def fit_thresholds_for_dataset(
    dataset_name: str, instances: Iterable[RawTrajectoryInstance], cfg: FutureRiskConfig = DEFAULT_CONFIG,
) -> ThresholdTable:
    """Algorithm 2, run once for a single dataset (S3.5.2 / S3.9 item 1)."""
    samples_by_type = collect_evasive_severity_samples(instances, cfg=cfg)
    tables = compute_threshold_tables(
        {dataset_name: samples_by_type}, percentile=cfg.threshold_percentile, n_min=cfg.n_min_samples,
    )
    return tables[dataset_name]


def preprocess_dataset(
    dataset_name: str,
    instances: List[RawTrajectoryInstance],
    output_dir: str,
    cfg: FutureRiskConfig = DEFAULT_CONFIG,
    threshold_table: Optional[ThresholdTable] = None,
) -> ThresholdTable:
    """End-to-end per-dataset preprocessing (S3.9 item 1): fit thresholds
    (independently per dataset -- never pool across datasets, S3.5.2), then
    generate + cache T_evt and tau_phy (and tau_3D, interaction type,
    evasive severity) for every instance.

    Caches one .npz per instance under `output_dir`, plus the threshold
    table itself (the reusable fixed preprocessing artifact).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if threshold_table is None:
        threshold_table = fit_thresholds_for_dataset(dataset_name, instances, cfg=cfg)
        save_threshold_tables({dataset_name: threshold_table}, str(out_dir / f"{dataset_name}_thresholds.json"))

    for instance in instances:
        pre = preprocess_instance(instance, threshold_table, cfg=cfg)
        _save_preprocessed_instance(pre, out_dir / f"{instance.instance_id}.npz")

    return threshold_table


_BOX_FIELDS = ("x", "y", "z", "heading", "length", "width", "height")


def _save_box(payload: dict, prefix: str, box: BoxTrajectory) -> None:
    for f in _BOX_FIELDS:
        payload[f"{prefix}__{f}"] = getattr(box, f)


def _load_box(data, prefix: str) -> BoxTrajectory:
    return BoxTrajectory(**{f: data[f"{prefix}__{f}"] for f in _BOX_FIELDS})


def _save_preprocessed_instance(pre: PreprocessedInstance, path: Path) -> None:
    payload = {"dataset": pre.dataset, "instance_id": pre.instance_id, "times": pre.times}
    for agent_id in pre.tau_3d:
        payload[f"tau3d__{agent_id}"] = pre.tau_3d[agent_id]
        payload[f"itype__{agent_id}"] = pre.interaction_type[agent_id]
        payload[f"sev__{agent_id}"] = pre.evasive_severity[agent_id]
        payload[f"tauphy__{agent_id}"] = pre.tau_phy[agent_id]
        payload[f"tevt__{agent_id}"] = np.array([pre.T_evt[agent_id]])
        payload[f"domtype__{agent_id}"] = np.array([pre.dominant_interaction_type.get(agent_id, -1)])
        if pre.agent_classes is not None:
            payload[f"class__{agent_id}"] = np.array(pre.agent_classes.get(agent_id, "other"))
        if pre.agent_collision_flags is not None:
            payload[f"collision__{agent_id}"] = pre.agent_collision_flags[agent_id]
        if pre.agent_boxes is not None:
            _save_box(payload, f"box__{agent_id}", pre.agent_boxes[agent_id])
    payload["agent_ids"] = np.array(list(pre.tau_3d.keys()))
    payload["has_raw"] = np.array(pre.ego_box is not None)
    if pre.ego_box is not None:
        _save_box(payload, "egobox", pre.ego_box)
    np.savez(path, **payload)


def load_preprocessed_instance(path: str) -> PreprocessedInstance:
    """Load one cached `.npz` (written by `_save_preprocessed_instance`).

    `np.load` on a `.npz` returns a lazy `NpzFile` that keeps the
    underlying file (and, for a compressed archive, a `zipfile.ZipFile`)
    open until explicitly `.close()`-d -- it is NOT closed by garbage
    collection promptly enough to rely on. Every array actually used below
    is materialized into an independent in-memory ndarray by `data[key]`
    (or by `_load_box`, which does the same), so nothing returned by this
    function holds a reference back into `data` -- closing it in `finally`,
    right after extraction, is always safe and avoids accumulating one
    open file handle per call across long loops over thousands of
    instances (found while diagnosing an OOM-style kill partway through
    `scripts/experiments/forward_ttc_baseline_eval.py`'s DeepAccident pass;
    every other caller of this function loops over instances the same way
    and benefits identically)."""
    data = np.load(path, allow_pickle=True)
    try:
        agent_ids = [str(a) for a in data["agent_ids"]]
        tau_3d, itype, sev, tau_phy, t_evt, dom_type = {}, {}, {}, {}, {}, {}
        has_raw = bool(data["has_raw"]) if "has_raw" in data else False
        agent_classes = {} if has_raw else None
        agent_collision_flags = {} if has_raw else None
        agent_boxes = {} if has_raw else None
        for agent_id in agent_ids:
            tau_3d[agent_id] = data[f"tau3d__{agent_id}"]
            itype[agent_id] = data[f"itype__{agent_id}"]
            sev[agent_id] = data[f"sev__{agent_id}"]
            tau_phy[agent_id] = data[f"tauphy__{agent_id}"]
            t_evt[agent_id] = float(data[f"tevt__{agent_id}"][0])
            if f"domtype__{agent_id}" in data:
                dom_type[agent_id] = int(data[f"domtype__{agent_id}"][0])
            if has_raw:
                agent_classes[agent_id] = str(data[f"class__{agent_id}"])
                agent_collision_flags[agent_id] = data[f"collision__{agent_id}"]
                agent_boxes[agent_id] = _load_box(data, f"box__{agent_id}")
        ego_box = _load_box(data, "egobox") if has_raw else None
        dataset_name = str(data["dataset"])
        instance_id = str(data["instance_id"])
        times = data["times"]
    finally:
        data.close()
    return PreprocessedInstance(
        dataset=dataset_name, instance_id=instance_id, times=times,
        tau_3d=tau_3d, interaction_type=itype, evasive_severity=sev, tau_phy=tau_phy, T_evt=t_evt,
        dominant_interaction_type=dom_type,
        ego_box=ego_box, agent_boxes=agent_boxes,
        agent_classes=agent_classes, agent_collision_flags=agent_collision_flags,
    )


def raw_instance_from_preprocessed(pre: PreprocessedInstance) -> RawTrajectoryInstance:
    """Reconstruct a (feature-building-sufficient) RawTrajectoryInstance
    from a cached PreprocessedInstance's raw-trajectory fields, so
    `features.py` never has to re-parse the original dataset archives for
    a cached instance. Raises if this instance was cached before the raw
    fields existed (`ego_box is None`) -- re-run `preprocess_dataset` to
    backfill.

    `lane_graph_available` is left False here: the lane-change proxy times
    were already consumed once (folded into `interaction_type`/
    `dominant_interaction_type` at preprocessing time) and aren't needed
    again for feature-tensor assembly.
    """
    if pre.ego_box is None:
        raise ValueError(
            f"'{pre.instance_id}' has no cached raw trajectory (preprocessed before this field existed); "
            "re-run preprocess_dataset to backfill."
        )
    return RawTrajectoryInstance(
        dataset=pre.dataset, instance_id=pre.instance_id, times=pre.times,
        ego_box=pre.ego_box, agent_boxes=pre.agent_boxes,
        agent_collision_flags=pre.agent_collision_flags,
        lane_graph_available=False,
        agent_classes=pre.agent_classes,
    )
