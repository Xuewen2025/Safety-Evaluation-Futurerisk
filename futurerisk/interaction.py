"""Interaction-type classification heuristic (S3.3 / Algorithm "Interaction
type heuristic").
"""
from __future__ import annotations

import numpy as np

from .config import FutureRiskConfig, DEFAULT_CONFIG

CAR_FOLLOWING = 0
CROSSING = 1
MERGING = 2
ONCOMING = 3
OTHER = 4

TYPE_NAMES = {
    CAR_FOLLOWING: "car-following",
    CROSSING: "crossing",
    MERGING: "merging/lane-change",
    ONCOMING: "oncoming",
    OTHER: "other",
}
NAME_TO_TYPE = {v: k for k, v in TYPE_NAMES.items()}


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def lateral_offset_frenet_like(ego_x, ego_y, ego_heading, agent_x, agent_y) -> np.ndarray:
    """Approximate lateral offset of agent j in ego's local (Frenet-like)
    frame: project the relative position onto the axis perpendicular to
    ego heading. This is exact for a straight local frame and a standard
    lightweight substitute for a full Frenet-frame projection onto a lane
    centerline when only raw poses are available."""
    dx = agent_x - ego_x
    dy = agent_y - ego_y
    # lateral (left-hand normal of ego heading)
    nx, ny = -np.sin(ego_heading), np.cos(ego_heading)
    return dx * nx + dy * ny


def classify_interaction_type(
    ego_heading: np.ndarray,
    agent_heading: np.ndarray,
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    agent_x: np.ndarray,
    agent_y: np.ndarray,
    lane_graph_available: bool = True,
    is_merging_or_lane_changing: np.ndarray | None = None,
    cfg: FutureRiskConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Vectorized version of Algorithm 1 (Interaction type heuristic),
    evaluated independently at each (t, j) -- inputs are arrays over the
    flattened (t, j) axis.

    Parameters
    ----------
    is_merging_or_lane_changing : bool array, same shape as ego_heading, or
        None. Required only when lane_graph_available=True; should already
        encode "ego or j changing/merging lanes within +-merge_window_s of
        t" (S3.3). If lane graph info is unavailable, pass
        lane_graph_available=False and this argument is ignored (S3.3
        fallback: drop the merging condition).

    Returns
    -------
    types : int array, one of {CAR_FOLLOWING, CROSSING, MERGING, ONCOMING,
        OTHER} per element (matching TYPE_NAMES).
    """
    ego_heading = np.asarray(ego_heading, dtype=float)
    agent_heading = np.asarray(agent_heading, dtype=float)
    delta_psi = np.abs(wrap_to_pi(agent_heading - ego_heading))  # radians, in [0, pi]

    d_lat = np.abs(lateral_offset_frenet_like(ego_x, ego_y, ego_heading, agent_x, agent_y))

    thr_cf = np.deg2rad(cfg.car_following_heading_deg)
    thr_onc = np.deg2rad(cfg.oncoming_heading_deg)
    thr_cross_lo = np.deg2rad(cfg.crossing_heading_low_deg)
    thr_cross_hi = np.deg2rad(cfg.crossing_heading_high_deg)
    half_lane = cfg.lane_half_width_m

    is_car_following = (delta_psi < thr_cf) & (d_lat < half_lane)
    is_oncoming = delta_psi > thr_onc
    is_crossing = (delta_psi >= thr_cross_lo) & (delta_psi <= thr_cross_hi)

    if lane_graph_available:
        if is_merging_or_lane_changing is None:
            raise ValueError(
                "is_merging_or_lane_changing must be provided when "
                "lane_graph_available=True"
            )
        is_merging = np.asarray(is_merging_or_lane_changing, dtype=bool)
        # Algorithm 1 branch order: car-following -> oncoming -> crossing ->
        # merging/lane-change -> other (first matching branch wins).
        types = np.full(delta_psi.shape, OTHER, dtype=int)
        types = np.where(is_merging, MERGING, types)
        types = np.where(is_crossing, CROSSING, types)
        types = np.where(is_oncoming, ONCOMING, types)
        types = np.where(is_car_following, CAR_FOLLOWING, types)
        return types

    # S3.3 fallback: "If lane-graph information is unavailable ... drop the
    # merging condition and fall back to {car-following, crossing,
    # oncoming, other} using only delta_psi and d_lat."
    types = np.full(delta_psi.shape, OTHER, dtype=int)
    types = np.where(is_crossing, CROSSING, types)
    types = np.where(is_oncoming, ONCOMING, types)
    types = np.where(is_car_following, CAR_FOLLOWING, types)
    return types


def merge_flag_from_lane_graph(t: np.ndarray, lane_change_times_ego: np.ndarray,
                                lane_change_times_agent: np.ndarray,
                                window_s: float = 1.5) -> np.ndarray:
    """Helper: build the boolean 'ego or j changing/merging lanes within
    +-window_s of t' flag from lists of lane-change event times for ego and
    for agent j. `t` is the array of query times (broadcastable); the two
    *_times arrays hold timestamps (seconds) at which a lane-change/merge
    maneuver is recorded for ego / agent j respectively (empty array if
    none)."""
    t = np.asarray(t, dtype=float)
    flag = np.zeros(t.shape, dtype=bool)
    for change_times in (lane_change_times_ego, lane_change_times_agent):
        change_times = np.asarray(change_times, dtype=float)
        if change_times.size == 0:
            continue
        diff = np.abs(t[..., None] - change_times[None, ...])
        flag |= (diff <= window_s).any(axis=-1)
    return flag


def dominant_interaction_type(types: np.ndarray, exclude_other: bool = False) -> int:
    """Reduce a time-varying c(t,j) trace to "a per-instance dominant type"
    (S3.6.1's x_ctx feature description explicitly allows either "one-hot,
    time-varying" *or* a per-instance dominant type) -- the mode over t.

    Used both as a possible x_ctx summary feature and, downstream, to group
    evaluation metrics by interaction type at the trajectory-instance level
    (S3.10-style discrimination/persistence breakdowns), since raw c(t,j) is
    per-(t,j) but most reporting wants one label per instance/agent pair.

    exclude_other: if True and at least one non-OTHER type occurs, ties with
    OTHER are broken in favor of the more informative type (OTHER is the
    catch-all/least-informative bucket).
    """
    types = np.asarray(types, dtype=int)
    if types.size == 0:
        return OTHER
    counts = np.bincount(types, minlength=len(TYPE_NAMES))
    if exclude_other and counts[OTHER] < counts.sum():
        counts = counts.copy()
        counts[OTHER] = -1  # never win a tie against a real type, still falls back if all zero
        if counts.max() <= 0:
            return OTHER
    return int(np.argmax(counts))
