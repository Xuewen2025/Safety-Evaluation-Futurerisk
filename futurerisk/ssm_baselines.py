"""Classic surrogate safety measures (SSMs) -- PET (Post-Encroachment Time)
and DRAC (Deceleration Rate to Avoid Crash) -- implemented as *comparison
baselines* for FutureRisk/CARM's R (experiment_design.txt S-one.3, S-two.3;
mentioned but not implemented in S3.10's own baseline list).

Neither is part of instruction.txt's own method. Both are computed here
using ONLY the fields every dataset adapter already produces on the shared,
dataset-agnostic `geometry.BoxTrajectory` interface (x, y, length, width,
common `dt`) -- the same universal inputs `evaluation.raw_ttc_baseline_score`
and `safety_prior`'s y_analytic already use. No dataset-specific field (e.g.
DeepAccident's native collision annotation, either adapter's lane-change
proxy) is read here, by design: these are meant to be directly comparable
across Bench2Drive and DeepAccident, the same way R itself is.
"""
from __future__ import annotations

import numpy as np

from .geometry import BoxTrajectory


def deceleration_rate_to_avoid_crash(ego: BoxTrajectory, agent: BoxTrajectory, dt: float) -> np.ndarray:
    """DRAC(t): the deceleration the closing party would need to apply,
    right now, to exactly avoid closing the remaining gap to zero (Cooper
    1997's standard SSM formula, generalized from pure longitudinal
    car-following to a general 2D closing-speed/gap pair the same way
    `data_pipeline._relative_closing_speed` already generalizes closing
    speed):

        DRAC(t) = v_close(t)^2 / (2 * gap(t))   if v_close(t) > 0 (closing)
                = 0                                otherwise

    Higher DRAC = harder braking required right now = more dangerous.
    Frames where the pair isn't closing get 0 (no braking needed), which is
    DRAC's own convention, not a missing-value placeholder."""
    dx = agent.x - ego.x
    dy = agent.y - ego.y
    gap = np.sqrt(dx ** 2 + dy ** 2)
    gap_safe = np.clip(gap, 1e-3, None)
    bearing_x, bearing_y = dx / gap_safe, dy / gap_safe

    ego_vx, ego_vy = np.gradient(ego.x, dt), np.gradient(ego.y, dt)
    agent_vx, agent_vy = np.gradient(agent.x, dt), np.gradient(agent.y, dt)
    rel_vx, rel_vy = ego_vx - agent_vx, ego_vy - agent_vy
    # (ego_v - agent_v) . bearing is positive exactly when the gap is
    # shrinking -- see data_pipeline._relative_closing_speed's docstring
    # for the full derivation (an earlier version of that same formula,
    # copied here, had this negated, which is backwards).
    v_close = rel_vx * bearing_x + rel_vy * bearing_y

    return np.where(v_close > 0, v_close ** 2 / (2 * gap_safe), 0.0)


def post_encroachment_time(
    ego: BoxTrajectory, agent: BoxTrajectory, times: np.ndarray, conflict_radius_scale: float = 1.0,
) -> float:
    """Scalar PET for one (ego, agent) pair over a trajectory instance
    (Allen et al. 1978's classic SSM: the time gap between the first party
    vacating a shared conflict point and the second party arriving there).

    Approximated (paths are rarely at the exact same place at the exact
    same logged frame) by a full pairwise search over (t1, t2) for the pair
    of times minimizing distance(ego(t1), agent(t2)) -- i.e. the closest
    the two *paths* (not necessarily time-synchronized) ever come to
    crossing the same spot. If that minimum distance is within a plausible
    "same spot" tolerance (the two boxes' combined half-diagonal, scaled by
    `conflict_radius_scale` -- same distance-plausibility-guard pattern as
    `adapters/deepaccident.py`'s native-collision patch), returns
    |times[t2*] - times[t1*]|; else the paths never plausibly cross and PET
    is undefined here (`+inf`, i.e. "no conflict point, no risk by this
    measure" -- consistent with `raw_ttc_baseline_score`'s +inf-TTC
    convention).

    O(T^2) is fine at trajectory-instance scale (T ~ 40-50 frames)."""
    ex, ey = ego.x[:, None], ego.y[:, None]
    ax, ay = agent.x[None, :], agent.y[None, :]
    dist = np.sqrt((ex - ax) ** 2 + (ey - ay) ** 2)  # (T, T), [t1_idx, t2_idx]
    i1, i2 = np.unravel_index(int(np.argmin(dist)), dist.shape)
    min_dist = float(dist[i1, i2])

    ego_half_diag = 0.5 * float(np.hypot(np.median(ego.length), np.median(ego.width)))
    agent_half_diag = 0.5 * float(np.hypot(np.median(agent.length), np.median(agent.width)))
    tolerance = conflict_radius_scale * (ego_half_diag + agent_half_diag)
    if min_dist > tolerance:
        return float("inf")
    return float(abs(times[i2] - times[i1]))


def drac_score(drac_max_over_time: np.ndarray, cap: float = 10.0) -> np.ndarray:
    """Trajectory-level DRAC -> a bounded [0,1] risk-like score for ranking
    metrics (AUROC/AUPRC only need a consistent ordering, not calibrated
    probabilities) -- same clip-and-normalize convention as
    `evaluation.raw_ttc_baseline_score`. `cap` (m/s^2) is a generous
    upper bound (a hard, near-emergency braking is roughly 8-9 m/s^2 for a
    typical vehicle); higher DRAC = higher score."""
    d = np.asarray(drac_max_over_time, dtype=float)
    return np.clip(d / cap, 0.0, 1.0)


def pet_score(pet_min_over_agents: np.ndarray, saturation_s: float = 5.0) -> np.ndarray:
    """Trajectory-level PET -> a bounded [0,1] risk-like score: lower PET
    means higher risk, so score = 1 - PET/saturation_s clipped to [0,1];
    +inf PET (no plausible conflict point) maps to 0, mirroring
    `raw_ttc_baseline_score`'s +inf-TTC -> 0-risk convention."""
    p = np.asarray(pet_min_over_agents, dtype=float)
    finite = np.isfinite(p)
    score = np.zeros_like(p)
    score[finite] = np.clip(1.0 - p[finite] / saturation_s, 0.0, 1.0)
    return score


def instance_ssm_scores(ego: BoxTrajectory, agent_boxes: dict, times: np.ndarray, dt: float) -> dict:
    """Aggregate PET/DRAC across every agent in one instance to the same
    "worst-case across agents" convention `raw_ttc_baseline_score`'s caller
    already uses for tau_3D (min over agents for TTC/PET -- lower is worse;
    max over agents for DRAC -- higher is worse). Returns raw (unscored)
    aggregates; pass through `pet_score`/`drac_score` to get [0,1] scores."""
    if not agent_boxes:
        return {"pet_min": float("inf"), "drac_max": 0.0}
    pets = [post_encroachment_time(ego, box, times) for box in agent_boxes.values()]
    dracs = [float(np.max(deceleration_rate_to_avoid_crash(ego, box, dt))) for box in agent_boxes.values()]
    return {"pet_min": float(np.min(pets)), "drac_max": float(np.max(dracs))}
