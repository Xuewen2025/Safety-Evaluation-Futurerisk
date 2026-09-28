"""Cross-domain generalization evaluation (S3.10): held out DeepAccident
test set, discrimination (AUROC/AUPRC), early-warning lead time, and
calibration (reliability diagram / ECE), plus the baseline comparison
against raw 3D-TTC thresholding and the analytic-only risk y_analytic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover - keep this module usable without sklearn
    _HAVE_SKLEARN = False


# --------------------------------------------------------------------------
# Discrimination: AUROC / AUPRC
# --------------------------------------------------------------------------

def _auroc_manual(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U statistic based AUROC, no sklearn dependency."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = scores[order]
    # average ranks for ties
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks_sorted[i:j + 1] = ranks_sorted[i:j + 1].mean()
        i = j + 1
    ranks[order] = ranks_sorted
    sum_ranks_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _auprc_manual(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under precision-recall curve, step
    interpolation), no sklearn dependency."""
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    precision = tp / np.clip(tp + fp, 1, None)
    recall = tp / n_pos
    # average precision = sum_n (R_n - R_{n-1}) * P_n
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def discrimination_metrics(risk_scores: Sequence[float], collision_labels: Sequence[int]) -> Dict[str, float]:
    """AUROC / AUPRC of trajectory-level R against the binary true-collision
    label (S3.10, metric 1)."""
    scores = np.asarray(risk_scores, dtype=float)
    labels = np.asarray(collision_labels, dtype=int)
    if _HAVE_SKLEARN:
        try:
            auroc = float(roc_auc_score(labels, scores))
            auprc = float(average_precision_score(labels, scores))
            return {"auroc": auroc, "auprc": auprc}
        except ValueError:
            pass  # e.g. single-class labels; fall through to manual (also returns nan)
    return {"auroc": _auroc_manual(scores, labels), "auprc": _auprc_manual(scores, labels)}


def group_discrimination_metrics(
    risk_scores: Sequence[float],
    collision_labels: Sequence[int],
    group_labels: Sequence,
    min_group_size: int = 10,
) -> Dict[str, Dict[str, float]]:
    """discrimination_metrics(), broken out per group (e.g. per dominant
    interaction type from `interaction.dominant_interaction_type`, or per
    dataset/agent-class) -- this is what "beyond collision rate" grained
    reporting looks like in practice: a single pooled AUROC can hide that a
    method discriminates well on car-following but poorly on crossing.

    Groups smaller than `min_group_size` are still reported but flagged via
    `n < min_group_size` so callers can grey them out rather than silently
    trusting a metric computed on a handful of instances; an "overall" key
    with metrics on the full (ungrouped) set is always included for
    reference.
    """
    scores = np.asarray(risk_scores, dtype=float)
    labels = np.asarray(collision_labels, dtype=int)
    groups = np.asarray(group_labels)
    assert len(scores) == len(labels) == len(groups)

    out: Dict[str, Dict[str, float]] = {
        "overall": {**discrimination_metrics(scores, labels), "n": float(len(scores))}
    }
    for g in sorted(set(groups.tolist()), key=str):
        mask = groups == g
        n = int(mask.sum())
        metrics = discrimination_metrics(scores[mask], labels[mask])
        metrics["n"] = float(n)
        metrics["low_sample_warning"] = float(n < min_group_size)
        out[str(g)] = metrics
    return out


def trajectory_level_score(r_of_t: np.ndarray, times: np.ndarray, lead_time: Optional[float] = None,
                            collision_time: Optional[float] = None) -> float:
    """Reduce a per-time risk trace r(t) to a single trajectory-level score:
    either max over time, or the value at a fixed lead time before the
    labeled accident (S3.10, metric 1 note)."""
    if lead_time is None or collision_time is None:
        return float(np.max(r_of_t))
    target_t = collision_time - lead_time
    idx = int(np.argmin(np.abs(times - target_t)))
    return float(r_of_t[idx])


# --------------------------------------------------------------------------
# Persistence: "how closely AND how persistently" (S3.1's stated motivation)
# --------------------------------------------------------------------------
#
# A single max-over-time risk score answers "how close did it get", but two
# collision-free trajectories with the same peak risk can differ a lot in
# how long they stayed close to that peak -- a 0.2s spike vs. a 3s sustained
# near-miss are not equally concerning. These functions expose that second
# axis explicitly instead of leaving it implicit inside the hazard head's
# cumulative F_k.

def risk_time_integral(r_of_t: np.ndarray, times: np.ndarray) -> float:
    """Trapezoidal integral of r(t) over [0, H]: total risk *exposure*
    accumulated over the trajectory, in risk-seconds. Two trajectories with
    the same max(r(t)) but different integrals differ in how long they
    spent near that peak."""
    return float(np.trapz(np.asarray(r_of_t, dtype=float), np.asarray(times, dtype=float)))


def critical_duration(margin_t: np.ndarray, times: np.ndarray, margin_threshold: float = 0.0) -> float:
    """Total wall-clock time (seconds) spent with margin m(t) at or below
    `margin_threshold` (default 0.0, i.e. actually unsafe per eq:margin's
    own sign convention: m<=0 means tau_3D<=tau_safe). Raise the threshold
    to also count "uncomfortably close but nominally still safe" dwell
    time."""
    margin_t = np.asarray(margin_t, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return 0.0
    dt = np.diff(times, prepend=times[0] - (times[1] - times[0]))
    below = margin_t <= margin_threshold
    return float(np.sum(dt[below]))


def persistence_metrics(
    r_of_t: np.ndarray, times: np.ndarray,
    margin_t: Optional[np.ndarray] = None, margin_threshold: float = 0.0,
) -> Dict[str, float]:
    """One-stop summary combining the "closeness" (peak) and "persistence"
    (duration/exposure) axes for a single trajectory instance, meant to be
    reported alongside (not instead of) the scalar R (S3.10)."""
    out = {
        "peak_risk": float(np.max(r_of_t)),
        "risk_time_integral": risk_time_integral(r_of_t, times),
        "mean_risk": float(np.mean(r_of_t)),
    }
    if margin_t is not None:
        out["critical_duration_s"] = critical_duration(margin_t, times, margin_threshold=margin_threshold)
        out["worst_margin"] = float(np.min(margin_t))
    return out


def persistence_summary_by_group(
    persistence_values: Sequence[Dict[str, float]], group_labels: Sequence,
) -> Dict[str, Dict[str, float]]:
    """Aggregate (e.g. mean/median `risk_time_integral`, `critical_duration_s`)
    across instances within each group (dominant interaction type, agent
    class, dataset, ...), mirroring `group_discrimination_metrics` for the
    persistence axis."""
    groups = np.asarray(group_labels)
    keys = sorted({k for pv in persistence_values for k in pv})
    out: Dict[str, Dict[str, float]] = {}
    for g in sorted(set(groups.tolist()), key=str):
        idx = np.where(groups == g)[0]
        summary = {"n": float(len(idx))}
        for k in keys:
            vals = np.array([persistence_values[i][k] for i in idx if k in persistence_values[i]])
            if vals.size:
                summary[f"mean_{k}"] = float(np.mean(vals))
                summary[f"median_{k}"] = float(np.median(vals))
        out[str(g)] = summary
    return out


# --------------------------------------------------------------------------
# Early-warning lead time
# --------------------------------------------------------------------------

def operating_point_threshold(
    r_of_t_noncolliding: List[np.ndarray], target_fpr: float = 0.05,
) -> float:
    """Choose a fixed operating-point threshold on r(t), from the
    *validation* set, such that non-colliding trajectories trigger a false
    positive at rate `target_fpr` (S3.10, metric 2). A trajectory "triggers"
    if r(t) crosses the threshold at any point."""
    max_r = np.array([np.max(r) for r in r_of_t_noncolliding])
    if len(max_r) == 0:
        raise ValueError("Need at least one non-colliding validation trajectory.")
    # threshold = the (1 - target_fpr) quantile of the per-trajectory max r,
    # so that exactly target_fpr fraction of non-colliding trajectories
    # would have crossed it.
    return float(np.quantile(max_r, 1 - target_fpr))


def early_warning_lead_time(
    r_of_t: np.ndarray, times: np.ndarray, threshold: float, collision_time: float,
) -> Optional[float]:
    """For one colliding trajectory: earliest t* at which r(t) crosses
    `threshold`; returns (t_collision - t*), or None if r(t) never crosses
    the threshold before the collision (S3.10, metric 2)."""
    crossed = np.where(r_of_t >= threshold)[0]
    if len(crossed) == 0:
        return None
    t_star = times[crossed[0]]
    if t_star > collision_time:
        return None
    return float(collision_time - t_star)


def lead_time_distribution(
    colliding_traces: List[Tuple[np.ndarray, np.ndarray, float]], threshold: float,
) -> np.ndarray:
    """colliding_traces: list of (r_of_t, times, collision_time). Returns
    the array of lead times for trajectories where a warning fired (NaN
    entries dropped)."""
    lead_times = []
    for r_of_t, times, collision_time in colliding_traces:
        lt = early_warning_lead_time(r_of_t, times, threshold, collision_time)
        if lt is not None:
            lead_times.append(lt)
    return np.array(lead_times)


# --------------------------------------------------------------------------
# Calibration: reliability diagram / ECE
# --------------------------------------------------------------------------

@dataclass
class ReliabilityDiagram:
    bin_edges: np.ndarray            # (n_bins+1,)
    bin_confidence: np.ndarray       # mean predicted risk per bin
    bin_accuracy: np.ndarray         # empirical collision frequency per bin
    bin_counts: np.ndarray           # samples per bin
    ece: float


def reliability_diagram(
    risk_scores: Sequence[float], collision_labels: Sequence[int], n_bins: int = 10,
) -> ReliabilityDiagram:
    """Reliability diagram + Expected Calibration Error of R against
    empirical collision frequency, binned by predicted risk (S3.10, metric 3)."""
    scores = np.asarray(risk_scores, dtype=float)
    labels = np.asarray(collision_labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, n_bins - 1)

    confidence = np.zeros(n_bins)
    accuracy = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            confidence[b] = scores[mask].mean()
            accuracy[b] = labels[mask].mean()

    total = counts.sum()
    ece = float(np.sum(counts / max(total, 1) * np.abs(accuracy - confidence)))
    return ReliabilityDiagram(edges, confidence, accuracy, counts, ece)


# --------------------------------------------------------------------------
# Baseline comparison (S3.10, "Baseline comparison")
# --------------------------------------------------------------------------

def raw_ttc_baseline_score(tau_3d_min_over_time: np.ndarray, ttc_cap: float = 5.0) -> np.ndarray:
    """Raw 3D-TTC thresholding baseline: convert min-TTC over an instance
    into a risk-like score by inverting and clipping (lower TTC = higher
    risk), for ranking-based metrics (AUROC/AUPRC don't need calibrated
    probabilities). +inf TTC (no near-miss) maps to 0 risk."""
    tau = np.asarray(tau_3d_min_over_time, dtype=float)
    finite = np.isfinite(tau)
    score = np.zeros_like(tau)
    score[finite] = np.clip(1.0 - tau[finite] / ttc_cap, 0.0, 1.0)
    return score


def compare_methods(
    method_scores: Dict[str, Sequence[float]],
    collision_labels: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    """Report discrimination metrics for every method in `method_scores`
    (e.g. {'CARM': R, 'analytic': y_analytic, 'raw_ttc': ..., 'PET': ...,
    'DRAC': ...}) against the same DeepAccident test set (S3.10)."""
    return {name: discrimination_metrics(scores, collision_labels) for name, scores in method_scores.items()}
