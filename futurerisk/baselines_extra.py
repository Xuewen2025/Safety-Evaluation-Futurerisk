"""RSS (Responsibility-Sensitive Safety, Shalev-Shwartz et al. 2017) and a
reproduction of Lu et al. (2022)'s unsupervised-representation-learning
surrogate-safety baseline -- the two comparison methods added to the
experiment design's Gap 1 (the rule-based and data-driven "extremes"
FutureRisk is positioned between in the paper's Related Work).

Both are computed from the SAME universal `BoxTrajectory`-level fields used
everywhere else in this package (position/heading/extent over a shared time
axis) -- no dataset-specific input, matching ssm_baselines.py's convention.

Honest scope notes (documented here, not just in the paper):
- RSS's full formulation has separate longitudinal/lateral sub-models and a
  proper "who is responsible" analysis; this implements the two
  longitudinal sub-formulas (same-direction and opposite-direction) from
  the paper's core safe-distance definitions, selected per-frame by the
  already-computed rule-based `interaction_type` (car-following ->
  same-direction; oncoming -> opposite-direction; crossing/merging/other ->
  same-direction as a generic fallback, since RSS's own paper does not
  cover intersection/crossing geometry in closed form either). Parameters
  (response time, accel/brake bounds) use commonly-cited example values
  from the RSS literature, not a specific vehicle's real specs.
- Lu et al. (2022) reproduction: no official code exists. This implements
  the paper's described recipe (embed sequences of surrogate safety
  measures via unsupervised representation learning, identify conflicts by
  clustering that embedding) as a small autoencoder over per-instance SSM
  time-series features + k-means, NOT a faithful re-implementation of
  their exact architecture/training details (which the paper does not
  fully specify either). Documented as a simplified reproduction in the
  paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .geometry import BoxTrajectory
from .interaction import CAR_FOLLOWING, ONCOMING


# --------------------------------------------------------------------------
# RSS (rule-based extreme)
# --------------------------------------------------------------------------

@dataclass
class RSSParams:
    """Commonly-cited example constants from the RSS literature (not a
    specific vehicle's certified specs)."""
    response_time: float = 1.0        # rho, seconds
    accel_max: float = 3.0            # a_max,accel, m/s^2
    brake_min: float = 4.0            # a_min,brake / a_max,brake (symmetric conservative assumption)


DEFAULT_RSS_PARAMS = RSSParams()


def rss_safe_distance_same_direction(v_rear: np.ndarray, v_front: np.ndarray, p: RSSParams = DEFAULT_RSS_PARAMS) -> np.ndarray:
    """RSS longitudinal safe distance, same-direction (rear car behind
    front car, both moving the same way -- the car-following case)."""
    rho, a_acc, a_brk = p.response_time, p.accel_max, p.brake_min
    d = (v_rear * rho + 0.5 * a_acc * rho ** 2 + (v_rear + rho * a_acc) ** 2 / (2 * a_brk)
         - v_front ** 2 / (2 * a_brk))
    return np.clip(d, 0.0, None)


def rss_safe_distance_opposite_direction(v1: np.ndarray, v2: np.ndarray, p: RSSParams = DEFAULT_RSS_PARAMS) -> np.ndarray:
    """RSS longitudinal safe distance, opposite-direction (oncoming) --
    both parties must be able to stop before reaching each other."""
    rho, a_acc, a_brk = p.response_time, p.accel_max, p.brake_min
    d1 = v1 * rho + 0.5 * a_acc * rho ** 2 + (v1 + rho * a_acc) ** 2 / (2 * a_brk)
    d2 = v2 * rho + 0.5 * a_acc * rho ** 2 + (v2 + rho * a_acc) ** 2 / (2 * a_brk)
    return d1 + d2


def rss_margin(
    ego: BoxTrajectory, agent: BoxTrajectory, dt: float, interaction_type: np.ndarray,
    p: RSSParams = DEFAULT_RSS_PARAMS,
) -> np.ndarray:
    """m_RSS(t) = actual_gap(t) - d_safe(t): positive = RSS-compliant,
    negative = RSS violation, per frame. `interaction_type` (T,) selects
    the same-direction vs. opposite-direction sub-formula per frame (see
    module docstring); crossing/merging/other default to same-direction."""
    dx = agent.x - ego.x
    dy = agent.y - ego.y
    center_dist = np.sqrt(dx ** 2 + dy ** 2)
    half_diag_ego = 0.5 * np.hypot(np.broadcast_to(ego.length, center_dist.shape),
                                    np.broadcast_to(ego.width, center_dist.shape))
    half_diag_agent = 0.5 * np.hypot(np.broadcast_to(agent.length, center_dist.shape),
                                      np.broadcast_to(agent.width, center_dist.shape))
    gap = np.clip(center_dist - half_diag_ego - half_diag_agent, 0.0, None)

    ego_speed = np.sqrt(np.gradient(ego.x, dt) ** 2 + np.gradient(ego.y, dt) ** 2)
    agent_speed = np.sqrt(np.gradient(agent.x, dt) ** 2 + np.gradient(agent.y, dt) ** 2)

    is_oncoming = interaction_type == ONCOMING
    # same-direction: whichever party's own forward motion points more
    # toward the other is treated as "rear" (responsible for keeping the
    # gap); a simple, dataset-agnostic proxy using the closing-speed sign
    # already used throughout this package.
    from .data_pipeline import _relative_closing_speed
    v_close = _relative_closing_speed(ego, agent, dt)
    ego_is_rear = v_close >= 0  # ego closing onto agent -> ego treated as rear
    v_rear = np.where(ego_is_rear, ego_speed, agent_speed)
    v_front = np.where(ego_is_rear, agent_speed, ego_speed)

    d_same = rss_safe_distance_same_direction(v_rear, v_front, p)
    d_opp = rss_safe_distance_opposite_direction(ego_speed, agent_speed, p)
    d_safe = np.where(is_oncoming, d_opp, d_same)
    return gap - d_safe


def rss_score(margin_min_over_time: np.ndarray, scale: float = 150.0) -> np.ndarray:
    """Trajectory-level RSS violation -> bounded [0,1] risk-like score,
    same clip-and-normalize convention as raw_ttc_baseline_score/DRAC:
    how deep (in meters) the worst RSS-margin violation went, normalized
    by `scale`. Non-negative margin (compliant) maps to 0.

    `scale=150` (empirically calibrated, not the initially-chosen 20):
    RSS's opposite-direction formula sums *both* parties' full stopping
    distances, and this package's instance-level score takes the worst
    violation across every agent in the instance, so violations routinely
    reach a hundred-plus meters even in ordinary interactions -- checked on
    a 500-instance DeepAccident sample: `scale=20` saturated 46% of
    instances to a score of exactly 1.0 (RSS is, by design, a hard
    compliant/non-compliant envelope rather than a graded severity measure,
    so *some* saturation is a real property being measured, not an
    artifact -- but 46% ties would make ranking-based metrics like
    AUROC/AUPRC mostly reflect coin-flip tie-breaking rather than real
    discrimination). `scale=150` still saturates 7% (a real residual RSS
    property) while keeping the rest resolvable, the best tradeoff found
    across {60, 100, 150, 200, 300}."""
    m = np.asarray(margin_min_over_time, dtype=float)
    return np.clip(-m / scale, 0.0, 1.0)


def instance_rss_score(ego: BoxTrajectory, agent_boxes: Dict[str, BoxTrajectory],
                        interaction_types: Dict[str, np.ndarray], dt: float) -> float:
    """Worst-case (max violation) RSS score across every agent in the
    instance -- same "worst agent drives the instance score" convention as
    ssm_baselines.instance_ssm_scores."""
    if not agent_boxes:
        return 0.0
    margins = [float(np.min(rss_margin(ego, box, dt, interaction_types.get(aid, np.zeros(len(box.x), dtype=int)))))
               for aid, box in agent_boxes.items()]
    return float(rss_score(np.array([min(margins)]))[0])


# --------------------------------------------------------------------------
# Lu et al. (2022) reproduction (data-driven extreme)
# --------------------------------------------------------------------------

def _resample_to_length(x: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a 1D array to length n (SSM sequences have
    variable length across instances; a fixed-length input is needed for
    the autoencoder)."""
    if len(x) == n:
        return x
    src_idx = np.linspace(0, len(x) - 1, num=len(x))
    tgt_idx = np.linspace(0, len(x) - 1, num=n)
    return np.interp(tgt_idx, src_idx, x)


def build_ssm_sequence_features(tau_3d: np.ndarray, drac_t: np.ndarray, evasive_severity: np.ndarray,
                                 seq_len: int = 16) -> np.ndarray:
    """Per-instance fixed-length feature vector: sequences of TTC, DRAC, and
    evasive-severity resampled to `seq_len` and concatenated -- "sequences
    of surrogate safety measures" per Lu et al.'s own description, using
    only quantities already computed elsewhere in this package."""
    ttc_capped = np.clip(np.nan_to_num(tau_3d, nan=10.0, posinf=10.0), 0.0, 10.0)
    feats = np.concatenate([
        _resample_to_length(ttc_capped, seq_len),
        _resample_to_length(np.nan_to_num(drac_t, nan=0.0, posinf=0.0), seq_len),
        _resample_to_length(np.nan_to_num(evasive_severity, nan=0.0), seq_len),
    ])
    return feats.astype(np.float32)


class LuEtAlReproduction:
    """Simplified reproduction: a small autoencoder embeds per-instance SSM
    sequences (unsupervised, reconstruction loss only, no label ever
    used); k-means then clusters the embeddings, and the risk score for a
    new instance is its distance to the *least severe* cluster centroid
    (ranked by that cluster's mean raw TTC -- higher mean TTC = safer
    cluster), normalized -- a label-free way to decide which cluster
    represents "conflict" vs. "normal" traffic, matching the paper's
    "identify traffic conflict by clustering" description without using
    the collision label anywhere in fitting.
    """

    def __init__(self, input_dim: int, latent_dim: int = 8, n_clusters: int = 4, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch = torch
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, input_dim),
        )
        self.n_clusters = n_clusters
        self.seed = seed
        self._fitted = False

    def fit(self, X: np.ndarray, ttc_summary: np.ndarray, epochs: int = 60, lr: float = 1e-3):
        """X: (N, input_dim) SSM-sequence features. ttc_summary: (N,) a
        scalar per instance (e.g. min raw TTC) used ONLY to rank cluster
        centroids by severity after clustering -- not used as a training
        signal for the autoencoder itself, and not the collision label."""
        import torch
        from sklearn.cluster import KMeans

        torch = self.torch
        X_t = torch.from_numpy(X)
        mu, sigma = X_t.mean(0, keepdim=True), X_t.std(0, keepdim=True).clamp_min(1e-6)
        self._mu, self._sigma = mu, sigma
        X_norm = (X_t - mu) / sigma

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            z = self.encoder(X_norm)
            recon = self.decoder(z)
            loss = ((recon - X_norm) ** 2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            Z = self.encoder(X_norm).numpy()
        km = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        labels = km.fit_predict(Z)
        # Rank clusters by mean raw-TTC-summary (higher TTC = safer); the
        # single safest cluster's centroid is the reference point distance
        # to which becomes the risk score.
        cluster_mean_ttc = np.array([ttc_summary[labels == k].mean() if (labels == k).any() else np.inf
                                      for k in range(self.n_clusters)])
        self._safest_centroid = km.cluster_centers_[int(np.argmax(cluster_mean_ttc))]
        self._km = km
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Risk score per instance: normalized embedding-space distance
        from the safest cluster's centroid (further from "safe" = riskier),
        min-max normalized over the batch being scored to [0,1] for
        ranking-based metrics (AUROC/AUPRC)."""
        import torch
        assert self._fitted, "call fit() first"
        torch = self.torch
        X_t = torch.from_numpy(X)
        X_norm = (X_t - self._mu) / self._sigma
        with torch.no_grad():
            Z = self.encoder(X_norm).numpy()
        dist = np.linalg.norm(Z - self._safest_centroid[None, :], axis=1)
        lo, hi = dist.min(), dist.max()
        if hi - lo < 1e-9:
            return np.zeros_like(dist)
        return (dist - lo) / (hi - lo)
