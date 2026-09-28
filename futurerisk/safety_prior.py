"""Analytic safe-TTC prior, context-dependent safety margin, and continuous
risk mapping (S3.4 / eq:margin, eq:phy-prior, eq:phi).
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def analytic_safe_ttc_prior(
    v_rel_close: np.ndarray,
    a_avail: float = 4.0,
    t_react: float = 0.5,
    eps_unc: float = 0.2,
    clip_range: tuple = (0.2, 8.0),
) -> np.ndarray:
    """eq:phy-prior.

    tau_phy^j(t) = t_react + v_rel_close^j(t) / a_avail + eps_unc(t)

    v_rel_close: closing-speed component along the current relative bearing
    (positive = closing). eps_unc is treated as the constant safety pad
    (S3.4 default 0.2s); pass an array if a time-varying pad is desired.
    """
    v_rel_close = np.asarray(v_rel_close, dtype=float)
    tau_phy = t_react + v_rel_close / a_avail + eps_unc
    return np.clip(tau_phy, clip_range[0], clip_range[1])


def analytic_safe_ttc_prior_for_class(
    v_rel_close: np.ndarray, agent_class: str, cfg=None,
) -> np.ndarray:
    """Object-class-aware convenience wrapper around
    `analytic_safe_ttc_prior`: looks up (a_avail, t_react, eps_unc,
    tau_phy_clip) for `agent_class` via
    `cfg.get_physical_prior_params()` (vehicle/truck/bus/cyclist/pedestrian
    each get physically distinct constants, S3.4) instead of one fixed
    triple shared by every agent regardless of what it is."""
    if cfg is None:
        from .config import DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG
    params = cfg.get_physical_prior_params(agent_class)
    return analytic_safe_ttc_prior(
        v_rel_close, a_avail=params.a_avail, t_react=params.t_react,
        eps_unc=params.eps_unc, clip_range=params.tau_phy_clip,
    )


def context_dependent_margin(tau_3d: np.ndarray, tau_safe: np.ndarray) -> np.ndarray:
    """eq:margin (per agent, per time): m^j(t) = tau_3D^j(t) - tau_safe^j(t).
    tau_3d may contain +inf entries (no predicted overlap); numpy handles
    inf - finite = inf correctly."""
    return np.asarray(tau_3d, dtype=float) - np.asarray(tau_safe, dtype=float)


def instance_margin(margin_per_agent_time) -> float:
    """m = min_j min_t m^j(t), eq:margin. Accepts any array-like (or list of
    per-agent arrays with possibly different lengths); returns the global
    scalar worst-case margin for one trajectory instance."""
    if isinstance(margin_per_agent_time, (list, tuple)):
        vals = [np.asarray(a, dtype=float) for a in margin_per_agent_time if len(a)]
        if not vals:
            return np.inf
        return float(min(np.min(v) for v in vals))
    arr = np.asarray(margin_per_agent_time, dtype=float)
    if arr.size == 0:
        return np.inf
    return float(np.min(arr))


def phi(margin: np.ndarray, T_m: float = 1.0) -> np.ndarray:
    """eq:phi risk-mapping function: phi(m) = sigma(-m / T_m).

    Monotonically *decreasing* in margin: phi(m) -> 1 as m -> -inf
    (deep intrusion / collision), phi(m) -> 0 as m -> +inf (large safe
    margin). This sign convention is what makes r(t) = phi(m(t)) behave as
    a risk score (high risk <=> small/negative margin)."""
    return sigmoid(-np.asarray(margin, dtype=float) / T_m)


def instantaneous_risk(margin_t: np.ndarray, T_m: float = 1.0) -> np.ndarray:
    """r(t) = phi(m(t)), eq:phi, evaluated at every t given the per-time
    (already agent-minimized) margin m(t)."""
    return phi(margin_t, T_m=T_m)


def analytic_trajectory_risk(margin_per_agent_time, T_m: float = 1.0) -> float:
    """R_analytic for one trajectory instance, eq:phi.

    Because phi is monotonically decreasing in its argument,
        max_{j,t} phi(m^j(t)) == phi( min_{j,t} m^j(t) ) == phi(m)
    with m the global worst-case margin from eq:margin. We compute it via
    the right-hand side (numerically identical, only one phi evaluation
    needed) rather than literally maximizing phi(-m^j(t)) pointwise, which
    would invert the intended sign (see note below).

    Implementation note on a sign subtlety in the source method
    description: the text also glosses R_analytic as
    "max_{j,t} phi(-m^j(t)) ... equivalently phi(-m)". Read completely
    literally that phrasing double-negates the margin and would make
    R_analytic *increase* with a larger (safer) margin, which contradicts
    phi's own definition and the paper's stated purpose (higher risk for
    closer calls). We implement the self-consistent quantity
    R_analytic = phi(m) = max_{j,t} phi(m^j(t)), matching eq:phi's r(t)
    definition and eq:margin's aggregation; flag this to the paper authors
    when reconciling notation.
    """
    m = instance_margin(margin_per_agent_time)
    return float(phi(np.array(m), T_m=T_m))


def analytic_risk_target(margin_per_agent_time, T_m: float = 1.0) -> float:
    """Alias used at training time: y_analytic (S3.7, "dense self-supervised
    analytic target" computed from eq:phi for every trajectory instance)."""
    return analytic_trajectory_risk(margin_per_agent_time, T_m=T_m)
