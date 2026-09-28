"""Forward-projected, deployable analytic TTC -- Li et al. (2023/2024,
"Beyond 1D and oversimplified kinematics: A generic analytical framework
for surrogate safety measures", arXiv:2312.07019) Section 4.1's 2-D
kinematic-bicycle-model instantiation of their generic SSM framework,
implemented as a comparison baseline (experiment_design.txt's Gap 1
rule-based cluster; see also `ssm_baselines.py`, `baselines_extra.py`).

Why this is a genuinely different baseline from what's already here
-------------------------------------------------------------------
`geometry.compute_3d_ttc` (this project's only existing "3D-TTC") is a
post-hoc ORACLE measure: it reads the already-observed future ground-truth
poses and finds when the oriented boxes actually overlap -- correct for
this project's post-hoc evaluation setting (S3.1), but not something a
real-time system could compute, since it peeks at the true future.

Li et al.'s framework instead formalizes the classic, DEPLOYABLE notion of
TTC: given only the state at the current instant, hold the control input
constant ("inspired by conventional TTC" -- their own words, Section 4.1),
forward-project the trajectory under an assumed vehicle-movement model,
and solve for the first time the projected shape overlaps another vehicle
(their Eq. 2, `g_v(chi_i(t_c), chi_j(t_c)) = 0`). This module implements
exactly that, using their own Section 4.1 vehicle-movement model (the 2-D
kinematic bicycle model, their Eqs. 42-49) and this project's own SAT-based
box-overlap test (`geometry.sat_overlap`) as the collision condition, so
the result is directly comparable to `compute_3d_ttc`'s oracle numbers
using the same geometry.

Judgment calls (documented, not hidden -- same convention as every other
baseline module in this package)
-----------------------------------------------------------------------
1. **(v, omega) instead of (v, delta, L).** Eqs. 46-49 only ever use the
   steering angle delta through the product omega = v*tan(delta)/L (the
   yaw rate). Rather than inventing a per-agent wheelbase `L` this project
   has no ground truth for and back-solving for `delta`, this estimates
   the yaw rate `omega` directly by finite-differencing the
   already-computed heading trajectory. Holding (v, omega) constant is
   mathematically IDENTICAL to holding (v, delta) constant with a fixed L
   -- both give the same closed-form circular-arc trajectory -- so this
   is not a weaker approximation of Eqs. 46-49, it is their exact solution
   parameterized more directly. This is the standard "constant-turn-rate-
   and-velocity" (CTRV) motion model.
2. **Acceleration is not modeled (a=0 held).** Section 4.1's control is
   u=(delta, a); this implements only the a=0 sub-case, which is exactly
   Hayward's (1972) own "continue at their present speed" assumption
   generalized to a curved path, and has an exact closed-form solution
   (a circular arc). Nonzero constant acceleration is only solvable in
   Li et al.'s own framework via their separate Taylor-linearization
   (their Eq. 50), which trades exactness for generality; since the a=0
   case is both exact and sufficient to test their headline 1D-vs-2D
   claim (see `ignore_turning` below), this module does not implement the
   linearized a!=0 case.
3. **No vertical (z) motion projected.** Section 4.1 is a Cartesian 2-D
   framework; z/height are held at their current-frame value for the
   optional vertical-overlap check (`geometry.vertical_overlap`), matching
   `compute_3d_ttc`'s own vertical check but without projecting z forward.
4. **Yaw-rate smoothing.** `omega` from a raw finite difference of heading
   is noisy (a rate estimated from already-noisy positions); a small
   centered moving-average smooths it before use, the same spirit as
   `geometry.heading_from_positions`'s own noise-handling for heading
   itself.

The `ignore_turning` flag exists specifically to reproduce Li et al.'s own
headline comparison: forcing `omega=0` for every frame reduces this exact
same function to the classic 1-D/longitudinal-only TTC (their Section 3.1,
Hayward 1972's original formulation) computed with the *same* geometry and
*same* collision condition as the 2-D case -- so the discrepancy between
`ignore_turning=False` and `ignore_turning=True` on the same instances is a
direct, like-for-like replication of their reported 1D-vs-higher-dimensional
divergence (they report up to ~300% for non-critical TTC, ~20% for
critical TTC<1.5s), on this project's own datasets.
"""
from __future__ import annotations

import numpy as np

from .geometry import BoxTrajectory, rect_corners, sat_overlap, vertical_overlap


def _smooth(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Centered moving-average smoothing, edge-padded (reflect) so the
    output keeps the same length. `window=1` is a no-op."""
    if window <= 1:
        return x
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def _ctrv_state(box: BoxTrajectory, dt: float, omega_smoothing: int = 3):
    """Per-frame (x, y, theta, v, omega) -- the CTRV state -- from an
    observed `BoxTrajectory`. `omega` (yaw rate) is estimated from the
    unwrapped heading trace (finite difference + smoothing, see module
    docstring point 4); `v` (speed) from finite-differenced position,
    matching this package's existing closing-speed convention
    (`data_pipeline._relative_closing_speed`)."""
    theta = box.heading
    v = np.hypot(np.gradient(box.x, dt), np.gradient(box.y, dt))
    omega = _smooth(np.gradient(np.unwrap(theta), dt), window=omega_smoothing)
    return box.x, box.y, theta, v, omega


def _ctrv_forward(x0, y0, theta0, v0, omega0, delta_t: np.ndarray, ignore_turning: bool):
    """Exact closed-form CTRV forward projection.

    x0, y0, theta0, v0, omega0: shape (T,) -- current state per anchor
    frame. delta_t: shape (n_delta,) -- forward time offsets (>= 0),
    shared across all anchor frames.

    Returns x, y, theta each of shape (T, n_delta): the projected pose at
    anchor-frame t, `delta_t[k]` seconds into that frame's own future,
    assuming constant (v0[t], omega0[t]) from that point on.
    """
    if ignore_turning:
        omega0 = np.zeros_like(omega0)

    theta0 = theta0[:, None]
    v0 = v0[:, None]
    omega0 = omega0[:, None]
    x0 = x0[:, None]
    y0 = y0[:, None]
    dt_grid = delta_t[None, :]

    theta = theta0 + omega0 * dt_grid

    straight = np.abs(omega0) < 1e-4
    # Circular-arc branch (guard div-by-zero; overwritten by the straight
    # branch wherever `straight` is True, so the garbage there is harmless).
    omega_safe = np.where(straight, 1.0, omega0)
    r = v0 / omega_safe
    x_arc = x0 + r * (np.sin(theta) - np.sin(theta0))
    y_arc = y0 - r * (np.cos(theta) - np.cos(theta0))

    x_straight = x0 + v0 * np.cos(theta0) * dt_grid
    y_straight = y0 + v0 * np.sin(theta0) * dt_grid

    x = np.where(straight, x_straight, x_arc)
    y = np.where(straight, y_straight, y_arc)
    return x, y, theta


def forward_ttc_ctrv(
    box_e: BoxTrajectory, box_j: BoxTrajectory, dt: float = 0.1, cap: float = 5.0,
    check_vertical: bool = True, ignore_turning: bool = False, omega_smoothing: int = 3,
) -> np.ndarray:
    """Forward-projected TTC, eq:3dttc's deployable/analytic counterpart.

    tau_fwd^j(t) = inf{delta >= 0 : B_e(t+delta | anchored at t)
                        overlaps B_j(t+delta | anchored at t), delta <= cap}

    where B_e/B_j(. | anchored at t) is the CTRV forward projection from
    each party's own state AT time t (not the observed future -- see
    module docstring), evaluated with this project's own SAT box-overlap
    test. +inf if no overlap occurs within the capped horizon.

    Same (T,)-shaped, +inf-for-no-conflict convention as
    `geometry.compute_3d_ttc`, so it is a drop-in alternative input to
    `evaluation.raw_ttc_baseline_score` and to the same worst-agent
    aggregation `ssm_baselines.instance_ssm_scores` and
    `geometry.compute_3d_ttc_all_agents` already use.
    """
    T = len(box_e)
    assert len(box_j) == T
    n_delta = int(round(cap / dt)) + 1
    delta_t = np.arange(n_delta) * dt

    xe, ye, the, ve, we = _ctrv_state(box_e, dt, omega_smoothing)
    xj, yj, thj, vj, wj = _ctrv_state(box_j, dt, omega_smoothing)

    xe_p, ye_p, the_p = _ctrv_forward(xe, ye, the, ve, we, delta_t, ignore_turning)
    xj_p, yj_p, thj_p = _ctrv_forward(xj, yj, thj, vj, wj, delta_t, ignore_turning)

    len_e = np.broadcast_to(box_e.length[:, None], xe_p.shape)
    wid_e = np.broadcast_to(box_e.width[:, None], xe_p.shape)
    len_j = np.broadcast_to(box_j.length[:, None], xj_p.shape)
    wid_j = np.broadcast_to(box_j.width[:, None], xj_p.shape)

    corners_e = rect_corners(xe_p, ye_p, the_p, len_e, wid_e)
    corners_j = rect_corners(xj_p, yj_p, thj_p, len_j, wid_j)
    overlap = sat_overlap(corners_e, corners_j)

    if check_vertical:
        # No z-projection (module docstring point 3): held at each anchor
        # frame's own current z/height, broadcast across the delta axis.
        z_e = np.broadcast_to(box_e.z[:, None], overlap.shape)
        h_e = np.broadcast_to(box_e.height[:, None], overlap.shape)
        z_j = np.broadcast_to(box_j.z[:, None], overlap.shape)
        h_j = np.broadcast_to(box_j.height[:, None], overlap.shape)
        overlap &= vertical_overlap(z_e, h_e, z_j, h_j)

    any_overlap = overlap.any(axis=1)
    first_idx = np.argmax(overlap, axis=1)  # 0 where no True exists; guarded below
    tau = np.where(any_overlap, first_idx * dt, np.inf)
    return tau


def forward_ttc_all_agents(
    box_e: BoxTrajectory, agent_boxes: dict, dt: float = 0.1, cap: float = 5.0,
    check_vertical: bool = True, ignore_turning: bool = False,
) -> dict:
    """tau_fwd^j(t) for every surrounding agent j -- mirrors
    `geometry.compute_3d_ttc_all_agents`'s interface exactly."""
    return {
        j: forward_ttc_ctrv(box_e, box_j, dt=dt, cap=cap, check_vertical=check_vertical, ignore_turning=ignore_turning)
        for j, box_j in agent_boxes.items()
    }
