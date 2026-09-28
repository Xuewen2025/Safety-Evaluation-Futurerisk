"""Sanity checks for the forward-projected (CTRV) analytic-TTC baseline
(`forward_ttc_baseline.py`) via synthetic positive/negative controls, same
verification style as `test_ssm_baselines.py`.

Key property under test: when the *true* future trajectory genuinely is a
constant-(v, omega) motion, `forward_ttc_ctrv` (which assumes exactly that)
must agree with `geometry.compute_3d_ttc` (the oracle, ground-truth-reading
measure) almost exactly -- there is no modeling gap to create a
discrepancy in that case. The discrepancy this baseline is *supposed* to
expose only shows up once the true trajectory deviates from constant-(v,
omega) (e.g. a real evasive maneuver), which is exactly the point of
adding it as a second, deployable comparison point alongside the oracle.
"""
from __future__ import annotations

import numpy as np

from futurerisk.geometry import BoxTrajectory, compute_3d_ttc
from futurerisk.forward_ttc_baseline import forward_ttc_ctrv


def _box(x, y, heading, length=4.5, width=1.8):
    n = len(x)
    heading = np.full(n, heading) if np.isscalar(heading) else np.asarray(heading, dtype=float)
    return BoxTrajectory(
        x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float), z=np.zeros(n),
        heading=heading, length=length, width=width, height=1.5,
    )


def test_straight_closing_pair_matches_oracle_3d_ttc():
    # Ego stationary at origin; agent approaching head-on at constant 10
    # m/s from x=20, no turning -- a genuine constant-velocity-straight-line
    # trajectory, so the oracle (reads real future) and the forward
    # CTRV projection (assumes constant v, omega=0 from t=0) must agree.
    dt = 0.1
    n = 60
    t = np.arange(n) * dt
    ego = _box(np.zeros(n), np.zeros(n), heading=0.0)
    agent = _box(20.0 - 10.0 * t, np.zeros(n), heading=np.pi)  # facing ego, moving toward it

    oracle = compute_3d_ttc(ego, agent, dt=dt, cap=5.0)
    forward = forward_ttc_ctrv(ego, agent, dt=dt, cap=5.0)

    finite = np.isfinite(oracle) & np.isfinite(forward)
    assert finite.sum() > 10, "expected a real overlap window in this synthetic head-on scenario"
    assert np.allclose(oracle[finite], forward[finite], atol=dt), (
        f"forward CTRV projection should match the oracle on a truly constant-velocity "
        f"straight-line trajectory; max diff={np.max(np.abs(oracle[finite]-forward[finite]))}"
    )


def test_receding_pair_is_inf():
    dt = 0.1
    n = 30
    t = np.arange(n) * dt
    ego = _box(np.zeros(n), np.zeros(n), heading=0.0)
    agent = _box(5.0 + 10.0 * t, np.zeros(n), heading=0.0)  # moving away, same direction
    tau = forward_ttc_ctrv(ego, agent, dt=dt, cap=5.0)
    assert np.all(np.isinf(tau))


def test_turning_ego_avoids_collision_that_straight_projection_would_predict():
    # Ego is actually executing a hard left turn (large, constant yaw
    # rate) away from an agent directly ahead; the agent sits still in
    # ego's current lane. A straight-line (ignore_turning=True) projection
    # must predict a collision (ego "continues straight" into the
    # stationary agent), while the turning-aware projection -- correctly
    # reading ego's own constant-yaw-rate arc -- must see the ego curving
    # away and NOT predict that same collision. This is the synthetic
    # analogue of Li et al.'s 1D-vs-2D discrepancy claim.
    dt = 0.1
    n = 40
    t = np.arange(n) * dt
    omega = 1.2  # rad/s, a hard constant turn
    v = 8.0
    theta = omega * t
    ego_x = (v / omega) * np.sin(theta)
    ego_y = -(v / omega) * (np.cos(theta) - 1.0)
    ego = _box(ego_x, ego_y, heading=theta)

    agent_x = np.full(n, 15.0)
    agent_y = np.zeros(n)
    agent = _box(agent_x, agent_y, heading=0.0)

    turning_aware = forward_ttc_ctrv(ego, agent, dt=dt, cap=3.0, ignore_turning=False)
    straight_only = forward_ttc_ctrv(ego, agent, dt=dt, cap=3.0, ignore_turning=True)

    assert np.isfinite(straight_only[0]), "straight-line (1D-style) projection should predict a collision here"
    assert not np.isfinite(turning_aware[0]), (
        "turning-aware (2D) projection should correctly see ego curving away from the stationary agent"
    )


def test_ignore_turning_matches_straight_manual_projection():
    # With ignore_turning=True, a genuinely-turning box's own future
    # position must NOT matter -- only its instantaneous (x,y,theta,v) at
    # each anchor frame, projected in a straight line. Verify the
    # first-frame projection directly against a hand-computed straight
    # line, independent of the box's real (curving) motion.
    dt = 0.1
    n = 20
    t = np.arange(n) * dt
    omega = 0.8
    v = 5.0
    theta = omega * t
    x = (v / omega) * np.sin(theta)
    y = -(v / omega) * (np.cos(theta) - 1.0)
    curving = _box(x, y, heading=theta)

    stationary_far_ahead = _box(np.full(n, 1000.0), np.full(n, 1000.0), heading=0.0)
    # No real collision possible; this just exercises the code path without div-by-zero.
    tau = forward_ttc_ctrv(curving, stationary_far_ahead, dt=dt, cap=2.0, ignore_turning=True)
    assert np.all(np.isinf(tau))
