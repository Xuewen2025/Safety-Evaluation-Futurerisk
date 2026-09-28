"""Sanity checks for the PET/DRAC baseline implementations
(`ssm_baselines.py`) via synthetic positive/negative controls -- same
verification style as `geometry.py`'s SAT-overlap tests."""
from __future__ import annotations

import numpy as np

from futurerisk.geometry import BoxTrajectory
from futurerisk.ssm_baselines import (
    deceleration_rate_to_avoid_crash, drac_score, pet_score, post_encroachment_time,
)


def _box(x, y, heading=0.0, length=4.5, width=1.8):
    n = len(x)
    return BoxTrajectory(
        x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float), z=np.zeros(n),
        heading=np.full(n, heading), length=length, width=width, height=1.5,
    )


def test_drac_closing_pair_matches_formula():
    dt = 0.1
    # ego stationary at x=0; agent approaching from x=20 at constant 10 m/s
    # (closing speed v_close=10, gap starts at 20) -- DRAC(0) = v^2/(2*gap)
    t = np.arange(30) * dt
    ego = _box(np.zeros(30), np.zeros(30))
    agent_x = 20.0 - 10.0 * t
    agent = _box(agent_x, np.zeros(30))
    drac = deceleration_rate_to_avoid_crash(ego, agent, dt)
    expected0 = 10.0 ** 2 / (2 * 20.0)
    assert abs(drac[0] - expected0) < 0.5, f"expected ~{expected0}, got {drac[0]}"
    assert drac[0] > 0


def test_drac_receding_pair_is_zero():
    dt = 0.1
    t = np.arange(30) * dt
    ego = _box(np.zeros(30), np.zeros(30))
    agent_x = 5.0 + 10.0 * t  # moving away
    agent = _box(agent_x, np.zeros(30))
    drac = deceleration_rate_to_avoid_crash(ego, agent, dt)
    assert np.allclose(drac, 0.0), "receding pair should have DRAC=0 everywhere (not closing)"


def test_pet_crossing_paths_at_different_times():
    # ego drives straight along x, passing through (0,0) at t=0.
    # agent drives straight along y, passing through (0,0) at t=2.0s --
    # paths cross at the origin, 2s apart -> PET should be ~2.0s.
    t = np.arange(-20, 41) * 0.1
    ego = _box(t * 5.0, np.zeros(len(t)))            # at (0,0) when t=0
    agent = _box(np.zeros(len(t)), (t - 2.0) * 5.0)  # at (0,0) when t=2.0
    pet = post_encroachment_time(ego, agent, t)
    assert abs(pet - 2.0) < 0.15, f"expected PET~2.0s, got {pet}"


def test_pet_parallel_never_crossing_is_inf():
    t = np.arange(30) * 0.1
    ego = _box(t * 5.0, np.zeros(30))
    agent = _box(t * 5.0, np.full(30, 50.0))  # 50m away laterally, same direction, never crosses
    pet = post_encroachment_time(ego, agent, t)
    assert pet == float("inf")


def test_scores_are_monotonic_and_bounded():
    assert drac_score(np.array([0.0, 4.0, 100.0]), cap=10.0).tolist() == [0.0, 0.4, 1.0]
    s = pet_score(np.array([0.0, 2.5, float("inf")]), saturation_s=5.0)
    assert s[0] == 1.0 and 0.0 < s[1] < 1.0 and s[2] == 0.0
    assert np.all(s[:-1] >= s[1:])  # monotonically decreasing in PET
