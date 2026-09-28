"""Synthetic end-to-end smoke test for the FutureRisk / CARM pipeline.

Not run automatically in this environment (numpy/torch are not installed
here -- see README.md). Intended to be run with `pytest` or
`python -m futurerisk.tests.test_smoke` in an environment with numpy, torch,
and matplotlib installed, to sanity-check shapes and value ranges end to
end on a synthetic two-car scenario (one collision-free "close call", one
clean pass).
"""
from __future__ import annotations

import numpy as np

from futurerisk.config import DEFAULT_CONFIG, FutureRiskConfig
from futurerisk.geometry import BoxTrajectory, compute_3d_ttc
from futurerisk.interaction import classify_interaction_type, dominant_interaction_type, TYPE_NAMES
from futurerisk.event_proxy import (
    longitudinal_deceleration, lateral_acceleration, evasive_severity,
    estimate_thresholds_for_dataset, critical_event_time,
)
from futurerisk.safety_prior import (
    analytic_safe_ttc_prior, analytic_safe_ttc_prior_for_class, context_dependent_margin,
    instance_margin, instantaneous_risk, analytic_trajectory_risk,
)
from futurerisk.evaluation import (
    risk_time_integral, critical_duration, persistence_metrics,
    group_discrimination_metrics, persistence_summary_by_group,
)


def make_close_call_scenario(cfg=DEFAULT_CONFIG):
    """Ego drives straight; a crossing agent nearly intersects ego's path
    at t~2s then brakes hard and passes just in front -- collision-free but
    with a large risk spike."""
    T = int(round(cfg.horizon_s / cfg.dt)) + 1
    t = np.arange(T) * cfg.dt

    ego_x = 10.0 * t
    ego_y = np.zeros(T)
    ego_heading = np.zeros(T)

    # agent crosses from the side, decelerating hard right before the
    # would-be intersection point (~x=20, t=2.0s)
    agent_y = -20.0 + 10.0 * t
    agent_x = np.full(T, 20.0)
    agent_heading = np.full(T, np.pi / 2)

    ego = BoxTrajectory(ego_x, ego_y, np.zeros(T), ego_heading, 4.5, 2.0, 1.5)
    agent = BoxTrajectory(agent_x, agent_y, np.zeros(T), agent_heading, 4.5, 2.0, 1.5)
    return t, ego, agent


def test_smoke():
    cfg = DEFAULT_CONFIG
    t, ego, agent = make_close_call_scenario(cfg)

    tau3d = compute_3d_ttc(ego, agent, dt=cfg.ttc_delta_step, cap=cfg.ttc_cap)
    assert tau3d.shape == t.shape

    types = classify_interaction_type(
        ego.heading, agent.heading, ego.x, ego.y, agent.x, agent.y,
        lane_graph_available=False, cfg=cfg,
    )
    assert set(types.tolist()) <= set(TYPE_NAMES.keys())

    ego_speed = np.gradient(ego.x, cfg.dt)
    a_lon = longitudinal_deceleration(ego_speed, cfg.dt)
    a_lat = lateral_acceleration(ego_speed, cfg.dt, heading=ego.heading)
    a_t = evasive_severity(a_lon, a_lat)
    assert (a_t >= 0).all()

    samples_by_type = {name: (a_t if types[0] == code else np.array([]))
                       for code, name in TYPE_NAMES.items()}
    # pad every type with enough synthetic samples so nothing merges into
    # "other" for this toy test
    for name in samples_by_type:
        if samples_by_type[name].size < cfg.n_min_samples:
            samples_by_type[name] = np.concatenate([samples_by_type[name], np.random.rand(cfg.n_min_samples)])
    table = estimate_thresholds_for_dataset(samples_by_type, percentile=cfg.threshold_percentile, n_min=cfg.n_min_samples)

    collision_flags = np.zeros(len(t), dtype=bool)  # collision-free by construction
    type_names = np.array([TYPE_NAMES[c] for c in types])
    t_evt = critical_event_time(t, collision_flags, a_t, type_names, table)
    assert isinstance(t_evt, float)

    dx = agent.x - ego.x
    dy = agent.y - ego.y
    dist = np.clip(np.sqrt(dx ** 2 + dy ** 2), 1e-3, None)
    rel_vx = np.gradient(ego.x, cfg.dt) - np.gradient(agent.x, cfg.dt)
    rel_vy = np.gradient(ego.y, cfg.dt) - np.gradient(agent.y, cfg.dt)
    v_close = -(rel_vx * dx / dist + rel_vy * dy / dist)
    tau_phy = analytic_safe_ttc_prior(v_close, a_avail=cfg.a_avail, t_react=cfg.t_react,
                                       eps_unc=cfg.eps_unc, clip_range=cfg.tau_phy_clip)
    assert ((tau_phy >= cfg.tau_phy_clip[0]) & (tau_phy <= cfg.tau_phy_clip[1])).all()

    margin_t = context_dependent_margin(tau3d, tau_phy)
    m = instance_margin(margin_t)
    r_t = instantaneous_risk(margin_t, T_m=cfg.T_m)
    R_analytic = analytic_trajectory_risk(margin_t, T_m=cfg.T_m)
    assert 0.0 <= R_analytic <= 1.0
    assert (r_t >= 0).all() and (r_t <= 1).all()
    print("Smoke test OK. R_analytic =", R_analytic, "worst margin =", m)


def test_persistence_and_grouped_reporting():
    """Sanity-check the two extensions: (A) persistence metrics + per-group
    discrimination reporting, (B) object-class-aware tau_phy."""
    cfg = DEFAULT_CONFIG
    t, ego, agent = make_close_call_scenario(cfg)
    tau3d = compute_3d_ttc(ego, agent, dt=cfg.ttc_delta_step, cap=cfg.ttc_cap)

    dx, dy = agent.x - ego.x, agent.y - ego.y
    dist = np.clip(np.sqrt(dx ** 2 + dy ** 2), 1e-3, None)
    rel_vx = np.gradient(ego.x, cfg.dt) - np.gradient(agent.x, cfg.dt)
    rel_vy = np.gradient(ego.y, cfg.dt) - np.gradient(agent.y, cfg.dt)
    v_close = -(rel_vx * dx / dist + rel_vy * dy / dist)

    # (B) object-class-aware tau_phy: pedestrian should demand a strictly
    # larger (more conservative) safe-TTC than a vehicle for the same
    # closing kinematics.
    tau_phy_vehicle = analytic_safe_ttc_prior_for_class(v_close, "vehicle", cfg=cfg)
    tau_phy_pedestrian = analytic_safe_ttc_prior_for_class(v_close, "pedestrian", cfg=cfg)
    tau_phy_unknown = analytic_safe_ttc_prior_for_class(v_close, "some_unclassified_blob", cfg=cfg)
    assert (tau_phy_pedestrian >= tau_phy_vehicle).all()
    # unknown/unclassified falls back to the "other" bucket, which defaults
    # to the same conservative profile as "pedestrian" (see config.py)
    assert np.allclose(tau_phy_unknown, tau_phy_pedestrian)

    margin_t = context_dependent_margin(tau3d, tau_phy_vehicle)
    r_t = instantaneous_risk(margin_t, T_m=cfg.T_m)

    # (A) persistence metrics for a single instance
    pm = persistence_metrics(r_t, t, margin_t=margin_t, margin_threshold=0.0)
    assert pm["peak_risk"] == float(np.max(r_t))
    assert pm["risk_time_integral"] >= 0.0
    assert pm["critical_duration_s"] >= 0.0

    types = classify_interaction_type(
        ego.heading, agent.heading, ego.x, ego.y, agent.x, agent.y,
        lane_graph_available=False, cfg=cfg,
    )
    dom_type = dominant_interaction_type(types)
    assert dom_type in TYPE_NAMES

    # (A) grouped discrimination metrics across a handful of synthetic
    # "instances" (reusing the same trace with noise, just to exercise the
    # grouping/aggregation machinery end to end).
    n_instances = 12
    rng = np.random.default_rng(0)
    risk_scores = np.clip(np.max(r_t) + rng.normal(0, 0.05, n_instances), 0, 1)
    collision_labels = (risk_scores > 0.5).astype(int)
    group_labels = [TYPE_NAMES[dom_type]] * (n_instances // 2) + ["other"] * (n_instances - n_instances // 2)
    grouped = group_discrimination_metrics(risk_scores, collision_labels, group_labels, min_group_size=3)
    assert "overall" in grouped
    assert TYPE_NAMES[dom_type] in grouped

    persistence_values = [persistence_metrics(r_t, t, margin_t=margin_t) for _ in range(n_instances)]
    grouped_persistence = persistence_summary_by_group(persistence_values, group_labels)
    assert TYPE_NAMES[dom_type] in grouped_persistence

    print("Extensions smoke test OK.",
          "tau_phy(vehicle) vs tau_phy(pedestrian) sample:", tau_phy_vehicle[0], tau_phy_pedestrian[0],
          "persistence:", pm)


if __name__ == "__main__":
    test_smoke()
    test_persistence_and_grouped_reporting()
