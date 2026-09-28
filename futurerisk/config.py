"""Central hyperparameter configuration for FutureRisk / CARM.

Every "e.g." default mentioned in the method description lives here so the
rest of the codebase never hard-codes a magic number twice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class PhysicalPriorParams:
    """Object-class-specific constants for the analytic safe-TTC prior
    (eq:phy-prior). A single fixed (a_avail, t_react, eps_unc) triple is
    only physically sensible for one kind of road user -- a pedestrian
    does not "brake" the way a car does, and a cyclist sits in between.
    Keeping the deterministic prior itself object-aware means CARM's
    learned correction (bounded to [0.5x, 2x], S3.6.3) only has to fine-tune
    around a reasonable per-class starting point rather than fight a
    systematically wrong one."""
    a_avail: float             # assumed available deceleration, m/s^2
    t_react: float             # fixed reaction time constant, s
    eps_unc: float             # constant safety pad, s
    tau_phy_clip: Tuple[float, float]


def _default_physical_prior_by_class() -> Dict[str, PhysicalPriorParams]:
    return {
        # Standard vehicle: matches the method description's own suggested
        # defaults (S3.4).
        "vehicle": PhysicalPriorParams(a_avail=4.0, t_react=0.5, eps_unc=0.2, tau_phy_clip=(0.2, 8.0)),
        # Larger/heavier vehicles brake less crisply than a typical car.
        "truck": PhysicalPriorParams(a_avail=3.0, t_react=0.6, eps_unc=0.25, tau_phy_clip=(0.2, 9.0)),
        "bus": PhysicalPriorParams(a_avail=3.0, t_react=0.6, eps_unc=0.25, tau_phy_clip=(0.2, 9.0)),
        # Cyclist: intermediate dynamics and predictability between a
        # vehicle and a pedestrian.
        "cyclist": PhysicalPriorParams(a_avail=3.0, t_react=0.6, eps_unc=0.25, tau_phy_clip=(0.25, 9.0)),
        # Pedestrian: "available deceleration" isn't a real braking action
        # (evasion is a side-step, not a stop), reaction is slower and far
        # less predictable, so we use a lower effective a_avail, a longer
        # t_react, and a larger uncertainty pad -- all pushing tau_phy
        # (the required safety margin) larger, i.e. more conservative.
        "pedestrian": PhysicalPriorParams(a_avail=2.0, t_react=0.8, eps_unc=0.3, tau_phy_clip=(0.3, 10.0)),
        # Unknown/unclassified object: fall back to the most conservative
        # (pedestrian-like) profile rather than silently assuming
        # vehicle-like dynamics for something we can't identify.
        "other": PhysicalPriorParams(a_avail=2.0, t_react=0.8, eps_unc=0.3, tau_phy_clip=(0.3, 10.0)),
    }


@dataclass
class FutureRiskConfig:
    # --- horizon / time discretization -----------------------------------
    dt: float = 0.1                 # trajectory resampling resolution (s), i.e. 10Hz
    horizon_s: float = 5.0          # H, seconds
    num_bins: int = 25              # K hazard bins over [0, H] (0.2s bins for H=5s)

    # --- 3D-TTC (sec:3dttc) -------------------------------------------------
    ttc_delta_step: float = 0.1     # discretized delta step, seconds
    ttc_cap: float = 5.0            # cap on forward-simulated delta, seconds

    # --- interaction type heuristic (sec:interaction-type) ------------------
    car_following_heading_deg: float = 20.0
    oncoming_heading_deg: float = 150.0
    crossing_heading_low_deg: float = 60.0
    crossing_heading_high_deg: float = 120.0
    lane_half_width_m: float = 1.75   # "half lane width"
    merge_window_s: float = 1.5       # +/- 1.5s around t for merge/lane-change flag

    # --- analytic physical safe-TTC prior (eq:phy-prior) ---------------------
    # Flat defaults kept for backward compatibility (== the "vehicle" row of
    # physical_prior_by_class below); prefer get_physical_prior_params() /
    # physical_prior_by_class for anything object-class-aware.
    a_avail: float = 4.0             # assumed available deceleration, m/s^2
    t_react: float = 0.5             # fixed reaction time constant, s
    eps_unc: float = 0.2             # constant safety pad, s
    tau_phy_clip: Tuple[float, float] = (0.2, 8.0)

    # Per-object-class physical prior constants (S3.4's "assumed available
    # deceleration"/"reaction time" are only meaningful per road-user type;
    # see PhysicalPriorParams). Keys are matched case-insensitively by
    # get_physical_prior_params(), with "other" as the fallback bucket.
    physical_prior_by_class: Dict[str, PhysicalPriorParams] = field(
        default_factory=_default_physical_prior_by_class
    )

    # --- risk mapping (eq:phi) ----------------------------------------------
    T_m: float = 1.0                 # temperature hyperparameter

    # --- harsh-evasive critical-event proxy (sec:proxy-event) ----------------
    threshold_percentile: float = 95.0
    n_min_samples: int = 500         # merge interaction type into "other" below this count

    # --- CARM heads (sec:carm) ------------------------------------------------
    delta_max: float = field(default_factory=lambda: math.log(2.0))  # bounds correction to [0.5x, 2x]
    sigma_min: float = 0.05
    sigma_max: float = 0.5
    composition_eps: float = 1e-4     # epsilon for logit-space numerical stability
    beta_residual: float = 2.0        # bounded residual logit-space scale

    # --- loss weights (eq:total-loss) -----------------------------------------
    lambda_evt: float = 1.0
    lambda_risk: float = 1.0
    lambda_rank: float = 0.5
    rank_margin_gamma: float = 0.1
    # Not in instruction.txt's spec -- see losses.safe_ttc_saturation_penalty's
    # docstring: mu_s/sigma_s get no direct supervision and were found to
    # collapse into their bounding nonlinearities' saturated tails (S-five.2
    # experiment). Small enough to only discourage runaway magnitude, not to
    # compete with L_evt/L_risk for the actual discriminative signal.
    lambda_saturation: float = 0.01

    # --- datasets --------------------------------------------------------------
    datasets: Tuple[str, ...] = ("DeepAccident", "nuPlan", "Bench2Drive")
    held_out_dataset: str = "DeepAccident"
    training_datasets: Tuple[str, ...] = ("nuPlan", "Bench2Drive")

    interaction_types: Tuple[str, ...] = (
        "car-following", "crossing", "merging/lane-change", "oncoming", "other",
    )

    @property
    def bin_edges(self):
        """b_0=0 < b_1 < ... < b_K=H, length K+1."""
        import numpy as np
        return np.linspace(0.0, self.horizon_s, self.num_bins + 1)

    def get_physical_prior_params(self, agent_class: str) -> PhysicalPriorParams:
        """Case-insensitive lookup into physical_prior_by_class, falling
        back to the "other" (most conservative) entry for an
        unrecognized/unclassified agent_class rather than silently
        defaulting to vehicle-like dynamics."""
        key = (agent_class or "other").strip().lower()
        table = self.physical_prior_by_class
        if key in table:
            return table[key]
        return table.get("other", table["vehicle"])


DEFAULT_CONFIG = FutureRiskConfig()
