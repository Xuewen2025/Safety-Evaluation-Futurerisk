"""Harsh-evasive critical-event proxy (S3.5): kinematic event signal a(t),
per-(dataset, interaction-type) threshold statistics (Algorithm 2), and the
critical-event time label T_evt (eq:tevt).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .interaction import TYPE_NAMES


# --------------------------------------------------------------------------
# Kinematic event signal (S3.5.1)
# --------------------------------------------------------------------------

def _finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference derivative, one-sided at the boundaries."""
    d = np.empty_like(x, dtype=float)
    if len(x) < 2:
        d[:] = 0.0
        return d
    d[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    return d


def _smooth(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Light moving-average smoothing used to make the longitudinal
    deceleration signal 'jerk-adjusted' (reduces single-frame noise before
    thresholding) rather than a raw noisy finite difference."""
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def longitudinal_deceleration(speed: np.ndarray, dt: float, smooth_window: int = 3) -> np.ndarray:
    """a_lon(t): jerk-adjusted longitudinal (de)acceleration magnitude
    signal, from a speed trace. Negative = decelerating."""
    accel = _finite_diff(np.asarray(speed, dtype=float), dt)
    return _smooth(accel, window=smooth_window)


def lateral_acceleration(speed: np.ndarray, dt: float,
                          curvature: Optional[np.ndarray] = None,
                          heading: Optional[np.ndarray] = None) -> np.ndarray:
    """a_lat(t), S3.5.1: v(t)^2 * kappa(t) if curvature is available,
    otherwise heading-rate * speed."""
    speed = np.asarray(speed, dtype=float)
    if curvature is not None:
        return speed ** 2 * np.asarray(curvature, dtype=float)
    if heading is None:
        raise ValueError("Provide either curvature or heading.")
    heading = np.asarray(heading, dtype=float)
    # wrap heading difference before differentiating to avoid +-pi jumps
    dh = np.diff(heading, prepend=heading[0])
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    heading_rate = dh / dt
    return heading_rate * speed


def evasive_severity(a_lon: np.ndarray, a_lat: np.ndarray) -> np.ndarray:
    """Combined evasive-severity scalar a(t), S3.5.1:
    a(t) = sqrt(max(-a_lon(t), 0)^2 + a_lat(t)^2)
    (only braking, not accelerating, counts toward the longitudinal term)."""
    a_lon = np.asarray(a_lon, dtype=float)
    a_lat = np.asarray(a_lat, dtype=float)
    braking = np.maximum(-a_lon, 0.0)
    return np.sqrt(braking ** 2 + a_lat ** 2)


# --------------------------------------------------------------------------
# Per-(dataset, interaction-type) threshold statistics (S3.5.2, Algorithm 2)
# --------------------------------------------------------------------------

@dataclass
class ThresholdTable:
    """theta_{d,c} lookup table for one dataset d."""
    dataset: str
    thresholds: Dict[str, float]        # interaction-type name -> theta
    merged_into_other: Dict[str, bool]  # which types got merged due to N_min

    def get(self, interaction_type_name: str) -> float:
        if self.merged_into_other.get(interaction_type_name, False):
            return self.thresholds["other"]
        return self.thresholds.get(interaction_type_name, self.thresholds["other"])


def estimate_thresholds_for_dataset(
    samples_by_type: Dict[str, np.ndarray],
    percentile: float = 95.0,
    n_min: int = 500,
) -> ThresholdTable:
    """Algorithm 2 (threshold estimation), run for a single dataset d.

    samples_by_type: interaction-type name -> 1D array of a(t) samples
    collected over all (t, j) pairs of that type *within this dataset only*
    (S3.5.2: "do NOT pool a(t) samples across DeepAccident / nuPlan /
    Bench2Drive"). This function itself only ever sees one dataset's
    samples -- the caller is responsible for keeping datasets separate,
    see `compute_threshold_tables`.
    """
    names = list(TYPE_NAMES.values())
    for name in samples_by_type:
        if name not in names:
            raise ValueError(f"Unknown interaction type '{name}'")

    merged = {name: False for name in names}
    # First pass: decide merges (insufficient samples -> merge into "other")
    other_samples = [np.asarray(samples_by_type.get("other", []), dtype=float)]
    for name in names:
        if name == "other":
            continue
        arr = np.asarray(samples_by_type.get(name, []), dtype=float)
        if arr.size < n_min:
            merged[name] = True
            other_samples.append(arr)

    thresholds: Dict[str, float] = {}
    for name in names:
        if name == "other":
            continue
        if merged[name]:
            continue
        arr = np.asarray(samples_by_type[name], dtype=float)
        thresholds[name] = float(np.percentile(arr, percentile))

    other_pool = np.concatenate(other_samples) if other_samples else np.array([])
    thresholds["other"] = float(np.percentile(other_pool, percentile)) if other_pool.size else float("inf")

    return ThresholdTable(dataset="", thresholds=thresholds, merged_into_other=merged)


def compute_threshold_tables(
    per_dataset_samples: Dict[str, Dict[str, np.ndarray]],
    percentile: float = 95.0,
    n_min: int = 500,
) -> Dict[str, ThresholdTable]:
    """Run Algorithm 2 independently per dataset (S3.5.2's central
    requirement). per_dataset_samples[d][c] = 1D array of a(t) samples for
    dataset d, interaction type c."""
    tables = {}
    for dataset, samples_by_type in per_dataset_samples.items():
        table = estimate_thresholds_for_dataset(samples_by_type, percentile=percentile, n_min=n_min)
        table.dataset = dataset
        tables[dataset] = table
    return tables


def save_threshold_tables(tables: Dict[str, ThresholdTable], path: str) -> None:
    """Persist the fixed preprocessing artifact (S3.5.2: 'a fixed
    preprocessing artifact, computed once, stored ... and reused for all
    subsequent label generation and training runs')."""
    payload = {d: asdict(t) for d, t in tables.items()}
    Path(path).write_text(json.dumps(payload, indent=2))


def load_threshold_tables(path: str) -> Dict[str, ThresholdTable]:
    payload = json.loads(Path(path).read_text())
    return {d: ThresholdTable(**t) for d, t in payload.items()}


# --------------------------------------------------------------------------
# Critical-event time label (S3.5.3, eq:tevt)
# --------------------------------------------------------------------------

def critical_event_time(
    times: np.ndarray,
    collision_flags: np.ndarray,
    a_t: np.ndarray,
    interaction_type_names: np.ndarray,
    threshold_table: ThresholdTable,
) -> float:
    """eq:tevt for a single agent j within one trajectory instance:

    T_evt^j = inf{ t in [0,H] : collision(t) OR a(t) > theta_{d,c(t,j)} }

    +inf if neither condition is ever met. Uses the dataset-and-type
    specific threshold table (already fitted for dataset d via Algorithm 2).
    """
    times = np.asarray(times, dtype=float)
    collision_flags = np.asarray(collision_flags, dtype=bool)
    a_t = np.asarray(a_t, dtype=float)
    assert len(times) == len(collision_flags) == len(a_t) == len(interaction_type_names)

    theta_t = np.array([threshold_table.get(name) for name in interaction_type_names])
    event_mask = collision_flags | (a_t > theta_t)
    if not event_mask.any():
        return float("inf")
    first = int(np.argmax(event_mask))
    return float(times[first])


def critical_event_times_all_agents(
    times: np.ndarray,
    collision_flags_by_agent: Dict, a_t_by_agent: Dict,
    interaction_type_names_by_agent: Dict,
    threshold_table: ThresholdTable,
) -> Dict[str, float]:
    """T_evt^j for every agent j in one trajectory instance ("computed once
    per trajectory instance, per agent", S3.5.3)."""
    out = {}
    for j in collision_flags_by_agent:
        out[j] = critical_event_time(
            times,
            collision_flags_by_agent[j],
            a_t_by_agent[j],
            interaction_type_names_by_agent[j],
            threshold_table,
        )
    return out
