"""Regression test for the TRACE adapter, run against the actual cloned
TRACE benchmark (not synthetic data).

Skips itself (rather than failing) if the dataset isn't present at the
expected path -- mirrors test_bench2drive_adapter.py's convention.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from futurerisk.adapters.trace import (
    build_instances_from_directory, build_instances_from_scenario, preprocess_trace,
)
from futurerisk.data_pipeline import load_preprocessed_instance

DATASET_DIR = Path(__file__).resolve().parents[2] / "external" / "carla-benchmark" / "Benchmark"


def _skip(reason: str) -> bool:
    print(f"SKIPPED: {reason}")
    return True


def test_build_instances_from_one_scenario():
    if not DATASET_DIR.exists():
        return _skip(f"{DATASET_DIR} not present")
    sims = sorted((DATASET_DIR / "simulations").glob("simulation_*.json"))
    if not sims:
        return _skip(f"no simulation_*.json under {DATASET_DIR}")

    instances = build_instances_from_scenario(str(sims[0]))
    assert len(instances) > 0, "expected at least one windowed instance from a real scenario"

    for inst in instances:
        n = len(inst.times)
        assert len(inst.ego_box) == n
        assert inst.times[0] == 0.0
        for aid, box in inst.agent_boxes.items():
            assert len(box) == n, f"agent {aid} box length mismatch with shared time axis"
            assert aid in inst.agent_classes
            assert aid in inst.agent_collision_flags
            assert len(inst.agent_collision_flags[aid]) == n
        assert set(inst.agent_classes.values()) <= {"vehicle", "truck", "bus", "cyclist", "other"}


def test_no_teleports_and_finite_kinematics():
    if not DATASET_DIR.exists():
        return _skip(f"{DATASET_DIR} not present")
    instances = build_instances_from_directory(str(DATASET_DIR))
    assert len(instances) > 0

    max_seen_speed = 0.0
    for inst in instances:
        dt = inst.times[1] - inst.times[0] if len(inst.times) > 1 else 0.1
        for box in list(inst.agent_boxes.values()) + [inst.ego_box]:
            assert np.all(np.isfinite(box.x)) and np.all(np.isfinite(box.y))
            vx = np.gradient(box.x, dt)
            vy = np.gradient(box.y, dt)
            speed = np.hypot(vx, vy)
            max_seen_speed = max(max_seen_speed, float(speed.max()))
    # Bound set above TRACE's own reported max (simulation_510202/vehicle2:
    # speed_kmh=230.5 => 64.0 m/s, verified directly against the source
    # JSON, not a reconstruction artifact) -- a value far beyond this would
    # indicate a bad arc-length interpolation (e.g. wraparound /
    # duplicate-vertex division-by-zero) rather than a genuinely fast crash.
    assert max_seen_speed < 90.0, f"suspiciously high speed {max_seen_speed} m/s in reconstructed trajectory"


def test_at_least_some_scenarios_produce_a_positive_window():
    """Sanity check on the whole-scenario reconstruction + windowing: since
    every TRACE scenario is an annotated real collision, at least some
    windows across the 52 scenarios must show the two reconstructed boxes
    overlapping (judgment call 2's simultaneous-impact timing is doing its
    job) -- otherwise Table 1' would have zero positives and AUROC/AUPRC
    would be undefined."""
    if not DATASET_DIR.exists():
        return _skip(f"{DATASET_DIR} not present")
    instances = build_instances_from_directory(str(DATASET_DIR))
    n_positive = sum(1 for inst in instances if inst.agent_collision_flags["vehicle2"].any())
    assert n_positive > 0, "expected at least one window with a geometric collision across all TRACE scenarios"


def test_full_preprocessing_pipeline(tmp_output_dir: str = "/tmp/futurerisk_test_trace_preprocessed"):
    if not DATASET_DIR.exists():
        return _skip(f"{DATASET_DIR} not present")

    table = preprocess_trace(str(DATASET_DIR), tmp_output_dir)
    # Unlike Bench2Drive/DeepAccident's much larger sample, TRACE's ~52
    # scenarios rarely clear cfg.n_min_samples (500) per interaction type,
    # so most/all types legitimately merge into "other" (event_proxy.py's
    # documented behavior) -- only "other" is guaranteed present.
    valid_names = {"car-following", "crossing", "merging/lane-change", "oncoming", "other"}
    assert "other" in table.thresholds
    assert set(table.thresholds.keys()) <= valid_names

    cached = sorted(glob.glob(f"{tmp_output_dir}/*.npz"))
    assert len(cached) > 0, "expected at least one cached preprocessed instance"

    pre = load_preprocessed_instance(cached[0])
    assert pre.dataset == "TRACE"
    for aid in pre.tau_3d:
        assert len(pre.tau_3d[aid]) == len(pre.tau_phy[aid]) == len(pre.evasive_severity[aid])
        assert aid in pre.T_evt


if __name__ == "__main__":
    test_build_instances_from_one_scenario()
    test_no_teleports_and_finite_kinematics()
    test_at_least_some_scenarios_produce_a_positive_window()
    test_full_preprocessing_pipeline()
    print("TRACE adapter tests OK (or skipped where dataset absent).")
