"""Regression test for the Bench2Drive adapter, run against the actual
downloaded Bench2Drive-Mini dataset (not synthetic data).

Skips itself (rather than failing) if the dataset isn't present at the
expected path -- this is meant to be run in an environment that has it
(e.g. after `dataset/Bench2Drive-mini/` has been populated per the
project's download instructions), not as a CI test with fixtures.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from futurerisk.adapters.bench2drive import (
    build_instances_from_clip, build_instances_from_directory, preprocess_bench2drive,
)
from futurerisk.data_pipeline import load_preprocessed_instance
from futurerisk.geometry import boxes_overlap

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset" / "Bench2Drive-mini"


def _skip(reason: str) -> bool:
    print(f"SKIPPED: {reason}")
    return True


def test_build_instances_from_one_clip():
    if not DATASET_DIR.exists():
        return _skip(f"{DATASET_DIR} not present")
    clips = sorted(DATASET_DIR.glob("*.tar.gz"))
    if not clips:
        return _skip(f"no .tar.gz clips under {DATASET_DIR}")

    instances = build_instances_from_clip(str(clips[0]))
    assert len(instances) > 0, "expected at least one windowed instance from a real clip"

    for inst in instances:
        n = len(inst.times)
        assert len(inst.ego_box) == n
        for aid, box in inst.agent_boxes.items():
            assert len(box) == n, f"agent {aid} box length mismatch with shared time axis"
            assert aid in inst.agent_classes
            assert aid in inst.agent_collision_flags
            assert len(inst.agent_collision_flags[aid]) == n
        # every agent-class value must be one we actually assign
        assert set(inst.agent_classes.values()) <= {"vehicle", "cyclist", "pedestrian"}


def test_no_id_recycling_teleports():
    """Regression test for the CARLA actor-id-reuse bug found while
    validating this adapter: no agent box should show a per-frame
    displacement implying a wildly unphysical speed."""
    if not DATASET_DIR.exists():
        return _skip(f"{DATASET_DIR} not present")
    instances = build_instances_from_directory(str(DATASET_DIR))
    assert len(instances) > 0

    max_seen_speed = 0.0
    for inst in instances:
        dt = inst.times[1] - inst.times[0] if len(inst.times) > 1 else 0.1
        for box in list(inst.agent_boxes.values()) + [inst.ego_box]:
            vx = np.gradient(box.x, dt)
            vy = np.gradient(box.y, dt)
            speed = np.hypot(vx, vy)
            max_seen_speed = max(max_seen_speed, float(speed.max()))
    # a genuine teleport (as originally observed: ~67m in one 0.1s frame,
    # i.e. ~670 m/s) would blow this far past any real vehicle/pedestrian
    assert max_seen_speed < 60.0, f"suspiciously high speed {max_seen_speed} m/s -- possible id-recycling regression"


def test_full_preprocessing_pipeline(tmp_output_dir: str = "/tmp/futurerisk_test_bench2drive_preprocessed"):
    if not DATASET_DIR.exists():
        return _skip(f"{DATASET_DIR} not present")

    table = preprocess_bench2drive(str(DATASET_DIR), tmp_output_dir)
    assert set(table.thresholds.keys()) == {
        "car-following", "crossing", "merging/lane-change", "oncoming", "other",
    }

    cached = sorted(glob.glob(f"{tmp_output_dir}/*.npz"))
    assert len(cached) > 0, "expected at least one cached preprocessed instance"

    pre = load_preprocessed_instance(cached[0])
    assert pre.dataset == "Bench2Drive"
    for aid in pre.tau_3d:
        assert len(pre.tau_3d[aid]) == len(pre.tau_phy[aid]) == len(pre.evasive_severity[aid])
        assert aid in pre.T_evt


if __name__ == "__main__":
    test_build_instances_from_one_clip()
    test_no_id_recycling_teleports()
    test_full_preprocessing_pipeline()
    print("Bench2Drive adapter tests OK (or skipped where dataset absent).")
