"""Regression test for the DeepAccident adapter, run against the actual
downloaded DeepAccident motion-data export (not synthetic data).

Skips itself (rather than failing) if the dataset isn't present at the
expected path -- same pattern as test_bench2drive_adapter.py.

Focuses in particular on the lane-graph integration (this module's
`_parse_lane_centerline`/`_nearest_lane_ids` additions): before this,
`lane_graph_available` was hardcoded False for every DeepAccident instance,
which made the MERGING interaction type structurally unreachable
(`classify_interaction_type`'s no-lane-graph fallback never returns it).
"""
from __future__ import annotations

from pathlib import Path

from futurerisk.adapters.deepaccident import build_instances_from_file, find_motion_files
from futurerisk.config import DEFAULT_CONFIG
from futurerisk.data_pipeline import fit_thresholds_for_dataset, preprocess_instance
from futurerisk.interaction import NAME_TO_TYPE

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset" / "DeepAccident-motion"


def _skip(reason: str) -> bool:
    print(f"SKIPPED: {reason}")
    return True


def test_lane_graph_available_and_merging_reachable():
    if not DATASET_DIR.exists():
        return _skip(f"{DATASET_DIR} not present")
    files = find_motion_files(str(DATASET_DIR))
    if not files:
        return _skip(f"no .tfrecords files under {DATASET_DIR}")

    instances = []
    for f in files[:5]:
        instances.extend(build_instances_from_file(f, cfg=DEFAULT_CONFIG, record_stride=10))
    assert len(instances) > 0, "expected at least one instance from the first few files"

    n_with_lane_graph = sum(1 for inst in instances if inst.lane_graph_available)
    assert n_with_lane_graph > 0, (
        "expected at least some instances to have a usable roadgraph "
        "(lane_graph_available=True); if this regresses to 0, the "
        "type==1 lane-centerline heuristic likely stopped matching this "
        "export's roadgraph_samples/type encoding"
    )
    for inst in instances:
        if inst.lane_graph_available:
            assert inst.ego_lane_change_times is not None
            assert inst.agent_lane_change_times is not None

    tbl = fit_thresholds_for_dataset("DeepAccident-test", instances, cfg=DEFAULT_CONFIG)
    seen_types = set()
    for inst in instances:
        pre = preprocess_instance(inst, tbl, cfg=DEFAULT_CONFIG)
        for types in pre.interaction_type.values():
            seen_types.update(int(t) for t in types)

    assert NAME_TO_TYPE["merging/lane-change"] in seen_types, (
        "MERGING should now be reachable now that lane_graph_available=True "
        "for at least some instances -- if this fails, the lane-graph wiring "
        "regressed back to always taking S3.3's no-lane-graph fallback path"
    )
