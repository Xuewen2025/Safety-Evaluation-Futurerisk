#!/usr/bin/env python3
"""Extract DeepAccident's own ground-truth collision labels
(collision_intensity/collision_ego_vehicle_id/collision_other_vehicle_id)
per instance and save them as a companion JSON next to the already-cached
preprocessed instances -- an independent label from our own tau_3D-derived
geometric collision flags, needed to fix the tautology described in
evaluate_crossdomain_deepaccident.py's findings (y_analytic/raw-TTC scores
are functions of tau_3D, so scoring them against a tau_3D-derived label is
circular).

This only re-runs the (fast, ~5min) raw-instance-building pass -- NOT the
expensive (~33min) threshold-fitting/tau_3D pass, since native labels don't
depend on any of that. Must use the same record_stride as the cached
preprocessing run so instance_ids line up 1:1.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from futurerisk.adapters.deepaccident import build_instances_from_file, collect_native_collision_labels, find_motion_files
from futurerisk.config import DEFAULT_CONFIG

MOTION_DATA_DIR = "dataset/DeepAccident-motion"
OUTPUT_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
RECORD_STRIDE = 10  # must match scripts/preprocess_deepaccident.py's RECORD_STRIDE
LOGS_DIR = Path("dataset/logs")

log = logging.getLogger("extract_deepaccident_native_labels")


def setup_logging(log_path: Path) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"extract_deepaccident_native_labels_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")

    files = find_motion_files(MOTION_DATA_DIR)
    log.info(f"Building raw instances from {len(files)} files (record_stride={RECORD_STRIDE}) ...")
    t0 = time.time()
    instances = []
    for i, path in enumerate(files):
        instances.extend(build_instances_from_file(path, cfg=DEFAULT_CONFIG, record_stride=RECORD_STRIDE))
        if (i + 1) % 40 == 0 or i + 1 == len(files):
            log.info(f"  ... {i+1}/{len(files)} files, {len(instances)} instances so far")

    labels = collect_native_collision_labels(instances)
    n_pos = sum(1 for v in labels.values() if v["collision_label"] == 1)
    log.info(f"Collected native labels for {len(labels)} instances, {n_pos} ({100*n_pos/len(labels):.1f}%)"
             f" with collision_label=1")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(labels, f)
    log.info(f"Wrote {OUTPUT_PATH}")
    log.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
