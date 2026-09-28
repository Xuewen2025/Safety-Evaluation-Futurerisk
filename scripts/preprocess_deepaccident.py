#!/usr/bin/env python3
"""Build RawTrajectoryInstances from every DeepAccident motion .tfrecords
file and run the full S3.9 preprocessing pipeline (threshold fitting +
T_evt/tau_phy generation + caching), exactly as already done for
Bench2Drive-Mini and Bench2Drive-Base.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from futurerisk.adapters.deepaccident import build_instances_from_file, find_motion_files
from futurerisk.config import DEFAULT_CONFIG
from futurerisk.data_pipeline import preprocess_dataset

MOTION_DATA_DIR = "dataset/DeepAccident-motion"
OUTPUT_DIR = "dataset/DeepAccident-preprocessed"
LOGS_DIR = Path("dataset/logs")

# Keep only every 10th record per file (~10 sampled records/file instead of
# up to 100) -- unthrottled, all-4-roles x all-records built 78,244
# instances (21x Bench2Drive-Base's 3,679) and the job never finished
# (server rebooted mid-run). Consecutive records mostly share scenario_id
# (near-duplicate time-offset snapshots, see adapter docstring), so this
# trades bounded redundancy loss for a ~10x cut, landing in the same
# order of magnitude as Bench2Drive-Base for a fair mixed-training split.
RECORD_STRIDE = 10

log = logging.getLogger("preprocess_deepaccident")


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
    log_path = LOGS_DIR / f"preprocess_deepaccident_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")

    files = find_motion_files(MOTION_DATA_DIR)
    log.info(f"[1/2] Building RawTrajectoryInstances from {len(files)} .tfrecords files in {MOTION_DATA_DIR}"
             f" (record_stride={RECORD_STRIDE}) ...")
    t0 = time.time()
    instances = []
    for i, path in enumerate(files):
        instances.extend(build_instances_from_file(path, cfg=DEFAULT_CONFIG, record_stride=RECORD_STRIDE))
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            log.info(f"      ... {i+1}/{len(files)} files done, {len(instances)} instances so far"
                     f" ({time.time()-t0:.1f}s elapsed)")
    t1 = time.time()
    n_clips = len(set(i.instance_id.split("__")[0] for i in instances))
    n_agents_total = sum(len(i.agent_boxes) for i in instances)
    n_with_collision = sum(1 for i in instances if any(f.any() for f in i.agent_collision_flags.values()))
    log.info(f"      -> {len(instances)} instances from {n_clips} scenario files"
             f" ({n_agents_total} total agent-instance pairs, {n_with_collision} instances with a"
             f" geometrically-detected collision) in {t1-t0:.1f}s")

    log.info(f"[2/2] Fitting thresholds + generating T_evt/tau_phy + caching to {OUTPUT_DIR} ...")
    table = preprocess_dataset("DeepAccident", instances, OUTPUT_DIR, cfg=DEFAULT_CONFIG)
    t2 = time.time()
    log.info(f"      thresholds: {table.thresholds}")
    log.info(f"      merged_into_other: {table.merged_into_other}")
    log.info(f"      preprocessing pass took {t2-t1:.1f}s")

    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")
    log.info(f"Cached instances directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
