#!/usr/bin/env python3
"""Build RawTrajectoryInstances from all 52 TRACE (github.com/
NahianSalsabil/carla-benchmark) scenarios and run the full S3.9
preprocessing pipeline (threshold fitting + T_evt/tau_phy generation +
caching), same convention as `preprocess_bench2drive_base.py`.

TRACE is a kinematic *reconstruction* from sparse scenario JSON (see
`futurerisk/adapters/trace.py`'s module docstring for every judgment call
involved), not a native recording -- this is cheap/fast (pure numpy, no
archive decompression), so this script runs in seconds, not the ~hour
Bench2Drive-Base needs.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from futurerisk.adapters.trace import build_instances_from_directory
from futurerisk.config import DEFAULT_CONFIG
from futurerisk.data_pipeline import preprocess_dataset

DATASET_DIR = "external/carla-benchmark/Benchmark"
OUTPUT_DIR = "dataset/TRACE-preprocessed"
LOGS_DIR = Path("dataset/logs")

log = logging.getLogger("preprocess_trace")


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
    log_path = LOGS_DIR / f"preprocess_trace_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")

    t0 = time.time()
    log.info(f"[1/2] Building RawTrajectoryInstances (kinematic reconstruction) from every scenario in {DATASET_DIR} ...")
    instances = build_instances_from_directory(DATASET_DIR)
    t1 = time.time()
    n_scenarios = len(set(i.instance_id.split("__w")[0] for i in instances))
    n_positive = sum(1 for i in instances if i.agent_collision_flags["vehicle2"].any())
    log.info(f"      -> {len(instances)} windowed instances from {n_scenarios} scenarios"
             f" ({n_positive} positive / {len(instances)-n_positive} negative) in {t1-t0:.1f}s")
    log.info("      NOTE: instances from the same scenario are correlated (52 independent"
             " underlying events, not one per window) -- account for this in any significance test.")

    log.info(f"[2/2] Fitting thresholds + generating T_evt/tau_phy + caching to {OUTPUT_DIR} ...")
    table = preprocess_dataset("TRACE", instances, OUTPUT_DIR, cfg=DEFAULT_CONFIG)
    t2 = time.time()
    log.info(f"      thresholds: {table.thresholds}")
    log.info(f"      merged_into_other: {table.merged_into_other}")
    log.info(f"      preprocessing pass took {t2-t1:.1f}s")

    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")
    log.info(f"Cached instances directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
