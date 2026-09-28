#!/usr/bin/env python3
"""Build RawTrajectoryInstances from all 1000 Bench2Drive-Base clips and run
the full S3.9 preprocessing pipeline (threshold fitting + T_evt/tau_phy
generation + caching), exactly as already validated on Bench2Drive-Mini.

This is I/O-bound (each clip's .tar.gz is read+decompressed once to reach
its anno/*.json.gz frames) -- expect this to take on the order of an hour
for 1000 clips (~30-45 min just for `build_instances_from_directory`,
extrapolated from Mini's ~2.6s/clip; threshold fitting + per-instance
labeling afterward are pure in-memory numpy ops on already-parsed data, so
comparatively fast).
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from futurerisk.adapters.bench2drive import build_instances_from_directory
from futurerisk.config import DEFAULT_CONFIG
from futurerisk.data_pipeline import preprocess_dataset

DATASET_DIR = "dataset/Bench2Drive-Base"
OUTPUT_DIR = "dataset/Bench2Drive-Base-preprocessed"
LOGS_DIR = Path("dataset/logs")

log = logging.getLogger("preprocess_bench2drive_base")


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
    log_path = LOGS_DIR / f"preprocess_bench2drive_base_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")

    t0 = time.time()
    log.info(f"[1/2] Building RawTrajectoryInstances from every clip in {DATASET_DIR} ...")
    instances = build_instances_from_directory(DATASET_DIR)
    t1 = time.time()
    n_clips = len(set(i.instance_id.split("__w")[0] for i in instances))
    n_agents_total = sum(len(i.agent_boxes) for i in instances)
    log.info(f"      -> {len(instances)} windowed instances from {n_clips} clips"
             f" ({n_agents_total} total agent-instance pairs) in {t1-t0:.1f}s")

    empty = sum(1 for i in instances if not i.agent_boxes)
    log.info(f"      instances with zero tracked agents: {empty} (kept for record, dropped later by CARMDataset)")

    log.info(f"[2/2] Fitting thresholds + generating T_evt/tau_phy + caching to {OUTPUT_DIR} ...")
    table = preprocess_dataset("Bench2Drive", instances, OUTPUT_DIR, cfg=DEFAULT_CONFIG)
    t2 = time.time()
    log.info(f"      thresholds: {table.thresholds}")
    log.info(f"      merged_into_other: {table.merged_into_other}")
    log.info(f"      preprocessing pass took {t2-t1:.1f}s")

    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")
    log.info(f"Cached instances directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
