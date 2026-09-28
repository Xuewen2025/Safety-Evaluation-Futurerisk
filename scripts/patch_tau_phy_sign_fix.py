#!/usr/bin/env python3
"""One-off patch: regenerate ONLY `tau_phy` in every already-cached
PreprocessedInstance (Bench2Drive-Mini/Base and DeepAccident), after fixing
`data_pipeline._relative_closing_speed`'s sign bug (see its docstring for
the full derivation/evidence -- found via `ssm_baselines.py`'s PET/DRAC
unit tests, which asserted the correct physical sign and failed against
the pre-fix formula).

Deliberately NOT a full `preprocess_dataset()` re-run:
- Thresholds, T_evt, interaction_type, evasive_severity are all
  independent of v_rel_close/tau_phy (T_evt depends only on
  evasive_severity + collision flags; interaction_type depends only on
  heading/position) -- refitting them would be wasted work, and for
  Bench2Drive specifically would be actively *wrong*: reconstructing raw
  instances via `raw_instance_from_preprocessed` always sets
  `lane_graph_available=False` (the lane-change info was already folded
  into `interaction_type` at the *original* build time, from the real tar
  archives with real road_id/lane_id), so re-running full
  `preprocess_instance` on a cache-reconstructed raw instance would
  silently degrade Bench2Drive's own MERGING classification back to the
  no-lane-graph fallback. Patching tau_phy in isolation avoids that
  entirely.
- This also avoids re-parsing Bench2Drive-Base's 312GB of tar archives
  (30-45 min) and DeepAccident's threshold-refitting pass (33 min) for a
  fix that only touches one derived quantity -- both datasets already
  cache raw trajectories (ego_box/agent_boxes/agent_classes) precisely so
  a fix like this doesn't need to re-parse source archives.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from futurerisk.config import DEFAULT_CONFIG
from futurerisk.data_pipeline import _relative_closing_speed, _save_preprocessed_instance, load_preprocessed_instance
from futurerisk.safety_prior import analytic_safe_ttc_prior_for_class

DIRS = [
    "dataset/Bench2Drive-mini-preprocessed",
    "dataset/Bench2Drive-Base-preprocessed",
    "dataset/DeepAccident-preprocessed",
]
LOGS_DIR = Path("dataset/logs")
log = logging.getLogger("patch_tau_phy_sign_fix")


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


def patch_one(path: Path, cfg=DEFAULT_CONFIG) -> bool:
    pre = load_preprocessed_instance(str(path))
    if pre.ego_box is None or pre.agent_boxes is None:
        return False  # cached before raw-trajectory fields existed; can't patch here, needs full re-preprocess
    dt = cfg.dt
    for agent_id, agent_box in pre.agent_boxes.items():
        v_close = _relative_closing_speed(pre.ego_box, agent_box, dt)
        agent_class = (pre.agent_classes or {}).get(agent_id, "other")
        pre.tau_phy[agent_id] = analytic_safe_ttc_prior_for_class(v_close, agent_class, cfg=cfg)
    _save_preprocessed_instance(pre, path)
    return True


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"patch_tau_phy_sign_fix_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")
    t0 = time.time()

    for d in DIRS:
        dpath = Path(d)
        if not dpath.exists():
            log.info(f"skip {d} (not present)")
            continue
        files = sorted(dpath.glob("*.npz"))
        n_ok, n_skip = 0, 0
        t_d0 = time.time()
        for i, f in enumerate(files):
            if patch_one(f):
                n_ok += 1
            else:
                n_skip += 1
            if (i + 1) % 2000 == 0:
                log.info(f"  {d}: {i+1}/{len(files)} done ({time.time()-t_d0:.1f}s elapsed)")
        log.info(f"{d}: patched {n_ok}, skipped (no raw cache, needs full re-preprocess) {n_skip},"
                 f" total {len(files)} in {time.time()-t_d0:.1f}s")

    log.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
