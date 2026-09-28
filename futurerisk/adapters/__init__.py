"""Dataset-specific adapters producing `data_pipeline.RawTrajectoryInstance`.

Each adapter module owns exactly one dataset's on-disk schema; nothing else
in the package should need to know about tar.gz layouts, per-frame JSON
keys, or similar. See `bench2drive.py` for the reference implementation.
"""
