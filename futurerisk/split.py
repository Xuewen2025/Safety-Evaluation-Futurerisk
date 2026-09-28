"""Train/val/test split utilities.

S3.9 describes a *multi-dataset* split scheme (nuPlan + Bench2Drive as the
training pool, DeepAccident held out entirely for the cross-domain test,
S3.10). Right now only Bench2Drive-Mini has been downloaded, so this module
instead splits *within* Bench2Drive at the clip level -- once nuPlan and
DeepAccident are added, DeepAccident's instances should simply bypass this
split entirely (never enter train/val) and be evaluated with
`evaluation.py` per S3.10, while this function keeps handling the
train/val/test division inside the training pool.

Split granularity: by **clip**, not by instance/window. Windows sliced
from the same clip (`adapters/bench2drive.build_instances_from_clip`) share
the same ego trajectory context and overlapping/neighboring agents -- a
window-level random split would leak information between train/val/test
(adjacent windows differing only by a partial frame shift landing in
different splits).
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence


def clip_name_from_instance_id(instance_id: str) -> str:
    """Inverse of `adapters.bench2drive`'s `f"{clip_name}__w{frame:05d}"`
    instance_id convention."""
    return instance_id.split("__w")[0]


def split_by_clip(
    clip_names: Sequence[str], train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 0,
) -> Dict[str, List[str]]:
    """Deterministically (given `seed`) shuffle and split clip names into
    train/val/test. With very few clips (as in Bench2Drive-Mini's 10),
    guarantees at least one clip in val and test whenever n >= 3."""
    names = sorted(set(clip_names))
    n = len(names)
    rng = random.Random(seed)
    rng.shuffle(names)

    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n < 3:
        # too few clips to hold out anything meaningfully -- everything to train
        return {"train": names, "val": [], "test": []}

    n_train = max(1, min(round(n * train_frac), n - 2))
    n_val = max(1, min(round(n * val_frac), n - n_train - 1))
    n_test = n - n_train - n_val
    return {
        "train": names[:n_train],
        "val": names[n_train:n_train + n_val],
        "test": names[n_train + n_val:],
    }


def split_instances_by_clip(
    instance_ids: Sequence[str], train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 0,
) -> Dict[str, List[str]]:
    """Same as `split_by_clip`, but takes instance ids and returns
    instance-id lists per split (grouping windows by their parent clip
    first, so the split stays clip-level under the hood)."""
    clip_names = {clip_name_from_instance_id(iid) for iid in instance_ids}
    clip_split = split_by_clip(clip_names, train_frac=train_frac, val_frac=val_frac, seed=seed)
    clip_to_split = {c: split for split, clips in clip_split.items() for c in clips}
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for iid in instance_ids:
        out[clip_to_split[clip_name_from_instance_id(iid)]].append(iid)
    return out
