"""v1 counterpart to `dataset.py`: torch Dataset/collate glue for the
ego/object/road feature schema in `features_v1.py`. Kept as a fully
separate module (not a parameter added to the existing `CARMDataset`) so
the original v0 pipeline/results are untouched.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DEFAULT_CONFIG, FutureRiskConfig
from .data_pipeline import PreprocessedInstance, RawTrajectoryInstance
from .features_v1 import D_CTX_V1, D_EGO, D_OBJ_SEQ, build_instance_batch_item_v1
from .training import Batch


class CARMDatasetV1(Dataset):
    def __init__(
        self,
        raw_instances: List[RawTrajectoryInstance],
        preprocessed_by_id: Dict[str, PreprocessedInstance],
        cfg: FutureRiskConfig = DEFAULT_CONFIG,
    ):
        self.cfg = cfg
        self.items = []
        for raw in raw_instances:
            pre = preprocessed_by_id.get(raw.instance_id)
            if pre is None or not raw.agent_boxes:
                continue
            self.items.append(build_instance_batch_item_v1(raw, pre, cfg))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def compute_pos_weight_v1(dataset: CARMDatasetV1) -> float:
    n_pos, n_neg = 0, 0
    for item in dataset.items:
        finite = np.isfinite(item["T_evt"])
        n_pos += int(finite.sum())
        n_neg += int((~finite).sum())
    if n_pos == 0:
        return 1.0
    return max(1.0, n_neg / n_pos)


def compute_risk_pos_weight_v1(dataset: CARMDatasetV1, pos_threshold: float = 0.5) -> float:
    n_pos = sum(1 for item in dataset.items if item["y_analytic"] > pos_threshold)
    n_neg = len(dataset.items) - n_pos
    if n_pos == 0:
        return 1.0
    return max(1.0, n_neg / n_pos)


def collate_batch_v1(
    items: List[dict], cfg: FutureRiskConfig = DEFAULT_CONFIG,
    pos_weight: float = 1.0, risk_pos_weight: float = 1.0,
) -> Batch:
    B = len(items)
    J = max(1, max(item["x_ego"].shape[0] for item in items))
    bin_edges = torch.tensor(cfg.bin_edges, dtype=torch.float32)

    x_ego = np.zeros((B, J, D_EGO), dtype=np.float32)
    X_obj = np.zeros((B, J, cfg.num_bins, D_OBJ_SEQ), dtype=np.float32)
    x_ctx_v1 = np.zeros((B, J, D_CTX_V1), dtype=np.float32)
    tau_phy_seq = np.full((B, J, cfg.num_bins), cfg.tau_phy_clip[1], dtype=np.float32)
    T_evt = np.full((B, J), np.inf, dtype=np.float32)
    agent_mask = np.zeros((B, J), dtype=bool)
    y_analytic = np.zeros((B,), dtype=np.float32)
    collision_label = np.zeros((B,), dtype=np.int64)
    instance_ids = []

    for b, item in enumerate(items):
        n = item["x_ego"].shape[0]
        if n > 0:
            x_ego[b, :n] = item["x_ego"]
            X_obj[b, :n] = item["X_obj"]
            x_ctx_v1[b, :n] = item["x_ctx_v1"]
            tau_phy_seq[b, :n] = item["tau_phy_seq"]
            T_evt[b, :n] = item["T_evt"]
            agent_mask[b, :n] = True
        y_analytic[b] = item["y_analytic"]
        collision_label[b] = item["collision_label"]
        instance_ids.append(item["instance_id"])

    # Reuse the same `Batch` dataclass as v0 -- field names x_kin/X_int/x_ctx
    # are generic tensor slots as far as CARM/training.py's per-agent
    # flattening logic is concerned; only their *semantic content* differs
    # here (ego info / object sequence / object+road context instead of the
    # original kin/int/ctx split), so no new dataclass is needed.
    batch = Batch(
        x_kin=torch.from_numpy(x_ego),
        X_int=torch.from_numpy(X_obj),
        x_ctx=torch.from_numpy(x_ctx_v1),
        tau_phy_seq=torch.from_numpy(tau_phy_seq),
        agent_mask=torch.from_numpy(agent_mask),
        T_evt=torch.from_numpy(T_evt),
        y_analytic=torch.from_numpy(y_analytic),
        bin_edges=bin_edges,
        pos_weight=pos_weight,
        risk_pos_weight=risk_pos_weight,
    )
    batch.instance_ids = instance_ids
    batch.collision_label = torch.from_numpy(collision_label)
    return batch


def make_collate_fn_v1(cfg: FutureRiskConfig = DEFAULT_CONFIG, pos_weight: float = 1.0, risk_pos_weight: float = 1.0):
    return lambda items: collate_batch_v1(items, cfg=cfg, pos_weight=pos_weight, risk_pos_weight=risk_pos_weight)
