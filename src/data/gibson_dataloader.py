"""
gibson_dataloader.py
-------------------------------
GibsonDataset的基础上创建Dataloader
"""
import os

import yaml
from torch.utils.data import DataLoader, random_split

from src.data.gibson_dataset import GibsonDataset


def build_gibson_dataloader(
    dataset_dir,
    L = 1,
    depth_suffix="depth40",
    batch_size=4,
    num_workers=4,
):
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = yaml.safe_load(f)

    train_set = GibsonDataset(
        dataset_dir=dataset_dir,
        scene_names=split["train"],
        L = L,
        depth_suffix=depth_suffix,
    )

    eval_set = GibsonDataset(
        dataset_dir=dataset_dir,
        scene_names=split["val"],
        L = L,
        depth_suffix=depth_suffix,
    )

    test_set = GibsonDataset(
        dataset_dir=dataset_dir,
        scene_names=split["test"],
        L = L,
        depth_suffix=depth_suffix,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,  # 训练集一般要打乱
        num_workers=num_workers,
        pin_memory=True,  # 如果用GPU训练，建议开
        drop_last=True,  # 可以防止最后一个 batch 尺寸不一致
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, eval_loader, test_loader


def build_gibson_dataloader_ddp(
    dataset_dir,
    L=1,
    depth_suffix="depth40",
    batch_size=4,
    num_workers=4,
    world_size=1,
    rank=0,
):
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = yaml.safe_load(f)

    train_set = GibsonDataset(
        dataset_dir=dataset_dir,
        scene_names=split["train"],
        L=L,
        depth_suffix=depth_suffix,
    )

    eval_set = GibsonDataset(
        dataset_dir=dataset_dir,
        scene_names=split["val"],
        L=L,
        depth_suffix=depth_suffix,
    )

    test_set = GibsonDataset(
        dataset_dir=dataset_dir,
        scene_names=split["test"],
        L=L,
        depth_suffix=depth_suffix,
    )

    # 关键：训练集用 DistributedSampler
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True
    )

    # 验证 / 测试也可以用 sampler（或只在 rank0 跑）
    eval_sampler = DistributedSampler(
        eval_set, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = DistributedSampler(
        test_set, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,      # 注意：不再用 shuffle=
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        sampler=eval_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, eval_loader, test_loader
