"""
gibson_dataset.py
-------------------------------
Gibson数据集的dataset类

数据组织形式
desdf/
├── gibson_f/
│   └── scene_name1/
│        └── desdf.npy
├── gibson_g/
└── gibson_t/
gibson_x/
├── scene_name1/   # 除gibson_t数据集之外，每个scene包含多条trajectory
│   ├── depth40.txt    # 40 条射线的gt_depth
│   ├── depth160.txt    # 160 条射线的gt_truth
│   ├── map.png    # 建筑物平面图
│   ├── poses.txt    # 每帧对应的camera pose
│   └── rgb/    # 每帧图像
"""
import os

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from src.utils.load_imgs import load_and_preprocess_imgs


# 适合 gibson_g 和 gibson_f ， gibson_t 要修改 图片路径
# gibson_f 共有 20843 组数据
class GibsonDataset(Dataset):
    """
    Args:
        L : Contains L frames per trajectories
        depth_suffix : gt_depth的txt文件，包括 depth40.txt 和 depth160.txt
    """
    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_suffix="depth40",
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.depth_suffix = depth_suffix
        # scene_start_id = [scene1_start_idx, scene2_start_idx, ...]
        self.scene_start_idx = []
        # gt_pose / gt_depth = [ scene[ frame(numpy) ], ...]
        self.gt_depth = []
        self.gt_pose = []
        self.rgbs = []
        self.load_scene_start_idx_and_depths_and_poses()
        # N : 数据集长度
        self.N = self.scene_start_idx[-1]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
        返回的tensor要在N那一维上反向，因为vggt将第一帧作为参考帧，而F3loc的将最后一帧作为参考帧
        return:
        data_dict:
            "depth": (N, 3, D) tensor
            "pose": (N, 3) tensor
            "img": (N, 3, H, W) tensor
        """
        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        data_dict = {}

        # get depths
        gt_depth = self.gt_depth[scene_idx][
            idx_within_scene * self.L: idx_within_scene * self.L + self.L
        ]
        gt_depth_stack = np.stack(gt_depth, axis=0)  # (N, ...)
        gt_depth = torch.from_numpy(gt_depth_stack)
        data_dict["depth"] = gt_depth.flip(0)

        # get poses
        poses = self.gt_pose[scene_idx][
            idx_within_scene * self.L: idx_within_scene * self.L + self.L
        ]
        poses_stack = np.stack(poses, axis=0)
        poses = torch.from_numpy(poses_stack)
        data_dict["pose"] = poses.flip(0)

        # get images
        img_path_list = []
        # 默认输入单帧图像，所以弃用下面的代码
        # for l in range(self.L):
        #     img_path = os.path.join(
        #         self.dataset_dir,
        #         scene_name,
        #         "rgb",
        #         str(idx_within_scene).zfill(5) + "-" + str(l) + ".png",
        #     )
        #     img_path_list.append(img_path)
        img_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "rgb",
            str(idx_within_scene).zfill(5) + "-0" + ".png",
        )
        img_path_list.append(img_path)
        # 返回tensor类型的数据 (N, 3, H, W)
        imgs = load_and_preprocess_imgs(img_path_list)
        data_dict["img"] = imgs.flip(0)

        return data_dict

    # path = dir + scene_name + depth_suffix.txt
    # gt_pose / gt_depth = [ scene[ frame(numpy) ], ...]
    # scene_start_id = [scene1_start_idx, scene2_start_idx, ...]
    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0

        for scene in self.scene_names:
            # read depth
            depth_file = os.path.join(
                self.dataset_dir, scene, self.depth_suffix + ".txt"
            )

            with open(depth_file, "r") as f:
                depth_txt = [line.strip() for line in f.readlines()]

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            traj_len = len(poses_txt)
            scene_depths = []
            scene_poses = []

            # 将每帧的 depth 和 pose 转换成 numpy， 并放到list中
            for state_id in range(traj_len):
                # depth
                depth = depth_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth]).astype(np.float32)
                scene_depths.append(depth)

                # pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

            start_idx += traj_len // self.L
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)

if __name__ == "__main__":
    dataset_dir = "/media/zhangcz-ubuntu/Seagate Basic/CL_Dataset/Gibson Floorplan Localization Dataset/gibson_f"
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = yaml.safe_load(f)
    dataset = GibsonDataset(
        dataset_dir,
        split["train"],
        L = 4,
    )

    print(len(dataset))
    print(dataset[0])