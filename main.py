from src.data.gibson_dataloader import build_gibson_dataloader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm   # ← 这里导入

if __name__ == "__main__":
    train_loader, eval_loader, test_loader = build_gibson_dataloader(
        dataset_dir="E:\CL_Dataset\Gibson Floorplan Localization Dataset\gibson_f",
        L = 4,
        depth_suffix="depth40",
        batch_size=128,
        num_workers=4,
    )

    depth_list = []

    # 训练集
    for data in tqdm(train_loader, desc="Loading train"):
        depth = data["depth"]
        depth_list.append(depth)

    # 验证集
    for data in tqdm(eval_loader, desc="Loading eval"):
        depth = data["depth"]
        depth_list.append(depth)

    # 测试集
    for data in tqdm(test_loader, desc="Loading test"):
        depth = data["depth"]
        depth_list.append(depth)

    # 展平 & Numpy
    flat_numpy = [t.reshape(-1).cpu().numpy() for t in depth_list]

    all_arr = np.concatenate(flat_numpy)
    global_min = all_arr.min()
    global_max = all_arr.max()

    print("全局最小值:", global_min)
    print("全局最大值:", global_max)

    # 可视化
    vec = all_arr
    idx = np.arange(len(vec))

    plt.figure(figsize=(12,4))
    plt.scatter(idx, vec, s=5)
    plt.title("1D Feature Vector Visualization")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.show()
