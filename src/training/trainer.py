# -*- coding: utf-8 -*-
"""
经典机器学习训练器 Trainer 类
包含：
- 模型训练
- 模型评估
- 模型保存
"""
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(model, train_loader, loss_fn, optimizer, epoches, device, log_dir):
    model.to(device).half()

    # 1. 创建 TensorBoard 写入器
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0  # 用来在 TensorBoard 上当作 x 轴

    for e in range(epoches):
        model.train()
        print("====={} / {} epoch=====".format(e + 1, epoches))
        pbar = tqdm(train_loader, desc=f"Epoch {e+1}/{epoches}", ncols=120)
        for i, data in enumerate(pbar):
            img = data["img"].to(device).half()
            depth = data["depth"][:, 0].to(device).half()

            out = model(img)
            loss = loss_fn(out, depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 2. 每个 step 记录一次 loss
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            if i % 20 == 0:
                print(f'epoch: {e + 1}, batch: {i + 1}, loss: {loss.item():.4f}')

        # 3. 也可以按 epoch 记录一次（可选）
        writer.add_scalar("epoch/train_loss_last_batch", loss.item(), e + 1)

        # 4. 保存模型
        torch.save(model, f"./checkpoints/train_model_{e}.pth")

    # 5. 训练结束关闭 writer
    writer.close()

def train_ddp(model, train_loader, loss_fn, optimizer, epoches, device, log_dir, rank):
    """model 已经是 DDP 包装过的，且在正确的 device 上。"""
    is_main = (rank == 0)
    model.to(device).half()

    # 只有 rank0 写 TensorBoard
    writer = SummaryWriter(log_dir=log_dir) if is_main else None
    global_step = 0

    for e in range(epoches):
        model.train()
        if is_main:
            print(f"====={e + 1} / {epoches} epoch=====")

        # 分布式 sampler 每个 epoch 设置一次种子
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(e)

        if is_main:
            pbar = tqdm(train_loader, desc=f"Epoch {e+1}/{epoches}", ncols=120)
        else:
            pbar = train_loader  # 其他 rank 不用 tqdm

        for i, data in enumerate(pbar):
            img = data["img"].to(device, non_blocking=True).half()
            depth = data["depth"][:, 0].to(device, non_blocking=True).half()

            out = model(img)
            loss = loss_fn(out, depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if is_main:
                # TensorBoard 只在 rank0 写
                writer.add_scalar("train/loss", loss.item(), global_step)
                global_step += 1

                if i % 20 == 0:
                    print(f'epoch: {e + 1}, batch: {i + 1}, loss: {loss.item():.4f}')

        # epoch 级别日志
        if is_main:
            writer.add_scalar("epoch/train_loss_last_batch", loss.item(), e + 1)
            # 只保存一次（注意 DDP 的保存方式）
            os.makedirs("./checkpoints", exist_ok=True)
            # 保存 state_dict，比直接 torch.save(model) 更推荐
            torch.save(
                model.module.state_dict(),  # DDP 包着的是真模型
                f"./checkpoints/train_model_{e}.pth",
            )

    if is_main and writer is not None:
        writer.close()
